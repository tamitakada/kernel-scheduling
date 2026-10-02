#define _GNU_SOURCE
#include "shared_state.h"
#include <cuda.h>
#include <cupti.h>
#include <nvrtc.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

static SchedulerSharedState* g_state = NULL;

/* ======================================================================
 * EXPORTED API: called from Python via ctypes.CDLL("./agent.so")
 * ====================================================================== */

void register_weight_region(const char* tag, void* ptr, size_t nbytes) {
    pthread_mutex_lock(&g_state->mutex);
    if (g_state->region_count < MAX_REGIONS) {
        int i = g_state->region_count++;
        g_state->regions[i].owner_pid = getpid();
        strncpy(g_state->regions[i].tag, tag, NAME_LEN - 1);
        g_state->regions[i].ptr = ptr;
        g_state->regions[i].nbytes = nbytes;
        g_state->regions[i].active = 1;
        fprintf(stderr, "[agent pid=%d] registered region tag=%s ptr=%p size=%zu\n",
                getpid(), tag, ptr, nbytes);
    } else {
        fprintf(stderr, "[agent pid=%d] WARNING: region table full, dropped tag=%s\n",
                getpid(), tag);
    }
    pthread_mutex_unlock(&g_state->mutex);
}

static int find_own_region(const char* tag, void** out_ptr, size_t* out_nbytes) {
    int found = 0;
    pid_t mypid = getpid();
    pthread_mutex_lock(&g_state->mutex);
    for (int i = 0; i < g_state->region_count; i++) {
        if (g_state->regions[i].active &&
            g_state->regions[i].owner_pid == mypid &&
            strcmp(g_state->regions[i].tag, tag) == 0) {
            *out_ptr = g_state->regions[i].ptr;
            *out_nbytes = g_state->regions[i].nbytes;
            found = 1;
            break;
        }
    }
    pthread_mutex_unlock(&g_state->mutex);
    return found;
}

/* ======================================================================
 * NVTX layer tagging -- same mechanism as before. Verify this resolves via
 * plain LD_PRELOAD in your build the same way you checked cudaLaunchKernel:
 *   nm -D libtorch*.so | grep -i nvtxRangePush
 * ====================================================================== */

#define MAX_NVTX_STACK 32
static __thread char t_nvtx_stack[MAX_NVTX_STACK][NAME_LEN];
static __thread int  t_nvtx_depth = 0;

typedef int (*nvtxRangePushA_t)(const char*);
static nvtxRangePushA_t real_nvtxRangePushA = NULL;

int nvtxRangePushA(const char* message) {
    if (!real_nvtxRangePushA) {
        real_nvtxRangePushA = (nvtxRangePushA_t)dlsym(RTLD_NEXT, "nvtxRangePushA");
    }
    if (t_nvtx_depth < MAX_NVTX_STACK) {
        strncpy(t_nvtx_stack[t_nvtx_depth], message, NAME_LEN - 1);
        t_nvtx_depth++;
    }
    return real_nvtxRangePushA ? real_nvtxRangePushA(message) : 0;
}

typedef int (*nvtxRangePop_t)(void);
static nvtxRangePop_t real_nvtxRangePop = NULL;

int nvtxRangePop(void) {
    if (!real_nvtxRangePop) {
        real_nvtxRangePop = (nvtxRangePop_t)dlsym(RTLD_NEXT, "nvtxRangePop");
    }
    if (t_nvtx_depth > 0) t_nvtx_depth--;
    return real_nvtxRangePop ? real_nvtxRangePop() : 0;
}

static const char* current_layer_tag(void) {
    return t_nvtx_depth > 0 ? t_nvtx_stack[t_nvtx_depth - 1] : "unknown";
}

// Set while WE are launching a kernel of our own (prefetch) through the
// real cuLaunchKernel, so cupti_callback can recognize the recursive
// re-entry and skip scheduling logic for it.
static __thread int t_in_scheduler_launch = 0;

/* ======================================================================
 * GPU IDLE TRACKING: the daemon increments gpu_inflight_count itself, at
 * the moment it releases a slot (see scheduler_daemon.c -- that's what
 * avoids a race). This callback is enqueued via cuLaunchHostFunc right
 * after the REAL kernel is actually submitted, and fires once that
 * specific kernel has genuinely finished executing -- at which point we
 * decrement, and if that brings the count to zero, wake the daemon so it
 * can release the next pending kernel.
 *
 * IMPORTANT: functions passed to cuLaunchHostFunc run on an internal CUDA
 * driver thread. They must be fast and must NOT make any CUDA API calls.
 * An atomic decrement + sem_post is fine; nothing else should go here.
 * ====================================================================== */

static void CUDA_CB launch_completion_callback(void* userdata) {
    (void)userdata;
    int remaining = atomic_fetch_sub(&g_state->gpu_inflight_count, 1) - 1;
    if (remaining == 0) {
        sem_post(&g_state->new_request_sem); // wake the daemon to re-check idle gate
    }
}

/* ======================================================================
 * PREFETCH KERNEL (NVRTC-compiled once) + scheduler-owned stream pool.
 * Executed locally by whichever agent the daemon instructs -- never by the
 * daemon itself, which has no CUDA context at all.
 * ====================================================================== */

static const char* k_prefetch_src =
"extern \"C\" __global__ void l2_prefetch_kernel(\n"
"    const float4* __restrict__ src, size_t n_float4, unsigned int* __restrict__ scratch) {\n"
"    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;\n"
"    size_t stride = (size_t)gridDim.x * blockDim.x;\n"
"    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);\n"
"    for (size_t i = idx; i < n_float4; i += stride) {\n"
"        float4 v = src[i];\n"
"        acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;\n"
"    }\n"
"    if (threadIdx.x == 0 && blockIdx.x == 0) {\n"
"        atomicAdd(scratch, (unsigned int)(acc.x + acc.y + acc.z + acc.w));\n"
"    }\n"
"}\n";

static CUfunction  g_prefetch_kernel_func = NULL;
static CUmodule    g_prefetch_module = NULL;
static CUdeviceptr g_prefetch_scratch = 0;

static void ensure_prefetch_kernel_compiled(void) {
    if (g_prefetch_kernel_func) return;

    nvrtcProgram prog;
    nvrtcCreateProgram(&prog, k_prefetch_src, "prefetch.cu", 0, NULL, NULL);
    if (nvrtcCompileProgram(prog, 0, NULL) != NVRTC_SUCCESS) {
        size_t log_size;
        nvrtcGetProgramLogSize(prog, &log_size);
        char* log = (char*)malloc(log_size);
        nvrtcGetProgramLog(prog, log);
        fprintf(stderr, "[agent pid=%d] NVRTC compile failed:\n%s\n", getpid(), log);
        free(log);
        nvrtcDestroyProgram(&prog);
        return;
    }

    size_t ptx_size;
    nvrtcGetPTXSize(prog, &ptx_size);
    char* ptx = (char*)malloc(ptx_size);
    nvrtcGetPTX(prog, ptx);
    nvrtcDestroyProgram(&prog);

    cuModuleLoadDataEx(&g_prefetch_module, ptx, 0, NULL, NULL);
    cuModuleGetFunction(&g_prefetch_kernel_func, g_prefetch_module, "l2_prefetch_kernel");
    free(ptx);

    cuMemAlloc(&g_prefetch_scratch, sizeof(unsigned int));
    fprintf(stderr, "[agent pid=%d] prefetch kernel compiled\n", getpid());
}

#define NUM_PREFETCH_STREAMS 4
static CUstream g_prefetch_streams[NUM_PREFETCH_STREAMS];
static int g_prefetch_streams_ready = 0;

static void ensure_prefetch_streams(void) {
    if (g_prefetch_streams_ready) return;
    for (int i = 0; i < NUM_PREFETCH_STREAMS; i++) {
        cuStreamCreate(&g_prefetch_streams[i], CU_STREAM_NON_BLOCKING);
    }
    g_prefetch_streams_ready = 1;
}

static void inject_prefetch(const char* region_tag) {
    void* ptr; size_t nbytes;
    if (!find_own_region(region_tag, &ptr, &nbytes)) return;

    ensure_prefetch_kernel_compiled();
    ensure_prefetch_streams();
    if (!g_prefetch_kernel_func) return;

    static int rr = 0;
    int stream_idx = (rr++) % NUM_PREFETCH_STREAMS;
    CUstream stream = g_prefetch_streams[stream_idx];

    CUdeviceptr target = (CUdeviceptr)(uintptr_t)ptr;
    size_t n_float4 = nbytes / sizeof(float4);
    void* args[] = { &target, &n_float4, &g_prefetch_scratch };

    int threads_per_block = 256;
    int num_blocks = (int)((n_float4 + threads_per_block - 1) / threads_per_block);
    if (num_blocks < 1) num_blocks = 1;
    if (num_blocks > 2048) num_blocks = 2048;

    // REENTRANCY GUARD: this cuLaunchKernel call would otherwise be caught
    // by our OWN CUPTI subscriber (it hooks the symbol globally, not just
    // calls originating from the app) and recurse into request_launch_
    // permission -- which could deadlock or spiral into the scheduler
    // deciding to prefetch a prefetch. Set the flag so cupti_callback skips
    // scheduling logic for this specific call.
    t_in_scheduler_launch = 1;
    cuLaunchKernel(g_prefetch_kernel_func,
                   num_blocks, 1, 1,
                   threads_per_block, 1, 1,
                   0, stream, args, NULL);
    t_in_scheduler_launch = 0;

    fprintf(stderr, "[agent pid=%d] injected prefetch tag=%s stream_idx=%d\n",
            getpid(), region_tag, stream_idx);
}

/* ======================================================================
 * CLIENT SIDE OF THE QUEUE: register this launch with the daemon, block
 * until released, then execute any prefetch instruction it attached.
 * ====================================================================== */

static void request_launch_permission(const char* kernel_name, const char* layer_tag) {
    pthread_mutex_lock(&g_state->mutex);
    int slot = -1;
    for (int i = 0; i < MAX_PENDING; i++) {
        if (!g_state->slots[i].active) { slot = i; break; }
    }
    if (slot == -1) {
        pthread_mutex_unlock(&g_state->mutex);
        fprintf(stderr, "[agent pid=%d] WARNING: pending queue full, launching unscheduled\n",
                getpid());
        return;
    }
    g_state->slots[slot].pid = getpid();
    strncpy(g_state->slots[slot].kernel_name, kernel_name, NAME_LEN - 1);
    strncpy(g_state->slots[slot].layer_tag, layer_tag, NAME_LEN - 1);
    g_state->slots[slot].active = 1;
    g_state->slots[slot].ready = 0;
    g_state->slots[slot].has_prefetch = 0;
    g_state->slots[slot].log_id = 0;
    pthread_mutex_unlock(&g_state->mutex);

    sem_post(&g_state->new_request_sem);   // wake the daemon
    sem_wait(&g_state->slots[slot].wake_sem); // block until IT releases THIS slot

    pthread_mutex_lock(&g_state->mutex);
    int has_prefetch = g_state->slots[slot].has_prefetch;
    char prefetch_tag[NAME_LEN];
    if (has_prefetch) strncpy(prefetch_tag, g_state->slots[slot].prefetch_tag, NAME_LEN - 1);
    g_state->slots[slot].active = 0;
    pthread_mutex_unlock(&g_state->mutex);

    if (has_prefetch) {
        inject_prefetch(prefetch_tag);
    }
}

/* ======================================================================
 * CUPTI CALLBACK
 * ====================================================================== */

static CUpti_SubscriberHandle g_subscriber;

static void CUPTIAPI cupti_callback(void* userdata, CUpti_CallbackDomain domain,
                                     CUpti_CallbackId cbid, const void* cbdata) {
    (void)userdata;
    if (domain != CUPTI_CB_DOMAIN_DRIVER_API) return;
    if (cbid != CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel &&
        cbid != CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx) return;

    // Skip our OWN prefetch launches entirely -- see inject_prefetch().
    if (t_in_scheduler_launch) return;

    const CUpti_CallbackData* data = (const CUpti_CallbackData*)cbdata;

    if (data->callbackSite == CUPTI_API_ENTER) {
        const char* layer_tag = current_layer_tag();
        const char* kernel_name = data->functionName ? data->functionName : "unknown";

        request_launch_permission(kernel_name, layer_tag);
        // returning here lets CUPTI/the driver proceed with the real launch

        // fprintf(stderr, "START kernel\n");

        return;
    }

    if (data->callbackSite == CUPTI_API_EXIT) {
        // The real kernel has just been submitted to its stream. Extract
        // the stream and enqueue a host-side callback that fires once this
        // SPECIFIC kernel has actually finished, so we can decrement the
        // in-flight count at the right time (not at submission time).
        CUstream stream = 0;
        int resolved = 0;

        if (cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel) {
            // Confirmed to match generated_cuda_meta.h's cuLaunchKernel_params_st.
            typedef struct { CUfunction f; unsigned int gx,gy,gz,bx,by,bz; unsigned int shmem;
                             CUstream hStream; void** kp; void** extra; } cuLaunchKernel_params_min;
            stream = ((cuLaunchKernel_params_min*)data->functionParams)->hStream;
            resolved = 1;
        } else if (cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx) {
            // Confirmed to match generated_cuda_meta.h's cuLaunchKernelEx_params_st.
            typedef struct { const CUlaunchConfig* config; CUfunction f;
                              void** kernelParams; void** extra; } cuLaunchKernelEx_params_min;
            const CUlaunchConfig* cfg = ((cuLaunchKernelEx_params_min*)data->functionParams)->config;
            if (cfg) {
                stream = cfg->hStream;
                resolved = 1;
            }
        }

        if (resolved) {
            // fprintf(stderr, "END kernel\n");

            CUresult hf_res = cuLaunchHostFunc(stream, launch_completion_callback, NULL);
            if (hf_res != CUDA_SUCCESS) {
                fprintf(stderr, "[agent pid=%d] WARNING: cuLaunchHostFunc failed (%d) for "
                                "stream=%p cbid=%d, decrementing immediately\n",
                                getpid(), hf_res, (void*)stream, (int)cbid);
                launch_completion_callback(NULL);
            }
        } else {
            // Only happens for an unhandled cbid, or a NULL CUlaunchConfig*
            // pointer for the Ex path (genuinely anomalous)
            fprintf(stderr, "[agent pid=%d] WARNING: couldn't resolve stream for "
                            "completion tracking (cbid=%d), decrementing immediately\n",
                            getpid(), (int)cbid);
            launch_completion_callback(NULL);
        }
    }
}

/* ======================================================================
 * INIT: open (never create) the daemon's shared memory; retry if the
 * daemon hasn't started yet.
 * ====================================================================== */

static void open_shared_state(void) {
    int fd = -1;
    for (int attempt = 0; attempt < 100; attempt++) {
        fd = shm_open(SHM_NAME, O_RDWR, 0666);
        if (fd >= 0) break;
        if (attempt == 0) {
            fprintf(stderr, "[agent pid=%d] waiting for scheduler_daemon to start...\n", getpid());
        }
        usleep(100 * 1000);
    }
    if (fd < 0) {
        fprintf(stderr, "[agent pid=%d] FATAL: scheduler_daemon never started "
                        "(shm_open %s failed: %s)\n", getpid(), SHM_NAME, strerror(errno));
        exit(1);
    }

    g_state = (SchedulerSharedState*)mmap(NULL, sizeof(SchedulerSharedState),
                                          PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);

    while (!atomic_load(&g_state->initialized)) {
        usleep(1000);
    }
    fprintf(stderr, "[agent pid=%d] attached to scheduler_daemon\n", getpid());
}

__attribute__((constructor))
static void agent_init(void) {
    open_shared_state();

    CUptiResult res = cuptiSubscribe(&g_subscriber, (CUpti_CallbackFunc)cupti_callback, NULL);
    if (res != CUPTI_SUCCESS) {
        fprintf(stderr, "[agent pid=%d] FATAL: cuptiSubscribe failed: %d\n", getpid(), res);
        return;
    }
    cuptiEnableCallback(1, g_subscriber, CUPTI_CB_DOMAIN_DRIVER_API,
                        CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel);
    cuptiEnableCallback(1, g_subscriber, CUPTI_CB_DOMAIN_DRIVER_API,
                        CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx);

    fprintf(stderr, "[agent pid=%d] CUPTI subscriber attached\n", getpid());
}