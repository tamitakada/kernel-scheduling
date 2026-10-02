#define _GNU_SOURCE
#include "shared_state.hpp"
#include <cuda.h>
#include <cupti.h>
#include <nvrtc.h>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

static SchedulerSharedState* g_state = nullptr;

/* ======================================================================
 * EXPORTED API: called from Python via ctypes.CDLL("./agent.so"). Must be
 * extern "C" -- ctypes looks these up by their exact, unmangled name.
 * ====================================================================== */

extern "C" void register_weight_region(const char* tag, void* ptr, size_t nbytes) {
    pthread_mutex_lock(&g_state->mutex);
    if (g_state->region_count < MAX_REGIONS) {
        int i = g_state->region_count++;
        g_state->regions[i].owner_pid = getpid();
        std::strncpy(g_state->regions[i].tag, tag, NAME_LEN - 1);
        g_state->regions[i].ptr = ptr;
        g_state->regions[i].nbytes = nbytes;
        g_state->regions[i].active = 1;
        std::fprintf(stderr, "[agent pid=%d] registered region tag=%s ptr=%p size=%zu\n",
                getpid(), tag, ptr, nbytes);
    } else {
        std::fprintf(stderr, "[agent pid=%d] WARNING: region table full, dropped tag=%s\n",
                getpid(), tag);
    }
    pthread_mutex_unlock(&g_state->mutex);
}

static bool find_own_region(const char* tag, void** out_ptr, size_t* out_nbytes) {
    bool found = false;
    pid_t mypid = getpid();
    pthread_mutex_lock(&g_state->mutex);
    for (int i = 0; i < g_state->region_count; i++) {
        if (g_state->regions[i].active &&
            g_state->regions[i].owner_pid == mypid &&
            std::strcmp(g_state->regions[i].tag, tag) == 0) {
            *out_ptr = g_state->regions[i].ptr;
            *out_nbytes = g_state->regions[i].nbytes;
            found = true;
            break;
        }
    }
    pthread_mutex_unlock(&g_state->mutex);
    return found;
}

/* ======================================================================
 * NVTX layer tagging. extern "C" is required here for a different reason
 * than register_weight_region: these function names must match the real
 * exported symbols from libnvToolsExt exactly, unmangled, for LD_PRELOAD
 * interposition to work at all. A mangled C++ name here would mean this
 * code silently never replaces the real symbol -- no error, just no effect.
 * Verify this resolves in your build the same way you checked
 * cudaLaunchKernel earlier: nm -D libtorch*.so | grep -i nvtxRangePush
 * ====================================================================== */

constexpr int MAX_NVTX_STACK = 32;
thread_local char t_nvtx_stack[MAX_NVTX_STACK][NAME_LEN];
thread_local int  t_nvtx_depth = 0;

using nvtxRangePushA_t = int (*)(const char*);
static nvtxRangePushA_t real_nvtxRangePushA = nullptr;

extern "C" int nvtxRangePushA(const char* message) {
    if (!real_nvtxRangePushA) {
        real_nvtxRangePushA = reinterpret_cast<nvtxRangePushA_t>(dlsym(RTLD_NEXT, "nvtxRangePushA"));
    }
    if (t_nvtx_depth < MAX_NVTX_STACK) {
        std::strncpy(t_nvtx_stack[t_nvtx_depth], message, NAME_LEN - 1);
        t_nvtx_depth++;
    }
    return real_nvtxRangePushA ? real_nvtxRangePushA(message) : 0;
}

using nvtxRangePop_t = int (*)();
static nvtxRangePop_t real_nvtxRangePop = nullptr;

extern "C" int nvtxRangePop() {
    if (!real_nvtxRangePop) {
        real_nvtxRangePop = reinterpret_cast<nvtxRangePop_t>(dlsym(RTLD_NEXT, "nvtxRangePop"));
    }
    if (t_nvtx_depth > 0) t_nvtx_depth--;
    return real_nvtxRangePop ? real_nvtxRangePop() : 0;
}

static const char* current_layer_tag() {
    return t_nvtx_depth > 0 ? t_nvtx_stack[t_nvtx_depth - 1] : "unknown";
}

// Set while WE are launching a kernel of our own (prefetch) through the
// real cuLaunchKernel, so cupti_callback can recognize the recursive
// re-entry and skip scheduling logic for it entirely.
thread_local bool t_in_scheduler_launch = false;

// Stashed between the ENTER and EXIT invocations of CUPTI's callback for
// the SAME launch call (they fire synchronously, back to back, on the same
// thread), so the EXIT handler knows which inflight slot this kernel's
// completion should eventually clear.
thread_local int t_current_inflight_slot = -1;

/* ======================================================================
 * GPU completion tracking: the daemon reserves a slot in inflight[] at
 * release time and tells us its index via PendingSlot::inflight_slot. Once
 * the real kernel we were released for actually finishes executing, this
 * callback clears exactly that slot -- not a generic counter -- so the
 * daemon (or anything inspecting shared memory) can always tell precisely
 * which kernel, from which process, is still running.
 *
 * IMPORTANT: functions passed to cuLaunchHostFunc run on an internal CUDA
 * driver thread. They must be fast and must NOT make any CUDA API calls.
 * ====================================================================== */

static void CUDA_CB launch_completion_callback(void* userdata) {
    int idx = static_cast<int>(reinterpret_cast<intptr_t>(userdata));
    if (idx >= 0 && idx < MAX_INFLIGHT) {
        g_state->inflight[idx].active.store(0);
    }
    // Always wake the daemon to re-check the idle gate. It's possible other
    // inflight slots are still active -- the daemon will simply see that on
    // re-scan and not release anything yet; this is simpler and race-free
    // compared to trying to determine "am I the last one" without holding
    // the lock from inside a driver callback thread.
    sem_post(&g_state->new_request_sem);
}

/* ======================================================================
 * PREFETCH KERNEL (NVRTC-compiled once) + scheduler-owned stream pool.
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

static CUfunction  g_prefetch_kernel_func = nullptr;
static CUmodule    g_prefetch_module = nullptr;
static CUdeviceptr g_prefetch_scratch = 0;

static void ensure_prefetch_kernel_compiled() {
    if (g_prefetch_kernel_func) return;

    nvrtcProgram prog;
    nvrtcCreateProgram(&prog, k_prefetch_src, "prefetch.cu", 0, nullptr, nullptr);
    if (nvrtcCompileProgram(prog, 0, nullptr) != NVRTC_SUCCESS) {
        size_t log_size;
        nvrtcGetProgramLogSize(prog, &log_size);
        char* log = static_cast<char*>(std::malloc(log_size));
        nvrtcGetProgramLog(prog, log);
        std::fprintf(stderr, "[agent pid=%d] NVRTC compile failed:\n%s\n", getpid(), log);
        std::free(log);
        nvrtcDestroyProgram(&prog);
        return;
    }

    size_t ptx_size;
    nvrtcGetPTXSize(prog, &ptx_size);
    char* ptx = static_cast<char*>(std::malloc(ptx_size));
    nvrtcGetPTX(prog, ptx);
    nvrtcDestroyProgram(&prog);

    cuModuleLoadDataEx(&g_prefetch_module, ptx, 0, nullptr, nullptr);
    cuModuleGetFunction(&g_prefetch_kernel_func, g_prefetch_module, "l2_prefetch_kernel");
    std::free(ptx);

    cuMemAlloc(&g_prefetch_scratch, sizeof(unsigned int));
    std::fprintf(stderr, "[agent pid=%d] prefetch kernel compiled\n", getpid());
}

constexpr int NUM_PREFETCH_STREAMS = 4;
static CUstream g_prefetch_streams[NUM_PREFETCH_STREAMS];
static bool g_prefetch_streams_ready = false;

static void ensure_prefetch_streams() {
    if (g_prefetch_streams_ready) return;
    for (auto& stream : g_prefetch_streams) {
        cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING);
    }
    g_prefetch_streams_ready = true;
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

    CUdeviceptr target = reinterpret_cast<CUdeviceptr>(ptr);
    size_t n_float4 = nbytes / sizeof(float4);
    void* args[] = { &target, &n_float4, &g_prefetch_scratch };

    int threads_per_block = 256;
    int num_blocks = static_cast<int>((n_float4 + threads_per_block - 1) / threads_per_block);
    if (num_blocks < 1) num_blocks = 1;
    if (num_blocks > 2048) num_blocks = 2048;

    // REENTRANCY GUARD: this cuLaunchKernel call would otherwise be caught
    // by our OWN CUPTI subscriber and recurse into request_launch_
    // permission. Skip scheduling logic for it entirely.
    t_in_scheduler_launch = true;
    cuLaunchKernel(g_prefetch_kernel_func,
                   num_blocks, 1, 1,
                   threads_per_block, 1, 1,
                   0, stream, args, nullptr);
    t_in_scheduler_launch = false;

    std::fprintf(stderr, "[agent pid=%d] injected prefetch tag=%s stream_idx=%d\n",
            getpid(), region_tag, stream_idx);
}

/* ======================================================================
 * CLIENT SIDE OF THE QUEUE: register this launch with the daemon, block
 * until released, record the inflight slot the daemon reserved for us, and
 * execute any prefetch instruction it attached.
 * ====================================================================== */

static void request_launch_permission(const char* kernel_name, const char* layer_tag) {
    pthread_mutex_lock(&g_state->mutex);
    int slot = -1;
    for (int i = 0; i < MAX_PENDING; i++) {
        if (!g_state->slots[i].active) { slot = i; break; }
    }
    if (slot == -1) {
        pthread_mutex_unlock(&g_state->mutex);
        std::fprintf(stderr, "[agent pid=%d] WARNING: pending queue full, launching unscheduled\n",
                getpid());
        t_current_inflight_slot = -1;
        return;
    }
    g_state->slots[slot].pid = getpid();
    std::strncpy(g_state->slots[slot].kernel_name, kernel_name, NAME_LEN - 1);
    std::strncpy(g_state->slots[slot].layer_tag, layer_tag, NAME_LEN - 1);
    g_state->slots[slot].active = 1;
    g_state->slots[slot].ready = 0;
    g_state->slots[slot].has_prefetch = 0;
    g_state->slots[slot].inflight_slot = -1;
    pthread_mutex_unlock(&g_state->mutex);

    sem_post(&g_state->new_request_sem);     // wake the daemon
    sem_wait(&g_state->slots[slot].wake_sem); // block until IT releases THIS slot

    pthread_mutex_lock(&g_state->mutex);
    int has_prefetch = g_state->slots[slot].has_prefetch;
    char prefetch_tag[NAME_LEN];
    if (has_prefetch) std::strncpy(prefetch_tag, g_state->slots[slot].prefetch_tag, NAME_LEN - 1);
    t_current_inflight_slot = g_state->slots[slot].inflight_slot; // carry through to EXIT
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

    const CUpti_CallbackData* data = static_cast<const CUpti_CallbackData*>(cbdata);

    if (data->callbackSite == CUPTI_API_ENTER) {
        const char* layer_tag = current_layer_tag();
        const char* kernel_name = data->functionName ? data->functionName : "unknown";

        request_launch_permission(kernel_name, layer_tag);
        // returning here lets CUPTI/the driver proceed with the real launch
        return;
    }

    if (data->callbackSite == CUPTI_API_EXIT) {
        // The real kernel has just been submitted to its stream. Extract
        // the stream and enqueue a host-side callback that fires once this
        // SPECIFIC kernel has actually finished, carrying the inflight
        // slot index through so the right slot gets cleared.
        //
        // Confirmed against generated_cuda_meta.h for CUDA 13.1: both
        // layouts below match the real cuLaunchKernel_params_st /
        // cuLaunchKernelEx_params_st exactly.
        CUstream stream = 0;
        bool resolved = false;

        if (cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel) {
            struct cuLaunchKernel_params_min {
                CUfunction f; unsigned int gx, gy, gz, bx, by, bz; unsigned int shmem;
                CUstream hStream; void** kp; void** extra;
            };
            stream = reinterpret_cast<const cuLaunchKernel_params_min*>(data->functionParams)->hStream;
            resolved = true;
        } else if (cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx) {
            struct cuLaunchKernelEx_params_min {
                const CUlaunchConfig* config; CUfunction f; void** kernelParams; void** extra;
            };
            const CUlaunchConfig* cfg =
                reinterpret_cast<const cuLaunchKernelEx_params_min*>(data->functionParams)->config;
            if (cfg) {
                stream = cfg->hStream;
                resolved = true;
            }
        }

        void* userdata_for_completion = reinterpret_cast<void*>(
            static_cast<intptr_t>(t_current_inflight_slot));

        if (resolved) {
            CUresult hf_res = cuLaunchHostFunc(stream, launch_completion_callback, userdata_for_completion);
            if (hf_res != CUDA_SUCCESS) {
                // NOTE: stream value 0 (the default/legacy stream) is
                // perfectly legitimate -- success here is tracked via
                // `resolved`, never by truthiness of stream itself.
                std::fprintf(stderr, "[agent pid=%d] WARNING: cuLaunchHostFunc failed (%d) for "
                                "stream=%p cbid=%d, clearing inflight slot immediately\n",
                                getpid(), hf_res, static_cast<void*>(stream), static_cast<int>(cbid));
                launch_completion_callback(userdata_for_completion);
            }
        } else {
            std::fprintf(stderr, "[agent pid=%d] WARNING: couldn't resolve stream for "
                            "completion tracking (cbid=%d), clearing inflight slot immediately\n",
                            getpid(), static_cast<int>(cbid));
            launch_completion_callback(userdata_for_completion);
        }
    }
}

/* ======================================================================
 * INIT: open (never create) the daemon's shared memory; retry if the
 * daemon hasn't started yet.
 * ====================================================================== */

static void open_shared_state() {
    int fd = -1;
    for (int attempt = 0; attempt < 100; attempt++) {
        fd = shm_open(SHM_NAME, O_RDWR, 0666);
        if (fd >= 0) break;
        if (attempt == 0) {
            std::fprintf(stderr, "[agent pid=%d] waiting for scheduler_daemon to start...\n", getpid());
        }
        usleep(100 * 1000);
    }
    if (fd < 0) {
        std::fprintf(stderr, "[agent pid=%d] FATAL: scheduler_daemon never started "
                        "(shm_open %s failed: %s)\n", getpid(), SHM_NAME, std::strerror(errno));
        std::exit(1);
    }

    g_state = static_cast<SchedulerSharedState*>(
        mmap(nullptr, sizeof(SchedulerSharedState), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
    close(fd);

    while (g_state->initialized.load() == 0) {
        usleep(1000);
    }
    std::fprintf(stderr, "[agent pid=%d] attached to scheduler_daemon\n", getpid());
}

__attribute__((constructor))
static void agent_init() {
    open_shared_state();

    CUptiResult res = cuptiSubscribe(&g_subscriber, (CUpti_CallbackFunc)cupti_callback, nullptr);
    if (res != CUPTI_SUCCESS) {
        std::fprintf(stderr, "[agent pid=%d] FATAL: cuptiSubscribe failed: %d\n", getpid(), res);
        return;
    }
    cuptiEnableCallback(1, g_subscriber, CUPTI_CB_DOMAIN_DRIVER_API,
                        CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel);
    cuptiEnableCallback(1, g_subscriber, CUPTI_CB_DOMAIN_DRIVER_API,
                        CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx);

    std::fprintf(stderr, "[agent pid=%d] CUPTI subscriber attached\n", getpid());
}
