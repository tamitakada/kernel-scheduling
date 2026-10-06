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
#include <algorithm>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

static SchedulerSharedState* g_state = nullptr;
static int kernel_count = 0;

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

constexpr int MAX_NVTX_STACK = 32;
thread_local char t_nvtx_stack[MAX_NVTX_STACK][NAME_LEN];
thread_local int  t_nvtx_depth = 0;

extern "C" void sched_push_tag(const char* name) {
    if (t_nvtx_depth < MAX_NVTX_STACK) {
        std::strncpy(t_nvtx_stack[t_nvtx_depth], name, NAME_LEN - 1);
        t_nvtx_stack[t_nvtx_depth][NAME_LEN - 1] = '\0';
        t_nvtx_depth++;
    }
}

extern "C" void sched_pop_tag() {
    if (t_nvtx_depth > 0) t_nvtx_depth--;
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

// Layouts of CUPTI's params for the two launch APIs (confirmed against
// generated_cuda_meta.h for CUDA 13.1). File scope so both the ENTER path
// (to find the app's stream, so the real kernel can wait on the prefetch)
// and the EXIT path (completion tracking) share one definition.
struct cuLaunchKernel_params_min {
    CUfunction f; unsigned int gx, gy, gz, bx, by, bz; unsigned int shmem;
    CUstream hStream; void** kp; void** extra;
};
struct cuLaunchKernelEx_params_min {
    const CUlaunchConfig* config; CUfunction f; void** kernelParams; void** extra;
};

// Extracts the stream the app is launching onto. Returns false only if it
// couldn't be determined. A stream VALUE of 0 is the legitimate
// default/legacy stream, so success is the return value, never "stream != 0".
static bool launch_stream(CUpti_CallbackId cbid, const void* params, CUstream* out) {
    if (cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel) {
        *out = reinterpret_cast<const cuLaunchKernel_params_min*>(params)->hStream;
        return true;
    }
    if (cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx) {
        const CUlaunchConfig* cfg =
            reinterpret_cast<const cuLaunchKernelEx_params_min*>(params)->config;
        if (cfg) { *out = cfg->hStream; return true; }
    }
    return false;
}

constexpr int NUM_PREFETCH_STREAMS = 4;
static CUstream g_prefetch_streams[NUM_PREFETCH_STREAMS];
// One event per prefetch stream, recorded right after each prefetch launch
// so the app's stream can wait on "this prefetch is done".
static CUevent g_prefetch_events[NUM_PREFETCH_STREAMS];
static bool g_prefetch_streams_ready = false;

static void ensure_prefetch_streams() {
    if (g_prefetch_streams_ready) return;
    for (int i = 0; i < NUM_PREFETCH_STREAMS; i++) {
        cuStreamCreate(&g_prefetch_streams[i], CU_STREAM_NON_BLOCKING);
        cuEventCreate(&g_prefetch_events[i], CU_EVENT_DISABLE_TIMING);
    }
    g_prefetch_streams_ready = true;
}

struct LaunchArgs {
    CUfunction f = nullptr;
    void** kp = nullptr;      // kernelParams: array of pointers to each argument's value
    void** extra = nullptr;   // alternative launch style: packed parameter buffer
};

static void launch_args(CUpti_CallbackId cbid, const void* params, LaunchArgs* out) {
    if (cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel) {
        const auto* p = reinterpret_cast<const cuLaunchKernel_params_min*>(params);
        out->f = p->f; out->kp = p->kp; out->extra = p->extra;
    } else if (cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx) {
        const auto* p = reinterpret_cast<const cuLaunchKernelEx_params_min*>(params);
        out->f = p->f; out->kp = p->kernelParams; out->extra = p->extra;
    }
}

constexpr size_t MAX_SCAN_PARAMS = 256;
constexpr size_t MAX_SCAN_BYTES  = 32768;

// Calls fn(value) for every 8-byte word of the launch's parameter memory.
template <class Fn>
static void for_each_param_word(const LaunchArgs& a, Fn&& fn) {
    auto scan = [&](const unsigned char* p, size_t n) {
        if (!p) return;
        if (n > MAX_SCAN_BYTES) n = MAX_SCAN_BYTES;
        for (size_t b = 0; b + sizeof(uintptr_t) <= n; b += sizeof(uintptr_t)) {
            uintptr_t v;
            std::memcpy(&v, p + b, sizeof(v));
            fn(v);
        }
    };
    if (a.kp && a.f) {
        for (size_t i = 0; i < MAX_SCAN_PARAMS; i++) {
            size_t off = 0, sz = 0;
            if (cuFuncGetParamInfo(a.f, i, &off, &sz) != CUDA_SUCCESS) break;  // past the last parameter
            scan(static_cast<const unsigned char*>(a.kp[i]), sz);
        }
    } else if (a.extra) {
        // {CU_LAUNCH_PARAM_BUFFER_POINTER, buf, CU_LAUNCH_PARAM_BUFFER_SIZE, &size, CU_LAUNCH_PARAM_END}
        const unsigned char* buf = nullptr;
        size_t size = 0;
        for (void** e = a.extra; *e != CU_LAUNCH_PARAM_END; e += 2) {
            if (*e == CU_LAUNCH_PARAM_BUFFER_POINTER) buf = static_cast<const unsigned char*>(e[1]);
            else if (*e == CU_LAUNCH_PARAM_BUFFER_SIZE) size = *static_cast<const size_t*>(e[1]);
        }
        scan(buf, size);
    }
}

// Local copy of THIS process's registered regions, rebuilt when the shared table grows.
struct OwnRegion { uintptr_t lo, hi; std::string tag; };
static std::mutex*             g_own_mu      = new std::mutex;
static std::vector<OwnRegion>* g_own_regions = new std::vector<OwnRegion>;
static int                     g_own_seen    = -1;

// Caller holds g_own_mu. (Lock order: g_own_mu, then g_state->mutex.)
static void refresh_own_regions_locked() {
    pid_t me = getpid();
    pthread_mutex_lock(&g_state->mutex);
    if (g_state->region_count != g_own_seen) {
        g_own_regions->clear();
        for (int i = 0; i < g_state->region_count; i++) {
            const auto& r = g_state->regions[i];
            if (r.active && r.owner_pid == me) {
                uintptr_t lo = reinterpret_cast<uintptr_t>(r.ptr);
                g_own_regions->push_back({lo, lo + r.nbytes, std::string(r.tag, strnlen(r.tag, NAME_LEN))});
            }
        }
        g_own_seen = g_state->region_count;
    }
    pthread_mutex_unlock(&g_state->mutex);
}

constexpr size_t PREFETCH_UNIT = 16;   // the prefetch kernel reads whole float4 (16-byte) units

static std::atomic<long> g_pf_kernels{0}, g_pf_skipped{0}, g_pf_args_hit{0}, g_pf_args_other{0}, g_pf_args_none{0};
static std::atomic<long long> g_pf_bytes_total{0};

struct PrefetchSeg { uintptr_t ptr; size_t nbytes; std::string tag; };

struct PrefetchPlan {
    std::vector<PrefetchSeg> segs;     // tensors to prefetch, already truncated to the byte cap
    size_t total_bytes = 0;            // sum of segs[].nbytes; never exceeds the cap
};

static bool starts_with(const std::string& s, const char* prefix) {
    return s.compare(0, std::strlen(prefix), prefix) == 0;
}

struct PrefetchSettings { bool enabled; size_t max_bytes; size_t min_bytes; };

static const PrefetchSettings& prefetch_settings() {
    static const PrefetchSettings cfg = [] {
        PrefetchSettings v{true, 10ull * 1024 * 1024, 64ull * 1024};
        if (const char* e = std::getenv("SCHED_PREFETCH")) v.enabled = std::atoi(e) != 0;
        if (const char* e = std::getenv("SCHED_PREFETCH_MAX_MB")) {
            double mb = std::atof(e);
            v.max_bytes = mb > 0 ? static_cast<size_t>(mb * 1024.0 * 1024.0) : 0;
        }
        if (const char* e = std::getenv("SCHED_PREFETCH_MIN_KB")) {
            double kb = std::atof(e);
            v.min_bytes = kb > 0 ? static_cast<size_t>(kb * 1024.0) : 0;
        }
        std::fprintf(stderr, "[agent pid=%d] prefetch %s: cap=%zu bytes per kernel, skip tensors < %zu bytes\n",
                     getpid(), v.enabled ? "ENABLED" : "DISABLED", v.max_bytes, v.min_bytes);
        return v;
    }();
    return cfg;
}

// "layer_<N>/<module>" -> layer index and module role. False for anything else
// ("unknown", plain "layer_<N>", "layer_<N>/").
static bool parse_hook_tag(const char* tag, int* layer, std::string* role) {
    int n = 0;
    if (std::sscanf(tag, "layer_%d/%n", layer, &n) != 1 || n == 0 || tag[n] == '\0') return false;
    *role = tag + n;
    return true;
}

// Fills `plan` with the tensors to prefetch for a kernel launched under this
// hook tag (empty plan = skip). See the block comment above.
static void select_prefetch_targets(const char* hook_tag, PrefetchPlan* plan) {
    const PrefetchSettings& cfg = prefetch_settings();
    if (!cfg.enabled) return;

    int layer = -1;
    std::string role;
    if (!parse_hook_tag(hook_tag, &layer, &role)) return;
    const std::string prefix = "layer" + std::to_string(layer) + "_" + role + "_";

    std::lock_guard<std::mutex> g(*g_own_mu);
    refresh_own_regions_locked();

    size_t remaining = cfg.max_bytes;
    for (const OwnRegion& r : *g_own_regions) {
        if (!starts_with(r.tag, prefix.c_str())) continue;
        size_t size = r.hi - r.lo;
        if (size < cfg.min_bytes) continue;
        size_t take = std::min(size, remaining);      // the hard cap
        take -= take % PREFETCH_UNIT;
        if (take == 0) continue;
        plan->segs.push_back({r.lo, take, r.tag});
        plan->total_bytes += take;
        remaining -= take;
    }
}

// ADVISORY only: does the kernel's argument memory contain the address of one
// of the tensors we are prefetching for it?
//   HIT        yes
//   OTHER:<t>  no, but it contains registered tensor <t> (so the hook tag may
//              be wrong for this kernel, or it is reading a different tensor)
//   NONE       no registered tensor found at all (inconclusive)
static std::string check_args_advisory(const LaunchArgs& a, const PrefetchPlan& plan) {
    std::lock_guard<std::mutex> g(*g_own_mu);
    bool hit = false;
    const OwnRegion* other = nullptr;
    for_each_param_word(a, [&](uintptr_t v) {
        for (const auto& r : *g_own_regions) {
            if (v < r.lo || v >= r.hi) continue;
            bool is_target = false;
            for (const auto& seg : plan.segs) {
                if (seg.tag == r.tag) { is_target = true; break; }
            }
            if (is_target) hit = true;
            else if (!other) other = &r;
            break;   // registered tensors don't overlap: the first hit is the one
        }
    });
    if (hit) return "HIT";
    if (other) return "OTHER:" + other->tag;
    return "NONE";
}

/* ======================================================================
 * GPU-SIDE KERNEL TIMING (CUDA events)
 *
 * Per launch we record, on the stream each kernel actually runs on:
 *   prefetch stream:  pf_start -> [prefetch kernel] -> pf_end
 *   app stream:       wait(pf_end) -> real_start -> [real kernel] -> real_end
 * real_start is recorded AFTER the prefetch wait, so the real kernel's
 * interval excludes the prefetch. Elapsed time between two events on one
 * stream is the kernel plus a few microseconds of event/launch overhead
 * (resolution about 0.5us), so treat sub-10us kernels as upper bounds.
 *
 * Events live in a pool and are resolved lazily: at each ENTER we recycle
 * slots whose real_end has completed (non-blocking cuEventQuery), and
 * sched_flush_timing() resolves the rest and writes the file. No CUDA call
 * is made from the cuLaunchHostFunc driver thread.
 * ====================================================================== */

constexpr int TIMING_POOL = 1024;   // max launches whose timing is unresolved at once

struct TimingSlot {
    CUevent real_start = nullptr, real_end = nullptr, pf_start = nullptr, pf_end = nullptr;
    int  state = 0;       // 0 = free, 1 = acquired (ENTER..EXIT), 2 = end recorded, awaiting resolve
    int  kernel_id = -1;
    bool has_pf = false;
    std::string name;
    std::string hook_tag;     // innermost forward-hook tag when the kernel was launched
    std::string args_check;   // advisory: "-" (nothing prefetched), HIT, OTHER:<tag> or NONE
    size_t      pf_bytes = 0; // bytes actually prefetched (after the cap)
    std::string pf_tensors;   // tensors prefetched, joined with "+"
};

struct TimingResult {
    int kernel_id; std::string name;
    float kernel_ms; bool has_pf; float pf_ms; float gap_ms;
    std::string hook_tag, args_check;
    size_t pf_bytes; std::string pf_tensors;
};

// Heap-allocated and never freed, so nothing depends on static destruction order.
static std::mutex*                g_timing_mu      = new std::mutex;
static TimingSlot*                g_timing         = new TimingSlot[TIMING_POOL];
static std::deque<int>*           g_timing_pending = new std::deque<int>;
static std::vector<TimingResult>* g_timing_results = new std::vector<TimingResult>;
static CUcontext                  g_timing_ctx     = nullptr;
static long                       g_timing_dropped = 0;

// The timing slot of the launch currently between ENTER and EXIT on this thread.
thread_local int t_timing_slot = -1;

static float event_ms(CUevent a, CUevent b) {
    float ms = 0.f;
    return cuEventElapsedTime(&ms, a, b) == CUDA_SUCCESS ? ms : -1.f;
}

// Caller holds g_timing_mu and guarantees the slot's events are complete.
static void timing_finish(const TimingSlot& t) {
    TimingResult r;
    r.kernel_id = t.kernel_id;
    r.name      = t.name;
    r.hook_tag   = t.hook_tag;
    r.args_check = t.args_check;
    r.pf_bytes   = t.pf_bytes;
    r.pf_tensors = t.pf_tensors;
    r.kernel_ms = event_ms(t.real_start, t.real_end);
    r.has_pf    = t.has_pf;
    r.pf_ms     = t.has_pf ? event_ms(t.pf_start, t.pf_end)    : 0.f;
    r.gap_ms    = t.has_pf ? event_ms(t.pf_end,   t.real_start) : 0.f;  // should be >= 0 if ordering holds
    g_timing_results->push_back(r);
}

static bool timing_create_events(TimingSlot& t) {
    if (t.real_start) return true;
    CUevent e[4] = {};
    for (auto& ev : e) {
        if (cuEventCreate(&ev, CU_EVENT_DEFAULT) != CUDA_SUCCESS) return false;
    }
    t.real_start = e[0]; t.real_end = e[1]; t.pf_start = e[2]; t.pf_end = e[3];
    return true;
}

// Recycles finished slots, oldest first. Non-blocking.
static void timing_resolve_ready() {
    std::lock_guard<std::mutex> g(*g_timing_mu);
    while (!g_timing_pending->empty()) {
        int idx = g_timing_pending->front();
        TimingSlot& t = g_timing[idx];
        // real_end completing implies pf_end did too: the app stream waits on pf_end first.
        if (cuEventQuery(t.real_end) != CUDA_SUCCESS) break;
        timing_finish(t);
        t.state = 0;
        g_timing_pending->pop_front();
    }
}

// Returns a slot index, or -1 if the pool is exhausted (that launch just goes untimed).
static int timing_acquire(int kernel_id, const char* name) {
    timing_resolve_ready();
    std::lock_guard<std::mutex> g(*g_timing_mu);
    if (!g_timing_ctx) cuCtxGetCurrent(&g_timing_ctx);
    for (int i = 0; i < TIMING_POOL; i++) {
        TimingSlot& t = g_timing[i];
        if (t.state != 0) continue;
        if (!timing_create_events(t)) break;
        t.state = 1;
        t.kernel_id = kernel_id;
        t.has_pf = false;
        t.name = name;
        t.hook_tag.clear();
        t.args_check = "-";
        t.pf_bytes = 0;
        t.pf_tensors.clear();
        return i;
    }
    g_timing_dropped++;
    return -1;
}

// Called at EXIT, after the real kernel has been queued on `stream`.
static void timing_mark_end(CUstream stream) {
    int idx = t_timing_slot;
    if (idx < 0) return;
    cuEventRecord(g_timing[idx].real_end, stream);
    std::lock_guard<std::mutex> g(*g_timing_mu);
    g_timing[idx].state = 2;
    g_timing_pending->push_back(idx);
    t_timing_slot = -1;
}

// Call from Python after your last llm.chat, while CUDA is still alive:
//   agent.sched_flush_timing()
// Blocks until every recorded kernel has finished, then appends rows to
// <SCHED_LOG_DIR or .>/agent_<pid>_timing.tsv (sort by kernel_id to match your log).
extern "C" void sched_flush_timing() {
    std::fprintf(stderr, "[agent pid=%d] prefetch: kernels_prefetched=%ld kernels_skipped=%ld args_hit=%ld args_other=%ld args_none=%ld prefetched_bytes_total=%lld\n",
                 getpid(), g_pf_kernels.load(), g_pf_skipped.load(), g_pf_args_hit.load(),
                 g_pf_args_other.load(), g_pf_args_none.load(), g_pf_bytes_total.load());

    // A ctypes call may run on a thread with no current CUDA context.
    CUcontext cur = nullptr;
    cuCtxGetCurrent(&cur);
    if (!cur && g_timing_ctx) cuCtxSetCurrent(g_timing_ctx);

    std::vector<int> todo;
    {
        std::lock_guard<std::mutex> g(*g_timing_mu);
        todo.assign(g_timing_pending->begin(), g_timing_pending->end());
        g_timing_pending->clear();
    }
    for (int idx : todo) cuEventSynchronize(g_timing[idx].real_end);  // no lock held while waiting

    std::vector<TimingResult> out;
    long dropped = 0;
    {
        std::lock_guard<std::mutex> g(*g_timing_mu);
        for (int idx : todo) { timing_finish(g_timing[idx]); g_timing[idx].state = 0; }
        out.swap(*g_timing_results);
        dropped = g_timing_dropped;
    }
    if (out.empty()) {
        std::fprintf(stderr, "[agent pid=%d] no timing rows to write (dropped=%ld)\n", getpid(), dropped);
        return;
    }

    const char* dir = std::getenv("SCHED_LOG_DIR");
    std::string path = std::string(dir ? dir : ".") + "/agent_" + std::to_string(getpid()) + "_timing.tsv";
    static bool header_written = false;
    FILE* f = std::fopen(path.c_str(), "a");
    if (!f) {
        std::fprintf(stderr, "[agent pid=%d] could not open %s for timing output\n", getpid(), path.c_str());
        return;
    }
    if (!header_written) {
        std::fputs("kernel_id\tkernel_us\tprefetch_us\tprefetch_to_kernel_gap_us\thook_tag\tprefetch_bytes\tprefetched_tensors\targs_check\tkernel_name\n", f);
        header_written = true;
    }
    for (const auto& r : out) {
        std::fprintf(f, "%d\t%.3f\t", r.kernel_id, r.kernel_ms * 1000.f);
        if (r.has_pf) std::fprintf(f, "%.3f\t%.3f\t", r.pf_ms * 1000.f, r.gap_ms * 1000.f);
        else          std::fputs("\t\t", f);
        std::fprintf(f, "%s\t%zu\t%s\t%s\t%s\n", r.hook_tag.c_str(), r.pf_bytes,
                     r.pf_tensors.c_str(), r.args_check.c_str(), r.name.c_str());
    }
    std::fclose(f);
    std::fprintf(stderr, "[agent pid=%d] wrote %zu timing rows to %s (dropped=%ld)\n",
                 getpid(), out.size(), path.c_str(), dropped);
}

static void inject_prefetch(const PrefetchPlan& plan, bool have_stream, CUstream app_stream) {
    if (plan.segs.empty()) return;

    ensure_prefetch_kernel_compiled();
    ensure_prefetch_streams();
    if (!g_prefetch_kernel_func) return;

    static int rr = 0;
    int stream_idx = (rr++) % NUM_PREFETCH_STREAMS;
    CUstream stream = g_prefetch_streams[stream_idx];
    TimingSlot* ts = (t_timing_slot >= 0) ? &g_timing[t_timing_slot] : nullptr;

    // REENTRANCY GUARD: these cuLaunchKernel calls would otherwise be caught
    // by our OWN CUPTI subscriber and recurse into request_launch_
    // permission. Skip scheduling logic for them entirely.
    if (ts) cuEventRecord(ts->pf_start, stream);
    t_in_scheduler_launch = true;
    for (const PrefetchSeg& seg : plan.segs) {
        CUdeviceptr target = static_cast<CUdeviceptr>(seg.ptr);
        size_t n_float4 = seg.nbytes / sizeof(float4);
        void* args[] = { &target, &n_float4, &g_prefetch_scratch };

        const unsigned int threads_per_block = 256;
        size_t blocks = (n_float4 + threads_per_block - 1) / threads_per_block;
        if (blocks < 1) blocks = 1;
        if (blocks > 2048) blocks = 2048;

        cuLaunchKernel(g_prefetch_kernel_func,
                       static_cast<unsigned int>(blocks), 1, 1,
                       threads_per_block, 1, 1,
                       0, stream, args, nullptr);
    }
    t_in_scheduler_launch = false;

    // Order the real kernel after the prefetch. The event captures "all work
    // on the prefetch stream so far" (the prefetch kernels); a wait on
    // app_stream snapshots that state at call time, so re-recording this
    // event for a later prefetch doesn't affect waits already queued.
    // With a timing slot, its pf_end event does double duty: it marks the end
    // of the prefetch for timing AND is what the app stream waits on.
    CUevent done = ts ? ts->pf_end : g_prefetch_events[stream_idx];
    if (ts) ts->has_pf = true;
    cuEventRecord(done, stream);
    if (have_stream) {
        cuStreamWaitEvent(app_stream, done, 0);
    } else {
        cuStreamSynchronize(stream); // can't see the app's stream: block the host instead
    }
    g_pf_bytes_total += static_cast<long long>(plan.total_bytes);

    // std::fprintf(stderr, "[agent pid=%d] injected prefetch tag=%s stream_idx=%d\n",
    //         getpid(), region_tag, stream_idx);
}

/* ======================================================================
 * CLIENT SIDE OF THE QUEUE: register this launch with the daemon, block
 * until released, record the inflight slot the daemon reserved for us, and
 * execute any prefetch instruction it attached.
 * ====================================================================== */

static void request_launch_permission(const char* kernel_name, const char* layer_tag,
                                      bool have_stream, CUstream app_stream,
                                      const LaunchArgs& args) {
    t_timing_slot = -1;   // no timing for this launch unless a slot is acquired below
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

    kernel_count++;
    int kernel_id = kernel_count;

    g_state->slots[slot].pid = getpid();
    std::strncpy(g_state->slots[slot].kernel_name, kernel_name, NAME_LEN - 1);
    std::strncpy(g_state->slots[slot].layer_tag, layer_tag, NAME_LEN - 1);
    g_state->slots[slot].active = 1;
    g_state->slots[slot].ready = 0;
    g_state->slots[slot].inflight_slot = -1;
    g_state->slots[slot].kernel_id = 0;
    pthread_mutex_unlock(&g_state->mutex);

    sem_post(&g_state->new_request_sem);     // wake the daemon
    sem_wait(&g_state->slots[slot].wake_sem); // block until IT releases THIS slot

    pthread_mutex_lock(&g_state->mutex);
    t_current_inflight_slot = g_state->slots[slot].inflight_slot; // carry through to EXIT
    g_state->slots[slot].active = 0;
    pthread_mutex_unlock(&g_state->mutex);

    // Prefetch decision, made entirely on the agent side from the kernel's
    // forward-hook tag (see the policy block above): the tensors owned by the
    // module this kernel runs in, capped. An empty plan means skip.
    PrefetchPlan plan;
    select_prefetch_targets(layer_tag, &plan);

    std::string args_check = "-";
    std::string prefetched_tensors;
    if (plan.segs.empty()) {
        g_pf_skipped++;
    } else {
        g_pf_kernels++;
        args_check = check_args_advisory(args, plan);          // advisory only: never gates the prefetch
        if (args_check == "HIT") g_pf_args_hit++;
        else if (args_check == "NONE") g_pf_args_none++;
        else g_pf_args_other++;
        for (const auto& seg : plan.segs) {
            if (!prefetched_tensors.empty()) prefetched_tensors += "+";
            prefetched_tensors += seg.tag;
        }
    }

    // Timing: acquire BEFORE the prefetch so inject_prefetch can record its events in the slot.
    t_timing_slot = have_stream ? timing_acquire(kernel_id, kernel_name) : -1;
    if (t_timing_slot >= 0) {
        g_timing[t_timing_slot].hook_tag   = layer_tag;
        g_timing[t_timing_slot].args_check = args_check;
        g_timing[t_timing_slot].pf_bytes   = plan.total_bytes;
        g_timing[t_timing_slot].pf_tensors = prefetched_tensors;
    }

    if (!plan.segs.empty()) {
        inject_prefetch(plan, have_stream, app_stream);
    }

    // real_start goes AFTER the prefetch wait, so the real kernel's interval excludes the prefetch.
    if (t_timing_slot >= 0) cuEventRecord(g_timing[t_timing_slot].real_start, app_stream);
}

/* ======================================================================
 * CUPTI CALLBACK
 * ====================================================================== */

static CUpti_SubscriberHandle g_subscriber;

static void CUPTIAPI cupti_callback(void* userdata, CUpti_CallbackDomain domain,
                                     CUpti_CallbackId cbid, const void* cbdata) {

    if (domain != CUPTI_CB_DOMAIN_DRIVER_API) return;
    if (cbid != CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel &&
        cbid != CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx) return;

    // Skip our OWN prefetch launches entirely -- see inject_prefetch().
    if (t_in_scheduler_launch) return;

    const CUpti_CallbackData* data = static_cast<const CUpti_CallbackData*>(cbdata);

    if (data->callbackSite == CUPTI_API_ENTER) { // on kernel launch
        const char* layer_tag = current_layer_tag();
        const char* kernel_name = (data->symbolName && data->symbolName[0]) ? data->symbolName : "unknown";
        // Find the app's stream so inject_prefetch can make it wait on the
        // prefetch before the real kernel is queued onto it.
        CUstream app_stream = 0;
        bool have_stream = launch_stream(cbid, data->functionParams, &app_stream);
        LaunchArgs args;
        launch_args(cbid, data->functionParams, &args);

        request_launch_permission(kernel_name, layer_tag, have_stream, app_stream, args);
        // returning here lets CUPTI/the driver proceed with the real launch
        return;
    }

    if (data->callbackSite == CUPTI_API_EXIT) {
        CUstream stream = 0;
        bool resolved = launch_stream(cbid, data->functionParams, &stream);

        void* userdata_for_completion = reinterpret_cast<void*>(
            static_cast<intptr_t>(t_current_inflight_slot));

        if (resolved) {
            timing_mark_end(stream);

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

static void open_shared_state() {
    int fd = shm_open(SHM_NAME, O_RDWR, 0666);
    if (fd < 0) {
        std::fprintf(stderr, "[agent pid=%d] FATAL: scheduler_daemon never started "
                        "(shm_open %s failed: %s)\n", getpid(), SHM_NAME, std::strerror(errno));
        std::exit(1);
    }

    g_state = static_cast<SchedulerSharedState*>(
        mmap(nullptr, sizeof(SchedulerSharedState), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
    close(fd);

    while (g_state->initialized.load() == 0) usleep(1000);

    std::fprintf(stderr, "[agent pid=%d] attached to scheduler_daemon\n", getpid());
}

__attribute__((constructor))
static void agent_init() {
    // load the shared memory state thru which agent communicates with scheduler
    open_shared_state();

    CUptiResult res = cuptiSubscribe(&g_subscriber, (CUpti_CallbackFunc)cupti_callback, nullptr);
    if (res != CUPTI_SUCCESS) {
        std::fprintf(stderr, "[agent pid=%d] FATAL: cuptiSubscribe failed: %d\n", getpid(), res);
        return;
    }

    // intercept CUDA kernel launches
    cuptiEnableCallback(1, g_subscriber, CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel);
    cuptiEnableCallback(1, g_subscriber, CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx);
    std::fprintf(stderr, "[agent pid=%d] CUPTI subscriber attached\n", getpid());
}
