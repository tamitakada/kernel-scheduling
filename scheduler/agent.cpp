#define _GNU_SOURCE
#define ENABLE_LOGGING 1

#include "shared_state.hpp"

#include <strings.h>
#include <cuda.h>
#include <cupti.h>
#include "kernel_key.hpp"
#include <nvrtc.h>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#include <atomic>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <algorithm>
#include <deque>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

static SchedulerSharedState* g_state = nullptr;
static int kernel_count = 0;

// True once this agent has subscribed to CUPTI, i.e. launches go through the scheduler and the SCHEDULER decides
// whether each one gets a prefetch. The Python-hook entry points (sched_prefetch*) are refused while this is set.
static std::atomic<bool> g_cupti_active{false};

/* Called from Python program to  
*/
extern "C" void register_weight_region(const char* tag, void* ptr, size_t nbytes) {
    pthread_mutex_lock(&g_state->mutex);

    if (g_state->region_count == MAX_REGIONS) {
        if (ENABLE_LOGGING)
            std::fprintf(stderr, "[agent pid=%d] WARNING: region table full at tag=%s\n", getpid(), tag);
        pthread_mutex_unlock(&g_state->mutex);
        return;
    }
   
    int i = g_state->region_count++;
    
    g_state->regions[i].owner_pid = getpid();
    std::strncpy(g_state->regions[i].tag, tag, NAME_LEN - 1);
    g_state->regions[i].ptr = ptr;
    g_state->regions[i].nbytes = nbytes;
    g_state->regions[i].active = 1;

    if (ENABLE_LOGGING)
        std::fprintf(stderr, "[agent pid=%d] registered region tag=%s ptr=%p size=%zu\n", getpid(), tag, ptr, nbytes);

    pthread_mutex_unlock(&g_state->mutex);
}

constexpr int MAX_NVTX_STACK = 32;

// The tag stack: front() is the innermost module currently running. Every push (= one module entry) gets a unique id,
// kept in a parallel stack, so the launch path can tell "another kernel of the module entry I already prefetched for"
// from "a new entry of the same module".
thread_local std::deque<std::string> t_tag_stack;
thread_local std::deque<uint64_t>    t_tag_entry;
thread_local uint64_t                t_tag_next_entry = 0;
extern "C" void sched_push_tag(const char* name) {
    t_tag_stack.push_front(name);
    t_tag_entry.push_front(++t_tag_next_entry);
}
extern "C" void sched_pop_tag() {
    if (t_tag_stack.empty()) return;                 // an unbalanced pop must not be undefined behaviour
    t_tag_stack.pop_front();
    t_tag_entry.pop_front();
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

// Host wall clock used for the start_ns / end_ns columns of the timing log: CLOCK_MONOTONIC, nanoseconds. It is system-wide, so
// rows from different processes (the LLM server, the Whisper server) can be compared directly, and it is the clock behind
// Python's time.monotonic_ns() / perf_counter_ns() on Linux.
static inline uint64_t mono_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull + static_cast<uint64_t>(ts.tv_nsec);
}

static void timing_stamp_end(int idx, uint32_t gen, uint64_t now_ns);   // defined with the timing slots below

// What the completion callback is told (one pointer-sized word): which daemon inflight slot to clear, and which timing slot
// (and which generation of it) to stamp with the kernel's end time. Fields are stored +1 so that -1 ("none") fits.
static_assert(sizeof(void*) == 8, "the completion word packs three fields into 64 bits");
static inline void* pack_completion(int inflight_idx, int timing_idx, uint32_t timing_gen) {
    return reinterpret_cast<void*>(static_cast<uintptr_t>(
        (static_cast<uint64_t>(timing_gen) << 32) |
        (static_cast<uint64_t>(static_cast<uint16_t>(timing_idx + 1)) << 16) |
        static_cast<uint64_t>(static_cast<uint16_t>(inflight_idx + 1))));
}

// Frees the daemon's idle gate. Also what the "could not track completion" fallbacks call: there the kernel may well still be
// running, so they must not stamp an end time.
static void release_inflight_slot(int idx) {
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

static void CUDA_CB launch_completion_callback(void* userdata) {
    const uint64_t now = mono_ns();          // first thing: this instant is the kernel's end_ns in the timing log
    const uint64_t u = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(userdata));
    timing_stamp_end(static_cast<int>((u >> 16) & 0xFFFF) - 1, static_cast<uint32_t>(u >> 32), now);
    const int infl = static_cast<int>(u & 0xFFFF) - 1;
    if (infl >= 0 && infl < MAX_INFLIGHT) g_state->inflight[infl].end_ns.store(now, std::memory_order_release);   // the daemon logs it
    release_inflight_slot(infl);
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

// The launch's grid and block dimensions, from either launch API's parameters (zeros if they cannot be read).
struct LaunchDims { uint32_t grid[3] = {0, 0, 0}, block[3] = {0, 0, 0}; };

static LaunchDims launch_dims(CUpti_CallbackId cbid, const void* params) {
    LaunchDims d;
    if (cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel) {
        const auto* p = reinterpret_cast<const cuLaunchKernel_params_min*>(params);
        d.grid[0] = p->gx; d.grid[1] = p->gy; d.grid[2] = p->gz;
        d.block[0] = p->bx; d.block[1] = p->by; d.block[2] = p->bz;
    } else if (cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx) {
        const CUlaunchConfig* c = reinterpret_cast<const cuLaunchKernelEx_params_min*>(params)->config;
        if (c) {
            d.grid[0] = c->gridDimX; d.grid[1] = c->gridDimY; d.grid[2] = c->gridDimZ;
            d.block[0] = c->blockDimX; d.block[1] = c->blockDimY; d.block[2] = c->blockDimZ;
        }
    }
    return d;
}

// The model this process runs ("qwen" / "whisper"), from SCHED_MODEL: it tells the daemon which kernel table to search.
static const std::string& model_tag() {
    static const std::string tag = [] { const char* e = std::getenv("SCHED_MODEL"); return std::string(e ? e : ""); }();
    return tag;
}

// ncu_function_base() demangles and parses a symbol of up to a few hundred characters: do it once per distinct symbol.
static const std::string& kernel_base_name(const char* symbol) {
    static thread_local std::unordered_map<std::string, std::string> cache;
    auto it = cache.find(symbol);
    if (it == cache.end()) it = cache.emplace(symbol, ncu_function_base(symbol)).first;
    return it->second;
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
// CUPTI path only: launches the scheduler told us NOT to prefetch for, and launches skipped because their module entry
// had already been prefetched (a module that launches several kernels is prefetched once, on the first one).
static std::atomic<long> g_pf_denied{0}, g_pf_dup{0};
thread_local uint64_t t_last_prefetch_entry = 0;
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
        PrefetchSettings v{true, 64ull * 1024 * 1024, 0};
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
 *
 * Wall-clock timestamps (start_ns, end_ns, CLOCK_MONOTONIC): start_ns is read just before real_start is queued, end_ns in the
 * completion callback, which fires right after real_end. The kernel's whole GPU execution therefore lies inside
 * [start_ns, end_ns], and end_ns - start_ns exceeds kernel_us by the launch and callback latency (tens of microseconds on an
 * idle GPU, more if the stream was busy). They are host readings, not GPU-clock times: that keeps rows from different
 * processes comparable (no per-process clock calibration, no drift). end_ns is blank if no callback was seen.
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
    uint64_t start_ns = 0;                   // mono_ns() just before real_start was queued
    std::atomic<uint64_t> end_ns{0};         // mono_ns() when the completion callback ran; 0 = not (yet) seen
    std::atomic<uint32_t> gen{0};            // bumped at every acquire, so a late callback cannot stamp a recycled slot
    uint64_t batch_id = 0;                   // the scheduler's batch this kernel was released in (0 = launched unscheduled)
    int      batch_size = 0, batch_pos = 0;
    uint32_t block[3] = {0, 0, 0};           // the launch's block dimensions
    uint32_t grid[3]  = {0, 0, 0};           // and grid dimensions
};

struct TimingResult {
    int kernel_id; std::string name;
    float kernel_ms; bool has_pf; float pf_ms; float gap_ms;
    std::string hook_tag, args_check;
    size_t pf_bytes; std::string pf_tensors;
    uint64_t start_ns, end_ns;
    uint64_t batch_id; int batch_size, batch_pos;
    uint32_t block[3], grid[3];
};

// Stored in end_ns by the fallbacks that free the gate WITHOUT a completion callback (cuLaunchHostFunc failed): no end time
// will ever come, so the slot may be recycled at once, and the row's end_ns stays blank.
static constexpr uint64_t NO_END_NS = ~0ull;

// Heap-allocated and never freed, so nothing depends on static destruction order.
static std::mutex*                g_timing_mu      = new std::mutex;
static TimingSlot*                g_timing         = new TimingSlot[TIMING_POOL];
static std::deque<int>*           g_timing_pending = new std::deque<int>;
static std::vector<TimingResult>* g_timing_results = new std::vector<TimingResult>;
static CUcontext                  g_timing_ctx     = nullptr;
static long                       g_timing_dropped = 0;

// The timing slot of the launch currently between ENTER and EXIT on this thread.
thread_local int t_timing_slot = -1;

// SCHED_LOG (the daemon and sched_hooks.py read the same variable): full (default) | decisions | off.
// The CUDA-event timing of every kernel, and the timing log built from it, exist only at "full": two cuEventRecord calls, a mutex and some string
// copies per launch are CPU time on the launching thread, and the agent is CPU-bound in eager mode. "decisions" and "off" skip all of it; the
// kernels are still intercepted and scheduled exactly as before (the daemon's own logging at "decisions" is a separate matter, see the daemon).
static bool timing_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("SCHED_LOG");
        if (!e || !*e || !strcasecmp(e, "full") || !strcmp(e, "2") || !strcasecmp(e, "all")) return true;
        if (!strcasecmp(e, "decisions") || !strcmp(e, "1") || !strcasecmp(e, "off") || !strcmp(e, "0") || !strcasecmp(e, "none")) return false;
        std::fprintf(stderr, "[agent pid=%d] WARNING: SCHED_LOG=%s is not one of full, decisions, off: using full\n", getpid(), e);
        return true;
    }();
    return on;
}

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
    r.start_ns   = t.start_ns;
    r.end_ns     = t.end_ns.load(std::memory_order_acquire);
    if (r.end_ns == NO_END_NS) r.end_ns = 0;
    r.batch_id = t.batch_id; r.batch_size = t.batch_size; r.batch_pos = t.batch_pos;
    std::memcpy(r.block, t.block, sizeof r.block);
    std::memcpy(r.grid, t.grid, sizeof r.grid);
    r.kernel_ms = event_ms(t.real_start, t.real_end);
    r.has_pf    = t.has_pf;
    r.pf_ms     = t.has_pf ? event_ms(t.pf_start, t.pf_end)    : 0.f;
    r.gap_ms    = t.has_pf ? event_ms(t.pf_end,   t.real_start) : 0.f;  // should be >= 0 if ordering holds
    g_timing_results->push_back(r);
}

// Runs on the driver's callback thread: atomics only, no lock, no CUDA call.
static void timing_stamp_end(int idx, uint32_t gen, uint64_t now_ns) {
    if (idx < 0 || idx >= TIMING_POOL) return;
    TimingSlot& t = g_timing[idx];
    if (t.gen.load(std::memory_order_acquire) == gen) t.end_ns.store(now_ns, std::memory_order_release);
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
        // The completion callback (it stamps end_ns) runs just after the end event. Recycle only once it has, so it can never
        // stamp a recycled slot; but give up after 2 s so a callback that never comes cannot block the pool.
        if (t.end_ns.load(std::memory_order_acquire) == 0 && mono_ns() - t.start_ns < 2000000000ull) break;
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
        t.start_ns = 0;
        t.batch_id = 0; t.batch_size = 0; t.batch_pos = 0;
        t.end_ns.store(0, std::memory_order_release);
        t.gen.fetch_add(1, std::memory_order_acq_rel);     // after end_ns is cleared: a stale callback is now rejected
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
    if (!timing_enabled()) {                  // SCHED_LOG is not "full": no timing slot was ever acquired, so there is nothing to wait for or write
        std::fprintf(stderr, "[agent pid=%d] SCHED_LOG is not full: no timing rows were recorded, nothing to write\n", getpid());
        return;
    }
    std::fprintf(stderr, "[agent pid=%d] prefetch: kernels_prefetched=%ld kernels_skipped=%ld scheduler_denied=%ld "
                         "same_module_entry=%ld args_hit=%ld args_other=%ld args_none=%ld prefetched_bytes_total=%lld\n",
                 getpid(), g_pf_kernels.load(), g_pf_skipped.load(), g_pf_denied.load(), g_pf_dup.load(),
                 g_pf_args_hit.load(), g_pf_args_other.load(), g_pf_args_none.load(), g_pf_bytes_total.load());

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
    // The completion callbacks run on a driver thread just after each end event; give the last ones a moment (bounded, shared).
    const uint64_t give_up = mono_ns() + 200000000ull;
    for (int idx : todo)
        while (g_timing[idx].end_ns.load(std::memory_order_acquire) == 0 && mono_ns() < give_up) usleep(50);

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
        std::fputs("kernel_id\tkernel_us\tprefetch_us\tprefetch_to_kernel_gap_us\thook_tag\tprefetch_bytes\tprefetched_tensors\targs_check\tkernel_name\tstart_ns\tend_ns\tbatch_id\tbatch_size\tbatch_pos\tblock\tmodel\tgrid\n", f);
        header_written = true;
    }
    for (const auto& r : out) {
        std::fprintf(f, "%d\t%.3f\t", r.kernel_id, r.kernel_ms * 1000.f);
        if (r.has_pf) std::fprintf(f, "%.3f\t%.3f\t", r.pf_ms * 1000.f, r.gap_ms * 1000.f);
        else          std::fputs("\t\t", f);
        std::fprintf(f, "%s\t%zu\t%s\t%s\t%s\t%llu\t", r.hook_tag.c_str(), r.pf_bytes,
                     r.pf_tensors.c_str(), r.args_check.c_str(), r.name.c_str(),
                     static_cast<unsigned long long>(r.start_ns));
        if (r.end_ns) std::fprintf(f, "%llu", static_cast<unsigned long long>(r.end_ns));
        std::fprintf(f, "\t%llu\t%d\t%d\t(%u, %u, %u)\t%s\t(%u, %u, %u)\n", static_cast<unsigned long long>(r.batch_id), r.batch_size, r.batch_pos, r.block[0], r.block[1], r.block[2],
                     model_tag().c_str(), r.grid[0], r.grid[1], r.grid[2]);
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
 * HOOK-TRIGGERED PREFETCH (no CUPTI)
 *
 * For running under Nsight Compute: ncu and this agent cannot both own CUPTI,
 * so with SCHED_NO_CUPTI=1 the agent does not subscribe and nothing here
 * intercepts launches. Instead a Python forward pre-hook calls
 * sched_prefetch() just before the module that should run with warm weights.
 * (Same policy and injection as the CUPTI path; stream = -1 means unknown.)
 * ====================================================================== */

static std::atomic<long> g_hook_calls{0}, g_hook_empty{0};

// With CUPTI active the scheduler owns the prefetch decision, so a Python hook must not prefetch behind its back.
static bool hook_prefetch_blocked() {
    if (!g_cupti_active.load()) return false;
    static std::atomic<bool> warned{false};
    if (!warned.exchange(true))
        std::fprintf(stderr, "[agent pid=%d] WARNING: sched_prefetch*() called while the CUPTI path is active; ignored "
                             "(the scheduler decides, and the tag hooks drive the prefetch). Use SCHED_NO_CUPTI=1 for "
                             "hook-triggered prefetch.\n", getpid());
    return true;
}

extern "C" long long sched_prefetch(const char* hook_tag, long long stream) {
    if (!hook_tag) return -1;
    if (hook_prefetch_blocked()) return 0;
    PrefetchPlan plan;
    select_prefetch_targets(hook_tag, &plan);
    const long n = g_hook_calls++;
    if (plan.segs.empty()) { g_hook_empty++; return 0; }

    g_pf_kernels++;
    t_timing_slot = -1;                       // no CUDA-event timing in this mode
    inject_prefetch(plan, stream >= 0, reinterpret_cast<CUstream>(static_cast<intptr_t>(stream)));
    if (n < 5)
        std::fprintf(stderr, "[agent pid=%d] sched_prefetch(%s, stream=%lld): %zu tensor(s), %zu bytes\n",
                     getpid(), hook_tag, stream, plan.segs.size(), plan.total_bytes);
    return static_cast<long long>(plan.total_bytes);
}

extern "C" long long sched_prefetch_ranges(const unsigned long long* ptrs, const unsigned long long* sizes,
                                           int n, long long stream) {
    if (!ptrs || !sizes || n <= 0) return -1;
    if (hook_prefetch_blocked()) return 0;
    const PrefetchSettings& cfg = prefetch_settings();
    const long call = g_hook_calls++;
    if (!cfg.enabled) { g_hook_empty++; return 0; }

    PrefetchPlan plan;
    size_t remaining = cfg.max_bytes;
    for (int i = 0; i < n && remaining > 0; i++) {
        uintptr_t lo = static_cast<uintptr_t>(ptrs[i]);
        size_t size = static_cast<size_t>(sizes[i]);
        if (lo == 0 || size < cfg.min_bytes) continue;
        const size_t skip = (PREFETCH_UNIT - lo % PREFETCH_UNIT) % PREFETCH_UNIT;   // float4 loads need 16-byte alignment
        if (size <= skip) continue;
        lo += skip;
        size -= skip;
        size_t take = std::min(size, remaining);                  // the hard cap, shared by all ranges of one call
        take -= take % PREFETCH_UNIT;
        if (take == 0) continue;
        plan.segs.push_back({lo, take, std::string()});
        plan.total_bytes += take;
        remaining -= take;
    }
    if (plan.segs.empty()) { g_hook_empty++; return 0; }

    g_pf_kernels++;
    t_timing_slot = -1;
    inject_prefetch(plan, stream >= 0, reinterpret_cast<CUstream>(static_cast<intptr_t>(stream)));
    if (call < 5)
        std::fprintf(stderr, "[agent pid=%d] sched_prefetch_ranges: %d range(s) given, %zu prefetched, %zu bytes, stream=%lld\n",
                     getpid(), n, plan.segs.size(), plan.total_bytes, stream);
    return static_cast<long long>(plan.total_bytes);
}

/* ======================================================================
 * CLIENT SIDE OF THE QUEUE: register this launch with the daemon, block
 * until released, record the inflight slot the daemon reserved for us, and
 * execute any prefetch instruction it attached.
 * ====================================================================== */

static void request_launch_permission(const char* kernel_name, const char* layer_tag,
                                      bool have_stream, CUstream app_stream,
                                      const LaunchArgs& args, uint64_t entry_seq, const LaunchDims& dims) {
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
    std::strncpy(g_state->slots[slot].kernel_base, kernel_base_name(kernel_name).c_str(), NAME_LEN - 1);
    g_state->slots[slot].kernel_base[NAME_LEN - 1] = '\0';
    std::strncpy(g_state->slots[slot].model, model_tag().c_str(), sizeof g_state->slots[slot].model - 1);
    g_state->slots[slot].model[sizeof g_state->slots[slot].model - 1] = '\0';
    std::memcpy(g_state->slots[slot].grid, dims.grid, sizeof dims.grid);
    std::memcpy(g_state->slots[slot].block, dims.block, sizeof dims.block);
    g_state->slots[slot].active = 1;
    g_state->slots[slot].ready = 0;
    g_state->slots[slot].inflight_slot = -1;
    g_state->slots[slot].kernel_id = 0;
    g_state->slots[slot].do_prefetch = -1;   // the daemon fills in 1 (prefetch) or 0 (don't) before it releases us
    g_state->slots[slot].agent_kernel_id = kernel_id;   // lets the daemon's decision log name this kernel: (pid, kernel_id)
    g_state->slots[slot].batch_id = 0;                  // the daemon fills these in before it releases us
    g_state->slots[slot].batch_size = 0;
    g_state->slots[slot].batch_pos = 0;
    g_state->slots[slot].request_ns = mono_ns();
    pthread_mutex_unlock(&g_state->mutex);

    sem_post(&g_state->new_request_sem);     // wake the daemon
    sem_wait(&g_state->slots[slot].wake_sem); // block until IT releases THIS slot

    pthread_mutex_lock(&g_state->mutex);
    t_current_inflight_slot = g_state->slots[slot].inflight_slot; // carry through to EXIT
    int do_prefetch = g_state->slots[slot].do_prefetch;           // the scheduler's decision; copy it BEFORE the slot is recycled
    const uint64_t batch_id = g_state->slots[slot].batch_id;      // which batch the scheduler released us in, likewise
    const int batch_size = g_state->slots[slot].batch_size, batch_pos = g_state->slots[slot].batch_pos;
    g_state->slots[slot].active = 0;
    pthread_mutex_unlock(&g_state->mutex);

    // WHETHER to prefetch is the scheduler's decision: do_prefetch, written by the daemon before it released us
    // (1 = prefetch, 0 = don't). WHAT to prefetch stays agent-side: the tensors owned by the module this kernel runs
    // in, found from its forward-hook tag (see the policy block above) and capped. A module entry is prefetched at most
    // once, on the first kernel the scheduler allows, however many kernels the module launches (a GEMM and its split-K
    // reduce, say) -- the same once-per-module-call behaviour as the hook-triggered path.
    if (do_prefetch != 0 && do_prefetch != 1) {
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true))
            std::fprintf(stderr, "[agent pid=%d] WARNING: the daemon released a kernel without setting do_prefetch "
                                 "(got %d): is the daemon binary or /dev/shm segment from an older build? "
                                 "Treating it as 'do not prefetch'.\n", getpid(), do_prefetch);
        do_prefetch = 0;
    }
    PrefetchPlan plan;
    bool selection_ran = false;
    if (do_prefetch == 0) {
        g_pf_denied++;
    } else if (entry_seq != 0 && entry_seq == t_last_prefetch_entry) {
        g_pf_dup++;
    } else {
        select_prefetch_targets(layer_tag, &plan);
        selection_ran = true;
    }

    std::string args_check = "-";
    std::string prefetched_tensors;
    if (plan.segs.empty()) {
        if (selection_ran) g_pf_skipped++;            // allowed, but this kernel's module owns nothing worth prefetching
    } else {
        g_pf_kernels++;
        t_last_prefetch_entry = entry_seq;
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
    t_timing_slot = (have_stream && timing_enabled()) ? timing_acquire(kernel_id, kernel_name) : -1;
    if (t_timing_slot >= 0) {
        g_timing[t_timing_slot].hook_tag   = layer_tag;
        g_timing[t_timing_slot].args_check = args_check;
        g_timing[t_timing_slot].pf_bytes   = plan.total_bytes;
        g_timing[t_timing_slot].pf_tensors = prefetched_tensors;
        g_timing[t_timing_slot].batch_id   = batch_id;
        g_timing[t_timing_slot].batch_size = batch_size;
        g_timing[t_timing_slot].batch_pos  = batch_pos;
        std::memcpy(g_timing[t_timing_slot].block, dims.block, sizeof dims.block);
        std::memcpy(g_timing[t_timing_slot].grid, dims.grid, sizeof dims.grid);
    }

    if (!plan.segs.empty()) {
        inject_prefetch(plan, have_stream, app_stream);
    }

    // real_start goes AFTER the prefetch wait, so the real kernel's interval excludes the prefetch.
    if (t_timing_slot >= 0) {
        g_timing[t_timing_slot].start_ns = mono_ns();     // BEFORE the event: the kernel cannot start earlier than this
        cuEventRecord(g_timing[t_timing_slot].real_start, app_stream);
    }
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
        const char* layer_tag = t_tag_stack.empty() ? "unknown" : t_tag_stack.front().c_str();
        const uint64_t entry_seq = t_tag_entry.empty() ? 0 : t_tag_entry.front();   // which entry of that module this is
        const char* kernel_name = (data->symbolName && data->symbolName[0]) ? data->symbolName : "unknown";
        // Find the app's stream so inject_prefetch can make it wait on the
        // prefetch before the real kernel is queued onto it.
        CUstream app_stream = 0;
        bool have_stream = launch_stream(cbid, data->functionParams, &app_stream);
        LaunchArgs args;
        launch_args(cbid, data->functionParams, &args);

        request_launch_permission(kernel_name, layer_tag, have_stream, app_stream, args, entry_seq,
                                  launch_dims(cbid, data->functionParams));
        // returning here lets CUPTI/the driver proceed with the real launch
        return;
    }

    if (data->callbackSite == CUPTI_API_EXIT) {
        CUstream stream = 0;
        bool resolved = launch_stream(cbid, data->functionParams, &stream);

        const int ts_idx = t_timing_slot;                                  // timing_mark_end() resets it
        const uint32_t ts_gen = ts_idx >= 0 ? g_timing[ts_idx].gen.load(std::memory_order_acquire) : 0;
        void* userdata_for_completion = pack_completion(t_current_inflight_slot, ts_idx, ts_gen);

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
                timing_stamp_end(ts_idx, ts_gen, NO_END_NS);              // no callback will come: recycle the slot, leave end_ns blank
                release_inflight_slot(t_current_inflight_slot);           // the kernel may still be running
            }
        } else {
            std::fprintf(stderr, "[agent pid=%d] WARNING: couldn't resolve stream for "
                            "completion tracking (cbid=%d), clearing inflight slot immediately\n",
                            getpid(), static_cast<int>(cbid));
            release_inflight_slot(t_current_inflight_slot);
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

    // SCHED_NO_CUPTI=1: leave CUPTI alone (so Nsight Compute can use it) and do not intercept launches. The weight
    // registry and sched_prefetch() still work; prefetches then come from Python hooks, not from the launch callback.
    if (const char* e = std::getenv("SCHED_NO_CUPTI"); e && std::atoi(e) != 0) {
        std::fprintf(stderr, "[agent pid=%d] SCHED_NO_CUPTI=1: NOT subscribing to CUPTI; launches are not intercepted, "
                             "prefetch is triggered by sched_prefetch() calls\n", getpid());
        return;
    }

    CUptiResult res = cuptiSubscribe(&g_subscriber, (CUpti_CallbackFunc)cupti_callback, nullptr);
    if (res != CUPTI_SUCCESS) {
        std::fprintf(stderr, "[agent pid=%d] FATAL: cuptiSubscribe failed: %d\n", getpid(), res);
        return;
    }

    // intercept CUDA kernel launches
    cuptiEnableCallback(1, g_subscriber, CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel);
    cuptiEnableCallback(1, g_subscriber, CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx);
    if (!timing_enabled())
        std::fprintf(stderr, "[agent pid=%d] SCHED_LOG=%s: no CUDA-event timing and no timing log (kernels are still intercepted and scheduled)\n", getpid(), std::getenv("SCHED_LOG"));
    g_cupti_active.store(true);
    std::fprintf(stderr, "[agent pid=%d] CUPTI subscriber attached\n", getpid());
}