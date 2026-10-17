#include "shared_state.hpp"

#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <algorithm>
#include <map>
#include <random>
#include <string>
#include <vector>

// SIGINT handler override to allow logging
static std::atomic<bool> g_stop{false};
extern "C" void on_signal(int) { g_stop = true; }

static SchedulerSharedState* state = nullptr;

// CLOCK_MONOTONIC in nanoseconds: the clock the agents use for request_ns / start_ns / end_ns, so every timestamp in the
// daemon log and in the agent logs can be compared directly (also across processes).
static inline uint64_t mono_ns() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull + static_cast<uint64_t>(ts.tv_nsec);
}

/* ======================================================================
 * POLICY -- the part to edit.
 * ====================================================================== */

static int policy_compute_priority(const char* kernel_name, const char* layer_tag) {
    (void)kernel_name;
    (void)layer_tag;
    return 0; // placeholder: all equal, so ties fall to the lowest slot index (NOT arrival order)
}

// Whether the agent should prefetch for this kernel launch (1) or not (0). Written into the kernel's slot before it is
// released; the agent does what it says. Placeholder policy: always prefetch.
static int policy_should_prefetch(const char* kernel_name, const char* layer_tag) {
    (void)kernel_name;
    (void)layer_tag;
    return 0;
}

// BATCHING. Whenever the GPU is free the daemon looks at every pending kernel and calls policy_select_batch() to choose which
// of them to release TOGETHER. The whole batch is released at once and the next batch is formed only after every kernel of it
// has finished. A batch of one is the previous behaviour (one kernel at a time), and it is the default.
//
// Settings (environment of the daemon):
//   SCHED_BATCH_MAX   most kernels per batch                       (default 1 = never batch)
//   SCHED_BATCH_PROB  chance of batching when the chance arises    (default 1.0)
//   SCHED_BATCH_SEED  seed of the draw behind SCHED_BATCH_PROB     (default 0: reproducible)
//
// The rule built in below: at most ONE kernel per process in a batch. Kernels of one process can depend on each other
// (stream order, data), kernels of different processes cannot, so only the latter are safe to run together. Whether they
// actually overlap on the GPU is up to the GPU: processes are time-sliced unless CUDA MPS is running.
struct BatchConfig { int max = 2; double prob = 1.0; };
struct Candidate { int slot; pid_t pid; int priority; };               // one pending kernel
struct BatchPlan { std::vector<int> pick; std::string reason; };        // indices into `pending`, in release order

static BatchPlan policy_select_batch(std::vector<Candidate>& pending, const BatchConfig& cfg, std::mt19937& rng) {
    // Same order as before batching existed: lowest priority value first, ties to the lowest slot index.
    std::stable_sort(pending.begin(), pending.end(), [](const Candidate& a, const Candidate& b) {
        return a.priority != b.priority ? a.priority < b.priority : a.slot < b.slot;
    });
    BatchPlan plan;
    plan.pick.push_back(0);
    if (cfg.max <= 1) { plan.reason = "single: SCHED_BATCH_MAX=1"; return plan; }

    std::vector<int> others;                       // other pending kernels that may join: one per process, in priority order
    std::vector<pid_t> used{pending[0].pid};
    for (size_t i = 1; i < pending.size() && static_cast<int>(others.size()) + 1 < cfg.max; i++) {
        if (std::find(used.begin(), used.end(), pending[i].pid) != used.end()) continue;
        others.push_back(static_cast<int>(i));
        used.push_back(pending[i].pid);
    }
    if (others.empty()) { plan.reason = "single: no pending kernel from another process"; return plan; }

    char buf[128];
    const double u = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
    if (cfg.prob >= 1.0 || u < cfg.prob) {
        for (int i : others) plan.pick.push_back(i);
        std::snprintf(buf, sizeof buf, "batched: %zu kernels from %zu processes", plan.pick.size(), plan.pick.size());
    } else {
        std::snprintf(buf, sizeof buf, "single: batching skipped (draw %.3f >= SCHED_BATCH_PROB %.3f)", u, cfg.prob);
    }
    plan.reason = buf;
    return plan;
}

/* ======================================================================
 * INFLIGHT TRACKING -- per-kernel slots in shared memory. Caller must
 * already hold state->mutex.
 * ====================================================================== */

// Returns the reserved slot's index, or -1 if the inflight table is full.
static int reserve_inflight_slot_locked(pid_t pid, const char* kernel_name, const char* layer_tag) {
    for (int i = 0; i < MAX_INFLIGHT; i++) {
        if (state->inflight[i].active.load() == 0) {
            state->inflight[i].pid = pid;
            std::strncpy(state->inflight[i].kernel_name, kernel_name, NAME_LEN - 1);
            std::strncpy(state->inflight[i].layer_tag, layer_tag, NAME_LEN - 1);
            state->inflight[i].end_ns.store(0);      // the agent writes the real end time when the kernel finishes
            state->inflight[i].active.store(1);
            return i;
        }
    }
    return -1;
}

/* ======================================================================
 * DECISION LOG: one row per kernel, written when its batch has finished
 * (so the row can carry the kernel's end time). Join with the agent logs on (pid, kernel_id).
 *
 *   batch_id, batch_size, batch_pos   the decision: which batch, how big, this kernel's place in the order of wake-ups
 *   decision_ns                       when the daemon fixed the batch and marked it released (the batch's launch time)
 *   post_ns                           the instant just before THIS kernel's wake-up was posted (a few us after decision_ns,
 *                                     in batch_pos order); decision_ns <= post_ns <= the agent's start_ns
 *   pid, kernel_id                    which kernel: the process and its own per-process launch counter (= kernel_id in the
 *                                     agent's agent_<pid>_timing.tsv); arrival_seq is the daemon's global arrival order
 *   kernel_name, layer_tag
 *   request_ns, queue_wait_us         when the launch was intercepted, and how long it waited for its release
 *   end_ns                            when the kernel finished (the agent's completion callback); blank if not known
 *   n_pending, pending_pids           the choice the policy had: how many kernels were waiting at the decision, and from whom
 *   do_prefetch, inflight_slot
 *   reason                            why the policy chose this batch
 * ====================================================================== */

struct Member {
    int slot = -1, inflight = -1, agent_kid = 0, arrival_seq = 0, do_prefetch = 0, pos = 0;
    pid_t pid = 0;
    std::string name, tag;
    uint64_t request_ns = 0, post_ns = 0;
};
struct Batch {
    uint64_t id = 0, decision_ns = 0;
    int n_pending = 0;
    std::string pending_pids, reason;
    std::vector<Member> m;
};

static FILE* g_log = nullptr;
static uint64_t g_last_flush_ns = 0;
static std::map<int, uint64_t> g_size_hist;   // batch size -> how many batches

static void append_rows(std::string& out, const Batch& b) {
    char buf[1024];
    for (const Member& m : b.m) {
        const uint64_t end = (m.inflight >= 0) ? state->inflight[m.inflight].end_ns.load(std::memory_order_acquire) : 0;
        const double wait_us = (m.request_ns && m.post_ns >= m.request_ns) ? (m.post_ns - m.request_ns) / 1000.0 : -1.0;
        std::snprintf(buf, sizeof buf, "%llu\t%zu\t%d\t%llu\t%llu\t%d\t%d\t%d\t",
                      (unsigned long long)b.id, b.m.size(), m.pos, (unsigned long long)b.decision_ns,
                      (unsigned long long)m.post_ns, (int)m.pid, m.agent_kid, m.arrival_seq);
        out += buf;
        out += m.name; out += '\t'; out += m.tag; out += '\t';
        std::snprintf(buf, sizeof buf, "%llu\t", (unsigned long long)m.request_ns);
        out += buf;
        if (wait_us >= 0) { std::snprintf(buf, sizeof buf, "%.3f", wait_us); out += buf; }
        out += '\t';
        if (end) { std::snprintf(buf, sizeof buf, "%llu", (unsigned long long)end); out += buf; }
        std::snprintf(buf, sizeof buf, "\t%d\t", b.n_pending);
        out += buf;
        out += b.pending_pids;
        std::snprintf(buf, sizeof buf, "\t%d\t%d\t", m.do_prefetch, m.inflight);
        out += buf;
        out += b.reason; out += '\n';
    }
}

static void write_log(const std::string& rows, bool force_flush) {
    if (!g_log || rows.empty()) { if (g_log && force_flush) std::fflush(g_log); return; }
    std::fputs(rows.c_str(), g_log);
    const uint64_t now = mono_ns();
    if (force_flush || now - g_last_flush_ns > 1000000000ull) { std::fflush(g_log); g_last_flush_ns = now; }
}

static void open_log() {
    const char* path_env = std::getenv("SCHED_DAEMON_LOG");
    const char* dir = std::getenv("SCHED_LOG_DIR");
    std::string path = path_env ? path_env : std::string(dir ? dir : ".") + "/scheduler_" + std::to_string(getpid()) + "_decisions.tsv";
    g_log = std::fopen(path.c_str(), "w");
    if (!g_log) { std::fprintf(stderr, "[daemon] WARNING: could not open %s for the decision log: %s\n", path.c_str(), std::strerror(errno)); return; }
    static char buf[1 << 20];
    std::setvbuf(g_log, buf, _IOFBF, sizeof buf);
    std::fputs("batch_id\tbatch_size\tbatch_pos\tdecision_ns\tpost_ns\tpid\tkernel_id\tarrival_seq\tkernel_name\tlayer_tag\t"
               "request_ns\tqueue_wait_us\tend_ns\tn_pending\tpending_pids\tdo_prefetch\tinflight_slot\treason\n", g_log);
    std::fprintf(stderr, "[daemon] decision log: %s\n", path.c_str());
}

/* ======================================================================
 * SETUP + MAIN LOOP
 * ====================================================================== */

static void init_shared_state() {
    int fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
    if (fd < 0) { perror("[daemon] shm_open"); std::exit(1); }
    if (ftruncate(fd, sizeof(SchedulerSharedState)) != 0) {
        perror("[daemon] ftruncate"); std::exit(1);
    }

    state = static_cast<SchedulerSharedState*>(
        mmap(nullptr, sizeof(SchedulerSharedState), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
    if (state == MAP_FAILED) { perror("[daemon] mmap"); std::exit(1); }
    close(fd);

    pthread_mutexattr_t mattr;
    pthread_mutexattr_init(&mattr);
    pthread_mutexattr_setpshared(&mattr, PTHREAD_PROCESS_SHARED);
    pthread_mutex_init(&state->mutex, &mattr);

    sem_init(&state->new_request_sem, /*pshared=*/1, 0);
    for (int i = 0; i < MAX_PENDING; i++) {
        sem_init(&state->slots[i].wake_sem, /*pshared=*/1, 0);
        state->slots[i].active = 0;
        state->slots[i].inflight_slot = -1;
        state->slots[i].do_prefetch = -1;
        state->slots[i].request_ns = 0;
        state->slots[i].agent_kernel_id = 0;
        state->slots[i].batch_id = 0;
        state->slots[i].batch_size = 0;
        state->slots[i].batch_pos = 0;
    }
    for (int i = 0; i < MAX_INFLIGHT; i++) {
        state->inflight[i].active.store(0);
        state->inflight[i].end_ns.store(0);
    }
    state->region_count = 0;

    // NOTE: do NOT rely on shm being zero-filled by default. shm_open with
    // O_CREAT silently reuses an existing segment from a prior run if one
    // is still present in /dev/shm, and ftruncate to the SAME size is a
    // no-op on content -- which is exactly why every field above is reset
    // explicitly rather than assumed to already be zero.

    state->initialized.store(1);
    std::fprintf(stderr, "[daemon] shared memory ready, pid=%d\n", getpid());
}

int main() {
    struct sigaction sa{};
    sa.sa_handler = on_signal;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;                      // no SA_RESTART, so blocking calls return EINTR
    sigaction(SIGINT,  &sa, nullptr);     // Ctrl+C

    BatchConfig cfg;
    if (const char* e = std::getenv("SCHED_BATCH_MAX"))  cfg.max  = std::max(1, std::min(MAX_INFLIGHT, std::atoi(e)));
    if (const char* e = std::getenv("SCHED_BATCH_PROB")) cfg.prob = std::max(0.0, std::min(1.0, std::atof(e)));
    std::mt19937 rng(std::getenv("SCHED_BATCH_SEED") ? static_cast<unsigned>(std::atol(std::getenv("SCHED_BATCH_SEED"))) : 0u);

    init_shared_state();
    open_log();
    std::fprintf(stderr, "[daemon] batching: SCHED_BATCH_MAX=%d%s, one kernel per process per batch%s\n", cfg.max,
                 cfg.max > 1 ? (std::string(" SCHED_BATCH_PROB=") + std::to_string(cfg.prob)).c_str() : "",
                 cfg.max > 1 ? "" : " (default: one kernel at a time)");

    int arrived_kernel_count = 0;
    uint64_t next_batch_id = 1;
    bool have_cur = false;
    Batch cur;

    std::fprintf(stderr, "[daemon] waiting for agent processes to attach...\n");

    while (!g_stop) {
        if (sem_wait(&state->new_request_sem) != 0) continue;       // EINTR (Ctrl+C): the loop condition ends it

        std::string rows;                       // decision-log rows for a batch that just finished
        Batch next;                             // the batch released in this iteration, if any
        bool released = false;

        pthread_mutex_lock(&state->mutex);

        // 1. A batch is running: the GPU is free again only when EVERY kernel of it has finished.
        if (have_cur) {
            bool all_done = true;
            for (const Member& m : cur.m)
                if (state->inflight[m.inflight].active.load() != 0) { all_done = false; break; }
            if (!all_done) { pthread_mutex_unlock(&state->mutex); continue; }   // leave the candidates pending, try again later
            append_rows(rows, cur);
            g_size_hist[static_cast<int>(cur.m.size())]++;
            have_cur = false;
        }

        // 2. Everything waiting for a release.
        std::vector<Candidate> pending;
        std::vector<int> arrival(MAX_PENDING, 0);
        for (int i = 0; i < MAX_PENDING; i++) {
            if (state->slots[i].active && !state->slots[i].ready) {
                if (!state->slots[i].kernel_id) state->slots[i].kernel_id = ++arrived_kernel_count;   // first sighting
                arrival[i] = state->slots[i].kernel_id;
                state->slots[i].priority = policy_compute_priority(state->slots[i].kernel_name, state->slots[i].layer_tag);
                pending.push_back({i, state->slots[i].pid, state->slots[i].priority});
            }
        }

        // 3. Decide, reserve an inflight slot for each member, and mark them all released.
        if (!pending.empty()) {
            const BatchPlan plan = policy_select_batch(pending, cfg, rng);
            next.id = next_batch_id;
            next.n_pending = static_cast<int>(pending.size());
            for (size_t i = 0; i < pending.size(); i++) {
                if (i) next.pending_pids += ',';
                next.pending_pids += std::to_string(static_cast<int>(pending[i].pid));
            }
            next.reason = plan.reason;
            for (int idx : plan.pick) {
                const int s = pending[idx].slot;
                const int infl = reserve_inflight_slot_locked(state->slots[s].pid, state->slots[s].kernel_name, state->slots[s].layer_tag);
                if (infl == -1) {                                     // inflight table full: release only what fits
                    if (next.m.empty()) next.reason += " (deferred: inflight table full)";
                    break;
                }
                Member m;
                m.slot = s; m.inflight = infl; m.pos = static_cast<int>(next.m.size());
                m.pid = state->slots[s].pid;
                m.agent_kid = state->slots[s].agent_kernel_id;
                m.arrival_seq = arrival[s];
                m.name = state->slots[s].kernel_name;
                m.tag = state->slots[s].layer_tag;
                m.request_ns = state->slots[s].request_ns;
                m.do_prefetch = policy_should_prefetch(state->slots[s].kernel_name, state->slots[s].layer_tag);
                next.m.push_back(m);
            }
            if (!next.m.empty()) {
                next.decision_ns = mono_ns();
                for (const Member& m : next.m) {
                    state->slots[m.slot].inflight_slot = m.inflight;
                    state->slots[m.slot].do_prefetch = m.do_prefetch;
                    state->slots[m.slot].batch_id = next.id;
                    state->slots[m.slot].batch_size = static_cast<int>(next.m.size());
                    state->slots[m.slot].batch_pos = m.pos;
                    state->slots[m.slot].ready = 1;
                }
                next_batch_id++;
                released = true;
            }
        }

        pthread_mutex_unlock(&state->mutex);

        // 4. Wake the batch's kernels as close together as possible; the time-critical part comes before any logging.
        if (released) {
            for (Member& m : next.m) {
                m.post_ns = mono_ns();                                  // just BEFORE the wake-up: decision_ns <= post_ns <= the agent's start_ns
                sem_post(&state->slots[m.slot].wake_sem);
            }
            for (const Member& m : next.m)
                std::fprintf(stderr, "[daemon] release pid=%d kernel=%s layer=%s inflight_slot=%d prefetch=%d batch=%llu size=%zu\n",
                             (int)m.pid, m.name.c_str(), m.tag.c_str(), m.inflight, m.do_prefetch,
                             (unsigned long long)next.id, next.m.size());
            cur = next;
            have_cur = true;
        }
        write_log(rows, false);
    }

    // Shutdown: log the batch that was still running (end times only where the kernels had finished) and flush.
    std::string rows;
    if (have_cur) { append_rows(rows, cur); g_size_hist[static_cast<int>(cur.m.size())]++; }
    write_log(rows, true);
    if (g_log) std::fclose(g_log);
    std::fprintf(stderr, "[daemon] batches by size:");
    for (const auto& kv : g_size_hist) std::fprintf(stderr, " %d x%llu", kv.first, (unsigned long long)kv.second);
    std::fprintf(stderr, "\n");
    return 0;
}