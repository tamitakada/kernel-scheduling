#include "shared_state.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

static SchedulerSharedState* state = nullptr;
static volatile sig_atomic_t g_stop = 0;
static void on_signal(int) { g_stop = 1; }
static uint64_t mono_ns() { timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return static_cast<uint64_t>(t.tv_sec) * 1000000000ull + static_cast<uint64_t>(t.tv_nsec); }   // the agents' clock
static double env_num(const char* n, double d) { const char* e = std::getenv(n); return (e && *e) ? std::atof(e) : d; }
static std::string env_str(const char* n, const char* d) { const char* e = std::getenv(n); return (e && *e) ? e : d; }

// ------------------------------------------------------------ configuration
static std::string g_model_a = "qwen", g_model_b = "whisper", g_log_dir = ".";
static int g_reps = 1;
static bool g_solo = false, g_log_singles = true, g_trace = false;
static uint64_t g_wait_ns = 3000000ull, g_hold_ns = 150000000ull, g_cool_ns = 2000000000ull;

// ------------------------------------------------------------ kernel types, and the pairs found so far
// A kernel TYPE is "short name|bx,by,bz". The agent already reduces a kernel's symbol to its short name (kernel_base in the slot).
static std::string type_key(const std::string& name, uint32_t bx, uint32_t by, uint32_t bz) { return name + "|" + std::to_string(bx) + "," + std::to_string(by) + "," + std::to_string(bz); }
static std::string slot_type(const PendingSlot& s) { return type_key(std::string(s.kernel_base, strnlen(s.kernel_base, NAME_LEN)), s.block[0], s.block[1], s.block[2]); }
static std::string pair_key(const std::string& type_a, const std::string& type_b) { return type_a + "\x1f" + type_b; }

struct TypeInfo { uint64_t released = 0; int open = 0; };                                   // open = how many types of the OTHER model it still has an unprofiled pair with
static std::unordered_map<std::string, TypeInfo> g_types[2];                                // [0] = types seen of model A, [1] = of model B
static std::unordered_map<std::string, int> g_done;                                         // pair -> gangs run so far (absent: none)
static size_t g_pairs_done = 0, g_gangs_done = 0;                                           // unique pairs ganged at least once; gangs of unprofiled pairs run in all
static std::vector<std::pair<int, std::string>> g_fresh;                                    // types seen for the first time since the console last printed them

// ------------------------------------------------------------ the policy
struct Cand { int slot; pid_t pid; int side; std::string type; uint64_t request_ns; };      // side: 0 = a kernel of model A, 1 = of model B, -1 = of neither
enum Kind { WAIT = 0, PROFILE, FILLER, ALONE_HOLD, ALONE };                                  // what a decision does
static const char* KIND[] = {"wait", "gang_profile", "gang_filler", "alone_other_waits", "alone"};
struct Decision { Kind kind = WAIT; std::vector<int> pick; uint64_t wake_ns = 0; std::string why; int rep = 0; };

static std::unordered_map<std::string, uint64_t> g_rest;                                    // type -> counts as being in no unprofiled pair until then (ns)
static uint64_t g_last_release_ns = 0;
static size_t g_n_filler = 0, g_n_alone_hold = 0, g_n_alone = 0;

static void discover(const Cand& c) {                                                       // register a type the first time it is seen
    if (c.side < 0) return;
    auto& mine = g_types[c.side]; auto& other = g_types[1 - c.side];
    if (mine.count(c.type)) return;
    TypeInfo t; t.open = static_cast<int>(other.size());                                    // a new type is unprofiled with every type of the other model seen so far ...
    for (auto& kv : other) kv.second.open++;                                                // ... and each of those has one more unprofiled pair
    mine.emplace(c.type, t); g_fresh.push_back({c.side, c.type});
}
static int gangs_of(const Cand& x, const Cand& y) {                                         // x and y wait in different processes: how often has this pair been ganged?
    const Cand& a = x.side == 0 ? x : y; const Cand& b = x.side == 0 ? y : x;
    const auto it = g_done.find(pair_key(a.type, b.type)); return it == g_done.end() ? 0 : it->second;
}
static bool pair_open(const Cand& x, const Cand& y) { return x.side >= 0 && y.side >= 0 && x.side != y.side && gangs_of(x, y) < g_reps; }
static bool in_open_pair(const Cand& c, uint64_t now) {                                     // c's type is in some unprofiled pair, and is not resting
    if (c.side < 0) return false;
    const auto it = g_types[c.side].find(c.type); if (it == g_types[c.side].end() || it->second.open <= 0) return false;
    const auto r = g_rest.find(c.type); return r == g_rest.end() || r->second <= now;
}
static int consume(const Cand& x, const Cand& y) {                                          // a gang of this pair is released: returns the rep number (1, 2, ...)
    const Cand& a = x.side == 0 ? x : y; const Cand& b = x.side == 0 ? y : x;
    const int n = ++g_done[pair_key(a.type, b.type)]; g_gangs_done++;
    if (n == 1) g_pairs_done++;
    if (n == g_reps) { g_types[0][a.type].open--; g_types[1][b.type].open--; }
    return n;
}

static Decision choose(std::vector<Cand>& w, uint64_t now) {                                // w: the waiting kernels (at most one per process); nothing is running
    Decision d;
    if (w.empty()) return d;
    std::stable_sort(w.begin(), w.end(), [](const Cand& a, const Cand& b) { return a.request_ns < b.request_ns || (a.request_ns == b.request_ns && a.slot < b.slot); });
    const Cand& f = w[0]; int si = -1;
    for (size_t i = 1; i < w.size(); i++) if (w[i].pid != f.pid) { si = static_cast<int>(i); break; }
    if (g_solo) { d.kind = ALONE; d.pick = {0}; d.why = "solo run: one kernel at a time"; return d; }
    for (const Cand& c : w) discover(c);
    for (size_t i = 0; i < w.size(); i++) if (w[i].side < 0) { d.kind = ALONE; d.pick = {static_cast<int>(i)}; d.why = "a process that is neither model: it is not part of the experiment, it runs alone at once"; return d; }   // rule 0
    auto rest = [&](const Cand& c) { g_rest[c.type] = now + g_cool_ns; };
    if (si >= 0) {
        const Cand& s = w[si];
        if (pair_open(f, s)) { d.kind = PROFILE; d.pick = {0, si}; d.rep = consume(f, s); d.why = "an unprofiled pair: gang"; return d; }                        // rule 1
        const bool f_open = in_open_pair(f, now), s_open = in_open_pair(s, now);
        const bool hold_f = f_open && (now - f.request_ns) < g_hold_ns, hold_s = !hold_f && s_open && (now - s.request_ns) < g_hold_ns;
        if (f_open && !hold_f) rest(f);                                                                                                                     // waited long enough
        if (!hold_f && s_open && !hold_s) rest(s);
        if (hold_f) { d.kind = ALONE_HOLD; d.pick = {si}; d.why = "pair already profiled; the first kernel is in another unprofiled pair: it waits, the other process runs alone"; }   // rule 2
        else if (hold_s) { d.kind = ALONE_HOLD; d.pick = {0}; d.why = "pair already profiled; the second kernel is in another unprofiled pair: it waits, the first process runs alone"; }   // rule 3
        else { d.kind = FILLER; d.pick = {0, si}; d.why = "no unprofiled pair for either: gang anyway"; }                                                    // rule 4
        return d;
    }
    if (in_open_pair(f, now)) {                                                                                                                             // rule 5
        if ((now - f.request_ns) < g_hold_ns && (now - g_last_release_ns) < g_wait_ns) { d.wake_ns = g_last_release_ns + g_wait_ns; d.why = "waiting for the other process's next kernel"; return d; }
        rest(f);
    }
    d.kind = ALONE; d.pick = {0}; d.why = "alone: no partner waiting"; return d;
}
static Decision decide(std::vector<Cand>& w, uint64_t now) {
    Decision d = choose(w, now);
    if (!d.pick.empty()) { g_last_release_ns = now; for (int i : d.pick) if (w[i].side >= 0) g_types[w[i].side][w[i].type].released++; }
    return d;
}

// ------------------------------------------------------------ shared memory (the same layout and handshake as scheduler_daemon)
static void init_shared_state() {
    const int fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666); if (fd < 0) { std::perror("[pairprof] shm_open"); std::exit(1); }
    if (ftruncate(fd, sizeof(SchedulerSharedState)) != 0) { std::perror("[pairprof] ftruncate"); std::exit(1); }
    state = static_cast<SchedulerSharedState*>(mmap(nullptr, sizeof(SchedulerSharedState), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
    if (state == MAP_FAILED) { std::perror("[pairprof] mmap"); std::exit(1); }
    close(fd);
    pthread_mutexattr_t ma; pthread_mutexattr_init(&ma); pthread_mutexattr_setpshared(&ma, PTHREAD_PROCESS_SHARED); pthread_mutex_init(&state->mutex, &ma);
    sem_init(&state->new_request_sem, 1, 0);
    for (int i = 0; i < MAX_PENDING; i++) {                                                   // never assume the segment is zero: a /dev/shm file of an earlier run is reused
        PendingSlot& s = state->slots[i]; sem_init(&s.wake_sem, 1, 0);
        s.active = 0; s.ready = 0; s.inflight_slot = -1; s.do_prefetch = 0; s.has_prefetch = 0; s.request_ns = 0; s.agent_kernel_id = 0; s.batch_id = 0; s.batch_size = 0; s.batch_pos = 0;
        s.kernel_base[0] = '\0'; s.model[0] = '\0'; for (int d = 0; d < 3; d++) { s.grid[d] = 0; s.block[d] = 0; }
    }
    for (int i = 0; i < MAX_INFLIGHT; i++) { state->inflight[i].active.store(0); state->inflight[i].end_ns.store(0); }
    state->region_count = 0; state->initialized.store(1);
    std::fprintf(stderr, "[pairprof] shared memory ready, pid=%d\n", getpid());
}
static int reserve_inflight(pid_t pid, const char* name, const std::vector<char>& held) {   // the agent clears this slot's `active` when the kernel has finished
    for (int i = 0; i < MAX_INFLIGHT; i++) if (!held[i] && state->inflight[i].active.load() == 0) {
        state->inflight[i].pid = pid; std::strncpy(state->inflight[i].kernel_name, name, NAME_LEN - 1); state->inflight[i].layer_tag[0] = '\0';
        state->inflight[i].end_ns.store(0); state->inflight[i].active.store(1); return i;
    }
    return -1;
}

// ------------------------------------------------------------ main loop
struct Running { int inflight; };
static std::string split_type(const std::string& t, std::string* block) { const size_t p = t.rfind('|'); *block = t.substr(p + 1); return t.substr(0, p); }
int main() {
    g_model_a = env_str("PAIRPROF_MODEL_A", "qwen"); g_model_b = env_str("PAIRPROF_MODEL_B", "whisper");
    g_reps = std::max(1, static_cast<int>(env_num("PAIRPROF_REPS", 1))); g_solo = env_num("PAIRPROF_SOLO", 0) != 0; g_trace = env_num("PAIRPROF_TRACE", 0) != 0;
    g_log_singles = env_str("PAIRPROF_LOG", "all") != "gangs";
    g_wait_ns = static_cast<uint64_t>(env_num("PAIRPROF_WAIT_US", 3000) * 1e3); g_hold_ns = static_cast<uint64_t>(env_num("PAIRPROF_MAX_HOLD_MS", 150) * 1e6); g_cool_ns = static_cast<uint64_t>(env_num("PAIRPROF_COOLDOWN_MS", 2000) * 1e6);
    g_log_dir = env_str("PAIRPROF_LOG_DIR", env_str("SCHED_LOG_DIR", ".").c_str());
    if (g_solo) std::fprintf(stderr, "[pairprof] SOLO run: one kernel at a time, nothing is ganged\n");
    else std::fprintf(stderr, "[pairprof] discovery mode: A = %s, B = %s; every pair of types seen is ganged %d time(s); patience %.1f ms, max hold %.0f ms, rest %.0f ms\n", g_model_a.c_str(), g_model_b.c_str(), g_reps, g_wait_ns / 1e6, g_hold_ns / 1e6, g_cool_ns / 1e6);
    struct sigaction sa{}; sa.sa_handler = on_signal; sigemptyset(&sa.sa_mask); sigaction(SIGINT, &sa, nullptr); sigaction(SIGTERM, &sa, nullptr); sigaction(SIGHUP, &sa, nullptr);
    init_shared_state();
    const std::string log_path = g_log_dir + "/pairprof_log_" + std::to_string(getpid()) + ".tsv";
    FILE* log = std::fopen(log_path.c_str(), "w"); if (!log) { std::perror("[pairprof] cannot open the log"); return 1; }
    std::fputs("batch_id\tkind\tpid\tkernel_id\tmodel\tkernel\tblock\trep\n", log); std::fprintf(stderr, "[pairprof] log: %s\n[pairprof] waiting for the agents...\n", log_path.c_str());

    std::vector<Running> running; uint64_t next_batch = 1, wake_ns = 0, last_progress = mono_ns(), t_start = mono_ns(); size_t n_gang_kernels = 0;
    while (!g_stop) {
        const uint64_t now0 = mono_ns();
        uint64_t wait_ns = 250000000ull;                                                      // nothing to do: wake up anyway every 250 ms
        if (wake_ns) wait_ns = wake_ns > now0 ? std::min<uint64_t>(wait_ns, wake_ns - now0) : 0;   // a waiting kernel's patience ends then
        if (wait_ns) {
            timespec until; clock_gettime(CLOCK_REALTIME, &until);
            until.tv_nsec += static_cast<long>(wait_ns % 1000000000ull); until.tv_sec += static_cast<time_t>(wait_ns / 1000000000ull);
            if (until.tv_nsec >= 1000000000L) { until.tv_sec++; until.tv_nsec -= 1000000000L; }
            sem_timedwait(&state->new_request_sem, &until);                                   // a new request, a finished kernel, or the timeout: all lead to a decision below
        }
        wake_ns = 0;

        std::vector<Cand> waiting; std::vector<int> released_slots; std::vector<std::string> rows, msgs;     // rows: for the log; msgs: for the console (both written after the unlock)
        pthread_mutex_lock(&state->mutex);
        for (size_t i = 0; i < running.size();) {                                             // retire the kernels that have finished
            if (state->inflight[running[i].inflight].active.load() == 0) running.erase(running.begin() + static_cast<long>(i)); else i++;
        }
        for (int i = 0; i < MAX_PENDING; i++) if (state->slots[i].active && !state->slots[i].ready) {
            const std::string model = state->slots[i].model;
            waiting.push_back({i, state->slots[i].pid, model == g_model_a ? 0 : model == g_model_b ? 1 : -1, slot_type(state->slots[i]), state->slots[i].request_ns});
        }
        if (running.empty() && !waiting.empty()) {                                            // the GPU is idle: decide
            const uint64_t now = mono_ns(); Decision d = decide(waiting, now); char b[640];
            for (const auto& ft : g_fresh) { std::snprintf(b, sizeof b, "[new type] %s: %s   (types seen: %s %zu, %s %zu)", ft.first == 0 ? g_model_a.c_str() : g_model_b.c_str(), ft.second.c_str(), g_model_a.c_str(), g_types[0].size(), g_model_b.c_str(), g_types[1].size()); msgs.push_back(b); }
            g_fresh.clear();
            if (g_trace) {
                std::string line = "[trace +" + std::to_string((now - t_start) / 1e6) + " ms] " + KIND[d.kind] + "  f=" + waiting[0].type;
                for (size_t i = 1; i < waiting.size(); i++) if (waiting[i].pid != waiting[0].pid) { line += "  s=" + waiting[i].type; break; }
                msgs.push_back(line + "   -- " + d.why);
            }
            wake_ns = d.wake_ns;
            if (!d.pick.empty()) {
                std::vector<char> held(MAX_INFLIGHT, 0); for (const Running& r : running) held[r.inflight] = 1;
                const uint64_t id = next_batch++; const int size = static_cast<int>(d.pick.size()); int pos = 0;
                for (int idx : d.pick) {
                    PendingSlot& s = state->slots[waiting[idx].slot];
                    const int infl = reserve_inflight(s.pid, s.kernel_name, held); if (infl < 0) break;     // cannot happen with two kernels in flight
                    held[infl] = 1; running.push_back({infl});
                    s.has_prefetch = 0; s.do_prefetch = 0; s.inflight_slot = infl; s.batch_id = id; s.batch_size = size; s.batch_pos = pos++; s.ready = 1;
                    released_slots.push_back(waiting[idx].slot);
                    if (d.kind == PROFILE || g_log_singles || d.kind == FILLER) {
                        std::snprintf(b, sizeof b, "%llu\t%s\t%d\t%d\t%s\t%s\t%u,%u,%u\t%s", static_cast<unsigned long long>(id), KIND[d.kind], static_cast<int>(s.pid), s.agent_kernel_id, s.model,
                                      std::string(s.kernel_base, strnlen(s.kernel_base, NAME_LEN)).c_str(), s.block[0], s.block[1], s.block[2], d.kind == PROFILE ? std::to_string(d.rep).c_str() : "");
                        rows.push_back(b);
                    }
                }
                if (d.kind == PROFILE || d.kind == FILLER) n_gang_kernels += 2;
                if (d.kind == PROFILE) {
                    const Cand& ca = waiting[d.pick[0]].side == 0 ? waiting[d.pick[0]] : waiting[d.pick[1]]; const Cand& cb = waiting[d.pick[0]].side == 0 ? waiting[d.pick[1]] : waiting[d.pick[0]];
                    if (g_trace || g_gangs_done <= 20) {
                        std::snprintf(b, sizeof b, "[gang %llu] %s: %s  +  %s: %s   rep %d/%d   (unique pairs profiled: %zu of %zu possible)", static_cast<unsigned long long>(id), g_model_a.c_str(), ca.type.c_str(), g_model_b.c_str(), cb.type.c_str(), d.rep, g_reps, g_pairs_done, g_types[0].size() * g_types[1].size());
                        msgs.push_back(b);
                    }
                } else if (d.kind == FILLER) g_n_filler++; else if (d.kind == ALONE_HOLD) g_n_alone_hold++; else g_n_alone++;
            }
        }
        pthread_mutex_unlock(&state->mutex);
        for (int s : released_slots) sem_post(&state->slots[s].wake_sem);                     // wake the released kernels as close together as possible
        for (const std::string& r : rows) std::fprintf(log, "%s\n", r.c_str());
        for (const std::string& m : msgs) std::fprintf(stderr, "%s\n", m.c_str());
        const uint64_t t = mono_ns();
        if (t - last_progress > 5000000000ull) {
            std::fflush(log); last_progress = t; const size_t possible = g_types[0].size() * g_types[1].size();
            std::fprintf(stderr, "[progress] types seen: %s %zu, %s %zu; unique pairs profiled: %zu of %zu possible (%.1f%%); kernels run: %zu in gangs, %zu alone while another waited, %zu alone otherwise\n",
                         g_model_a.c_str(), g_types[0].size(), g_model_b.c_str(), g_types[1].size(), g_pairs_done, possible, possible ? 100.0 * static_cast<double>(g_pairs_done) / static_cast<double>(possible) : 0.0, n_gang_kernels, g_n_alone_hold, g_n_alone);
        }
    }
    std::fclose(log);
    if (!g_solo) {
        const std::string tp = g_log_dir + "/pairprof_types_" + std::to_string(getpid()) + ".tsv", op = g_log_dir + "/pairprof_open_" + std::to_string(getpid()) + ".tsv"; std::string blk;
        if (FILE* f = std::fopen(tp.c_str(), "w")) {
            std::fputs("model\tkernel\tblock\treleased\tpartners_seen\tpartners_unprofiled\n", f);
            for (int side = 0; side < 2; side++) for (const auto& kv : g_types[side]) { const std::string n = split_type(kv.first, &blk); std::fprintf(f, "%s\t%s\t%s\t%llu\t%zu\t%d\n", side == 0 ? g_model_a.c_str() : g_model_b.c_str(), n.c_str(), blk.c_str(), static_cast<unsigned long long>(kv.second.released), g_types[1 - side].size(), kv.second.open); }
            std::fclose(f);
        }
        size_t never = 0;
        if (FILE* f = std::fopen(op.c_str(), "w")) {
            std::fputs("name_a\tblock_a\tname_b\tblock_b\tgangs_done\n", f); std::string bb;
            for (const auto& ka : g_types[0]) for (const auto& kb : g_types[1]) {
                const auto it = g_done.find(pair_key(ka.first, kb.first)); const int n = it == g_done.end() ? 0 : it->second; if (n >= g_reps) continue;
                if (++never <= 200000) { const std::string na = split_type(ka.first, &blk), nb = split_type(kb.first, &bb); std::fprintf(f, "%s\t%s\t%s\t%s\t%d\n", na.c_str(), blk.c_str(), nb.c_str(), bb.c_str(), n); }
            }
            std::fclose(f);
        }
        const size_t possible = g_types[0].size() * g_types[1].size();
        std::fprintf(stderr, "[pairprof] stopped: %zu %s types x %zu %s types = %zu possible pairs; %zu profiled (%zu gangs); %zu never met (%s)\n", g_types[0].size(), g_model_a.c_str(), g_types[1].size(), g_model_b.c_str(), possible, g_pairs_done, g_gangs_done, never, op.c_str());
    }
    return 0;
}
