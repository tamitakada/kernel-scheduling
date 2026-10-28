// pairprof_daemon.cpp -- a SCALED-DOWN fork of scheduler_daemon that does ONE thing: pair profiling, with NO list of pairs: it discovers them as it goes.
//
// WHAT IT MEASURES. A kernel TYPE is the kernel's short name plus its block dimensions. Every type is registered the first time it is seen, separately for model A and
// model B. A PAIR is one type of model A with one type of model B. The aim is to run as many UNIQUE pairs together ("gang") as possible, with nothing else on the GPU, and
// to log each run. The kernel times come from the agent's own timing log, as always (agent_<pid>_timing.tsv); this daemon only writes which kernel was which type and which
// kernels were released together.
//
// WHAT IT DOES NOT DO. No pairs file, no kernel tables, no IPC/SM/L2 limits, no estimates, no prefetch, no continuous batching. The agents, the Python hooks and the
// shared-memory layout (shared_state.hpp) are the existing, unchanged ones.
//
// "UNPROFILED PAIR" = a pair of types seen so far that has not been ganged PAIRPROF_REPS times yet. It is a moving target: a type seen for the first time makes a new
// unprofiled pair with EVERY type of the other model seen so far (and the same the other way round).
//
// THE POLICY, in full. A decision is made only when NOTHING is running (the GPU is idle), so every kernel runs either alone or in a gang of exactly two.
//   0. a kernel of a process that is neither model A nor model B runs alone at once (it is not part of the experiment). For the rest:
//   f = the oldest waiting kernel,  s = the oldest waiting kernel of the OTHER process.
//   1. f and s both wait, and (f, s) is an unprofiled pair                 -> release both together. This is the measurement; the pair loses one rep.
//   2. both wait, the pair is already profiled, but f's type still has an unprofiled pair with a type of the other model that is STILL BEING LAUNCHED (see below)
//                                                                          -> f keeps waiting; s (and then the next kernels of s's process) run ALONE,
//                                                                             one at a time, until a kernel of that process makes an unprofiled pair with f.
//                                                                             (That process may also show a type never seen before: it makes new unprofiled pairs.)
//   3. else, if s's type has one                                           -> the same with the roles swapped: s waits, f runs alone.
//   4. else (neither type has one)                                         -> release both together anyway: nothing to learn from them, the run goes on.
//   5. only f waits, and f's type has one                                  -> wait a moment for the other process's next kernel (PAIRPROF_WAIT_US);
//                                                                             if it does not come, f runs alone.
// WHAT KEEPS A HOLD FROM STARVING A PROCESS. Without limits, holding could stall a process almost for ever: most pairs involve some type that never comes back (kernels of
// the start-up, of a request that ended), and waiting for those cannot succeed. Three limits, all cheap:
//   * "still being launched": only types seen waiting within the last PAIRPROF_ALIVE_MS count as partners worth waiting for. A pair with a type that stopped appearing is
//     still ganged if its kernels happen to meet (rule 1), but no kernel is held for it.
//   * one kernel waits at most PAIRPROF_MAX_HOLD_MS; its type then rests PAIRPROF_COOLDOWN_MS (counts as having no partner), and every further failed hold of the type
//     doubles that rest (up to PAIRPROF_COOLDOWN_MAX_MS); a gang of the type puts it back to the start.
//   * a process is not held while it has been held more than PAIRPROF_MAX_HOLD_FRACTION of the last ~2 s: then rules 2, 3 and 5 do not apply to it.
// (PAIRPROF_MAX_HOLD_MS=0 switches the holding off altogether: gangs only where kernels meet by chance.)
//
// WARM-UP. A server starts up with a burst of launches (model load, memory profiling, sampler warm-up) and then sits idle until the first request. Those launches are not what
// is being profiled, and their kernel types would never come back. So, PER PROCESS, nothing is ganged, held or registered as a type until the process's warm-up is over: its kernels
// run alone, at once. Warm-up is over when the process has had an IDLE GAP of at least PAIRPROF_WARMUP_GAP_S seconds (the idle time between the server being ready and the first
// request; start-up has shorter pauses of its own, 2-3 s) AND has made at least PAIRPROF_WARMUP_LAUNCHES launches (a floor, to skip the first real requests as well: 0 = none).
// `kill -USR1 <pid of this daemon>` ends warm-up for every process at once. While a process is still warming up the console says what it is waiting for, every 20 s.
//
// ENVIRONMENT (all optional)
//   PAIRPROF_WARMUP_GAP_S   idle gap that ends a process's warm-up (default 10; 0 = this rule is off)     PAIRPROF_WARMUP_LAUNCHES   launches a process must have made first (default 0)
//                           (both 0: no warm-up skipping at all, as before)
//   PAIRPROF_MODEL_A / _B the SCHED_MODEL names of the two processes                          (default qwen / whisper)
//   PAIRPROF_REPS         gangs wanted per pair                                                (default 1)
//   PAIRPROF_SOLO         1: release ONE kernel at a time, no gangs: the reference run for solo kernel times
//   PAIRPROF_LOG          all: log every released kernel (default); gangs: only the kernels released in a gang
//   PAIRPROF_TRACE        1: print EVERY decision and every gang (very verbose), to watch the policy work
//   PAIRPROF_WAIT_US (3000)  PAIRPROF_MAX_HOLD_MS (150)  PAIRPROF_COOLDOWN_MS (2000)  PAIRPROF_LOG_DIR (default SCHED_LOG_DIR, else .)
//   PAIRPROF_COOLDOWN_MAX_MS (60000)  PAIRPROF_ALIVE_MS (500)  PAIRPROF_MAX_HOLD_FRACTION (0.5)       (the three limits above; ALIVE_MS=1e12, MAX_HOLD_FRACTION=1e9 and
//                                                                                                 COOLDOWN_MAX_MS=COOLDOWN_MS switch them off: the behaviour before they existed)
//
// OUTPUT   pairprof_log_<pid>.tsv     one row per released kernel: batch_id, kind, pid, kernel_id (= the agent log's kernel_id), model, kernel, block, rep, grid
//                                       (the grid is not part of the type; it is logged so the report can look up each kernel's exact solo time)
//          pairprof_types_<pid>.tsv   at exit: every type seen, how often it was released, and how many pairs with it are still unprofiled
//          pairprof_open_<pid>.tsv    at exit: the pairs of seen types that never met (at most 200,000 rows)
//          On the console: a line for every new type, the first 20 gangs, and a progress line every 5 s.
//
// BUILD    g++ -std=c++17 -O2 -o pairprof_daemon pairprof_daemon.cpp -lpthread -lrt        RUN  rm -f /dev/shm/kernel_sched_shm_*; ./pairprof_daemon
#include "shared_state.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

static SchedulerSharedState* state = nullptr;
static volatile sig_atomic_t g_stop = 0;
static void on_signal(int) { g_stop = 1; }
static volatile sig_atomic_t g_force_warm = 0;
static void on_usr1(int) { g_force_warm = 1; }                                              // kill -USR1: warm-up is over for every process
static uint64_t mono_ns() { timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return static_cast<uint64_t>(t.tv_sec) * 1000000000ull + static_cast<uint64_t>(t.tv_nsec); }   // the agents' clock
static double env_num(const char* n, double d) { const char* e = std::getenv(n); return (e && *e) ? std::atof(e) : d; }
static std::string env_str(const char* n, const char* d) { const char* e = std::getenv(n); return (e && *e) ? e : d; }

// ------------------------------------------------------------ configuration
static std::string g_model_a = "qwen", g_model_b = "whisper", g_log_dir = ".";
static int g_reps = 1;
static bool g_solo = false, g_log_singles = true, g_trace = false;
static uint64_t g_wait_ns = 3000000ull, g_hold_ns = 150000000ull, g_cool_ns = 2000000000ull;
static uint64_t g_cool_max_ns = 60000000000ull, g_alive_ns = 500000000ull;                 // PAIRPROF_COOLDOWN_MAX_MS, PAIRPROF_ALIVE_MS
static double g_budget = 0.5;                                                              // PAIRPROF_MAX_HOLD_FRACTION
static uint64_t g_warm_gap_ns = 10000000000ull, g_warm_launches = 0;                      // PAIRPROF_WARMUP_GAP_S, PAIRPROF_WARMUP_LAUNCHES

// ------------------------------------------------------------ warm-up, per process
struct WarmState { uint64_t last_ns = 0, launches = 0, last_status = 0; int max_id = 0; bool gap_seen = false, warm = false; std::string model; };
static std::unordered_map<pid_t, WarmState> g_warm;
// Called for every kernel waiting in a process; true once that process's warm-up is over. A launch is counted once (its agent_kernel_id is higher than any seen), whatever
// number of times the loop looks at it while it waits.
static bool warm_observe(pid_t pid, const char* model, int kernel_id, uint64_t request_ns, std::vector<std::string>& msgs) {
    WarmState& w = g_warm[pid];
    if (w.warm) return true;                                                                 // the common case from then on
    auto over = [&](const char* why) { w.warm = true; char b[300]; std::snprintf(b, sizeof b, "[pairprof] warm-up of %s (pid %d) is over after %llu launches (%s): profiling starts for it now", model, static_cast<int>(pid), static_cast<unsigned long long>(w.launches), why); msgs.push_back(b); };
    if (g_warm_gap_ns == 0 && g_warm_launches == 0) { w.warm = true; return true; }         // warm-up skipping is off
    w.model = model; if (!w.last_status) w.last_status = mono_ns();                      // the first status line comes 20 s after the process's first launch, not at once
    if (g_force_warm) { over("SIGUSR1"); return true; }
    if (kernel_id > w.max_id) {                                                              // a launch not counted yet
        w.max_id = kernel_id; w.launches++;
        if (g_warm_gap_ns && w.last_ns && request_ns > w.last_ns && request_ns - w.last_ns >= g_warm_gap_ns) w.gap_seen = true;   // idle gap: start-up is done
        if (request_ns > w.last_ns) w.last_ns = request_ns;
        if ((g_warm_gap_ns == 0 || w.gap_seen) && w.launches >= g_warm_launches) over(g_warm_gap_ns && g_warm_launches ? "an idle gap and the launch floor" : g_warm_gap_ns ? "an idle gap" : "the launch floor");
    }
    return w.warm;
}

// ------------------------------------------------------------ kernel types, and the pairs found so far
// A kernel TYPE is "short name|bx,by,bz". The agent already reduces a kernel's symbol to its short name (kernel_base in the slot).
static std::string type_key(const std::string& name, uint32_t bx, uint32_t by, uint32_t bz) { return name + "|" + std::to_string(bx) + "," + std::to_string(by) + "," + std::to_string(bz); }
static std::string slot_type(const PendingSlot& s) { return type_key(std::string(s.kernel_base, strnlen(s.kernel_base, NAME_LEN)), s.block[0], s.block[1], s.block[2]); }
struct TypeInfo { std::string key; uint64_t released = 0, last_seen_ns = 0, rest_until_ns = 0; int fails = 0; };   // fails: holds for this type that ended without a gang
static std::vector<TypeInfo> g_ty[2];                                                       // the types seen so far, per model, in order of discovery: a type's id is its index
static std::unordered_map<std::string, int> g_id[2];                                        // type key -> id
static std::vector<std::vector<uint16_t>> g_cnt;                                            // g_cnt[id_a][id_b] = gangs run so far for that pair of types
static size_t g_pairs_done = 0, g_gangs_done = 0;                                           // unique pairs ganged at least once; gangs of unprofiled pairs run in all
static std::vector<std::pair<int, std::string>> g_fresh;

// ------------------------------------------------------------ the policy
struct Cand { int slot; pid_t pid; int side; std::string type; uint64_t request_ns; int id = -1; };   // side: 0 = a kernel of model A, 1 = of model B, -1 = of neither; id: set by discover()
enum Kind { WAIT = 0, PROFILE, FILLER, ALONE_HOLD, ALONE };                                  // what a decision does
static const char* KIND[] = {"wait", "gang_profile", "gang_filler", "alone_other_waits", "alone"};
struct Decision { Kind kind = WAIT; std::vector<int> pick; uint64_t wake_ns = 0; std::string why; int rep = 0; int held = -1; };   // held: the model whose kernel is kept waiting, if any

static uint64_t g_last_release_ns = 0;
static size_t g_n_filler = 0, g_n_alone_hold = 0, g_n_alone = 0;
static double g_held_frac[2] = {0, 0};                                                      // how much of the last ~2 s each model's process was held (decaying average)
static int g_prev_held = -1; static uint64_t g_prev_ns = 0;

static void discover(Cand& c, uint64_t now) {                                               // register a type the first time it is seen; remember when it was last seen
    if (c.side < 0) return;
    auto it = g_id[c.side].find(c.type);
    if (it == g_id[c.side].end()) {
        it = g_id[c.side].emplace(c.type, static_cast<int>(g_ty[c.side].size())).first;
        TypeInfo t; t.key = c.type; g_ty[c.side].push_back(t);
        if (c.side == 0) g_cnt.emplace_back(g_ty[1].size(), 0); else for (auto& row : g_cnt) row.push_back(0);   // a new type is unprofiled with every type of the other model
        g_fresh.push_back({c.side, c.type});
    }
    c.id = it->second; g_ty[c.side][c.id].last_seen_ns = now;
}
static int gangs_of(const Cand& x, const Cand& y) { const Cand& a = x.side == 0 ? x : y; const Cand& b = x.side == 0 ? y : x; return g_cnt[a.id][b.id]; }
static bool pair_open(const Cand& x, const Cand& y) { return x.side >= 0 && y.side >= 0 && x.side != y.side && x.id >= 0 && y.id >= 0 && gangs_of(x, y) < g_reps; }
static bool has_open_partner(const Cand& c, uint64_t now) {                                 // c's type has an unprofiled pair with a type of the other model that is still being launched
    if (c.side < 0 || c.id < 0 || g_ty[c.side][c.id].rest_until_ns > now) return false;
    const int o = 1 - c.side;
    for (size_t j = 0; j < g_ty[o].size(); j++) {
        if (now > g_ty[o][j].last_seen_ns && now - g_ty[o][j].last_seen_ns > g_alive_ns) continue;    // it stopped appearing: not worth waiting for
        if ((c.side == 0 ? g_cnt[c.id][j] : g_cnt[j][c.id]) < g_reps) return true;
    }
    return false;
}
static int consume(const Cand& x, const Cand& y) {                                          // a gang of this pair is released: returns the rep number (1, 2, ...)
    const Cand& a = x.side == 0 ? x : y; const Cand& b = x.side == 0 ? y : x;
    const int n = ++g_cnt[a.id][b.id]; g_gangs_done++;
    if (n == 1) g_pairs_done++;
    for (TypeInfo* t : {&g_ty[0][a.id], &g_ty[1][b.id]}) { t->fails = 0; t->rest_until_ns = 0; }    // a gang: whatever held it was worth it
    return n;
}
static void give_up(const Cand& c, uint64_t now) {                                          // a hold of c ended without a gang: the type rests, longer every time
    TypeInfo& t = g_ty[c.side][c.id]; t.fails++;
    uint64_t rest = g_cool_ns; for (int i = 1; i < t.fails && rest < g_cool_max_ns; i++) rest *= 2;
    t.rest_until_ns = now + std::min(rest, g_cool_max_ns);
}
static void update_held(uint64_t now) {                                                     // the decaying average of how long each process has been held
    if (g_prev_ns && now > g_prev_ns) {
        const double dec = std::exp(-static_cast<double>(now - g_prev_ns) / 2e9);
        for (int s = 0; s < 2; s++) g_held_frac[s] = g_held_frac[s] * dec + (g_prev_held == s ? 1.0 - dec : 0.0);
    }
    g_prev_ns = now;
}

static Decision choose(std::vector<Cand>& w, uint64_t now) {                                // w: the waiting kernels (at most one per process); nothing is running
    Decision d;
    if (w.empty()) return d;
    std::stable_sort(w.begin(), w.end(), [](const Cand& a, const Cand& b) { return a.request_ns < b.request_ns || (a.request_ns == b.request_ns && a.slot < b.slot); });
    const Cand& f = w[0]; int si = -1;
    for (size_t i = 1; i < w.size(); i++) if (w[i].pid != f.pid) { si = static_cast<int>(i); break; }
    if (g_solo) { d.kind = ALONE; d.pick = {0}; d.why = "solo run: one kernel at a time"; return d; }
    for (Cand& c : w) discover(c, now);
    update_held(now);
    for (size_t i = 0; i < w.size(); i++) if (w[i].side < 0) { d.kind = ALONE; d.pick = {static_cast<int>(i)}; d.why = "a process that is neither model: it is not part of the experiment, it runs alone at once"; return d; }   // rule 0
    auto may_hold = [&](const Cand& c) { return (now - c.request_ns) < g_hold_ns && g_held_frac[c.side] < g_budget; };
    if (si >= 0) {
        const Cand& s = w[si];
        if (pair_open(f, s)) { d.kind = PROFILE; d.pick = {0, si}; d.rep = consume(f, s); d.why = "an unprofiled pair: gang"; return d; }                        // rule 1
        const bool f_open = has_open_partner(f, now), s_open = has_open_partner(s, now);
        const bool hold_f = f_open && may_hold(f), hold_s = !hold_f && s_open && may_hold(s);
        if (f_open && !hold_f && (now - f.request_ns) >= g_hold_ns) give_up(f, now);                                                                         // waited long enough
        if (s_open && !hold_f && !hold_s && (now - s.request_ns) >= g_hold_ns) give_up(s, now);
        if (hold_f) { d.kind = ALONE_HOLD; d.pick = {si}; d.held = f.side; d.why = "pair already profiled; the first kernel has an unprofiled partner type still being launched: it waits, the other process runs alone"; }   // rule 2
        else if (hold_s) { d.kind = ALONE_HOLD; d.pick = {0}; d.held = s.side; d.why = "pair already profiled; the second kernel has an unprofiled partner type still being launched: it waits, the first process runs alone"; }   // rule 3
        else { d.kind = FILLER; d.pick = {0, si}; d.why = f_open || s_open ? "a hold limit is reached: gang anyway" : "no unprofiled partner for either: gang anyway"; }                  // rule 4
        return d;
    }
    if (has_open_partner(f, now)) {                                                                                                                         // rule 5
        if (may_hold(f) && (now - g_last_release_ns) < g_wait_ns) { d.wake_ns = g_last_release_ns + g_wait_ns; d.held = f.side; d.why = "waiting for the other process's next kernel"; return d; }
        if ((now - f.request_ns) >= g_hold_ns || (now - g_last_release_ns) >= g_wait_ns) give_up(f, now);
    }
    d.kind = ALONE; d.pick = {0}; d.why = "alone: no partner waiting"; return d;
}
static Decision decide(std::vector<Cand>& w, uint64_t now) {
    Decision d = choose(w, now);
    if (!d.pick.empty()) { g_last_release_ns = now; for (int i : d.pick) if (w[i].side >= 0 && w[i].id >= 0) g_ty[w[i].side][w[i].id].released++; }       // id < 0: never registered (solo mode, or not one of the two models)
    g_prev_held = d.held;
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
    g_warm_gap_ns = static_cast<uint64_t>(env_num("PAIRPROF_WARMUP_GAP_S", 10) * 1e9); g_warm_launches = static_cast<uint64_t>(env_num("PAIRPROF_WARMUP_LAUNCHES", 0));
    g_reps = std::max(1, static_cast<int>(env_num("PAIRPROF_REPS", 1))); g_solo = env_num("PAIRPROF_SOLO", 0) != 0; g_trace = env_num("PAIRPROF_TRACE", 0) != 0;
    g_log_singles = env_str("PAIRPROF_LOG", "all") != "gangs";
    g_wait_ns = static_cast<uint64_t>(env_num("PAIRPROF_WAIT_US", 3000) * 1e3); g_hold_ns = static_cast<uint64_t>(env_num("PAIRPROF_MAX_HOLD_MS", 150) * 1e6); g_cool_ns = static_cast<uint64_t>(env_num("PAIRPROF_COOLDOWN_MS", 2000) * 1e6);
    g_cool_max_ns = std::max(g_cool_ns, static_cast<uint64_t>(env_num("PAIRPROF_COOLDOWN_MAX_MS", 60000) * 1e6)); g_alive_ns = static_cast<uint64_t>(env_num("PAIRPROF_ALIVE_MS", 500) * 1e6); g_budget = env_num("PAIRPROF_MAX_HOLD_FRACTION", 0.5);
    g_log_dir = env_str("PAIRPROF_LOG_DIR", env_str("SCHED_LOG_DIR", ".").c_str());
    if (g_solo) std::fprintf(stderr, "[pairprof] SOLO run: one kernel at a time, nothing is ganged\n");
    else std::fprintf(stderr, "[pairprof] discovery mode: A = %s, B = %s; every pair of types seen is ganged %d time(s); patience %.1f ms, max hold %.0f ms, rest %.0f ms (doubling up to %.0f ms), a partner type counts while seen within %.0f ms, a process is held at most %.0f%% of the time\n", g_model_a.c_str(), g_model_b.c_str(), g_reps, g_wait_ns / 1e6, g_hold_ns / 1e6, g_cool_ns / 1e6, g_cool_max_ns / 1e6, g_alive_ns / 1e6, 100.0 * g_budget);
    if (!g_solo) { if (g_warm_gap_ns == 0 && g_warm_launches == 0) std::fprintf(stderr, "[pairprof] warm-up skipping is OFF\n");
                   else std::fprintf(stderr, "[pairprof] warm-up skipping: a process is profiled only after %s%s%s (kill -USR1 %d ends it for all)\n", g_warm_gap_ns ? ("an idle gap of " + std::to_string(g_warm_gap_ns / 1000000000ull) + " s").c_str() : "",
                                     g_warm_gap_ns && g_warm_launches ? " and " : "", g_warm_launches ? ("at least " + std::to_string(g_warm_launches) + " launches").c_str() : "", static_cast<int>(getpid())); }
    struct sigaction sa{}; sa.sa_handler = on_signal; sigemptyset(&sa.sa_mask); struct sigaction su{}; su.sa_handler = on_usr1; sigemptyset(&su.sa_mask); sigaction(SIGUSR1, &su, nullptr); sigaction(SIGINT, &sa, nullptr); sigaction(SIGTERM, &sa, nullptr); sigaction(SIGHUP, &sa, nullptr);
    init_shared_state();
    const std::string log_path = g_log_dir + "/pairprof_log_" + std::to_string(getpid()) + ".tsv";
    FILE* log = std::fopen(log_path.c_str(), "w"); if (!log) { std::perror("[pairprof] cannot open the log"); return 1; }
    std::fputs("batch_id\tkind\tpid\tkernel_id\tmodel\tkernel\tblock\trep\tgrid\n", log); std::fprintf(stderr, "[pairprof] log: %s\n[pairprof] waiting for the agents...\n", log_path.c_str());

    std::unordered_set<std::string> warned;                                                   // model names already complained about
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
            const std::string model = state->slots[i].model; const int model_side = model == g_model_a ? 0 : model == g_model_b ? 1 : -1;
            const int side = (model_side >= 0 && !g_solo && !warm_observe(state->slots[i].pid, state->slots[i].model, state->slots[i].agent_kernel_id, state->slots[i].request_ns, msgs)) ? -1 : model_side;   // in warm-up: runs alone at once, is not a type
            if (model_side < 0 && !g_solo && warned.insert(model).second) {                                // without this the whole experiment would silently run every kernel alone
                char wb[512]; std::snprintf(wb, sizeof wb, "[pairprof] WARNING: pid %d reports SCHED_MODEL='%s', which is neither model A ('%s') nor model B ('%s'): its kernels run alone and are NEVER paired. "
                                            "Set PAIRPROF_MODEL_A / PAIRPROF_MODEL_B to the SCHED_MODEL names of your two servers.", static_cast<int>(state->slots[i].pid), model.empty() ? "(not set)" : model.c_str(), g_model_a.c_str(), g_model_b.c_str());
                msgs.push_back(wb);
            }
            waiting.push_back({i, state->slots[i].pid, side, slot_type(state->slots[i]), state->slots[i].request_ns});
        }
        if (running.empty() && !waiting.empty()) {                                            // the GPU is idle: decide
            const uint64_t now = mono_ns(); Decision d = decide(waiting, now); char b[640];
            for (const auto& ft : g_fresh) { std::snprintf(b, sizeof b, "[new type] %s: %s   (types seen: %s %zu, %s %zu)", ft.first == 0 ? g_model_a.c_str() : g_model_b.c_str(), ft.second.c_str(), g_model_a.c_str(), g_ty[0].size(), g_model_b.c_str(), g_ty[1].size()); msgs.push_back(b); }
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
                        std::snprintf(b, sizeof b, "%llu\t%s\t%d\t%d\t%s\t%s\t%u,%u,%u\t%s\t%u,%u,%u", static_cast<unsigned long long>(id), KIND[d.kind], static_cast<int>(s.pid), s.agent_kernel_id, s.model,
                                      std::string(s.kernel_base, strnlen(s.kernel_base, NAME_LEN)).c_str(), s.block[0], s.block[1], s.block[2], d.kind == PROFILE ? std::to_string(d.rep).c_str() : "", s.grid[0], s.grid[1], s.grid[2]);
                        rows.push_back(b);
                    }
                }
                if (d.kind == PROFILE || d.kind == FILLER) n_gang_kernels += 2;
                if (d.kind == PROFILE) {
                    const Cand& ca = waiting[d.pick[0]].side == 0 ? waiting[d.pick[0]] : waiting[d.pick[1]]; const Cand& cb = waiting[d.pick[0]].side == 0 ? waiting[d.pick[1]] : waiting[d.pick[0]];
                    if (g_trace || g_gangs_done <= 20) {
                        std::snprintf(b, sizeof b, "[gang %llu] %s: %s  +  %s: %s   rep %d/%d   (unique pairs profiled: %zu of %zu possible)", static_cast<unsigned long long>(id), g_model_a.c_str(), ca.type.c_str(), g_model_b.c_str(), cb.type.c_str(), d.rep, g_reps, g_pairs_done, g_ty[0].size() * g_ty[1].size());
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
        if (!g_solo) for (auto& kv : g_warm) {                                              // a process still warming up: say what it is waiting for, every 20 s
            WarmState& w = kv.second; if (w.warm || !w.launches || t - w.last_status < 20000000000ull) continue; w.last_status = t;
            std::string need = (g_warm_gap_ns && !w.gap_seen) ? "an idle gap of " + std::to_string(g_warm_gap_ns / 1000000000ull) + " s" : std::string();
            if (g_warm_launches && w.launches < g_warm_launches) need += std::string(need.empty() ? "" : " and ") + "launch #" + std::to_string(g_warm_launches);
            std::fprintf(stderr, "[pairprof] %s (pid %d) is still in warm-up: %llu launches so far; waiting for %s. (kill -USR1 %d starts profiling now)\n", w.model.c_str(), static_cast<int>(kv.first), static_cast<unsigned long long>(w.launches), need.c_str(), static_cast<int>(getpid()));
        }
        if (t - last_progress > 5000000000ull) {
            std::fflush(log); last_progress = t; const size_t possible = g_ty[0].size() * g_ty[1].size();
            std::fprintf(stderr, "[progress] types seen: %s %zu, %s %zu; unique pairs profiled: %zu of %zu possible (%.1f%%); kernels run: %zu in gangs, %zu alone while another waited, %zu alone otherwise\n",
                         g_model_a.c_str(), g_ty[0].size(), g_model_b.c_str(), g_ty[1].size(), g_pairs_done, possible, possible ? 100.0 * static_cast<double>(g_pairs_done) / static_cast<double>(possible) : 0.0, n_gang_kernels, g_n_alone_hold, g_n_alone);
        }
    }
    std::fclose(log);
    if (!g_solo) {
        const std::string tp = g_log_dir + "/pairprof_types_" + std::to_string(getpid()) + ".tsv", op = g_log_dir + "/pairprof_open_" + std::to_string(getpid()) + ".tsv"; std::string blk;
        if (FILE* f = std::fopen(tp.c_str(), "w")) {
            std::fputs("model\tkernel\tblock\treleased\tpartners_seen\tpartners_unprofiled\n", f);
            for (int side = 0; side < 2; side++) for (size_t i = 0; i < g_ty[side].size(); i++) {
                size_t open = 0; for (size_t j = 0; j < g_ty[1 - side].size(); j++) if ((side == 0 ? g_cnt[i][j] : g_cnt[j][i]) < g_reps) open++;
                const std::string n = split_type(g_ty[side][i].key, &blk);
                std::fprintf(f, "%s\t%s\t%s\t%llu\t%zu\t%zu\n", side == 0 ? g_model_a.c_str() : g_model_b.c_str(), n.c_str(), blk.c_str(), static_cast<unsigned long long>(g_ty[side][i].released), g_ty[1 - side].size(), open);
            }
            std::fclose(f);
        }
        size_t never = 0;
        if (FILE* f = std::fopen(op.c_str(), "w")) {
            std::fputs("name_a\tblock_a\tname_b\tblock_b\tgangs_done\n", f); std::string bb;
            for (size_t i = 0; i < g_ty[0].size(); i++) for (size_t j = 0; j < g_ty[1].size(); j++) {
                if (g_cnt[i][j] >= g_reps) continue;
                if (++never <= 200000) { const std::string na = split_type(g_ty[0][i].key, &blk), nb = split_type(g_ty[1][j].key, &bb); std::fprintf(f, "%s\t%s\t%s\t%s\t%d\n", na.c_str(), blk.c_str(), nb.c_str(), bb.c_str(), static_cast<int>(g_cnt[i][j])); }
            }
            std::fclose(f);
        }
        const size_t possible = g_ty[0].size() * g_ty[1].size();
        std::fprintf(stderr, "[pairprof] stopped: %zu %s types x %zu %s types = %zu possible pairs; %zu profiled (%zu gangs); %zu never met (%s)\n", g_ty[0].size(), g_model_a.c_str(), g_ty[1].size(), g_model_b.c_str(), possible, g_pairs_done, g_gangs_done, never, op.c_str());
    }
    return 0;
}
