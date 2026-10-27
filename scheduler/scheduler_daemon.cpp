// scheduler_daemon.cpp -- decides when each intercepted kernel launch may go ahead. Three policies, chosen with SCHED_MODE (required):
//   SCHED_MODE=always   ALWAYS continuous batching: whenever a slot is free, the oldest waiting kernel whose process has none running is released, up to SCHED_BATCH_MAX (default 2) at once.
//   SCHED_MODE=none     NO batching: one kernel at a time, oldest first (the same policy with a batch of 1).
//   SCHED_MODE=eff      the EFFICIENCY / DURATION policy: no batching by default; a kernel joins the running one(s) only if the pair was measured to gain (efficiency >= SCHED_PAIR_MIN_EFF)
//                       or either kernel is short (< SCHED_PAIR_SHORT_US). Needs SCHED_PAIR_EFF_FILE and SCHED_KERNEL_US_FILE (make_pair_eff.py writes them); see the block before policy_select_eff.
//   SCHED_MODE=hold     the HOLD-LIST policy, the efficiency policy turned around: batch like "always", EXCEPT a pair that was measured to be SLOWER overlapped than one after the other
//                       (efficiency below SCHED_HOLD_BELOW_EFF, default 1.0) is never run together: one waits for the other. A pair that was not measured is batched. Reads the same two files
//                       as eff (SCHED_PAIR_EFF_FILE, SCHED_KERNEL_US_FILE); a pair with a short kernel (< SCHED_PAIR_SHORT_US) is never held (one flagged line).
// Other settings: SCHED_BATCH_MAX, SCHED_LOG (full | decisions | off), SCHED_LOG_DIR, SCHED_DAEMON_LOG.
// REMOVED: the constraint-based policy (IPC / SM / L2 limits from profile tables), the whitelist policy (SCHED_PAIR_LIST) and the CSV tables. SCHED_CONSTRAINTS and SCHED_PAIR_LIST now stop the
// daemon with an error instead of being ignored. The decision log and batching log lost the columns only those policies filled.
// KEPT: the prefetch machinery, as a TOOLBOX that no policy calls (section "PREFETCH TOOLBOX"). The agents still obey do_prefetch; this daemon always sets it to 0.
// Build: g++ -std=c++17 -O2 -o scheduler_daemon scheduler_daemon.cpp -lpthread -lrt     (shared_state.hpp and kernel_key.hpp next to it)
#include "shared_state.hpp"

#include <unordered_map>
#include <unordered_set>
#include <strings.h>
#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cctype>
#include <ctime>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <vector>
#include <numeric>

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
 * KERNEL KEY and KERNEL STATS. The key of a kernel is its name and its BLOCK dimensions (the grid is not part of it). KernelStats is what the prefetch toolbox's
 * duration estimates read (estimate_duration_us); nothing fills it now, because the profile-table loader was removed.
 * ====================================================================== */

struct KernelKey {                                  // the table key: name and block dimensions; the grid is NOT part of it
    std::string name;
    std::array<uint32_t, 3> block;
    bool operator<(const KernelKey& o) const { return std::tie(name, block) < std::tie(o.name, o.block); }
};

struct KernelStats {                                // the remaining columns of a row; NaN where the cell was empty
    long   launches         = 0;
    double sm_util          = 0;
    double sm_util_std      = 0;
    double occupancy        = 0;
    double occupancy_std    = 0;
    double avg_ipc          = 0;
    double avg_ipc_std      = 0;
    double l2_tput          = 0;
    double l2_tput_std      = 0;
    double blocks_per_wave  = NAN;              // the Blocks/Wave column: blocks the GPU runs at once for this kernel = SMs x max resident blocks per SM (NaN if there is none)
    double duration_per_wave_us = NAN;          // the Duration/Wave column (us per wave of the profiled launch), NaN if there is none or the cell is empty
    double duration_per_block_us = NAN;         // the Duration / grid volume column (us per block of the profiled grid), NaN if there is none or the cell is empty
    double duration_us      = NAN;              // the Duration column, converted to microseconds; NaN if the CSV has none or the cell is empty
};

// Each statistic is a column and its "std" column.
// The two models the scheduler knows. A kernel of any other process (another SCHED_MODEL, or none) is never batched by the efficiency policy; it still runs, alone
// when it is first in the queue, and it is in the decision log with its model tag.
static bool is_known_model(const std::string& model) { return model == "qwen" || model == "whisper"; }

static uint64_t g_other_launches = 0;            // launches of processes that are neither qwen nor whisper: counted, reported at exit

static std::string dims_str(const std::array<uint32_t, 3>& d) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "(%u, %u, %u)", d[0], d[1], d[2]);
    return buf;
}

static KernelKey slot_key(const PendingSlot& s) {
    KernelKey k;
    const size_t n = strnlen(s.kernel_base, NAME_LEN);
    k.name = n ? std::string(s.kernel_base, n) : "unknown";
    for (int i = 0; i < 3; i++) k.block[i] = s.block[i];
    return k;
}

static std::array<uint32_t, 3> slot_grid(const PendingSlot& s) { return {s.grid[0], s.grid[1], s.grid[2]}; }   // logged, not part of the key

static std::string slot_model(const PendingSlot& s) {
    return std::string(s.model, strnlen(s.model, sizeof s.model));
}

// The limits of the removed constraint policy. KEPT because plan_prefetch (the prefetch toolbox) still tests a kernel against them.
static constexpr double IPC_BATCH_LIMIT = 4.0;
static constexpr double SM_UTIL_BATCH_LIMIT = 100.0;   // percent
static constexpr double L2_BATCH_LIMIT = 100.0;        // percent

struct Candidate { 
    int slot; 
    pid_t pid; 
    uint64_t request_ns; 
    double ipc, sm_util, l2;                        // inputs of the prefetch toolbox (plan_prefetch); NaN: nothing sets them now
    bool known_model;
    double dur_us = NAN;                            // its expected duration in microseconds (prefetch toolbox), NaN if unknown
    std::string sig;                                // the efficiency policy only: model|kernel|block|grid (empty otherwise)
};   // one pending kernel (known_model: its process is qwen or whisper)

struct Release {                                    // what one decision looked like, shared by the kernels it released
    uint64_t id = 0, decision_ns = 0;
    int size = 0, n_pending = 0;
    std::string pending_pids;
};
struct Member {
    int slot = -1, inflight = -1, agent_kid = 0, arrival_seq = 0, do_prefetch = 0, pos = 0;
    pid_t pid = 0;
    std::string name, tag, base, grid, block, model;
    std::string sig;                      // the efficiency policy only: see Candidate
    double ipc = NAN, sm_util = NAN, l2 = NAN;
    uint64_t request_ns = 0, post_ns = 0;
    double dur_us = NAN;                  // profiled duration (us), NaN if unknown
    uint64_t est_end_ns = 0;              // when the daemon expects it to finish (0: it cannot tell); fixed when it is released
    bool pf_planned = false;              // released early on purpose, so that its prefetch fills the wait for the kernels it could not join
    size_t pf_bytes = 0;                  // what the agent is expected to prefetch for it (0: nothing, or no prefetch was asked for)
    uint64_t target_end_ns = 0;           // pf_planned: the estimated end of the kernels in the way, which its prefetch was timed to end with
    std::shared_ptr<const Release> rel;
};

// Why a decision did not batch (counted in the batching log). BATCHED means it did.
enum BatchCause { BATCHED = 0, BATCHING_OFF, NOT_ENOUGH_IN_QUEUE, INFLIGHT_FULL, PAIR_NOT_LISTED, PAIR_HELD, N_CAUSES };
static const char* const CAUSE_NAME[N_CAUSES] = {"batched", "batching_disabled", "not_enough_in_queue", "inflight_table_full", "pair_not_listed", "pair_held"};
static constexpr int LIM_IPC = 1, LIM_SM = 2, LIM_L2 = 4;       // the bits of a set of limits

struct SelectPlan {
    std::vector<int> pick;                          // indices into `pending`, in release order
    BatchCause cause = BATCHED;                     // BATCHED if two or more kernels, else why not
};

// Which policy runs (SCHED_MODE).
enum Mode { MODE_NONE, MODE_ALWAYS, MODE_EFF, MODE_HOLD };
static Mode g_mode = MODE_ALWAYS;

// SCHED_MODE=always (and none, with a batch of 1): a free slot is always filled. The oldest waiting kernel whose process has none running is released, then the oldest
// waiting kernel of another such process, up to max_batch running at once -- whatever the kernels are (no table, no limits, no efficiency). Nothing is ever held, so no
// prefetch is planned (do_prefetch is 0). What stays: one kernel per process in flight (the next launch of a process would only queue behind its previous one in the same
// stream), the oldest-first order, and max_batch.
static SelectPlan policy_select_always(std::vector<Member>& curr, std::vector<Candidate>& pending, int max_batch) {
    SelectPlan plan;
    if (static_cast<int>(curr.size()) >= max_batch || pending.empty()) return plan;
    std::stable_sort(pending.begin(), pending.end(), [](const Candidate& a, const Candidate& b) {
        return (a.request_ns < b.request_ns) || (a.request_ns == b.request_ns && a.slot < b.slot);
    });
    auto process_running = [&](pid_t p) { for (const Member& m : curr) if (m.pid == p) return true; return false; };
    size_t first = 0;
    while (first < pending.size() && process_running(pending[first].pid)) first++;
    if (first == pending.size()) { plan.cause = NOT_ENOUGH_IN_QUEUE; return plan; }     // every waiting kernel belongs to a process that already has one running
    plan.pick.push_back(static_cast<int>(first));
    if (static_cast<int>(curr.size()) + 1 == max_batch) {
        if (curr.empty()) plan.cause = BATCHING_OFF;                                     // SCHED_BATCH_MAX=1
        return plan;
    }
    for (size_t i = 0; i < pending.size(); i++) {
        if (i == first || pending[i].pid == pending[first].pid || process_running(pending[i].pid)) continue;
        plan.pick.push_back(static_cast<int>(i));                                        // the oldest eligible partner: no table, no limits
        break;
    }
    if (plan.pick.size() == 1 && curr.empty()) plan.cause = NOT_ENOUGH_IN_QUEUE;         // nobody eligible to join: it runs alone
    return plan;
}

static std::string pair_key(const std::string& a, const std::string& b) { return a < b ? a + "\x1f" + b : b + "\x1f" + a; }     // unordered pair
// ======================= SCHED_PAIR_EFF_FILE: batch a pair only if it was MEASURED to gain, or one of its kernels is short =======================
// Two tab-separated files, each with a header line (columns are found by NAME, extra columns are ignored, lines starting with # are skipped); make_pair_eff.py writes them.
//   SCHED_PAIR_EFF_FILE   pair_eff.tsv :  model_a kernel_a block_a [grid_a]  model_b kernel_b block_b [grid_b]  efficiency [gangs]
//   SCHED_KERNEL_US_FILE  kernel_us.tsv:  model kernel block [grid]  solo_us [launches]
// block and grid are x,y,z ("(x, y, z)" is read too). A grid of * (or no grid column) means every grid: a TYPE row; a row with a grid is a SHAPE row. A pair row needs BOTH grids or NEITHER. A kernel is looked up by its
// SHAPE (model, kernel, block, grid) first and by its TYPE (grid *) second, because efficiency and duration depend on the grid (how many SMs the kernel fills).
// A pair of kernels can be batched when
//   1. either kernel is short: its solo time is known and < SCHED_PAIR_SHORT_US (default 20 us). This holds for ANY partner, profiled or not.   [the SHORT-KERNEL RULE: one line, flagged below]
//   2. or the pair's efficiency is known and >= SCHED_PAIR_MIN_EFF (default 1.25): from the shape-pair row if it has >= SCHED_PAIR_MIN_GANGS gangs (default 3), else from the type-pair row.
// Everything else (an unprofiled pair, or a profiled one below the threshold, of two kernels that are not short) is NOT batched: it waits for the running kernel, as in any pair-list policy.
// A kernel with no duration row counts as NOT short. The daemon does not look at launch numbers: it only matches what a kernel IS, so it also works when the runs are not aligned.
static double g_eff_min = 1.25, g_short_us = 20.0; static int g_eff_min_gangs = 3;
static double g_hold_below = 1.0;                                  // SCHED_MODE=hold: a pair with a KNOWN efficiency below this is held
struct EffRow { double eff; int gangs; };
static std::unordered_map<std::string, EffRow> g_eff;              // pair_key(kernel key a, kernel key b) -> measured efficiency and the gangs behind it
static std::unordered_map<std::string, double> g_kus;              // kernel key -> median solo time in us
static std::unordered_map<std::string, bool> g_elig;               // pair_key(shape sig a, shape sig b) -> answer (it never changes, and this is the hot path)

static std::string xyz(const std::array<uint32_t, 3>& d) { return std::to_string(d[0]) + "," + std::to_string(d[1]) + "," + std::to_string(d[2]); }
static std::string kkey(const std::string& model, const std::string& kernel, const std::string& block, const std::string& grid) { return model + "|" + kernel + "|" + block + "|" + grid; }
static std::string eff_sig(const std::string& model, const std::string& kernel, const std::array<uint32_t, 3>& block, const std::array<uint32_t, 3>& grid) { return kkey(model, kernel, xyz(block), xyz(grid)); }
static std::string type_of(const std::string& sig) { const size_t p = sig.rfind('|'); return p == std::string::npos ? sig : sig.substr(0, p) + "|*"; }     // the same kernel, any grid
static bool norm_dims(const std::string& s, bool allow_star, std::string* out) {            // "(x, y, z)" or "x,y,z" -> "x,y,z"; "*" when allowed
    if (allow_star && s == "*") { *out = "*"; return true; }
    std::vector<unsigned long> v; for (size_t i = 0; i < s.size();) { if (std::isdigit(static_cast<unsigned char>(s[i]))) { char* e = nullptr; v.push_back(std::strtoul(s.c_str() + i, &e, 10)); i = static_cast<size_t>(e - s.c_str()); } else i++; }
    if (v.size() != 3) return false;
    *out = std::to_string(v[0]) + "," + std::to_string(v[1]) + "," + std::to_string(v[2]); return true;
}
static bool read_tsv(const char* path, std::vector<std::string>* header, std::vector<std::vector<std::string>>* rows) {
    FILE* f = std::fopen(path, "r"); if (!f) return false;
    std::string line; int c; header->clear(); rows->clear();
    auto flush = [&]() { while (!line.empty() && (line.back() == '\r')) line.pop_back(); if (line.empty() || line[0] == '#') { line.clear(); return; }
                         std::vector<std::string> cells; size_t from = 0; for (size_t t; (t = line.find('\t', from)) != std::string::npos; from = t + 1) cells.push_back(line.substr(from, t - from));
                         cells.push_back(line.substr(from)); if (header->empty()) *header = cells; else rows->push_back(cells); line.clear(); };
    while ((c = std::fgetc(f)) != EOF) { if (c == '\n') flush(); else line.push_back(static_cast<char>(c)); }
    flush(); std::fclose(f); return !header->empty();
}
static int col_of(const std::vector<std::string>& h, const char* name) { for (size_t i = 0; i < h.size(); i++) if (h[i] == name) return static_cast<int>(i); return -1; }
static bool load_pair_eff(const char* path, size_t* n_type, size_t* n_shape, size_t* n_gain, std::string* err) {
    std::vector<std::string> h; std::vector<std::vector<std::string>> rows;
    if (!read_tsv(path, &h, &rows)) { *err = "cannot read it"; return false; }
    const char* need[] = {"model_a", "kernel_a", "block_a", "model_b", "kernel_b", "block_b", "efficiency"}; int ix[7];
    for (int i = 0; i < 7; i++) if ((ix[i] = col_of(h, need[i])) < 0) { *err = std::string("no column '") + need[i] + "' in its header"; return false; }
    const int ga = col_of(h, "grid_a"), gb = col_of(h, "grid_b"), gg = col_of(h, "gangs"); size_t bad = 0; *n_type = *n_shape = *n_gain = 0;
    for (const auto& r : rows) {
        auto cell = [&](int i) -> std::string { return i >= 0 && static_cast<size_t>(i) < r.size() ? r[static_cast<size_t>(i)] : std::string(); };
        std::string ba, bb, xa = "*", xb = "*";
        if (!norm_dims(cell(ix[2]), false, &ba) || !norm_dims(cell(ix[5]), false, &bb) || (ga >= 0 && !norm_dims(cell(ga), true, &xa)) || (gb >= 0 && !norm_dims(cell(gb), true, &xb))) { if (++bad <= 3) std::fprintf(stderr, "[daemon] WARNING: %s: skipping a row with a bad block or grid\n", path); continue; }
        if ((xa == "*") != (xb == "*")) { if (++bad <= 3) std::fprintf(stderr, "[daemon] WARNING: %s: skipping a row with a grid for one kernel and * for the other (give both grids, or neither)\n", path); continue; }
        char* e = nullptr; const std::string es = cell(ix[6]); const double eff = std::strtod(es.c_str(), &e); if (es.empty() || e == es.c_str()) { if (++bad <= 3) std::fprintf(stderr, "[daemon] WARNING: %s: skipping a row with no efficiency\n", path); continue; }
        const int gangs = gg >= 0 && !cell(gg).empty() ? std::atoi(cell(gg).c_str()) : 1000000;          // no gangs column: every row counts as well measured
        const std::string k = pair_key(kkey(cell(ix[0]), cell(ix[1]), ba, xa), kkey(cell(ix[3]), cell(ix[4]), bb, xb)); const auto it = g_eff.find(k);
        if (it == g_eff.end() || gangs > it->second.gangs) g_eff[k] = {eff, gangs};
        ((xa == "*" && xb == "*") ? *n_type : *n_shape)++; if (g_mode == MODE_HOLD ? eff < g_hold_below : eff >= g_eff_min) (*n_gain)++;
    }
    if (bad > 3) std::fprintf(stderr, "[daemon] WARNING: %s: %zu rows skipped in all\n", path, bad);
    return true;
}
static bool load_kernel_us(const char* path, size_t* n_rows, size_t* n_short, std::string* err) {
    std::vector<std::string> h; std::vector<std::vector<std::string>> rows;
    if (!read_tsv(path, &h, &rows)) { *err = "cannot read it"; return false; }
    const char* need[] = {"model", "kernel", "block", "solo_us"}; int ix[4];
    for (int i = 0; i < 4; i++) if ((ix[i] = col_of(h, need[i])) < 0) { *err = std::string("no column '") + need[i] + "' in its header"; return false; }
    const int gc = col_of(h, "grid"); size_t bad = 0; *n_rows = *n_short = 0;
    for (const auto& r : rows) {
        auto cell = [&](int i) -> std::string { return i >= 0 && static_cast<size_t>(i) < r.size() ? r[static_cast<size_t>(i)] : std::string(); };
        std::string b, g = "*"; char* e = nullptr; const std::string us = cell(ix[3]); const double v = std::strtod(us.c_str(), &e);
        if (!norm_dims(cell(ix[2]), false, &b) || (gc >= 0 && !norm_dims(cell(gc), true, &g)) || us.empty() || e == us.c_str()) { if (++bad <= 3) std::fprintf(stderr, "[daemon] WARNING: %s: skipping a bad row\n", path); continue; }
        g_kus[kkey(cell(ix[0]), cell(ix[1]), b, g)] = v; (*n_rows)++; if (v < g_short_us) (*n_short)++;
    }
    if (bad > 3) std::fprintf(stderr, "[daemon] WARNING: %s: %zu rows skipped in all\n", path, bad);
    return true;
}
[[maybe_unused]] static bool kernel_short(const std::string& sig) {                      // is this kernel's solo time known and under SCHED_PAIR_SHORT_US? its shape's, else its type's
    auto it = g_kus.find(sig); if (it == g_kus.end()) it = g_kus.find(type_of(sig));
    return it != g_kus.end() && it->second < g_short_us;
}
static const EffRow* lookup_eff(const std::string& a, const std::string& b) {              // the measured efficiency of two kernel shapes: the shape-pair row if enough gangs measured it, else the type-pair row (any grid), else none
    const auto s = g_eff.find(pair_key(a, b));
    if (s != g_eff.end() && s->second.gangs >= g_eff_min_gangs) return &s->second;
    const auto t = g_eff.find(pair_key(type_of(a), type_of(b)));
    return t != g_eff.end() ? &t->second : nullptr;
}
static bool eff_eligible_compute(const std::string& a, const std::string& b) {                 // SCHED_MODE=eff: true = the pair may run together
    if (kernel_short(a) || kernel_short(b)) return true;                                  // <<<<<<<< SHORT-KERNEL RULE: batch ANY pair with a kernel under SCHED_PAIR_SHORT_US, profiled or not. COMMENT OUT THIS ONE LINE to batch only pairs measured to gain.
    const EffRow* e = lookup_eff(a, b);
    return e && e->eff >= g_eff_min;
}
static bool hold_eligible_compute(const std::string& a, const std::string& b) {                // SCHED_MODE=hold: true = the pair may run together, false = HELD
    if (kernel_short(a) || kernel_short(b)) return true;                                  // <<<<<<<< SHORT-KERNEL EXEMPTION: never HOLD a pair with a kernel under SCHED_PAIR_SHORT_US (a hold costs a daemon round trip, ~100 us, more than such a kernel runs; the pair file leaves them out anyway). COMMENT OUT THIS ONE LINE to hold them too when listed.
    const EffRow* e = lookup_eff(a, b);
    return !(e && e->eff < g_hold_below);                                                 // held only when the efficiency is KNOWN and below the threshold: a pair that was not measured is batched
}
static bool pair_eligible_compute(const std::string& a, const std::string& b) { return g_mode == MODE_HOLD ? hold_eligible_compute(a, b) : eff_eligible_compute(a, b); }
static bool pair_eligible(const std::string& a, const std::string& b) {
    if (a.empty() || b.empty()) return g_mode == MODE_HOLD;                               // a kernel of an unlabelled process: eff never batches it, hold (default: batch) does                                             // a kernel of a process that is neither model: never batched
    const std::string k = pair_key(a, b); const auto it = g_elig.find(k); if (it != g_elig.end()) return it->second;
    const bool r = pair_eligible_compute(a, b); g_elig.emplace(k, r); return r;
}
// ==============================================================================================================================================================================

// SCHED_MODE=eff. The default is NO batching: one kernel at a time over both processes, oldest first. A kernel runs together with the one(s) already running only if the
// pair is eligible (pair_eligible above): measured to gain, or one of the two is short. One kernel per process stays.
static SelectPlan policy_select_eff(std::vector<Member>& curr, std::vector<Candidate>& pending, int max_batch) {
    SelectPlan plan;
    if (static_cast<int>(curr.size()) >= max_batch || pending.empty()) return plan;
    std::stable_sort(pending.begin(), pending.end(), [](const Candidate& a, const Candidate& b) {
        return (a.request_ns < b.request_ns) || (a.request_ns == b.request_ns && a.slot < b.slot);
    });
    auto process_running = [&](pid_t p) { for (const Member& m : curr) if (m.pid == p) return true; return false; };
    auto joins_running = [&](const std::string& sig) { for (const Member& m : curr) if (!pair_eligible(m.sig, sig)) return false; return true; };
    size_t first = 0;
    while (first < pending.size() && process_running(pending[first].pid)) first++;
    if (first == pending.size()) { plan.cause = NOT_ENOUGH_IN_QUEUE; return plan; }
    if (!joins_running(pending[first].sig)) { plan.cause = PAIR_NOT_LISTED; return plan; }          // something is running and this is not a listed partner for it: wait (the default)
    plan.pick.push_back(static_cast<int>(first));
    if (static_cast<int>(curr.size()) + 1 == max_batch) { if (curr.empty()) plan.cause = BATCHING_OFF; return plan; }
    bool other_waiting = false;
    for (size_t i = 0; i < pending.size(); i++) {
        if (i == first || pending[i].pid == pending[first].pid || process_running(pending[i].pid)) continue;
        other_waiting = true;
        if (pair_eligible(pending[first].sig, pending[i].sig) && joins_running(pending[i].sig)) { plan.pick.push_back(static_cast<int>(i)); break; }
    }
    if (plan.pick.size() == 1 && curr.empty()) plan.cause = other_waiting ? PAIR_NOT_LISTED : NOT_ENOUGH_IN_QUEUE;
    return plan;
}

// SCHED_MODE=hold. The default is to BATCH, as in "always": a kernel runs together with the one(s) already running unless the pair is HELD (pair_eligible is false: its efficiency is KNOWN and
// below SCHED_HOLD_BELOW_EFF, i.e. overlapping was measured to be slower than running the two one after the other). A held kernel waits for the running kernel(s) to finish; a kernel BEHIND it
// that is not held may go ahead (no head-of-line blocking: with three or more processes that is the choice of a better partner). With nothing held it decides exactly like "always".
static SelectPlan policy_select_hold(std::vector<Member>& curr, std::vector<Candidate>& pending, int max_batch) {
    SelectPlan plan;
    if (static_cast<int>(curr.size()) >= max_batch || pending.empty()) return plan;
    std::stable_sort(pending.begin(), pending.end(), [](const Candidate& a, const Candidate& b) {
        return (a.request_ns < b.request_ns) || (a.request_ns == b.request_ns && a.slot < b.slot);
    });
    auto process_running = [&](pid_t p) { for (const Member& m : curr) if (m.pid == p) return true; return false; };
    auto joins_running = [&](const std::string& sig) { for (const Member& m : curr) if (!pair_eligible(m.sig, sig)) return false; return true; };
    bool held_any = false; size_t first = pending.size();
    for (size_t i = 0; i < pending.size(); i++) {                                         // the oldest kernel that may run with everything running; a held one is skipped, it waits
        if (process_running(pending[i].pid)) continue;
        if (!joins_running(pending[i].sig)) { held_any = true; continue; }
        first = i; break;
    }
    if (first == pending.size()) { plan.cause = held_any ? PAIR_HELD : NOT_ENOUGH_IN_QUEUE; return plan; }
    plan.pick.push_back(static_cast<int>(first));
    if (static_cast<int>(curr.size()) + 1 == max_batch) { if (curr.empty()) plan.cause = BATCHING_OFF; return plan; }
    for (size_t i = 0; i < pending.size(); i++) {                                         // a partner for it: the oldest kernel of another process that is not held against the first or the running ones
        if (i == first || pending[i].pid == pending[first].pid || process_running(pending[i].pid)) continue;
        if (!pair_eligible(pending[first].sig, pending[i].sig) || !joins_running(pending[i].sig)) { held_any = true; continue; }
        plan.pick.push_back(static_cast<int>(i)); break;
    }
    if (plan.pick.size() == 1 && curr.empty()) plan.cause = held_any ? PAIR_HELD : NOT_ENOUGH_IN_QUEUE;
    return plan;
}

static SelectPlan policy_select(std::vector<Member>& curr, std::vector<Candidate>& pending, int max_batch) {
    if (g_mode == MODE_HOLD) return policy_select_hold(curr, pending, max_batch);
    if (g_mode == MODE_EFF) return policy_select_eff(curr, pending, max_batch);
    return policy_select_always(curr, pending, max_batch);              // MODE_ALWAYS, and MODE_NONE (its max_batch is 1)
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-variable"
/* ======================================================================
 * PREFETCH TOOLBOX -- KEPT, BUT NOT CALLED BY ANY POLICY. The text below describes the prefetch scheduling of the removed constraint policy, which used these
 * functions; they are the starting point for putting prefetch injection into a batching policy. They compile (so they do not rot) and are unused. What they need that
 * nothing supplies now: KernelStats / Candidate::ipc, sm_util, l2 / Member::est_end_ns (profile numbers), g_kernels_with_capacity (set by a table loader), and
 * a call to load_prefetch_config() at startup. The agents still obey do_prefetch.
 *
 * A kernel B that only the LIMITS keep out of the running work (a slot is free, B is next in line and in its table, but A + B would exceed
 * a cap on IPC, SM util or L2 throughput) waits until the kernels in the way finish. Instead B is released EARLY with do_prefetch=1: the agent
 * runs B's prefetch on a side stream and B waits for it, so B starts when its prefetch ends. Released one prefetch time before the kernels in
 * the way are expected to end, only the prefetch overlaps them, B starts about when they are done, and its data
 * is in L2. It also hides the time the daemon needs to notice that they finished.
 *
 * Nothing is done (B waits as before) when the batch is limited by something else (the slot count, B's process already running, an unmatched
 * kernel), when the agent would not prefetch for B, when an estimate is missing, when a kernel that should have ended by now has not, or when the
 * estimated remaining time of the kernels in B's way cannot cover the prefetch plus SCHED_PREFETCH_OVERHEAD_US (the prefetch could not end with them
 * and would only hold B back).
 *
 * Estimates
 *   - when a running kernel ends: release + its own prefetch + its estimated duration. That is the table's duration PER BLOCK (Duration / grid volume) times the
 *     volume x*y*z of the grid the launch really uses, or a Duration/Wave column times the number of waves of that grid, or the plain Duration column for whatever grid
 *     was profiled (see estimate_duration_us for the order and the formulas);
 *   - how long B's prefetch takes: bytes / SCHED_PREFETCH_BW_GBPS (+ SCHED_PREFETCH_FIXED_US). The bytes are what the AGENT will prefetch, which
 *     the daemon works out the same way the agent does (select_prefetch_targets): B's hook tag must be "layer_<N>/<role>", the tensors its process
 *     registered as layer<N>_<role>_..., each at least SCHED_PREFETCH_MIN_KB, in order, up to SCHED_PREFETCH_MAX_MB. The daemon cannot see the
 *     agents' environment: give the servers the same SCHED_PREFETCH_MAX_MB / _MIN_KB / SCHED_PREFETCH (the agent's own cap defaults to 64 MB).
 *   - The agent prefetches once per module entry, on the first kernel it is allowed to. Prefetching helps the FIRST kernel of an entry (the GEMM
 *     that reads the weights), so only a kernel whose tag differs from its process's previous kernel is planned for. Entries are told apart by
 *     the tag changing; a module entered twice in a row looks like one entry.
 * ====================================================================== */

struct PrefetchCfg {
    bool plan = true;                        // SCHED_PREFETCH_PLAN       0 turns the whole feature off
    bool agent_prefetch = true;              // SCHED_PREFETCH            as the agents' (0: they never prefetch)
    size_t cap_bytes = 10ull * 1024 * 1024;  // SCHED_PREFETCH_MAX_MB     as the agents' (their default is 64)
    size_t min_bytes = 0;                    // SCHED_PREFETCH_MIN_KB     as the agents'
    double bw_gbps = 933.1;                  // SCHED_PREFETCH_BW_GBPS    the bandwidth a prefetch runs at
    double fixed_us = 0;                     // SCHED_PREFETCH_FIXED_US   what a prefetch costs besides bytes / bandwidth
    double bias_us = 0;                      // SCHED_PLAN_BIAS_US        moves every planned release by this much (positive: later). The launch latency of the blocker and of B cancel, so
                                             //                           what is left to correct is the error of the Duration and prefetch estimates: read it off the log (lag of B's start after the end of the kernel it waited for)
    bool require_cover = true;               // SCHED_PREFETCH_REQUIRE_COVER 0 brings back the earlier behaviour: release B at once even when the remaining time is shorter than the prefetch
                                             //                           (it still started B soon after the blocker, because waiting for the blocker's completion to be noticed takes longer)
    int sms = 0;                             // SCHED_GPU_SMS             multiprocessors of the GPU: with the block size and the profile's achieved occupancy it gives the blocks per wave
    double max_threads_per_sm = 2048;        // SCHED_GPU_MAX_THREADS_PER_SM  resident threads one SM can hold (2048 on recent NVIDIA GPUs)
    double blocks_per_wave = 0;              // SCHED_BLOCKS_PER_WAVE     blocks the GPU runs at once, the same for every kernel: used instead of the derivation above if set
    double min_grid_volume = 1.0;            // SCHED_GRID_MIN_VOLUME      the estimate uses at least this grid volume. 1 is pure proportionality. A grid that fits the GPU in one wave takes about as long as
                                             //                           a full wave, so raising it to the number of blocks the GPU runs at once stops small grids from being estimated far too short
                                             //                           (only right if the per-block value was profiled with a big grid; too high a value gives overestimates, which are the safe side)
    double overhead_us = 0;                  // SCHED_PREFETCH_OVERHEAD_US an early release is made only if the estimated remaining time of the kernels in B's way is at least the prefetch time
                                             //                           PLUS this: the overhead (daemon -> agent -> GPU) that the prefetch also has to fit in. It does not move the release time
    double dur_scale = 1.0;                  // SCHED_DURATION_SCALE      multiplies every Duration from the tables. Durations measured one kernel at a time are SHORTER than when kernels
                                             //                           share the GPU, and an underestimate releases B while the kernel in its way still runs (an overestimate only makes the plan not happen)
    double spin_us = 100;                    // SCHED_SPIN_US             this long before a planned release the daemon polls instead of sleeping
    int limits = LIM_IPC | LIM_SM | LIM_L2;  // SCHED_PREFETCH_LIMITS     "ipc,sm,l2": plan only if EVERY limit that B fails is in this list
};
static PrefetchCfg g_pf;

static void load_prefetch_config() {
    auto num = [](const char* name, double& v) { if (const char* e = std::getenv(name)) v = std::atof(e); };
    if (const char* e = std::getenv("SCHED_PREFETCH_PLAN")) g_pf.plan = std::atoi(e) != 0;
    if (const char* e = std::getenv("SCHED_PREFETCH")) g_pf.agent_prefetch = std::atoi(e) != 0;
    double mb = static_cast<double>(g_pf.cap_bytes) / (1024.0 * 1024.0), kb = static_cast<double>(g_pf.min_bytes) / 1024.0;
    num("SCHED_PREFETCH_MAX_MB", mb); num("SCHED_PREFETCH_MIN_KB", kb);
    g_pf.cap_bytes = mb > 0 ? static_cast<size_t>(mb * 1024.0 * 1024.0) : 0;
    g_pf.min_bytes = kb > 0 ? static_cast<size_t>(kb * 1024.0) : 0;
    num("SCHED_PREFETCH_BW_GBPS", g_pf.bw_gbps); num("SCHED_PREFETCH_FIXED_US", g_pf.fixed_us);
    num("SCHED_PLAN_BIAS_US", g_pf.bias_us); num("SCHED_SPIN_US", g_pf.spin_us); num("SCHED_DURATION_SCALE", g_pf.dur_scale); num("SCHED_PREFETCH_OVERHEAD_US", g_pf.overhead_us);
    if (const char* e = std::getenv("SCHED_PREFETCH_REQUIRE_COVER")) g_pf.require_cover = std::atoi(e) != 0;
    if (g_pf.overhead_us < 0) g_pf.overhead_us = 0;
    num("SCHED_GRID_MIN_VOLUME", g_pf.min_grid_volume);
    if (g_pf.min_grid_volume < 1.0) g_pf.min_grid_volume = 1.0;
    if (const char* e = std::getenv("SCHED_GPU_SMS")) g_pf.sms = std::max(0, std::atoi(e));
    num("SCHED_GPU_MAX_THREADS_PER_SM", g_pf.max_threads_per_sm); num("SCHED_BLOCKS_PER_WAVE", g_pf.blocks_per_wave);
    if (g_pf.max_threads_per_sm < 1) g_pf.max_threads_per_sm = 2048;
    if (g_pf.blocks_per_wave < 0) g_pf.blocks_per_wave = 0;
    if (g_pf.dur_scale <= 0) g_pf.dur_scale = 1.0;
    if (g_pf.bw_gbps <= 0) g_pf.bw_gbps = 933.1;
    if (g_pf.spin_us < 0) g_pf.spin_us = 0;
    if (const char* e = std::getenv("SCHED_PREFETCH_LIMITS")) {
        g_pf.limits = 0;
        const std::string v = e;
        if (v.find("ipc") != std::string::npos) g_pf.limits |= LIM_IPC;
        if (v.find("sm") != std::string::npos) g_pf.limits |= LIM_SM;
        if (v.find("l2") != std::string::npos) g_pf.limits |= LIM_L2;
    }
}

static uint64_t prefetch_ns(size_t bytes) {                              // bytes / (GB/s) = ns
    return bytes ? static_cast<uint64_t>(g_pf.fixed_us * 1000.0 + static_cast<double>(bytes) / g_pf.bw_gbps) : 0;
}

// How many blocks of this kernel the GPU runs at once (one wave) = SMs x max resident blocks per SM. In order of preference:
//   1. the table's own Blocks/Wave for this kernel: exact, and the same capacity the profile's Duration/Wave was computed with (launch__grid_size / launch__waves_per_multiprocessor);
//   2. SCHED_BLOCKS_PER_WAVE, one number for every kernel;
//   3. SCHED_GPU_SMS x blocks per SM derived from the block size and the profile's ACHIEVED occupancy. This is only an approximation: achieved occupancy is below the
//      theoretical limit, and far below it when the profiled grid was small, so the wave comes out too small, the number of waves too large and the estimate too long
//      (the safe side, but it can be several times too long). Prefer 1 or 2.
// 0 if none of them is available.
static size_t g_kernels_with_capacity = 0;       // kernels that state their own blocks per wave (a table loader would set it)
static bool wave_model_configured() { return g_kernels_with_capacity > 0 || g_pf.blocks_per_wave > 0 || g_pf.sms > 0; }
static double blocks_per_wave(const KernelStats& st, const std::array<uint32_t, 3>& block) {
    if (!std::isnan(st.blocks_per_wave) && st.blocks_per_wave > 0) return st.blocks_per_wave;
    if (g_pf.blocks_per_wave > 0) return g_pf.blocks_per_wave;
    if (g_pf.sms <= 0) return 0;
    const double threads = std::max(1.0, static_cast<double>(block[0]) * block[1] * block[2]);
    const double occ = (st.occupancy > 0 ? std::min(st.occupancy, 100.0) : 100.0) / 100.0;
    const double per_sm = std::min(32.0, std::max(1.0, std::floor(g_pf.max_threads_per_sm * occ / threads + 0.5)));
    return static_cast<double>(g_pf.sms) * per_sm;
}

// How long a kernel is expected to run with THIS launch's grid; the table was profiled with one grid and the launch may use another.
//   1. Duration/Wave column and the blocks per wave known:  duration per wave x max(1, ceil(blocks / blocks per wave)). This inverts how the column is made,
//      Duration/Wave = duration / max(1, ceil(launch__waves_per_multiprocessor)): a wave only partly filled takes about as long as a full one, so a small grid costs one wave.
//   2. else the Duration / grid volume column: the per-block duration x the launch's volume x*y*z (at least SCHED_GRID_MIN_VOLUME).
//   3. else the plain Duration column, for whatever grid it was profiled with.
// NaN if the table has none of them.
static double estimate_duration_us(const KernelStats& st, const std::array<uint32_t, 3>& block, const std::array<uint32_t, 3>& grid) {
    const double volume = static_cast<double>(grid[0]) * grid[1] * grid[2];
    if (!std::isnan(st.duration_per_wave_us) && volume >= 1.0) {
        const double per_wave_capacity = blocks_per_wave(st, block);
        if (per_wave_capacity > 0) return st.duration_per_wave_us * std::max(1.0, std::ceil(volume / per_wave_capacity));
    }
    if (!std::isnan(st.duration_per_block_us) && volume >= 1.0) return st.duration_per_block_us * std::max(volume, g_pf.min_grid_volume);
    return st.duration_us;
}

// What the agent will prefetch for a kernel launched under this hook tag if it is allowed to (its select_prefetch_targets). Caller holds state->mutex.
static size_t predict_prefetch_bytes(pid_t pid, const char* tag) {
    if (!g_pf.agent_prefetch) return 0;
    int layer = -1, n = 0;
    if (std::sscanf(tag, "layer_%d/%n", &layer, &n) != 1 || n == 0 || tag[n] == '\0') return 0;
    const std::string prefix = "layer" + std::to_string(layer) + "_" + (tag + n) + "_";
    size_t remaining = g_pf.cap_bytes, total = 0;
    for (int i = 0; i < state->region_count && i < MAX_REGIONS; i++) {
        const SharedWeightRegion& r = state->regions[i];
        if (!r.active || r.owner_pid != pid) continue;
        if (std::strncmp(r.tag, prefix.c_str(), prefix.size()) != 0) continue;
        if (r.nbytes < g_pf.min_bytes) continue;
        size_t take = std::min(r.nbytes, remaining);
        take -= take % 16;                                               // the agent prefetches whole float4 units
        if (take == 0) continue;
        total += take; remaining -= take;
    }
    return total;
}

struct PfEntry { std::string tag; bool done = false; };                  // per process: the module entry of its latest released kernel, and whether it was prefetched
static std::map<pid_t, PfEntry> g_pf_entry;
static bool first_of_entry(pid_t pid, const char* tag) { auto it = g_pf_entry.find(pid); return it == g_pf_entry.end() || it->second.tag != tag; }
// Bytes the agent will prefetch if this kernel is released with do_prefetch=1: nothing if its module entry has had its prefetch already.
static size_t predicted_bytes(pid_t pid, const char* tag) {
    auto it = g_pf_entry.find(pid);
    if (it != g_pf_entry.end() && it->second.tag == tag && it->second.done) return 0;
    return predict_prefetch_bytes(pid, tag);
}
static void note_released(pid_t pid, const char* tag, bool prefetched) {
    PfEntry& e = g_pf_entry[pid];
    if (e.tag != tag) { e.tag = tag; e.done = false; }
    if (prefetched) e.done = true;
}

struct PfDecision { bool release_now = false; uint64_t wake_ns = 0, target_end_ns = 0; size_t bytes = 0; };
struct ArmedPlan { int slot = -1, kernel_id = 0; uint64_t t_fit = 0; };           // the plan that passed the remaining-time test, until it is carried out
static ArmedPlan g_armed;
static uint64_t g_pf_declined = 0;                                                // kernels (each once) not launched early because the remaining time was too short
static std::vector<int> g_pf_declined_kid(MAX_PENDING, 0);

// `b` is the oldest kernel whose process is free, in its table, and kept out of `curr` only by the limits in `mask`. Either release it now with a
// prefetch (release_now), or say when to (wake_ns), or neither (it waits for the kernels in the way, as without this feature).
static PfDecision plan_prefetch(const std::vector<Member>& curr, const Candidate& b, int mask, uint64_t now, const char* tag, int kernel_id) {
    PfDecision d;
    if (!g_pf.plan || (mask & ~g_pf.limits)) return d;                   // switched off, or a limit that B fails is not one we prefetch for
    if (!first_of_entry(b.pid, tag)) return d;                           // prefetching helps the first kernel of a module entry only
    d.bytes = predicted_bytes(b.pid, tag);
    if (d.bytes == 0) return d;                                          // the agent would prefetch nothing for it
    std::vector<const Member*> by_end;
    for (const Member& m : curr) { if (m.est_end_ns == 0) return d; by_end.push_back(&m); }     // a running kernel with no estimate: wait for it
    std::sort(by_end.begin(), by_end.end(), [](const Member* x, const Member* y) { return x->est_end_ns < y->est_end_ns; });
    double ipc = 0, sm = 0, l2 = 0;                                      // what runs now
    for (const Member* m : by_end) { ipc += m->ipc; sm += m->sm_util; l2 += m->l2; }
    uint64_t t_fit = 0;                                                  // when enough of it has ended for B to fit with the rest (same test as policy_select)
    for (const Member* m : by_end) {
        ipc -= m->ipc; sm -= m->sm_util; l2 -= m->l2;
        if (IPC_BATCH_LIMIT - ipc >= b.ipc && SM_UTIL_BATCH_LIMIT - sm >= b.sm_util && L2_BATCH_LIMIT - l2 >= b.l2) { t_fit = m->est_end_ns; break; }
    }
    if (!t_fit) return d;                                                // does not fit even on an empty GPU: it is released when the GPU is idle
    const uint64_t lead = prefetch_ns(d.bytes);
    const uint64_t need = lead + static_cast<uint64_t>(g_pf.overhead_us * 1000.0);
    // The estimated remaining time of the kernels in the way must cover the prefetch and its overhead, or the prefetch could not end with them and would
    // only hold B back: then B is not launched early. This is judged when B is first looked at; a plan that passed stays valid until its time comes (waking
    // up a few microseconds late must not cancel it), as long as the blockers' estimated end is the same.
    const bool armed = g_armed.slot == b.slot && g_armed.kernel_id == kernel_id && g_armed.t_fit == t_fit;
    if (g_pf.require_cover && !armed && t_fit < now + need) {
        if (g_pf_declined_kid[b.slot] != kernel_id) { g_pf_declined_kid[b.slot] = kernel_id; g_pf_declined++; }
        return d;
    }
    if (now > t_fit + lead) return d;                                    // they should have ended by now and have not: the estimate is no basis, wait for them
    g_armed = {b.slot, kernel_id, t_fit};
    d.target_end_ns = t_fit;
    const int64_t t_rel_s = static_cast<int64_t>(t_fit) - static_cast<int64_t>(lead) + static_cast<int64_t>(g_pf.bias_us * 1000.0);
    const uint64_t t_rel = t_rel_s > 0 ? static_cast<uint64_t>(t_rel_s) : 0;
    if (t_rel <= now) d.release_now = true; else d.wake_ns = t_rel;
    return d;
}

#pragma GCC diagnostic pop

/* ======================================================================
 * INFLIGHT TRACKING -- per-kernel slots in shared memory. Caller must
 * already hold state->mutex.
 * ====================================================================== */

// Returns the reserved slot's index, or -1 if the inflight table is full.
// FIX: `held` marks the slots of kernels that are still in `running`. With continuous batching such a kernel may have finished a moment ago, so its slot
// reads free, but its end time is read from that slot when it is retired; a new kernel handed that slot would reset it (end_ns, name) first.
static int reserve_inflight_slot_locked(pid_t pid, const char* kernel_name, const char* layer_tag, const std::vector<char>& held) {
    for (int i = 0; i < MAX_INFLIGHT; i++) {
        if (!held[i] && state->inflight[i].active.load() == 0) {
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
 * DECISION LOG: one row per kernel, written when that kernel has finished
 * (so the row can carry its end time). Join with the agent logs on (pid, kernel_id).
 *
 *   batch_id, batch_size, batch_pos   the decision: kernels with the same batch_id were released by one decision;
 *                                     batch_size is how many, batch_pos this kernel's place in the order of wake-ups
 *   decision_ns                       when the daemon fixed that decision and marked its kernels released (their launch time)
 *   post_ns                           the instant just before THIS kernel's wake-up was posted (decision_ns <= post_ns <= the
 *                                     agent's start_ns)
 *   pid, kernel_id                    which kernel: the process and its own per-process launch counter (= kernel_id in the
 *                                     agent's agent_<pid>_timing.tsv); arrival_seq is the daemon's global arrival order
 *   kernel_name, layer_tag
 *   request_ns, queue_wait_us         when the launch was intercepted, and how long it waited for its release
 *   end_ns                            when the kernel finished (the agent's completion callback); blank if not known
 *   n_pending, pending_pids           the choice the policy had: how many kernels were waiting at the decision, and from whom
 *   inflight_slot
 *   kernel_base, grid, block          the kernel's short name and launch dimensions as the agent reported them
 *   model                             the model tag the kernel's process reported (SCHED_MODE's models: qwen, whisper)
 * Gone with the removed policies (always blank or constant now): do_prefetch, reason, ipc, sm_util, l2_throughput, est_dur_us, pf_planned,
 * pf_pred_bytes, pf_est_us, pf_target_end_ns, est_end_ns.
 * ====================================================================== */

static FILE* g_log = nullptr;

// SCHED_LOG: how much is logged per kernel. full (the default) and decisions: the decision log and the batching log; off: neither (the batching
// summary is still written: one counter bump per decision, rewritten every 0.5 s). The agents and sched_hooks.py
// read the same variable (set it the same for the daemon and both servers): at decisions and off they also skip their CUDA-event timing and
// timing log, which is where most of the per-kernel CPU cost of logging is. Building and formatting a row costs the daemon about 6 us per decision.
enum { LOG_OFF = 0, LOG_DECISIONS = 1, LOG_FULL = 2 };
static int g_log_level = LOG_FULL;
static void parse_log_level() {
    const char* e = std::getenv("SCHED_LOG");
    if (e && *e) {
        if (!strcasecmp(e, "full") || !strcmp(e, "2") || !strcasecmp(e, "all")) g_log_level = LOG_FULL;
        else if (!strcasecmp(e, "decisions") || !strcmp(e, "1")) g_log_level = LOG_DECISIONS;
        else if (!strcasecmp(e, "off") || !strcmp(e, "0") || !strcasecmp(e, "none")) g_log_level = LOG_OFF;
        else std::fprintf(stderr, "[daemon] WARNING: SCHED_LOG=%s is not one of full, decisions, off: using full\n", e);
    }
    std::fprintf(stderr, "[daemon] log level (SCHED_LOG): %s\n", g_log_level == LOG_OFF ? "off: no decision log, no batching log (the batching summary is still written)"
                                                         : g_log_level == LOG_DECISIONS ? "decisions: decision and batching logs; agents record no timing" : "full");
}
static std::map<int, uint64_t> g_size_hist;         // kernels released by one decision -> how many decisions
static int g_largest_batch = 0;

static void append_row(std::string& out, const Member& m) {
    char buf[1024];
    const Release& r = *m.rel;
    const uint64_t end = (m.inflight >= 0) ? state->inflight[m.inflight].end_ns.load(std::memory_order_acquire) : 0;
    const double wait_us = (m.request_ns && m.post_ns >= m.request_ns) ? (m.post_ns - m.request_ns) / 1000.0 : -1.0;
    std::snprintf(buf, sizeof buf, "%llu\t%d\t%d\t%llu\t%llu\t%d\t%d\t%d\t",
                  (unsigned long long)r.id, r.size, m.pos, (unsigned long long)r.decision_ns,
                  (unsigned long long)m.post_ns, (int)m.pid, m.agent_kid, m.arrival_seq);
    out += buf;
    out += m.name; out += '\t'; out += m.tag; out += '\t';
    std::snprintf(buf, sizeof buf, "%llu\t", (unsigned long long)m.request_ns);
    out += buf;
    if (wait_us >= 0) { std::snprintf(buf, sizeof buf, "%.3f", wait_us); out += buf; }
    out += '\t';
    if (end) { std::snprintf(buf, sizeof buf, "%llu", (unsigned long long)end); out += buf; }
    std::snprintf(buf, sizeof buf, "\t%d\t", r.n_pending);
    out += buf;
    out += r.pending_pids;
    std::snprintf(buf, sizeof buf, "\t%d\t", m.inflight);
    out += buf;
    out += m.base; out += '\t'; out += m.grid; out += '\t'; out += m.block; out += '\t'; out += m.model;
    out += '\n';
}

// Flushed every time rows are written (once per finished batch): a daemon that is killed (kill -9, a closed terminal) must not
// take its log with it.
static void write_log(const std::string& rows) {
    if (!g_log || rows.empty()) return;
    std::fputs(rows.c_str(), g_log);
    std::fflush(g_log);
}

// A log that cannot be written is an error, not a warning: the run would go on and leave nothing behind.
static bool open_log() {
    if (g_log_level == LOG_OFF) return true;         // g_log stays null: write_log() and the row builders' call sites skip it
    const char* path_env = std::getenv("SCHED_DAEMON_LOG");
    const char* dir = std::getenv("SCHED_LOG_DIR");
    std::string path = path_env ? path_env : std::string(dir ? dir : ".") + "/scheduler_" + std::to_string(getpid()) + "_decisions.tsv";
    g_log = std::fopen(path.c_str(), "w");
    if (!g_log) {
        std::fprintf(stderr, "[daemon] ERROR: cannot open %s for the decision log: %s (check SCHED_LOG_DIR / SCHED_DAEMON_LOG)\n", path.c_str(), std::strerror(errno));
        return false;
    }
    static char buf[1 << 20];
    std::setvbuf(g_log, buf, _IOFBF, sizeof buf);
    std::fputs("batch_id\tbatch_size\tbatch_pos\tdecision_ns\tpost_ns\tpid\tkernel_id\tarrival_seq\tkernel_name\tlayer_tag\t"
               "request_ns\tqueue_wait_us\tend_ns\tn_pending\tpending_pids\tinflight_slot\t"
               "kernel_base\tgrid\tblock\tmodel\n", g_log);
    std::fflush(g_log);                              // the file has its header from the first moment
    char abs[PATH_MAX];
    std::fprintf(stderr, "[daemon] decision log (one row per kernel, written as batches finish): %s\n", realpath(path.c_str(), abs) ? abs : path.c_str());
    return true;
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
        state->slots[i].kernel_base[0] = '\0';
        state->slots[i].model[0] = '\0';
        for (int d = 0; d < 3; d++) { state->slots[i].grid[d] = 0; state->slots[i].block[d] = 0; }
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

struct BatchingStats {
    uint64_t decisions = 0, cause[N_CAUSES] = {0};
};
static BatchingStats g_bs;
static FILE* g_blog = nullptr;
static std::string g_summary_path;
static bool g_summary_dirty = false;
static uint64_t g_summary_written_ns = 0;

static void add_batching_row(std::string& out, uint64_t batch_id, uint64_t decision_ns, int batch_size, BatchCause cause,
                             int n_pending, const Candidate& head, const std::string& head_model, const std::string& head_base) {
    char buf[512];
    std::snprintf(buf, sizeof buf, "%llu\t%llu\t%d\t%s\t%s\t%d\t%d\t",
                  (unsigned long long)batch_id, (unsigned long long)decision_ns, batch_size, cause == BATCHED ? "batched" : "not_batched",
                  cause == BATCHED ? "-" : CAUSE_NAME[cause], n_pending, static_cast<int>(head.pid));
    out += buf;
    out += head_model; out += '\t'; out += head_base; out += '\n';
}

static void account_decision(BatchCause cause) {
    g_bs.decisions++;
    g_bs.cause[cause]++;
    g_summary_dirty = true;
}

static void write_batching_rows(const std::string& rows) {
    if (!g_blog || rows.empty()) return;
    std::fputs(rows.c_str(), g_blog);
    std::fflush(g_blog);
}

// Rewritten through a temporary file and renamed, like the unmatched file.
static bool write_batching_summary() {
    const std::string tmp = g_summary_path + ".tmp";
    FILE* f = std::fopen(tmp.c_str(), "w");
    if (!f) { std::fprintf(stderr, "[daemon] ERROR: cannot write %s: %s\n", tmp.c_str(), std::strerror(errno)); return false; }
    auto row = [&](const char* group, const std::string& category, uint64_t n, uint64_t total) {
        std::fprintf(f, "%s,%s,%llu,", group, category.c_str(), (unsigned long long)n);
        if (total) std::fprintf(f, "%.1f", 100.0 * static_cast<double>(n) / static_cast<double>(total));
        std::fputc('\n', f);
    };
    std::fputs("group,category,count,percent_of_group\n", f);
    const uint64_t not_batched = g_bs.decisions - g_bs.cause[BATCHED];
    row("decisions", "batched", g_bs.cause[BATCHED], g_bs.decisions);
    row("decisions", "not_batched", not_batched, g_bs.decisions);
    for (int c = 1; c < N_CAUSES; c++) row("not_batched_cause", CAUSE_NAME[c], g_bs.cause[c], not_batched);
    row("mode", g_mode == MODE_EFF ? "pair_eff_policy" : g_mode == MODE_HOLD ? "hold_list_policy" : "constraint_checking_off", g_bs.decisions, g_bs.decisions);     // which policy produced the file; always and none keep the value the old no-constraints baseline had
    if (std::fclose(f) != 0 || std::rename(tmp.c_str(), g_summary_path.c_str()) != 0) {
        std::fprintf(stderr, "[daemon] ERROR: cannot write %s: %s\n", g_summary_path.c_str(), std::strerror(errno));
        return false;
    }
    g_summary_dirty = false;
    g_summary_written_ns = mono_ns();
    return true;
}

// Both log files exist from the first moment (the .tsv with its header, the summary with zeros); a path that cannot be written is an error.
static bool open_batching_log() {
    const char* dir = std::getenv("SCHED_LOG_DIR");
    const std::string base = std::string(dir ? dir : ".") + "/scheduler_" + std::to_string(getpid());
    const std::string path = base + "_batching.tsv";
    g_summary_path = base + "_batching_summary.csv";
    g_blog = (g_log_level == LOG_OFF) ? nullptr : std::fopen(path.c_str(), "w");   // off: no batching log, the summary below is still written
    if (!g_blog && g_log_level != LOG_OFF) { std::fprintf(stderr, "[daemon] ERROR: cannot open %s for the batching log: %s (check SCHED_LOG_DIR)\n", path.c_str(), std::strerror(errno)); return false; }
    if (g_blog) std::fputs("batch_id\tdecision_ns\tbatch_size\toutcome\tcause\tn_pending\thead_pid\thead_model\thead_kernel_base\n", g_blog);
    if (g_blog) std::fflush(g_blog);
    if (!write_batching_summary()) return false;
    char a[PATH_MAX], b[PATH_MAX];
    if (g_blog) std::fprintf(stderr, "[daemon] batching log (one row per decision: batched or not, and why): %s\n", realpath(path.c_str(), a) ? a : path.c_str());
    std::fprintf(stderr, "[daemon] batching summary (counts by cause, rewritten as the run goes and at exit): %s\n", realpath(g_summary_path.c_str(), b) ? b : g_summary_path.c_str());
    return true;
}

// Called whenever the loop wakes, whether for work or after 250 ms of nothing: the files are at most about 0.75 s behind.
static void housekeeping() {
    const uint64_t now = mono_ns();
    if (g_summary_dirty && now - g_summary_written_ns >= 500000000ull) write_batching_summary();
}

int main() {
    struct sigaction sa{};
    sa.sa_handler = on_signal;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;                      // no SA_RESTART, so blocking calls return EINTR
    sigaction(SIGINT,  &sa, nullptr);     // Ctrl+C
    sigaction(SIGTERM, &sa, nullptr);     // kill
    sigaction(SIGHUP,  &sa, nullptr);     // the terminal the daemon runs in is closed

    // ---- the policy: SCHED_MODE (required). The switches of the removed policies are an error, not silently ignored.
    for (const char* gone : {"SCHED_CONSTRAINTS", "SCHED_PAIR_LIST"})
        if (std::getenv(gone)) { std::fprintf(stderr, "[daemon] ERROR: %s no longer exists: the policy it selected was removed. Set SCHED_MODE to always, none or eff.\n", gone); return 1; }
    for (const char* gone : {"SCHED_QWEN_CSV", "SCHED_WHISPER_CSV", "SCHED_UNMATCHED_CSV"})
        if (std::getenv(gone)) std::fprintf(stderr, "[daemon] WARNING: %s is ignored: the profile tables were removed\n", gone);
    const char* mode_env = std::getenv("SCHED_MODE");
    if (!mode_env || !*mode_env) {
        std::fprintf(stderr, "[daemon] ERROR: set SCHED_MODE to always (always continuous batching), none (no batching) eff (batch only pairs measured to gain, or with a kernel under SCHED_PAIR_SHORT_US) or hold (batch everything except pairs measured slower overlapped than one after the other)\n");
        return 1;
    }
    if (!strcasecmp(mode_env, "always")) g_mode = MODE_ALWAYS;
    else if (!strcasecmp(mode_env, "none")) g_mode = MODE_NONE;
    else if (!strcasecmp(mode_env, "eff")) g_mode = MODE_EFF;
    else if (!strcasecmp(mode_env, "hold")) g_mode = MODE_HOLD;
    else { std::fprintf(stderr, "[daemon] ERROR: SCHED_MODE=%s is not always, none, eff or hold\n", mode_env); return 1; }

    int max_batch = 2;
    if (const char* e = std::getenv("SCHED_BATCH_MAX")) max_batch = std::max(1, std::min(MAX_INFLIGHT, std::atoi(e)));
    if (g_mode == MODE_NONE) {
        if (std::getenv("SCHED_BATCH_MAX") && max_batch != 1) std::fprintf(stderr, "[daemon] WARNING: SCHED_BATCH_MAX=%d is ignored: SCHED_MODE=none runs one kernel at a time\n", max_batch);
        max_batch = 1;
    }
    const char* eff_path = std::getenv("SCHED_PAIR_EFF_FILE");
    const char* us_path = std::getenv("SCHED_KERNEL_US_FILE");
    if (g_mode == MODE_EFF || g_mode == MODE_HOLD) {                                                // the efficiency policy: see the block before policy_select_eff
        if (!eff_path || !*eff_path) { std::fprintf(stderr, "[daemon] ERROR: SCHED_MODE=%s needs SCHED_PAIR_EFF_FILE (and SCHED_KERNEL_US_FILE); make_pair_eff.py writes both\n", mode_env); return 1; }
        if (const char* v = std::getenv("SCHED_PAIR_MIN_EFF")) g_eff_min = std::atof(v);
        if (const char* v = std::getenv("SCHED_HOLD_BELOW_EFF")) g_hold_below = std::atof(v);
        if (const char* v = std::getenv("SCHED_PAIR_SHORT_US")) g_short_us = std::atof(v);
        if (const char* v = std::getenv("SCHED_PAIR_MIN_GANGS")) g_eff_min_gangs = std::max(1, std::atoi(v));
        size_t nt = 0, ns = 0, ng = 0, nk = 0, nshort = 0; std::string err;
        if (!load_pair_eff(eff_path, &nt, &ns, &ng, &err)) { std::fprintf(stderr, "[daemon] ERROR: SCHED_PAIR_EFF_FILE=%s: %s\n", eff_path, err.c_str()); return 1; }
        if (us_path && *us_path) { if (!load_kernel_us(us_path, &nk, &nshort, &err)) { std::fprintf(stderr, "[daemon] ERROR: SCHED_KERNEL_US_FILE=%s: %s\n", us_path, err.c_str()); return 1; } }
        else std::fprintf(stderr, "[daemon] WARNING: no SCHED_KERNEL_US_FILE: no kernel counts as short, so the short-kernel rule does nothing\n");
        if (g_mode == MODE_HOLD) std::fprintf(stderr, "[daemon] SCHED_PAIR_EFF_FILE=%s: %zu type-pair rows, %zu shape-pair rows, %zu with efficiency below %g (these are HELD; a shape row needs >= %d gangs to count); SCHED_KERNEL_US_FILE: %zu rows, %zu under %g us\n",
                                               eff_path, nt, ns, ng, g_hold_below, g_eff_min_gangs, nk, nshort, g_short_us);
        else std::fprintf(stderr, "[daemon] SCHED_PAIR_EFF_FILE=%s: %zu type-pair rows, %zu shape-pair rows, %zu with efficiency >= %g (a shape row needs >= %d gangs); SCHED_KERNEL_US_FILE: %zu rows, %zu under %g us\n",
                     eff_path, nt, ns, ng, g_eff_min, g_eff_min_gangs, nk, nshort, g_short_us);
    } else if ((eff_path && *eff_path) || (us_path && *us_path))
        std::fprintf(stderr, "[daemon] WARNING: SCHED_PAIR_EFF_FILE / SCHED_KERNEL_US_FILE are ignored: only SCHED_MODE=eff and SCHED_MODE=hold use them\n");

    init_shared_state();
    parse_log_level();
    if (!open_log()) return 1;
    if (!open_batching_log()) return 1;
        std::fprintf(stderr, "[daemon] pid %d. Stop it with  kill -INT %d  (Ctrl+C only reaches it if it runs in the foreground; SIGTERM and SIGHUP also write the logs; kill -9 loses only what is not yet written)\n",
                     static_cast<int>(getpid()), static_cast<int>(getpid()));
    if (g_mode == MODE_EFF)
        std::fprintf(stderr, "[daemon] SCHED_MODE=eff (efficiency / duration policy): NO batching by default (one kernel at a time, oldest first); a kernel joins a running one only if the pair was measured to gain (efficiency >= %g) or either kernel is under %g us; up to %d at once. Prefetch is not used (do_prefetch=0).\n", g_eff_min, g_short_us, max_batch);
    else if (g_mode == MODE_HOLD)
        std::fprintf(stderr, "[daemon] SCHED_MODE=hold (hold-list policy): BATCHING like always, up to %d running at once, EXCEPT a pair whose measured efficiency is below %g (overlapping was slower than one after the other) is never run together: one waits. A pair that was not measured, or has a kernel under %g us, is batched. Prefetch is not used (do_prefetch=0).\n", max_batch, g_hold_below, g_short_us);
    else if (g_mode == MODE_NONE)
        std::fprintf(stderr, "[daemon] SCHED_MODE=none: NO batching. One kernel at a time over all processes, oldest first. Prefetch is not used (do_prefetch=0).\n");
    else
        std::fprintf(stderr, "[daemon] SCHED_MODE=always: ALWAYS continuous batching. Whenever a slot is free the oldest waiting kernel whose process has none running is released, then the oldest of another such process, up to %d running at once, whatever the kernels are. One kernel per process in flight. Prefetch is not used (do_prefetch=0).\n", max_batch);

    int arrived_kernel_count = 0;
    uint64_t next_batch_id = 1;
    std::vector<Member> running;                      // kernels released and not yet seen to finish
    std::vector<KernelKey> keys(MAX_PENDING);         // slot -> key (name, block) of the kernel pending there (only pending slots are refreshed)

    std::fprintf(stderr, "[daemon] waiting for agent processes to attach...\n");

    while (!g_stop) {
        {
            uint64_t wait_ns = 250000000ull;           // 250 ms of nothing, or a signal (the loop condition ends it)
            timespec until;
            clock_gettime(CLOCK_REALTIME, &until);
            until.tv_nsec += static_cast<long>(wait_ns % 1000000000ull);
            until.tv_sec += static_cast<time_t>(wait_ns / 1000000000ull);
            if (until.tv_nsec >= 1000000000L) { until.tv_sec++; until.tv_nsec -= 1000000000L; }
            if (sem_timedwait(&state->new_request_sem, &until) != 0) { housekeeping(); continue; }
        }

        std::string rows, brows;                // log rows: of kernels that finished, and of the decisions made now
        std::vector<Member> released;           // what this iteration releases
        pthread_mutex_lock(&state->mutex);

        // 1. Retire every kernel that has finished: its row is complete now. Its inflight slot is only reused below.
        for (size_t i = 0; i < running.size();) {
            if (state->inflight[running[i].inflight].active.load() == 0) {
                if (g_log) append_row(rows, running[i]);
                running.erase(running.begin() + static_cast<long>(i));
            } 
            else i++;
        }

        std::vector<Candidate> pending;
        std::vector<int> arrival(MAX_PENDING, 0);
        for (int i = 0; i < MAX_PENDING; i++) {
            if (state->slots[i].active && !state->slots[i].ready) {
                const bool first_sighting = !state->slots[i].kernel_id;
                if (first_sighting) state->slots[i].kernel_id = ++arrived_kernel_count;
                arrival[i] = state->slots[i].kernel_id;
                keys[i] = slot_key(state->slots[i]);
                const std::string model = slot_model(state->slots[i]);
                if (first_sighting && !is_known_model(model)) g_other_launches++;           // another process: counted
                const std::string sig = ((g_mode == MODE_EFF && is_known_model(model)) || (g_mode == MODE_HOLD && !model.empty())) ? eff_sig(model, keys[i].name, keys[i].block, slot_grid(state->slots[i])) : std::string();
                pending.push_back({i, state->slots[i].pid, state->slots[i].request_ns, static_cast<double>(NAN), static_cast<double>(NAN), static_cast<double>(NAN),
                                   is_known_model(model), static_cast<double>(NAN), sig});
            }
        }

        if (!pending.empty()) {
            SelectPlan plan = policy_select(running, pending, max_batch);

            const Candidate& head = pending[plan.pick.empty() ? 0 : plan.pick[0]];   // FIX: the first kernel released (a kernel whose process is running waits, so not always pending[0])
            const std::string head_model = slot_model(state->slots[head.slot]);
            const std::string head_base = keys[head.slot].name;
            
            auto rel = std::make_shared<Release>();
            rel->n_pending = static_cast<int>(pending.size());
            for (size_t i = 0; i < pending.size(); i++) {
                if (i) rel->pending_pids += ',';
                rel->pending_pids += std::to_string(static_cast<int>(pending[i].pid));
            }

            std::vector<char> held_slots(MAX_INFLIGHT, 0);                       // FIX: inflight slots of kernels that are not retired yet
            for (const Member& m : running) if (m.inflight >= 0) held_slots[m.inflight] = 1;

            // mark inflight true & add member to running batch
            for (int idx : plan.pick) {
                const int s = pending[idx].slot;
                const int infl = reserve_inflight_slot_locked(state->slots[s].pid, state->slots[s].kernel_name, state->slots[s].layer_tag, held_slots);
                
                assert(infl != -1);

                Member m;
                m.slot = s; m.inflight = infl; m.pos = static_cast<int>(released.size());
                m.pid = state->slots[s].pid;
                m.agent_kid = state->slots[s].agent_kernel_id;
                m.arrival_seq = arrival[s];
                m.name = state->slots[s].kernel_name;
                m.tag = state->slots[s].layer_tag;
                m.model = slot_model(state->slots[s]);
                m.base = keys[s].name;
                m.grid = dims_str(slot_grid(state->slots[s]));
                m.block = dims_str(keys[s].block);
                m.sig = pending[idx].sig;
                m.request_ns = state->slots[s].request_ns;
                m.do_prefetch = 0;                                               // no policy asks for a prefetch (see the PREFETCH TOOLBOX)
                released.push_back(m);
            }

            if (!released.empty()) {
                rel->id = next_batch_id++;
                rel->size = static_cast<int>(released.size());
                rel->decision_ns = mono_ns();
                for (Member& m : released) {
                    m.rel = rel;
                    state->slots[m.slot].inflight_slot = m.inflight;
                    state->slots[m.slot].do_prefetch = m.do_prefetch;
                    state->slots[m.slot].batch_id = rel->id;
                    state->slots[m.slot].batch_size = rel->size;
                    state->slots[m.slot].batch_pos = m.pos;
                    state->slots[m.slot].ready = 1;
                }
                g_size_hist[rel->size]++;
                BatchCause cause = plan.cause;                                   // why this batch is what it is
                if (released.size() >= 2 || !running.empty()) cause = BATCHED;   // FIX: a kernel released into running work is batched with it
                else if (cause == BATCHED) cause = INFLIGHT_FULL;                // the policy wanted more, the inflight table was full
                if (g_blog) add_batching_row(brows, rel->id, rel->decision_ns, rel->size, cause, rel->n_pending, head, head_model, head_base);
                account_decision(cause);
            }
        }

        pthread_mutex_unlock(&state->mutex);

        // 3. Wake the released kernels as close together as possible; the time-critical part comes before any logging.
        for (Member& m : released) {
            m.post_ns = mono_ns();                      // just BEFORE the wake-up: decision_ns <= post_ns <= the agent's start_ns
            sem_post(&state->slots[m.slot].wake_sem);
        }
        for (const Member& m : released)
            std::fprintf(stderr, "[daemon] release pid=%d model=%s kernel=%s grid=%s block=%s layer=%s batch=%llu size=%d\n",
                         (int)m.pid, m.model.c_str(), m.base.c_str(), m.grid.c_str(), m.block.c_str(), m.tag.c_str(),
                         (unsigned long long)m.rel->id, m.rel->size);
        for (Member& m : released) running.push_back(std::move(m));
        g_largest_batch = std::max(g_largest_batch, static_cast<int>(running.size()));
        write_log(rows);
        write_batching_rows(brows);
        housekeeping();
    }

    // Shutdown: log the kernels of a batch still running (end times only where they had finished) and flush.
    std::string rows;
    for (const Member& m : running) if (g_log) append_row(rows, m);
    write_log(rows);
    if (g_log) std::fclose(g_log);
    std::fprintf(stderr, "[daemon] decisions by number of kernels released together:");
    for (const auto& kv : g_size_hist) std::fprintf(stderr, " %d x%llu", kv.first, (unsigned long long)kv.second);
    std::fprintf(stderr, "; largest batch: %d\n", g_largest_batch);
    write_batching_summary();
    if (g_blog) std::fclose(g_blog);
    {
        const uint64_t nb = g_bs.decisions - g_bs.cause[BATCHED];
        std::fprintf(stderr, "[daemon] %llu decisions: %llu batched, %llu not batched:", (unsigned long long)g_bs.decisions, (unsigned long long)g_bs.cause[BATCHED], (unsigned long long)nb);
        for (int c = 1; c < N_CAUSES; c++) std::fprintf(stderr, " %s %llu%s", CAUSE_NAME[c], (unsigned long long)g_bs.cause[c], c + 1 < N_CAUSES ? "," : "");
        std::fprintf(stderr, "  (%s)\n", g_summary_path.c_str());
    }
    if (g_other_launches)
        std::fprintf(stderr, "[daemon] %llu launches came from processes that are neither qwen nor whisper (see the decision log, model column; the efficiency policy never batches them)\n", (unsigned long long)g_other_launches);
    return 0;
}