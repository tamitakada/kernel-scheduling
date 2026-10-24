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
 * POLICY -- the part to edit.
 * ====================================================================== */

// Whether the agent should prefetch for this kernel launch (1) or not (0). Written into the kernel's slot before it is
// released; the agent does what it says. Placeholder policy: always prefetch.
static int policy_should_prefetch(const char* kernel_name, const char* layer_tag) {
    (void)kernel_name;
    (void)layer_tag;
    return 0;
}

/* ======================================================================
 * KERNEL TABLES -- measured statistics of kernels, one table per model, each read once at startup from its own CSV
 * (SCHED_QWEN_CSV and SCHED_WHISPER_CSV).
 *
 * The columns: Kernel, Block ("(x, y, z)"), Launches, then pairs of a statistic and its "std" (a Grid column may be there too;
 * it is not used). The KEY of a table is the kernel's name and its BLOCK dimensions: the grid is not part of it, so a kernel
 * is found whatever its grid. Several rows can share a name and block (they differ in grid); they become ONE entry whose
 * launches are added up and whose every statistic is the launch-weighted mean of the rows that have it. Kernel is NCU's short
 * name (no namespaces, templates or parameters); the agent reduces every symbol it sees to that form (kernel_key.hpp) and
 * sends it with the launch dimensions.
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
struct StatPair { const char* mean_col; const char* std_col; double KernelStats::*mean; double KernelStats::*sd; };
static const StatPair STAT_PAIRS[] = {
    {"SM Util %", "SM Util % std", &KernelStats::sm_util, &KernelStats::sm_util_std},
    {"Achieved Occ %", "Achieved Occ % std", &KernelStats::occupancy, &KernelStats::occupancy_std},
    {"IPC (active)", "IPC (active) std", &KernelStats::avg_ipc, &KernelStats::avg_ipc_std},
    {"L2 Throughput %", "L2 Throughput % std", &KernelStats::l2_tput, &KernelStats::l2_tput_std},
};

// Two separate maps. A kernel is looked up ONLY in the map of the model whose process launched it: the agent tags every launch
// with its process's SCHED_MODEL ("qwen" or "whisper"), so a Whisper kernel is never searched for in the Qwen map or vice versa.
typedef std::map<KernelKey, KernelStats> KernelTable;
static KernelTable g_qwen_table, g_whisper_table;
static bool g_have_qwen = false, g_have_whisper = false;       // whether a CSV was given for that model

// The table to search for a model tag, or nullptr if there is none (unknown tag, empty tag, or no CSV given for that model).
static const KernelTable* table_for(const std::string& model) {
    if (model == "qwen" && g_have_qwen) return &g_qwen_table;
    if (model == "whisper" && g_have_whisper) return &g_whisper_table;
    return nullptr;
}

// The two processes the scheduler has tables for. A kernel of any other process (another SCHED_MODEL, or none) can never be
// found, so it is not listed in the unmatched-kernels file (it still runs, alone when it is first in the queue, and it is in the
// decision log with its model tag).
static bool is_known_model(const std::string& model) { return model == "qwen" || model == "whisper"; }

struct Unmatched { uint64_t launches = 0; std::string example_symbol, example_grid; };   // example_*: of the first launch
// "qwen" / "whisper" -> the kernels of that model's process that are not in that model's table (the unmatched-kernels file)
static std::map<std::string, std::map<KernelKey, Unmatched>> g_unmatched;
static uint64_t g_other_launches = 0;            // launches of processes that are neither: not listed there
static bool g_unmatched_dirty = false;           // the unmatched file is out of date

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

static void note_unmatched(const std::string& model, const KernelKey& k, const char* raw_symbol, const std::string& grid) {
    Unmatched& u = g_unmatched[model][k];
    if (u.launches++ == 0) { u.example_symbol = raw_symbol; u.example_grid = grid; }
    g_unmatched_dirty = true;
}

// A CSV file as rows of fields: quoted fields may contain commas, "" is a quote, \r\n and \n both end a row.
static bool read_csv_rows(const std::string& path, std::vector<std::vector<std::string>>& rows) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    std::vector<std::string> row;
    std::string field;
    bool quoted = false, started = false;
    for (size_t i = text.compare(0, 3, "\xEF\xBB\xBF") == 0 ? 3 : 0; i < text.size(); i++) {
        const char c = text[i];
        if (quoted) {
            if (c != '"') field += c;
            else if (i + 1 < text.size() && text[i + 1] == '"') { field += '"'; i++; }
            else quoted = false;
        } else if (c == '"') { quoted = true; started = true; }
        else if (c == ',') { row.push_back(field); field.clear(); started = true; }
        else if (c == '\r') {}
        else if (c == '\n') {
            if (started || !field.empty()) { row.push_back(field); rows.push_back(row); }
            row.clear(); field.clear(); started = false;
        } else field += c;
    }
    if (started || !field.empty()) { row.push_back(field); rows.push_back(row); }
    return true;
}

static bool parse_dims(const std::string& s, std::array<uint32_t, 3>& out) {
    unsigned a, b, c;
    if (std::sscanf(s.c_str(), " ( %u , %u , %u )", &a, &b, &c) != 3) return false;
    out = {a, b, c};
    return true;
}

static size_t g_kernels_with_duration = 0;       // kernels, over all tables, that have a duration (plain or per block)
static size_t g_kernels_with_ratio = 0;          // ... of which have a duration per grid volume
static size_t g_kernels_with_capacity = 0;       // kernels, over all tables, that state their own blocks per wave
static size_t g_kernels_with_wave = 0;           // kernels, over all tables, that have a duration per wave (usable only if the number of blocks per wave is known)

// Fills `table` from the CSV. Returns false (after saying why) if the file cannot be used. `label` names the model in messages.
static bool load_kernel_table(const std::string& path, const char* label, KernelTable& table) {
    std::vector<std::vector<std::string>> rows;
    if (!read_csv_rows(path, rows)) { std::fprintf(stderr, "[daemon] ERROR: cannot read the kernel table %s: %s\n", path.c_str(), std::strerror(errno)); return false; }
    if (rows.empty()) { std::fprintf(stderr, "[daemon] ERROR: the kernel table %s is empty\n", path.c_str()); return false; }

    std::map<std::string, size_t> col;
    for (size_t i = 0; i < rows[0].size(); i++) col[rows[0][i]] = i;
    for (const char* need : {"Kernel", "Block", "Launches", "IPC (active)", "SM Util %", "L2 Throughput %"}) {
        if (!col.count(need)) {
            std::fprintf(stderr, "[daemon] ERROR: the kernel table %s has no column \"%s\". Its columns are:", path.c_str(), need);
            for (const auto& h : rows[0]) std::fprintf(stderr, " \"%s\"", h.c_str());
            std::fprintf(stderr, "\n");
            return false;
        }
    }
    // The profiled duration of a kernel, if the CSV has a column for it: the first header that starts with "Duration" (any case). Its unit is
    // read from the parentheses ("Duration (us)", "(ns)", "(ms)", "(s)"), microseconds if there are none. The daemon uses it to estimate when a
    // running kernel ends (see PREFETCH SCHEDULING); without it everything else works as before.
    // A second column, Duration / grid volume (any header with "duration" and a "/" or " per ", e.g. "Duration / Grid Volume (us)"), gives the duration PER BLOCK
    // of the profiled grid; the daemon multiplies it by the volume x*y*z of the grid a launch really uses. Columns with "std" in the name are ignored.
    std::string dur_col, ratio_col, wave_col, cap_col;
    double dur_scale_us = 1.0, ratio_scale_us = 1.0, wave_scale_us = 1.0;
    auto unit_scale = [](const std::string& low) { return low.find("(ns)") != std::string::npos ? 1e-3 : low.find("(ms)") != std::string::npos ? 1e3 : low.find("(s)") != std::string::npos ? 1e6 : 1.0; };
    for (const auto& h : rows[0]) {
        std::string low = h;
        for (char& ch : low) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        if (low.find("duration") == std::string::npos && low.find("std") == std::string::npos && low.find("wave") != std::string::npos && low.find("block") != std::string::npos) {
            if (cap_col.empty()) cap_col = h;                                  // "Blocks/Wave", "Blocks per Wave", "Wave Size (blocks)": the capacity of one wave, in blocks
            continue;
        }
        if (low.find("duration") == std::string::npos || low.find("std") != std::string::npos) continue;
        const bool per = low.find('/') != std::string::npos || low.find(" per ") != std::string::npos;
        if (per && low.find("wave") != std::string::npos) { if (wave_col.empty()) { wave_col = h; wave_scale_us = unit_scale(low); } }     // "Duration/Wave (us)": per WAVE, never per block
        else if (per) { if (ratio_col.empty()) { ratio_col = h; ratio_scale_us = unit_scale(low); } }
        else if (dur_col.empty() && low.compare(0, 8, "duration") == 0) { dur_col = h; dur_scale_us = unit_scale(low); }
    }
    size_t with_duration = 0, with_ratio = 0, with_wave = 0, with_cap = 0;
    int skipped = 0;
    size_t used = 0;
    for (size_t r = 1; r < rows.size(); r++) {
        const auto& row = rows[r];
        auto cell = [&](const char* name) -> std::string { auto c = col.find(name); return c != col.end() && c->second < row.size() ? row[c->second] : std::string(); };
        KernelKey key;
        key.name = cell("Kernel");
        if (key.name.empty() || !parse_dims(cell("Block"), key.block)) { skipped++; continue; }
        KernelStats st;
        st.launches = std::atol(cell("Launches").c_str());
        for (const StatPair& sp : STAT_PAIRS) {
            const std::string m = cell(sp.mean_col), sd = cell(sp.std_col);
            if (!m.empty()) st.*sp.mean = std::strtod(m.c_str(), nullptr);
            if (!sd.empty()) st.*sp.sd = std::strtod(sd.c_str(), nullptr);
        }
        if (!dur_col.empty()) {
            const std::string v = cell(dur_col.c_str());
            if (!v.empty()) { st.duration_us = std::strtod(v.c_str(), nullptr) * dur_scale_us; with_duration++; }
        }
        if (!ratio_col.empty()) {
            const std::string v = cell(ratio_col.c_str());
            if (!v.empty()) { st.duration_per_block_us = std::strtod(v.c_str(), nullptr) * ratio_scale_us; with_ratio++; }
        }
        if (!wave_col.empty()) {
            const std::string v = cell(wave_col.c_str());
            if (!v.empty()) { st.duration_per_wave_us = std::strtod(v.c_str(), nullptr) * wave_scale_us; with_wave++; }
        }
        if (!cap_col.empty()) {
            const std::string v = cell(cap_col.c_str());
            if (!v.empty() && std::strtod(v.c_str(), nullptr) > 0) { st.blocks_per_wave = std::strtod(v.c_str(), nullptr); with_cap++; }
        }
        table[key] = st;
        used++;
    }
    std::fprintf(stderr, "[daemon] %s kernel table: %zu kernels (distinct name and block) from %zu rows in %s", label, table.size(), used, path.c_str());
    if (skipped) std::fprintf(stderr, "; %d unreadable rows skipped", skipped);
    if (dur_col.empty() && ratio_col.empty() && wave_col.empty()) std::fprintf(stderr, "; no Duration column");
    if (!dur_col.empty()) std::fprintf(stderr, "; %zu with a duration (column \"%s\")", with_duration, dur_col.c_str());
    if (!ratio_col.empty()) std::fprintf(stderr, "; %zu with a duration per grid volume (column \"%s\")", with_ratio, ratio_col.c_str());
    g_kernels_with_duration += with_duration + with_ratio;
    if (!wave_col.empty()) std::fprintf(stderr, "; %zu with a duration per wave (column \"%s\")", with_wave, wave_col.c_str());
    if (!cap_col.empty()) std::fprintf(stderr, "; %zu with their own blocks per wave (column \"%s\")", with_cap, cap_col.c_str());
    g_kernels_with_ratio += with_ratio;
    g_kernels_with_wave += with_wave;
    g_kernels_with_capacity += with_cap;
    std::fprintf(stderr, "\n");
    return !table.empty();
}

static constexpr double IPC_BATCH_LIMIT = 4.0;
static constexpr double SM_UTIL_BATCH_LIMIT = 100.0;   // percent
static constexpr double L2_BATCH_LIMIT = 100.0;        // percent

struct Candidate { 
    int slot; 
    pid_t pid; 
    uint64_t request_ns; 
    bool in_table; 
    double ipc, sm_util, l2; 
    bool known_model; 
    double dur_us = NAN;                            // its profiled duration in microseconds, NaN if unknown
    std::string sig;                                // pair-list policy only: model|kernel|block-count bucket (empty otherwise)
};   // one pending kernel (known_model: its process is qwen or whisper)

struct Release {                                    // what one decision looked like, shared by the kernels it released
    uint64_t id = 0, decision_ns = 0;
    int size = 0, n_pending = 0;
    std::string pending_pids, reason;
};
struct Member {
    int slot = -1, inflight = -1, agent_kid = 0, arrival_seq = 0, do_prefetch = 0, pos = 0;
    pid_t pid = 0;
    std::string name, tag, base, grid, block, model;
    std::string sig;                      // pair-list policy only: see Candidate
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
enum BatchCause { BATCHED = 0, BATCHING_OFF, FIRST_NOT_IN_TABLE, FIRST_OTHER_PROCESS, NOT_ENOUGH_IN_QUEUE, PARTNERS_NOT_FOUND, LIMIT_VIOLATED, INFLIGHT_FULL, PAIR_NOT_LISTED, N_CAUSES };
static const char* const CAUSE_NAME[N_CAUSES] = {"batched", "batching_disabled", "first_not_in_table", "first_other_process",
                                                 "not_enough_in_queue", "partners_not_found", "limit_violated", "inflight_table_full", "pair_not_listed"};
static constexpr int LIM_IPC = 1, LIM_SM = 2, LIM_L2 = 4;       // the bits of a set of limits

static std::string limits_str(int mask) {                       // 3 -> "ipc+sm"
    std::string t;
    if (mask & LIM_IPC) t += "ipc";
    if (mask & LIM_SM) t += std::string(t.empty() ? "" : "+") + "sm";
    if (mask & LIM_L2) t += std::string(t.empty() ? "" : "+") + "l2";
    return t.empty() ? "-" : t;
}

struct SelectPlan {
    std::vector<int> pick;                          // indices into `pending`, in release order
    std::string reason;
    BatchCause cause = BATCHED;                     // BATCHED if two or more kernels, else why not
    int others_waiting = 0;                         // other processes with a kernel waiting: the possible partners
    int considered = 0, not_found = 0, rejected = 0, joined = 0;   // partners the policy looked at, and what became of them
    int rej_ipc = 0, rej_sm = 0, rej_l2 = 0;        // refused partners, per limit that failed (one partner can fail several)
    int first_rejected_mask = 0;                    // the limits the first refused partner failed
    int blocked = -1;                               // index in `pending` of a kernel that only the LIMITS kept out of the running work (-1: none)
    int blocked_mask = 0;                           // ... and the limits it fails
};

// SCHED_CONSTRAINTS=off: the baseline with NO constraint checking. A free slot is always filled: the oldest waiting kernel whose process has none running is
// released, then the oldest waiting kernel of another such process, up to max_batch running at once -- whether or not the kernels are in their model's table
// (an unmatched kernel is batched like any other), and whatever the IPC / SM / L2 sums. Nothing is ever held for a limit, so no prefetch is planned (and
// none is requested: do_prefetch is 0). What stays: one kernel per process in flight (the next launch of a process would only queue behind its previous one
// in the same stream), the oldest-first order, and max_batch. Default on: the policy below is unchanged.
static bool g_constraints = true;

static SelectPlan policy_select_unconstrained(std::vector<Member>& curr, std::vector<Candidate>& pending, int max_batch) {
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

// SCHED_PAIR_LIST=file: the whitelist policy. The default is NO batching: one kernel at a time over both processes, oldest first. A second kernel runs
// with the one(s) already running only if the two are a LISTED pair (a pair of kernel types the logs showed to overlap with high efficiency). A kernel
// is identified by model | kernel base name | its number of thread blocks, in 5 buckets (< 8, < 32, < 128, < 512, more): the efficiency of an overlap
// depends on how much of the GPU a launch fills, so the bucket is part of the key. Tables, limits and prefetch are not used. One per process stays.
static std::unordered_set<std::string> g_pairlist;
static bool g_pairlist_on = false;
static std::string pair_sig(const std::string& model, const std::string& base, const std::array<uint32_t, 3>& grid) {
    static const uint64_t EDGES[4] = {8, 32, 128, 512};
    const uint64_t blocks = static_cast<uint64_t>(grid[0]) * grid[1] * grid[2]; int b = 0;
    while (b < 4 && blocks >= EDGES[b]) b++;                                   // np.digitize(blocks, [8, 32, 128, 512]), as in make_pair_list.py
    return model + "|" + base + "|" + std::to_string(b);
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
static bool g_eff_on = false;
static double g_eff_min = 1.25, g_short_us = 20.0; static int g_eff_min_gangs = 3;
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
        ((xa == "*" && xb == "*") ? *n_type : *n_shape)++; if (eff >= g_eff_min) (*n_gain)++;
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
static bool pair_eligible_compute(const std::string& a, const std::string& b) {
    if (kernel_short(a) || kernel_short(b)) return true;                                  // <<<<<<<< SHORT-KERNEL RULE: batch ANY pair with a kernel under SCHED_PAIR_SHORT_US, profiled or not. COMMENT OUT THIS ONE LINE to batch only pairs measured to gain.
    const EffRow* e = nullptr; const auto s = g_eff.find(pair_key(a, b));
    if (s != g_eff.end() && s->second.gangs >= g_eff_min_gangs) e = &s->second;           // the exact shape pair, if enough gangs measured it
    if (!e) { const auto t = g_eff.find(pair_key(type_of(a), type_of(b))); if (t != g_eff.end()) e = &t->second; }       // else the type pair (any grid)
    return e && e->eff >= g_eff_min;
}
static bool pair_eligible(const std::string& a, const std::string& b) {
    if (a.empty() || b.empty()) return false;                                             // a kernel of a process that is neither model: never batched
    const std::string k = pair_key(a, b); const auto it = g_elig.find(k); if (it != g_elig.end()) return it->second;
    const bool r = pair_eligible_compute(a, b); g_elig.emplace(k, r); return r;
}
// ==============================================================================================================================================================================

static bool pair_listed(const std::string& a, const std::string& b) { if (g_eff_on) return pair_eligible(a, b); return !a.empty() && !b.empty() && g_pairlist.count(pair_key(a, b)) > 0; }
static bool load_pair_list(const char* path, size_t* n_pairs) {       // tab-separated: sig_a, sig_b, ... (a header line starting with sig_a and # lines are skipped)
    FILE* f = std::fopen(path, "r"); if (!f) return false;
    char buf[16384]; *n_pairs = 0;
    while (std::fgets(buf, sizeof buf, f)) {
        std::string l = buf; while (!l.empty() && (l.back() == '\n' || l.back() == '\r')) l.pop_back();
        if (l.empty() || l[0] == '#') continue;
        const size_t t1 = l.find('\t'); if (t1 == std::string::npos) continue;
        const size_t t2 = l.find('\t', t1 + 1); const std::string a = l.substr(0, t1), b = l.substr(t1 + 1, t2 == std::string::npos ? std::string::npos : t2 - t1 - 1);
        if (a == "sig_a" || a.empty() || b.empty()) continue;
        g_pairlist.insert(pair_key(a, b)); (*n_pairs)++;
    }
    std::fclose(f); return true;
}

static SelectPlan policy_select_pairlist(std::vector<Member>& curr, std::vector<Candidate>& pending, int max_batch) {
    SelectPlan plan;
    if (static_cast<int>(curr.size()) >= max_batch || pending.empty()) return plan;
    std::stable_sort(pending.begin(), pending.end(), [](const Candidate& a, const Candidate& b) {
        return (a.request_ns < b.request_ns) || (a.request_ns == b.request_ns && a.slot < b.slot);
    });
    auto process_running = [&](pid_t p) { for (const Member& m : curr) if (m.pid == p) return true; return false; };
    auto joins_running = [&](const std::string& sig) { for (const Member& m : curr) if (!pair_listed(m.sig, sig)) return false; return true; };
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
        if (pair_listed(pending[first].sig, pending[i].sig) && joins_running(pending[i].sig)) { plan.pick.push_back(static_cast<int>(i)); break; }
    }
    if (plan.pick.size() == 1 && curr.empty()) plan.cause = other_waiting ? PAIR_NOT_LISTED : NOT_ENOUGH_IN_QUEUE;
    return plan;
}

static SelectPlan policy_select(std::vector<Member>& curr, std::vector<Candidate>& pending, int max_batch) {
    if (g_pairlist_on) return policy_select_pairlist(curr, pending, max_batch);
    if (!g_constraints) return policy_select_unconstrained(curr, pending, max_batch);
    SelectPlan plan;

    // don't exceed max batch size, check for empty queue
    if (curr.size() >= max_batch || pending.empty()) return plan; 

    std::stable_sort(pending.begin(), pending.end(), [](const Candidate& a, const Candidate& b) {
        return (a.request_ns < b.request_ns) || (a.request_ns == b.request_ns && a.slot < b.slot);
    });

    // FIX: a running kernel with no usable numbers (not in its table) runs by itself. Left to the budget below its NaN would make rem_stats NaN, every
    // comparison with it false, the "can't be serviced" test would never fire, and the oldest kernel would be released into it.
    for (const Member& m : curr)
        if (std::isnan(m.ipc) || std::isnan(m.sm_util) || std::isnan(m.l2)) return plan;

    std::array<double, 3> rem_stats = std::accumulate(curr.begin(), curr.end(), 
        std::array<double, 3>{4.0,100.0,100.0}, 
        [](std::array<double, 3> acc, const Member& next) { 
            acc[0] -= next.ipc; acc[1] -= next.sm_util; acc[2] -= next.l2;
            return acc;
        });

    // if oldest kernel has no profiling data, force no batching
    if (!pending[0].in_table) {
        plan.cause = max_batch <= 1 ? BATCHING_OFF : pending[0].known_model ? FIRST_NOT_IN_TABLE : FIRST_OTHER_PROCESS;   // FIX: the causes as before
        // FIX: it has to be released, or it (and everything behind it) waits forever. Nothing running: it runs now, by itself. Otherwise it waits
        // for what is running to finish, and nothing else is admitted meanwhile.
        if (curr.empty()) plan.pick.push_back(0);
        return plan;
    }

    // FIX: one kernel per process in flight. A process's next launch is usually already waiting while its previous kernel runs; released now it would only
    // queue behind it in the same stream, take a slot, and keep another process's kernel out. `first` is the oldest kernel whose process has none running.
    auto process_running = [&](pid_t p) { for (const Member& m : curr) if (m.pid == p) return true; return false; };
    size_t first = 0;
    while (first < pending.size() && process_running(pending[first].pid)) first++;
    if (first == pending.size()) { plan.cause = NOT_ENOUGH_IN_QUEUE; return plan; }
    // FIX: a kernel without profiling data cannot join running work: it waits until it is the oldest and the GPU is idle
    if (!pending[first].in_table) { plan.cause = PARTNERS_NOT_FOUND; return plan; }

    // if the oldest kernel can't be serviced, wait for curr batch to finish
    if (rem_stats[0] < pending[first].ipc || rem_stats[1] < pending[first].sm_util || rem_stats[2] < pending[first].l2) {
        plan.cause = LIMIT_VIOLATED;
        plan.blocked = static_cast<int>(first);                              // the prefetch planner looks at this kernel
        plan.blocked_mask = (rem_stats[0] < pending[first].ipc ? LIM_IPC : 0) | (rem_stats[1] < pending[first].sm_util ? LIM_SM : 0) | (rem_stats[2] < pending[first].l2 ? LIM_L2 : 0);
        plan.first_rejected_mask = plan.blocked_mask;                        // the batching log names the limits
        return plan;
    }

    plan.pick.push_back(first); // always release the oldest kernel
    rem_stats[0] -= pending[first].ipc;
    rem_stats[1] -= pending[first].sm_util;
    rem_stats[2] -= pending[first].l2;

    // don't exceed max batch size
    if (curr.size() + 1 == max_batch) {
        if (curr.empty()) plan.cause = BATCHING_OFF;     // FIX: that is SCHED_BATCH_MAX=1
        return plan;
    }

    bool other_waiting = false, other_in_table = false;
    for (size_t i = 0; i < pending.size(); i++) {
        if (i == first || pending[i].pid == pending[first].pid || process_running(pending[i].pid)) continue; // at most 1 from same proc (FIX: and none from a process that has one running)
        other_waiting = true;
        if (!pending[i].in_table) continue; // skip no profiling stats
        other_in_table = true;
        
        if (rem_stats[0] >= pending[i].ipc && \
            rem_stats[1] >= pending[i].sm_util && \
            rem_stats[2] >= pending[i].l2) {

            plan.pick.push_back(i); // picks at most 2 kernels for batch
            break;
        }
    }

    // FIX: why a kernel released onto an idle GPU ran alone (one released into running work counts as batched: its cause stays BATCHED).
    // This replaces "pending.size() == 1 -> NOT_ENOUGH_IN_QUEUE" and "pick.size() == 1 -> LIMIT_VIOLATED", the second of which overwrote the first.
    if (plan.pick.size() == 1 && curr.empty())
        plan.cause = !other_waiting ? NOT_ENOUGH_IN_QUEUE : !other_in_table ? PARTNERS_NOT_FOUND : LIMIT_VIOLATED;
    return plan;
}

/* ======================================================================
 * PREFETCH SCHEDULING
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
 *   do_prefetch, inflight_slot
 *   reason                            why the policy decided as it did
 *   kernel_base, grid, block, ipc     the kernel's short name and launch dimensions as the agent reported them (the table key is
 *                                     the name and the block; the grid is logged for information) and its "IPC (active)" from
 *                                     its model's table; ipc is blank when the kernel is not in it
 *   model                             the model tag the kernel's process reported (SCHED_MODEL): which table was searched
 *   sm_util, l2_throughput            the kernel's "SM Util %" and "L2 Throughput %" from that table (blank if not in it): with
 *                                     ipc, the numbers whose sums over a batch the policy limits
 * ====================================================================== */



static FILE* g_log = nullptr;

// SCHED_LOG: how much is logged per kernel. full (the default) and decisions: the decision log and the batching log; off: neither (the batching
// summary and the unmatched-kernel list are still written: one counter bump per decision, rewritten every 0.5 s). The agents and sched_hooks.py
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
    std::fprintf(stderr, "[daemon] log level (SCHED_LOG): %s\n", g_log_level == LOG_OFF ? "off: no decision log, no batching log (the summary and unmatched-kernel files are still written)"
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
    std::snprintf(buf, sizeof buf, "\t%d\t%d\t", m.do_prefetch, m.inflight);
    out += buf;
    out += r.reason;
    out += '\t'; out += m.base; out += '\t'; out += m.grid; out += '\t'; out += m.block; out += '\t';
    if (!std::isnan(m.ipc)) { std::snprintf(buf, sizeof buf, "%.4f", m.ipc); out += buf; }
    out += '\t'; out += m.model;
    out += '\t'; if (!std::isnan(m.sm_util)) { std::snprintf(buf, sizeof buf, "%.4f", m.sm_util); out += buf; }
    out += '\t'; if (!std::isnan(m.l2)) { std::snprintf(buf, sizeof buf, "%.4f", m.l2); out += buf; }
    out += '\t'; if (!std::isnan(m.dur_us)) { std::snprintf(buf, sizeof buf, "%.3f", m.dur_us); out += buf; }                       // est_dur_us
    std::snprintf(buf, sizeof buf, "\t%d\t%zu\t", m.pf_planned ? 1 : 0, m.pf_bytes); out += buf;                                    // pf_planned, pf_pred_bytes
    if (m.pf_bytes) { std::snprintf(buf, sizeof buf, "%.3f", static_cast<double>(prefetch_ns(m.pf_bytes)) / 1000.0); out += buf; }  // pf_est_us
    out += '\t'; if (m.target_end_ns) { std::snprintf(buf, sizeof buf, "%llu", (unsigned long long)m.target_end_ns); out += buf; }  // pf_target_end_ns
    out += '\t'; if (m.est_end_ns) { std::snprintf(buf, sizeof buf, "%llu", (unsigned long long)m.est_end_ns); out += buf; }        // est_end_ns
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
               "request_ns\tqueue_wait_us\tend_ns\tn_pending\tpending_pids\tdo_prefetch\tinflight_slot\treason\t"
               "kernel_base\tgrid\tblock\tipc\tmodel\tsm_util\tl2_throughput\test_dur_us\tpf_planned\tpf_pred_bytes\tpf_est_us\tpf_target_end_ns\test_end_ns\n", g_log);
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

static std::string csv_field(const std::string& s) {
    if (s.find_first_of(",\"\r\n") == std::string::npos) return s;
    std::string q = "\"";
    for (char c : s) { if (c == '"') q += '"'; q += c; }
    return q + "\"";
}

static std::string g_unmatched_path;
static uint64_t g_unmatched_written_ns = 0, g_unmatched_launches = 0;
static size_t g_unmatched_kernels = 0;

// Every kernel of the qwen and whisper processes that is not in its model's table, most launched first, in the table's own layout (Kernel, Block,
// Launches) behind a Model column, plus the grid and symbol of its first launch, so it can be profiled and merged into that model's table. The file is written through
// a temporary file and renamed, so a reader (or a daemon killed in the middle) never sees half of it.
static bool write_unmatched_csv() {
    struct Row { const std::string* model; const KernelKey* key; const Unmatched* u; };
    std::vector<Row> v;
    uint64_t total = 0;
    for (const auto& [model, kernels] : g_unmatched)
        for (const auto& [key, u] : kernels) { v.push_back({&model, &key, &u}); total += u.launches; }
    std::stable_sort(v.begin(), v.end(), [](const Row& a, const Row& b) { return a.u->launches > b.u->launches; });
    const std::string tmp = g_unmatched_path + ".tmp";
    FILE* f = std::fopen(tmp.c_str(), "w");
    if (!f) { std::fprintf(stderr, "[daemon] ERROR: cannot write %s for the unmatched kernels: %s (check SCHED_UNMATCHED_CSV / SCHED_LOG_DIR)\n", tmp.c_str(), std::strerror(errno)); return false; }
    std::fputs("Model,Kernel,Block,Launches,Example grid,Example symbol (cut to 127 characters)\n", f);
    for (const Row& r : v)
        std::fprintf(f, "%s,%s,%s,%llu,%s,%s\n", csv_field(*r.model).c_str(), csv_field(r.key->name).c_str(), csv_field(dims_str(r.key->block)).c_str(),
                     (unsigned long long)r.u->launches, csv_field(r.u->example_grid).c_str(), csv_field(r.u->example_symbol).c_str());
    if (std::fclose(f) != 0 || std::rename(tmp.c_str(), g_unmatched_path.c_str()) != 0) {
        std::fprintf(stderr, "[daemon] ERROR: cannot write %s: %s\n", g_unmatched_path.c_str(), std::strerror(errno));
        return false;
    }
    g_unmatched_dirty = false;
    g_unmatched_written_ns = mono_ns();
    g_unmatched_kernels = v.size();
    g_unmatched_launches = total;
    return true;
}

/* ======================================================================
 * BATCHING LOG -- why each decision did or did not batch.
 *
 * scheduler_<pid>_batching.tsv has one row per decision (every batch that was formed):
 *   batch_id, decision_ns, batch_size   as in the decision log
 *   outcome, cause                      batched / not_batched, and for not_batched why (the causes are CAUSE_NAME above):
 *                                         batching_disabled    SCHED_BATCH_MAX=1
 *                                         first_not_in_table   the first kernel in the queue is from a qwen/whisper process and is not in its table
 *                                         first_other_process  the first kernel's process is neither qwen nor whisper
 *                                         not_enough_in_queue  no kernel of another process was waiting (the queue held only the first
 *                                                              kernel, or only kernels of its process)
 *                                         partners_not_found   kernels of other processes were waiting, but none is in its table
 *                                         limit_violated       a kernel that could be judged was refused by IPC, SM util or L2
 *                                         inflight_table_full  the batch was cut short because the inflight table was full
 *   first_refused_limits                the limits the first refused partner failed (ipc, sm, l2, ipc+sm, ...), "-" if none was refused
 *   n_pending                           kernels waiting at the decision
 *   others_waiting                      other processes with a kernel waiting: the possible partners
 *   partners_considered, partners_not_found, partners_refused
 *                                       partners the policy looked at, how many of those were not in their table, how many were refused
 *   refused_ipc, refused_sm, refused_l2 refused partners per limit that failed (a partner can fail several)
 *   head_pid, head_model, head_kernel_base, head_found
 *                                       the first kernel in the queue, and whether it was in its model's table
 * scheduler_<pid>_batching_summary.csv holds the counts (group, category, count, percent_of_group), rewritten as the run goes
 * and at exit; the same counts can be had by grouping the rows of the .tsv.
 * ====================================================================== */

struct BatchingStats {
    uint64_t decisions = 0, cause[N_CAUSES] = {0}, first_limits[8] = {0};   // first_limits: limit_violated decisions by the limits of the first refused partner
    uint64_t considered = 0, not_found = 0, joined = 0, rejected = 0, rej_ipc = 0, rej_sm = 0, rej_l2 = 0;   // partner evaluations
    uint64_t pf_planned = 0;                        // decisions that released a kernel early so that its prefetch fills the wait (see PREFETCH SCHEDULING)
};
static BatchingStats g_bs;
static FILE* g_blog = nullptr;
static std::string g_summary_path;
static bool g_summary_dirty = false;
static uint64_t g_summary_written_ns = 0;

static void add_batching_row(std::string& out, uint64_t batch_id, uint64_t decision_ns, int batch_size, BatchCause cause, const SelectPlan& plan,
                             int n_pending, const Candidate& head, const std::string& head_model, const std::string& head_base, bool pf_planned) {
    char buf[512];
    std::snprintf(buf, sizeof buf, "%llu\t%llu\t%d\t%s\t%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t",
                  (unsigned long long)batch_id, (unsigned long long)decision_ns, batch_size, cause == BATCHED ? "batched" : "not_batched",
                  cause == BATCHED ? "-" : CAUSE_NAME[cause], limits_str(plan.first_rejected_mask).c_str(), n_pending, plan.others_waiting,
                  plan.considered, plan.not_found, plan.rejected, plan.rej_ipc, plan.rej_sm, plan.rej_l2, static_cast<int>(head.pid));
    out += buf;
    out += head_model; out += '\t'; out += head_base; out += '\t'; out += head.in_table ? '1' : '0'; out += pf_planned ? "\t1\n" : "\t0\n";
}

static void account_decision(BatchCause cause, const SelectPlan& plan) {
    g_bs.decisions++;
    g_bs.cause[cause]++;
    if (cause == LIMIT_VIOLATED) g_bs.first_limits[plan.first_rejected_mask & 7]++;
    g_bs.considered += plan.considered; g_bs.not_found += plan.not_found; g_bs.joined += plan.joined; g_bs.rejected += plan.rejected;
    g_bs.rej_ipc += plan.rej_ipc; g_bs.rej_sm += plan.rej_sm; g_bs.rej_l2 += plan.rej_l2;
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
    for (int m = 1; m < 8; m++) row("limit_violated_by_the_first_refused_partner", limits_str(m), g_bs.first_limits[m], g_bs.cause[LIMIT_VIOLATED]);
    row("refused_partners_by_limit", "ipc", g_bs.rej_ipc, g_bs.rejected);
    row("refused_partners_by_limit", "sm", g_bs.rej_sm, g_bs.rejected);
    row("refused_partners_by_limit", "l2", g_bs.rej_l2, g_bs.rejected);
    row("partners_looked_at", "joined", g_bs.joined, g_bs.considered);
    row("partners_looked_at", "refused", g_bs.rejected, g_bs.considered);
    row("partners_looked_at", "not_in_table", g_bs.not_found, g_bs.considered);
    row("mode", g_eff_on ? "pair_eff_policy" : g_pairlist_on ? "pair_list_policy" : (g_constraints ? "constraint_checking_on" : "constraint_checking_off"), g_bs.decisions, g_bs.decisions);     // which policy produced this run
    row("prefetch", "planned_releases", g_bs.pf_planned, g_bs.decisions);
    row("prefetch", "declined_remaining_time_too_short", g_pf_declined, g_bs.decisions);
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
    if (g_blog) std::fputs("batch_id\tdecision_ns\tbatch_size\toutcome\tcause\tfirst_refused_limits\tn_pending\tothers_waiting\tpartners_considered\tpartners_not_found\t"
               "partners_refused\trefused_ipc\trefused_sm\trefused_l2\thead_pid\thead_model\thead_kernel_base\thead_found\tpf_planned\n", g_blog);
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
    if (g_unmatched_dirty && now - g_unmatched_written_ns >= 500000000ull) write_unmatched_csv();
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

    int max_batch = 2;
    if (const char* e = std::getenv("SCHED_BATCH_MAX")) max_batch = std::max(1, std::min(MAX_INFLIGHT, std::atoi(e)));
    if (const char* e = std::getenv("SCHED_CONSTRAINTS"); e && *e) {      // on (default) | off: off is the baseline that batches up to SCHED_BATCH_MAX with no checks
        if (!strcasecmp(e, "off") || !strcmp(e, "0") || !strcasecmp(e, "false") || !strcasecmp(e, "no") || !strcasecmp(e, "none")) g_constraints = false;
        else if (strcasecmp(e, "on") && strcmp(e, "1") && strcasecmp(e, "true") && strcasecmp(e, "yes"))
            std::fprintf(stderr, "[daemon] WARNING: SCHED_CONSTRAINTS=%s is not on or off: using on\n", e);
    }
    // One CSV per model (Kernel, Block, Launches, ..., IPC (active), SM Util %, L2 Throughput %, ...). Give at least one; a model without one has no
    // table, so all of its kernels count as not found.
    const char* qwen_path = std::getenv("SCHED_QWEN_CSV");
    const char* whisper_path = std::getenv("SCHED_WHISPER_CSV");
    if (const char* e = std::getenv("SCHED_PAIR_LIST"); e && *e) {      // the whitelist policy: see policy_select_pairlist
        size_t np = 0;
        if (!load_pair_list(e, &np)) { std::fprintf(stderr, "[daemon] ERROR: cannot read SCHED_PAIR_LIST=%s\n", e); return 1; }
        g_pairlist_on = true; std::fprintf(stderr, "[daemon] SCHED_PAIR_LIST=%s: %zu listed pairs\n", e, np);
    }
    if (const char* e = std::getenv("SCHED_PAIR_EFF_FILE"); e && *e) {      // the efficiency policy: see the block before pair_listed
        if (const char* v = std::getenv("SCHED_PAIR_MIN_EFF")) g_eff_min = std::atof(v);
        if (const char* v = std::getenv("SCHED_PAIR_SHORT_US")) g_short_us = std::atof(v);
        if (const char* v = std::getenv("SCHED_PAIR_MIN_GANGS")) g_eff_min_gangs = std::max(1, std::atoi(v));
        size_t nt = 0, ns = 0, ng = 0, nk = 0, nshort = 0; std::string err; const char* us = std::getenv("SCHED_KERNEL_US_FILE");
        if (std::getenv("SCHED_PAIR_LIST")) std::fprintf(stderr, "[daemon] WARNING: SCHED_PAIR_LIST is ignored: SCHED_PAIR_EFF_FILE is set\n");
        if (!load_pair_eff(e, &nt, &ns, &ng, &err)) { std::fprintf(stderr, "[daemon] ERROR: SCHED_PAIR_EFF_FILE=%s: %s\n", e, err.c_str()); return 1; }
        if (us && *us) { if (!load_kernel_us(us, &nk, &nshort, &err)) { std::fprintf(stderr, "[daemon] ERROR: SCHED_KERNEL_US_FILE=%s: %s\n", us, err.c_str()); return 1; } }
        else std::fprintf(stderr, "[daemon] WARNING: no SCHED_KERNEL_US_FILE: no kernel counts as short, so the short-kernel rule does nothing\n");
        g_pairlist_on = true; g_eff_on = true;
        std::fprintf(stderr, "[daemon] SCHED_PAIR_EFF_FILE=%s: %zu type-pair rows, %zu shape-pair rows, %zu with efficiency >= %g (a shape row needs >= %d gangs); SCHED_KERNEL_US_FILE: %zu rows, %zu under %g us\n", e, nt, ns, ng, g_eff_min, g_eff_min_gangs, nk, nshort, g_short_us);
    }
    if (g_constraints && !g_pairlist_on && (!qwen_path || !*qwen_path) && (!whisper_path || !*whisper_path)) {     // without constraints the tables are not used: none is needed
        std::fprintf(stderr, "[daemon] ERROR: set SCHED_QWEN_CSV and/or SCHED_WHISPER_CSV to the CSV of profiled kernels of that model\n");
        return 1;
    }
    if (qwen_path && *qwen_path) { if (!load_kernel_table(qwen_path, "Qwen", g_qwen_table)) return 1; g_have_qwen = true; }
    else if (g_constraints && !g_pairlist_on) std::fprintf(stderr, "[daemon] no SCHED_QWEN_CSV: kernels of a process with SCHED_MODEL=qwen all count as not found\n");
    if (whisper_path && *whisper_path) { if (!load_kernel_table(whisper_path, "Whisper", g_whisper_table)) return 1; g_have_whisper = true; }
    else if (g_constraints && !g_pairlist_on) std::fprintf(stderr, "[daemon] no SCHED_WHISPER_CSV: kernels of a process with SCHED_MODEL=whisper all count as not found\n");
    std::set<std::string> warned_models;              // model tags that have no table, each reported once
    load_prefetch_config();
    if (g_kernels_with_wave && !wave_model_configured())
        std::fprintf(stderr, "[daemon] WARNING: a kernel table has a Duration/Wave column but nothing says how many blocks the GPU runs at once (no Blocks/Wave column, no SCHED_BLOCKS_PER_WAVE, no SCHED_GPU_SMS), so the per-wave values are NOT used\n");
    if (g_pf.plan && g_kernels_with_duration + (wave_model_configured() ? g_kernels_with_wave : 0) == 0) {
        std::fprintf(stderr, "[daemon] prefetch scheduling is OFF: no kernel table has a usable duration (a Duration or Duration / grid volume column, or a Duration/Wave column together with a Blocks/Wave column, SCHED_BLOCKS_PER_WAVE or SCHED_GPU_SMS): it is needed to estimate when a running kernel ends\n");
        g_pf.plan = false;
    } else if (g_pf.plan) {
        if (g_kernels_with_wave && wave_model_configured()) {
            char cap[400];
            if (g_kernels_with_capacity) std::snprintf(cap, sizeof cap, "the table's Blocks/Wave column (%zu kernels)%s", g_kernels_with_capacity,
                                                      g_pf.blocks_per_wave > 0 ? "; any other kernel: SCHED_BLOCKS_PER_WAVE" : g_pf.sms > 0 ? "; any other kernel: APPROXIMATELY SCHED_GPU_SMS x blocks per SM from the block size and achieved occupancy" : "; a kernel without one has no per-wave estimate");
            else if (g_pf.blocks_per_wave > 0) std::snprintf(cap, sizeof cap, "SCHED_BLOCKS_PER_WAVE = %.0f for every kernel", g_pf.blocks_per_wave);
            else std::snprintf(cap, sizeof cap, "APPROXIMATELY %d SMs x blocks per SM derived from the block size and the profile's achieved occupancy (%.0f threads per SM); add a Blocks/Wave column for exact values", g_pf.sms, g_pf.max_threads_per_sm);
            std::fprintf(stderr, "[daemon] duration estimate: duration per wave x max(1, ceil(grid blocks / blocks per wave)); blocks per wave: %s; a kernel with no per-wave value uses its per-block or plain Duration\n", cap);
        } else if (g_kernels_with_ratio) std::fprintf(stderr, "[daemon] duration estimate: duration per block x the launch's grid volume x*y*z (at least %.0f); a kernel with no per-block value uses its plain Duration\n", g_pf.min_grid_volume);
        else std::fprintf(stderr, "[daemon] duration estimate: the plain Duration column as it is, whatever grid the launch uses (no per-block column, so nothing is scaled)\n");
        std::fprintf(stderr, "[daemon] prefetch scheduling ON: a kernel kept out only by %s is released early, with a prefetch of at most %.1f MiB (%.2f us at %.1f GB/s%s), "
                             "duration scale %.2f, plan bias %.0f us, a prefetch must fit in the blocker's remaining time %s (margin %.0f us), polling the last %.0f us. The agents must run with the same SCHED_PREFETCH_MAX_MB (their own default is 64) and "
                             "SCHED_PREFETCH_MIN_KB; this daemon assumes %.1f MiB, %.0f KB, agent prefetch %s\n",
                     limits_str(g_pf.limits).c_str(), static_cast<double>(g_pf.cap_bytes) / (1024.0 * 1024.0), static_cast<double>(prefetch_ns(g_pf.cap_bytes)) / 1000.0, g_pf.bw_gbps,
                     g_pf.fixed_us > 0 ? " + fixed cost" : "", g_pf.dur_scale, g_pf.bias_us, g_pf.require_cover ? "(required)" : "(NOT required)", g_pf.overhead_us, g_pf.spin_us, static_cast<double>(g_pf.cap_bytes) / (1024.0 * 1024.0),
                     static_cast<double>(g_pf.min_bytes) / 1024.0, g_pf.agent_prefetch ? "on" : "OFF");
    }

    {   // where the unmatched kernels go: created now (header only), so it exists from the start and a bad path stops the daemon now
        const char* env = std::getenv("SCHED_UNMATCHED_CSV");
        const char* dir = std::getenv("SCHED_LOG_DIR");
        g_unmatched_path = env ? env : std::string(dir ? dir : ".") + "/unmatched_kernels_" + std::to_string(getpid()) + ".csv";
    }
    if (!write_unmatched_csv()) return 1;
    init_shared_state();
    parse_log_level();
    if (!open_log()) return 1;
    if (!open_batching_log()) return 1;
    {
        char abs[PATH_MAX];
        std::fprintf(stderr, "[daemon] unmatched kernels (not found in their model's table), rewritten as they appear and at exit: %s\n",
                     realpath(g_unmatched_path.c_str(), abs) ? abs : g_unmatched_path.c_str());
        std::fprintf(stderr, "[daemon] pid %d. Stop it with  kill -INT %d  (Ctrl+C only reaches it if it runs in the foreground; SIGTERM and SIGHUP also write the logs; kill -9 loses only what is not yet written)\n",
                     static_cast<int>(getpid()), static_cast<int>(getpid()));
    }
    if (g_eff_on)
        std::fprintf(stderr, "[daemon] pair-efficiency policy: NO batching by default (one kernel at a time, oldest first); a kernel joins a running one only if the pair was measured to gain (efficiency >= %g) or either kernel is under %g us. Tables, limits and prefetch are not used (do_prefetch=0).\n", g_eff_min, g_short_us);
    else if (g_pairlist_on)
        std::fprintf(stderr, "[daemon] pair-list policy: NO batching by default (one kernel at a time, oldest first); a second kernel joins a running one only if the two are a listed pair (%zu pairs). Tables, limits and prefetch are not used (do_prefetch=0).\n", g_pairlist.size());
    else if (!g_constraints)
        std::fprintf(stderr, "[daemon] SCHED_CONSTRAINTS=off: NO constraint checking. Whenever a slot is free the oldest waiting kernel whose process has none running is released, then the oldest of another such process, up to %d at once, one per process -- in its table or not, whatever the IPC / SM / L2 sums. Nothing is held for a limit, so no prefetch is planned or requested (do_prefetch=0). The kernel tables are not used.\n", max_batch);
    else std::fprintf(stderr, "[daemon] continuous batching: up to %d kernels run at once, one per process; the oldest waiting kernel whose process has none running joins when it is in its model's table and the sums over what runs stay within IPC %.1f, SM utilization %.0f%% and L2 throughput %.0f%%. A kernel that is not in its table runs by itself, and while it is the oldest nothing new is admitted\n",
                 max_batch, IPC_BATCH_LIMIT, SM_UTIL_BATCH_LIMIT, L2_BATCH_LIMIT);

    int arrived_kernel_count = 0;
    uint64_t next_batch_id = 1, released_in_table = 0, released_not_in_table = 0;
    std::vector<Member> running;                      // kernels released and not yet seen to finish
    std::vector<KernelKey> keys(MAX_PENDING);         // slot -> table key of the kernel pending there (only pending slots are refreshed)

    std::fprintf(stderr, "[daemon] waiting for agent processes to attach...\n");

    uint64_t plan_wake_ns = 0;                        // a prefetch-planned release is due at this time (0: none); set by the decision below
    while (!g_stop) {
        const uint64_t spin_ns = static_cast<uint64_t>(g_pf.spin_us * 1000.0);
        const uint64_t now0 = mono_ns();
        if (plan_wake_ns && plan_wake_ns <= now0 + spin_ns) {
            // The planned time is (almost) here: poll instead of sleeping, so the release is not late by a wake-up. An event (a kernel finished, a new
            // request) ends the polling early; the decision below is made either way and plans again if it still has to.
            while (mono_ns() < plan_wake_ns && !g_stop) if (sem_trywait(&state->new_request_sem) == 0) break;
        } else {
            uint64_t wait_ns = 250000000ull;           // 250 ms of nothing, or a signal (the loop condition ends it)
            if (plan_wake_ns) wait_ns = std::min<uint64_t>(wait_ns, plan_wake_ns - spin_ns - now0);
            timespec until;
            clock_gettime(CLOCK_REALTIME, &until);
            until.tv_nsec += static_cast<long>(wait_ns % 1000000000ull);
            until.tv_sec += static_cast<time_t>(wait_ns / 1000000000ull);
            if (until.tv_nsec >= 1000000000L) { until.tv_sec++; until.tv_nsec -= 1000000000L; }
            if (sem_timedwait(&state->new_request_sem, &until) != 0) { housekeeping(); continue; }
        }
        plan_wake_ns = 0;

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
                const KernelTable* table = table_for(model);                     // ONLY this model's table is searched
                if (g_constraints && !g_pairlist_on && !table && warned_models.insert(model).second)
                    std::fprintf(stderr, "[daemon] WARNING: pid %d runs model \"%s\", which has no kernel table (SCHED_MODEL must be qwen or whisper, with its SCHED_*_CSV set): its kernels all count as not found\n",
                                    (int)state->slots[i].pid, model.c_str());
                const auto hit = table ? table->find(keys[i]) : KernelTable::const_iterator();
                const bool found = table && hit != table->end();
                if (first_sighting && !found) {                                  // once per launch
                    if (is_known_model(model)) note_unmatched(model, keys[i], state->slots[i].kernel_name, dims_str(slot_grid(state->slots[i])));
                    else g_other_launches++;                                     // another process: counted, not listed
                }
                pending.push_back({i, state->slots[i].pid, state->slots[i].request_ns,
                                    found, found ? hit->second.avg_ipc : static_cast<double>(NAN),
                                    found ? hit->second.sm_util : static_cast<double>(NAN), found ? hit->second.l2_tput : static_cast<double>(NAN),
                                    is_known_model(model), found ? estimate_duration_us(hit->second, keys[i].block, slot_grid(state->slots[i])) : static_cast<double>(NAN),
                                    (g_pairlist_on && is_known_model(model)) ? (g_eff_on ? eff_sig(model, keys[i].name, keys[i].block, slot_grid(state->slots[i])) : pair_sig(model, keys[i].name, slot_grid(state->slots[i]))) : std::string()});
            }
        }

        if (!pending.empty()) {
            SelectPlan plan = policy_select(running, pending, max_batch);
            PfDecision pfd;
            bool pf_planned = false;
            if (plan.pick.empty() && plan.blocked >= 0) {                        // kept out of the running work only by the limits: can a prefetch fill the wait?
                const Candidate& b = pending[plan.blocked];
                pfd = plan_prefetch(running, b, plan.blocked_mask, mono_ns(), state->slots[b.slot].layer_tag, state->slots[b.slot].kernel_id);
                if (pfd.release_now) { plan.pick.push_back(plan.blocked); pf_planned = true; }
                else plan_wake_ns = pfd.wake_ns;
            }
            
            const Candidate& head = pending[plan.pick.empty() ? 0 : plan.pick[0]];   // FIX: the first kernel released (a kernel whose process is running waits, so not always pending[0])
            const std::string head_model = slot_model(state->slots[head.slot]);
            const std::string head_base = keys[head.slot].name;
            
            auto rel = std::make_shared<Release>();
            rel->n_pending = static_cast<int>(pending.size());
            for (size_t i = 0; i < pending.size(); i++) {
                if (i) rel->pending_pids += ',';
                rel->pending_pids += std::to_string(static_cast<int>(pending[i].pid));
            }
            rel->reason = plan.reason;

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
                m.ipc = pending[idx].ipc;
                m.sm_util = pending[idx].sm_util;
                m.l2 = pending[idx].l2;
                m.request_ns = state->slots[s].request_ns;
                m.do_prefetch = pf_planned ? 1 : ((g_constraints && !g_pairlist_on) ? policy_should_prefetch(state->slots[s].kernel_name, state->slots[s].layer_tag) : 0);   // no constraints / pair list: no prefetch
                m.dur_us = pending[idx].dur_us;
                m.pf_planned = pf_planned;
                m.target_end_ns = pf_planned ? pfd.target_end_ns : 0;
                if (m.do_prefetch == 1) m.pf_bytes = predicted_bytes(m.pid, state->slots[s].layer_tag);      // before note_released: it updates the entry
                note_released(m.pid, state->slots[s].layer_tag, m.pf_bytes > 0);
                (pending[idx].in_table ? released_in_table : released_not_in_table)++;
                released.push_back(m);
            }

            if (!released.empty()) {
                rel->id = next_batch_id++;
                rel->size = static_cast<int>(released.size());
                rel->decision_ns = mono_ns();
                for (Member& m : released) {
                    m.rel = rel;
                    m.est_end_ns = std::isnan(m.dur_us) ? 0 : rel->decision_ns + prefetch_ns(m.pf_bytes) + static_cast<uint64_t>(m.dur_us * g_pf.dur_scale * 1000.0);
                    state->slots[m.slot].inflight_slot = m.inflight;
                    state->slots[m.slot].do_prefetch = m.do_prefetch;
                    state->slots[m.slot].batch_id = rel->id;
                    state->slots[m.slot].batch_size = rel->size;
                    state->slots[m.slot].batch_pos = m.pos;
                    state->slots[m.slot].ready = 1;
                }
                g_size_hist[rel->size]++;
                BatchCause cause = plan.cause;                                   // why this batch is what it is
                if (pf_planned) g_bs.pf_planned++;                               // released early for its prefetch: it did NOT batch, the limits kept it out (cause stays limit_violated)
                else if (released.size() >= 2 || !running.empty()) cause = BATCHED;   // FIX: a kernel released into running work is batched with it
                else if (cause == BATCHED) cause = INFLIGHT_FULL;                // the policy wanted more, the inflight table was full
                if (g_blog) add_batching_row(brows, rel->id, rel->decision_ns, rel->size, cause, plan, rel->n_pending, head, head_model, head_base, pf_planned);
                account_decision(cause, plan);
            }
        }

        pthread_mutex_unlock(&state->mutex);

        // 3. Wake the released kernels as close together as possible; the time-critical part comes before any logging.
        for (Member& m : released) {
            m.post_ns = mono_ns();                      // just BEFORE the wake-up: decision_ns <= post_ns <= the agent's start_ns
            sem_post(&state->slots[m.slot].wake_sem);
        }
        for (const Member& m : released)
            std::fprintf(stderr, "[daemon] release pid=%d model=%s kernel=%s grid=%s block=%s ipc=%.3f layer=%s batch=%llu size=%d prefetch=%d%s\n",
                         (int)m.pid, m.model.c_str(), m.base.c_str(), m.grid.c_str(), m.block.c_str(), m.ipc, m.tag.c_str(),
                         (unsigned long long)m.rel->id, m.rel->size, m.do_prefetch, m.pf_planned ? " (planned early)" : "");
        for (Member& m : released) running.push_back(std::move(m));
        g_largest_batch = std::max(g_largest_batch, static_cast<int>(running.size()));
        write_log(rows);
        write_batching_rows(brows);
        housekeeping();
    }

    // Shutdown: log the kernels of a batch still running (end times only where they had finished), flush, and write the unmatched kernels.
    std::string rows;
    for (const Member& m : running) if (g_log) append_row(rows, m);
    write_log(rows);
    if (g_log) std::fclose(g_log);
    std::fprintf(stderr, "[daemon] decisions by number of kernels released together:");
    for (const auto& kv : g_size_hist) std::fprintf(stderr, " %d x%llu", kv.first, (unsigned long long)kv.second);
    std::fprintf(stderr, "; largest batch: %d; launches released: %llu in their table, %llu not in it (each alone)\n",
                 g_largest_batch, (unsigned long long)released_in_table, (unsigned long long)released_not_in_table);
    write_batching_summary();
    if (g_blog) std::fclose(g_blog);
    {
        const uint64_t nb = g_bs.decisions - g_bs.cause[BATCHED];
        std::fprintf(stderr, "[daemon] %llu decisions: %llu batched, %llu not batched:", (unsigned long long)g_bs.decisions, (unsigned long long)g_bs.cause[BATCHED], (unsigned long long)nb);
        for (int c = 1; c < N_CAUSES; c++) std::fprintf(stderr, " %s %llu%s", CAUSE_NAME[c], (unsigned long long)g_bs.cause[c], c + 1 < N_CAUSES ? "," : "");
        std::fprintf(stderr, "  (%s)\n", g_summary_path.c_str());
    }
    write_unmatched_csv();
    std::fprintf(stderr, "[daemon] %zu distinct kernels (%llu launches) of the qwen and whisper processes were not in their model's table: %s\n", g_unmatched_kernels, (unsigned long long)g_unmatched_launches, g_unmatched_path.c_str());
    if (g_other_launches)
        std::fprintf(stderr, "[daemon] %llu launches from processes that are neither qwen nor whisper are not in that file (they ran alone; see the decision log, model column)\n", (unsigned long long)g_other_launches);
    return 0;
}