#include "shared_state.hpp"
#include "scheduler_logger.hpp"

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <map>
#include <vector>

// SIGINT handler override to allow logging
static std::atomic<bool> g_stop{false};
extern "C" void on_signal(int) { g_stop = true; }

static SchedulerSharedState* state = nullptr;

static int policy_compute_priority(const char* kernel_name, const char* layer_tag) {
    (void)kernel_name;
    (void)layer_tag;
    return 0; // placeholder: FIFO across all attached processes
}

/* ======================================================================
 * INFLIGHT TRACKING -- per-kernel slots, not a single counter. Caller must
 * already hold state->mutex for both of these.
 * ====================================================================== */

static bool any_inflight_locked(std::map<int,int>& kernel_id_to_slot_idx, SchedulerLogger& logger) {
    std::vector<int> finished_ids;
    bool any_active = false;
    for (const auto& [kid, sidx] : kernel_id_to_slot_idx) {
        if (state->inflight[sidx].active.load() == 0) {
            logger.log_kernel_exec_end(kid);
            finished_ids.push_back(kid);
        }
        else any_active = true;
    }

    for (const auto& kid : finished_ids)
        kernel_id_to_slot_idx.erase(kid);

    return any_active;
}

// Returns the reserved slot's index, or -1 if the inflight table is full.
static int reserve_inflight_slot_locked(pid_t pid, const char* kernel_name, const char* layer_tag) {
    for (int i = 0; i < MAX_INFLIGHT; i++) {
        if (state->inflight[i].active.load() == 0) {
            state->inflight[i].pid = pid;
            std::strncpy(state->inflight[i].kernel_name, kernel_name, NAME_LEN - 1);
            std::strncpy(state->inflight[i].layer_tag, layer_tag, NAME_LEN - 1);
            state->inflight[i].active.store(1);
            return i;
        }
    }
    return -1;
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
    }
    for (int i = 0; i < MAX_INFLIGHT; i++) {
        state->inflight[i].active.store(0);
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

    init_shared_state();

    SchedulerLogger logger;
    int arrived_kernel_count = 0;

    std::map<int, int> kernel_id_to_slot_idx;

    std::fprintf(stderr, "[daemon] waiting for agent processes to attach...\n");

    while (!g_stop) {
        sem_wait(&state->new_request_sem);

        pthread_mutex_lock(&state->mutex);

        int n_pending = 0;
        for (int i = 0; i < MAX_PENDING; i++)
            if (state->slots[i].active && !state->slots[i].ready)
                n_pending++;
        
        int best = -1;
        for (int i = 0; i < MAX_PENDING; i++) {
            if (state->slots[i].active && !state->slots[i].ready) {
                if (!state->slots[i].kernel_id) { // log arrival if first time
                    arrived_kernel_count++;
                    state->slots[i].kernel_id = arrived_kernel_count;
                    logger.log_kernel_arrival(state->slots[i].kernel_id, state->slots[i].kernel_name, n_pending);
                }

                state->slots[i].priority = policy_compute_priority(
                    state->slots[i].kernel_name, state->slots[i].layer_tag);
                if (best == -1 || state->slots[i].priority < state->slots[best].priority) {
                    best = i;
                }
            }
        }

        if (best != -1 && any_inflight_locked(kernel_id_to_slot_idx, logger)) {
            best = -1; // GPU busy -- leave this candidate pending, try again later
        }

        if (best != -1) {
            int slot_idx = reserve_inflight_slot_locked(
                state->slots[best].pid, state->slots[best].kernel_name, state->slots[best].layer_tag);

            if (slot_idx == -1) {
                // Inflight table full -- defer rather than release untracked.
                best = -1;
            } else {
                kernel_id_to_slot_idx[state->slots[best].kernel_id] = slot_idx;
                logger.log_kernel_exec_start(state->slots[best].kernel_id, 1);

                state->slots[best].inflight_slot = slot_idx;

                std::fprintf(stderr,
                    "[daemon] release pid=%d kernel=%s layer=%s inflight_slot=%d\n",
                    state->slots[best].pid, state->slots[best].kernel_name,
                    state->slots[best].layer_tag,
                    slot_idx);
                state->slots[best].ready = 1;
            }
        }

        pthread_mutex_unlock(&state->mutex);

        if (best != -1) {
            sem_post(&state->slots[best].wake_sem);
        }
    }
    return 0;
}