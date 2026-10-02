#include "shared_state.hpp"
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

static SchedulerSharedState* state = nullptr;

/* ======================================================================
 * POLICY HOOKS -- this is the single global scheduling process, so these
 * see every pending kernel from every attached agent, and every registered
 * weight region from every agent, at once. Replace with your actual ideas.
 * ====================================================================== */

static int policy_compute_priority(const char* kernel_name, const char* layer_tag) {
    (void)kernel_name;
    (void)layer_tag;
    return 0; // placeholder: FIFO across all attached processes
}

// Caller already holds state->mutex.
static bool find_region_locked(pid_t owner_pid, const char* tag) {
    for (int i = 0; i < state->region_count; i++) {
        if (state->regions[i].active &&
            state->regions[i].owner_pid == owner_pid &&
            std::strcmp(state->regions[i].tag, tag) == 0) {
            return true;
        }
    }
    return false;
}

static const char* policy_choose_prefetch_locked(PendingSlot& slot) {
    if (std::strncmp(slot.layer_tag, "layer_", 6) == 0) {
        int layer_idx = std::atoi(slot.layer_tag + 6);
        static char next_tag[NAME_LEN];
        std::snprintf(next_tag, sizeof(next_tag), "layer%d_qkv_proj", layer_idx + 1);
        if (find_region_locked(slot.pid, next_tag)) {
            return next_tag;
        }
    }
    return nullptr;
}

/* ======================================================================
 * INFLIGHT TRACKING -- per-kernel slots, not a single counter. Caller must
 * already hold state->mutex for both of these.
 * ====================================================================== */

static bool any_inflight_locked() {
    for (int i = 0; i < MAX_INFLIGHT; i++) {
        if (state->inflight[i].active.load() != 0) return true;
    }
    return false;
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
    init_shared_state();
    std::fprintf(stderr, "[daemon] waiting for agent processes to attach...\n");

    while (true) {
        sem_wait(&state->new_request_sem);

        pthread_mutex_lock(&state->mutex);

        int best = -1;
        for (int i = 0; i < MAX_PENDING; i++) {
            if (state->slots[i].active && !state->slots[i].ready) {
                state->slots[i].priority = policy_compute_priority(
                    state->slots[i].kernel_name, state->slots[i].layer_tag);
                if (best == -1 || state->slots[i].priority < state->slots[best].priority) {
                    best = i;
                }
            }
        }

        // IDLE GATE: only release if NOTHING is currently executing. Scans
        // the inflight[] array rather than checking a single counter --
        // this is what gives you visibility into exactly which kernel(s)
        // are running, not just a count.
        if (best != -1 && any_inflight_locked()) {
            best = -1; // GPU busy -- leave this candidate pending, try again later
        }

        if (best != -1) {
            int slot_idx = reserve_inflight_slot_locked(
                state->slots[best].pid, state->slots[best].kernel_name, state->slots[best].layer_tag);

            if (slot_idx == -1) {
                // Inflight table full -- defer rather than release untracked.
                best = -1;
            } else {
                state->slots[best].inflight_slot = slot_idx;

                const char* tag = policy_choose_prefetch_locked(state->slots[best]);
                if (tag) {
                    state->slots[best].has_prefetch = 1;
                    std::strncpy(state->slots[best].prefetch_tag, tag, NAME_LEN - 1);
                } else {
                    state->slots[best].has_prefetch = 0;
                }

                std::fprintf(stderr,
                    "[daemon] release pid=%d kernel=%s layer=%s prefetch=%s inflight_slot=%d\n",
                    state->slots[best].pid, state->slots[best].kernel_name,
                    state->slots[best].layer_tag,
                    state->slots[best].has_prefetch ? state->slots[best].prefetch_tag : "(none)",
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
