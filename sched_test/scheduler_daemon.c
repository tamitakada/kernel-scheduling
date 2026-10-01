// scheduler_daemon.c
//
// Standalone scheduling process. Owns the shared-memory segment, runs the
// arbitration loop that decides WHEN each pending kernel (from any attached
// agent process) is released, and WHAT (if anything) that agent should
// prefetch before letting it go.
//
// This process never touches CUDA. It only reads/writes plain host-side
// bookkeeping in shared memory -- the actual kernel launches (both the real
// kernel and any prefetch kernel) always execute inside the agent process
// that owns the relevant GPU memory, because a CUDA device pointer is only
// valid within the context that allocated it. See shared_state.h.
//
// BUILD (no CUDA libraries needed):
//   gcc -o scheduler_daemon scheduler_daemon.c -lpthread -lrt
//
// RUN (start this FIRST, before any vLLM process):
//   ./scheduler_daemon

#define _GNU_SOURCE
#include "shared_state.h"
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

static SchedulerSharedState* state = NULL;

/* ======================================================================
 * POLICY HOOKS -- this is the single global scheduling process, so these
 * two functions see every pending kernel from every attached agent, and
 * every registered weight region from every agent, at once. Replace with
 * your actual ideas.
 * ====================================================================== */

static int policy_compute_priority(const char* kernel_name, const char* layer_tag) {
    (void)kernel_name;
    (void)layer_tag;
    return 0; // placeholder: FIFO across all attached processes
}

// Caller already holds state->mutex.
static int find_region_locked(pid_t owner_pid, const char* tag) {
    for (int i = 0; i < state->region_count; i++) {
        if (state->regions[i].active &&
            state->regions[i].owner_pid == owner_pid &&
            strcmp(state->regions[i].tag, tag) == 0) {
            return 1;
        }
    }
    return 0;
}

// Decide whether the process about to be released should prefetch
// something of ITS OWN first. Example: prefetch "the next layer's
// qkv_proj" whenever releasing the current layer's kernel.
static const char* policy_choose_prefetch_locked(PendingSlot* slot) {
    if (strncmp(slot->layer_tag, "layer_", 6) == 0) {
        int layer_idx = atoi(slot->layer_tag + 6);
        static char next_tag[NAME_LEN];
        snprintf(next_tag, sizeof(next_tag), "layer%d_qkv_proj", layer_idx + 1);
        if (find_region_locked(slot->pid, next_tag)) {
            return next_tag;
        }
    }
    return NULL;
}

/* ======================================================================
 * SETUP + MAIN LOOP
 * ====================================================================== */

static void init_shared_state(void) {
    int fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
    if (fd < 0) { perror("[daemon] shm_open"); exit(1); }
    if (ftruncate(fd, sizeof(SchedulerSharedState)) != 0) {
        perror("[daemon] ftruncate"); exit(1);
    }

    state = (SchedulerSharedState*)mmap(NULL, sizeof(SchedulerSharedState),
                                        PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (state == MAP_FAILED) { perror("[daemon] mmap"); exit(1); }
    close(fd);

    pthread_mutexattr_t mattr;
    pthread_mutexattr_init(&mattr);
    pthread_mutexattr_setpshared(&mattr, PTHREAD_PROCESS_SHARED);
    pthread_mutex_init(&state->mutex, &mattr);

    sem_init(&state->new_request_sem, /*pshared=*/1, 0);
    for (int i = 0; i < MAX_PENDING; i++) {
        sem_init(&state->slots[i].wake_sem, /*pshared=*/1, 0);
        state->slots[i].active = 0;
    }
    state->region_count = 0;

    atomic_store(&state->initialized, 1);
    fprintf(stderr, "[daemon] shared memory ready, pid=%d\n", getpid());
}

int main(void) {
    init_shared_state();
    fprintf(stderr, "[daemon] waiting for agent processes to attach...\n");

    while (1) {
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

        // IDLE GATE: only release if nothing is currently executing. This
        // check-and-increment happens in the SAME locked section as the
        // release decision, so there's no window for a second slot to be
        // released before this one's increment is visible -- the agent
        // only decrements once the kernel it actually launches completes
        // (see the cuLaunchHostFunc callback in agent.c).
        if (best != -1 && atomic_load(&state->gpu_inflight_count) != 0) {
            best = -1; // GPU busy -- leave this candidate pending, try again later
        }

        if (best != -1) {
            atomic_fetch_add(&state->gpu_inflight_count, 1);

            const char* tag = policy_choose_prefetch_locked(&state->slots[best]);
            if (tag) {
                state->slots[best].has_prefetch = 1;
                strncpy(state->slots[best].prefetch_tag, tag, NAME_LEN - 1);
            } else {
                state->slots[best].has_prefetch = 0;
            }
            fprintf(stderr, "[daemon] release pid=%d kernel=%s layer=%s prefetch=%s\n",
                    state->slots[best].pid, state->slots[best].kernel_name,
                    state->slots[best].layer_tag,
                    state->slots[best].has_prefetch ? state->slots[best].prefetch_tag : "(none)");
            state->slots[best].ready = 1;
        }

        pthread_mutex_unlock(&state->mutex);

        if (best != -1) {
            sem_post(&state->slots[best].wake_sem);
        }
    }
    return 0;
}
