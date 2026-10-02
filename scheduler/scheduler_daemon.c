#define _GNU_SOURCE
#include "shared_state.h"
#include "scheduler_logging.h"
#include <errno.h>
#include <signal.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>


// SIGINT handler override to allow logging
static volatile sig_atomic_t stop = 0;
static void on_sigint(int sig) { stop = 1; }

static SchedulerSharedState* state = NULL;

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

static void init_shared_state() {
    int fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
    if (fd < 0) { 
        perror("[daemon] shm_open"); 
        exit(1); 
    }

    if (ftruncate(fd, sizeof(SchedulerSharedState))) {
        perror("[daemon] ftruncate"); 
        exit(1);
    }

    state = (SchedulerSharedState*) mmap(
        NULL, 
        sizeof(SchedulerSharedState), 
        PROT_READ | PROT_WRITE, MAP_SHARED, 
        fd, 
        0
    );

    if (state == MAP_FAILED) { 
        perror("[daemon] mmap"); 
        exit(1); 
    }
    close(fd);

    pthread_mutexattr_t mattr;
    pthread_mutexattr_init(&mattr);
    pthread_mutexattr_setpshared(&mattr, PTHREAD_PROCESS_SHARED);
    pthread_mutex_init(&state->mutex, &mattr);

    sem_init(&state->new_request_sem, /*pshared=*/1, 0);
    for (int i = 0; i < MAX_PENDING; i++) {
        sem_init(&state->slots[i].wake_sem, /*pshared=*/1, 0);
        state->slots[i].active = 0;
        state->slots[i].log_id = 0;
    }
    state->region_count = 0;

    atomic_store(&state->gpu_inflight_count, 0);
    atomic_store(&state->initialized, 1);
    fprintf(stderr, "[daemon] shared memory ready, pid=%d\n", getpid());
}

int main() {
    struct sigaction sa = {0};
    sa.sa_handler = on_sigint;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, NULL);

    init_shared_state();
    fprintf(stderr, "[daemon] waiting for agent processes to attach...\n");

    while (!stop) {
        sem_wait(&state->new_request_sem);

        pthread_mutex_lock(&state->mutex);

        uint8_t n_pending = 0;
        for (int i = 0; i < MAX_PENDING; i++)
            if (state->slots[i].active && !state->slots[i].ready)
                n_pending++;

        int n_launch = 0;
        int to_launch[10] = {0};
        for (int i = 0; i < MAX_PENDING; i++) {
            if (state->slots[i].active && !state->slots[i].ready) {
                if (!state->slots[i].log_id) {
                    state->slots[i].log_id = log_kernel_arrival(
                        state->slots[i].kernel_name, n_pending);
                }

                state->slots[i].priority = policy_compute_priority(
                    state->slots[i].kernel_name, state->slots[i].layer_tag);
                
                if (n_launch < 2) { //|| state->slots[i].priority < state->slots[best].priority) {
                    to_launch[n_launch] = i; 
                    n_launch++;
                }
            }
        }

        if (n_launch && atomic_load(&state->gpu_inflight_count) != 0) {
            n_launch = 0; // GPU busy -- leave this candidate pending, try again later
        }

        if (n_launch) {
            atomic_fetch_add(&state->gpu_inflight_count, n_launch);

            /*const char* tag = policy_choose_prefetch_locked(&state->slots[best]);
            if (tag) {
                state->slots[best].has_prefetch = 1;
                strncpy(state->slots[best].prefetch_tag, tag, NAME_LEN - 1);
            } else {
                state->slots[best].has_prefetch = 0;
            }*/
            for (int i = 0; i < n_launch; i++) {
                // fprintf(stderr, "[daemon] release pid=%d kernel=%s layer=%s prefetch=%s\n",
                //     state->slots[i].pid, state->slots[i].kernel_name,
                //     state->slots[i].layer_tag,
                //     state->slots[i].has_prefetch ? state->slots[i].prefetch_tag : "(none)");
                state->slots[i].ready = 1;
                log_kernel_exec_start(state->slots[i].log_id, n_launch);
            }
        }

        pthread_mutex_unlock(&state->mutex);

        for (int i = 0; i < n_launch; i++)
            sem_post(&state->slots[i].wake_sem);
    }

    write_log();

    return 0;
}
