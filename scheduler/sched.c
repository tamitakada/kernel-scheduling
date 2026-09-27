#include "sched.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stdio.h>


int main() {
    int fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
    ftruncate(fd, sizeof(SchedulerSharedState));
    SchedulerSharedState* state = mmap(NULL, sizeof(SchedulerSharedState),
                                        PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

    // CRITICAL: pshared attrs so these primitives work across process boundaries
    pthread_mutexattr_t mattr;
    pthread_mutexattr_init(&mattr);
    pthread_mutexattr_setpshared(&mattr, PTHREAD_PROCESS_SHARED);
    pthread_mutex_init(&state->mutex, &mattr);

    sem_init(&state->new_request_sem, /*pshared=*/1, 0);
    for (int i = 0; i < MAX_PENDING; i++) {
        sem_init(&state->slots[i].wake_sem, /*pshared=*/1, 0);
        state->slots[i].active = 0;
    }

    fprintf(stderr, "[daemon] ready, waiting for agents...\n");
    
    // prioritize P1 for first 1K kernel calls, then interleave
    
    int n_executed = 0;
    pid_t priority_pid = 0;
    
    while (1) {
        sem_wait(&state->new_request_sem);  // wake whenever ANY process posts a request

        pthread_mutex_lock(&state->mutex);
        int best = -1;
        
        int n_pending = 0;
        for (int i = 0; i < MAX_PENDING; i++) {
            if (state->slots[i].active && !state->slots[i].ready) {
                n_pending++;
                
                if (priority_pid == 0 && n_executed < 1000)
                    priority_pid = state->slots[i].pid;
                
                if (n_executed < 1000 && priority_pid == state->slots[i].pid)
                    best = i;
                else if (n_executed >= 1000 && priority_pid != state->slots[i].pid)
                    best = i;
            }
        }
        
        printf("%d pending, %d PID priority, %d executed",
               n_pending, priority_pid, n_executed);
        
        if (best != -1) state->slots[best].ready = 1;
        pthread_mutex_unlock(&state->mutex);

        if (best != -1) {
            n_executed++;
            fprintf(stderr, "[daemon] releasing pid=%d kernel=%s\n",
                    state->slots[best].pid, state->slots[best].kernel_name);
            sem_post(&state->slots[best].wake_sem);
        }
    }
}
