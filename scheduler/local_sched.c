#include "sched.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <cupti.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>


static SchedulerSharedState* state = NULL;

void request_launch_permission(const char* kernel_name, int priority) {
    pthread_mutex_lock(&state->mutex);
    int slot = -1;
    for (int i = 0; i < MAX_PENDING; i++) {
        if (!state->slots[i].active) { slot = i; break; }
    }
    state->slots[slot].pid = getpid();
    state->slots[slot].priority = priority;
    strncpy(state->slots[slot].kernel_name, kernel_name, sizeof(state->slots[slot].kernel_name) - 1);
    state->slots[slot].active = 1;
    state->slots[slot].ready = 0;
    pthread_mutex_unlock(&state->mutex);

    sem_post(&state->new_request_sem);        // wake the daemon
    sem_wait(&state->slots[slot].wake_sem);    // block until daemon releases THIS slot

    pthread_mutex_lock(&state->mutex);
    state->slots[slot].active = 0;             // free the slot
    pthread_mutex_unlock(&state->mutex);
}

static CUpti_SubscriberHandle subscriber;
static long launchCount = 0;

static void CUPTIAPI callback(void* userdata, CUpti_CallbackDomain domain,
                               CUpti_CallbackId cbid, const void* cbdata) {
    
    if (domain != CUPTI_CB_DOMAIN_DRIVER_API) return;
    const CUpti_CallbackData* data = (const CUpti_CallbackData*)cbdata;

    if (cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel ||
        cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx) {
        
        if (data->callbackSite == CUPTI_API_ENTER) {
            launchCount++;
            fprintf(stderr, "[sched] ENTER launch #%ld: %s (correlationId=%u)\n",
                    launchCount, data->functionName, data->correlationId);
            
            int priority = 1;
            request_launch_permission(data->functionName, priority);
        }
        else if (data->callbackSite == CUPTI_API_EXIT) {
            fprintf(stderr, "[sched] EXIT  launch: %s (correlationId=%u)\n",
                    data->functionName, data->correlationId);
        }
    }
}

__attribute__((constructor))
static void init_cupti(void) {
    int fd = shm_open(SHM_NAME, O_RDWR, 0666);
    if (fd < 0) {
        fprintf(stderr, "[agent] FATAL: could not open shared memory — is scheduler_daemon running?\n");
        return;
    }
    state = mmap(NULL, sizeof(SchedulerSharedState), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    
    CUptiResult res;

    res = cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)callback, NULL);
    if (res != CUPTI_SUCCESS) {
        fprintf(stderr, "[sched] cuptiSubscribe failed: %d\n", res);
        return;
    }

    cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API,
                        CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel);
    cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API,
                        CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx);

    fprintf(stderr, "[sched] CUPTI subscriber attached\n");
}
