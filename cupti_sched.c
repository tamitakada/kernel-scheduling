// cupti_sched.c
#include <cupti.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <unistd.h>

#define MAX_PENDING 64

typedef struct {
    char name[256];
    int priority;          // lower = higher priority, your policy decides
    int ready_to_launch;    // set by scheduler when it's this one's turn
    pthread_cond_t cond;
    pthread_mutex_t mutex;
    int active;
} PendingLaunch;

static PendingLaunch pending[MAX_PENDING];
static pthread_mutex_t queue_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t scheduler_wakeup = PTHREAD_COND_INITIALIZER;

// Called from CUPTI ENTER callback — BLOCKS until scheduler releases it
void request_launch_permission(const char* kernel_name, int priority) {
    pthread_mutex_lock(&queue_mutex);

    // find a free slot, register this launch
    int slot = -1;
    for (int i = 0; i < MAX_PENDING; i++) {
        if (!pending[i].active) { slot = i; break; }
    }
    pending[slot].active = 1;
    pending[slot].ready_to_launch = 0;
    pending[slot].priority = priority;
    strncpy(pending[slot].name, kernel_name, sizeof(pending[slot].name) - 1);
    pthread_mutex_init(&pending[slot].mutex, NULL);
    pthread_cond_init(&pending[slot].cond, NULL);

    pthread_cond_signal(&scheduler_wakeup);  // tell scheduler "new request"
    pthread_mutex_unlock(&queue_mutex);

    // now block THIS thread until scheduler says go
    pthread_mutex_lock(&pending[slot].mutex);
    while (!pending[slot].ready_to_launch) {
        pthread_cond_wait(&pending[slot].cond, &pending[slot].mutex);
    }
    pthread_mutex_unlock(&pending[slot].mutex);

    pthread_mutex_lock(&queue_mutex);
    pending[slot].active = 0;  // free the slot
    pthread_mutex_unlock(&queue_mutex);
}

static int should_wait = 0;

// Separate scheduler thread — decides ORDER
void* scheduler_loop(void* arg) {
    while (1) {
        printf("SCHED LOOP...\n");
        // sleep(3);
        pthread_mutex_lock(&queue_mutex);

        // pick the best candidate per YOUR policy (e.g. lowest priority number)
        int best = -1;
        int num_pending = 0;
        for (int i = 0; i < MAX_PENDING; i++) {
            if (pending[i].active && !pending[i].ready_to_launch) {
                num_pending++;
                if (best == -1 || pending[i].priority < pending[best].priority) {
                    best = i;
                }
            }
        }
        printf("Sched saw %d pending\n", num_pending);

        /*if (num_pending) {
            should_wait = !should_wait;
            if (should_wait) {
                pthread_mutex_unlock(&queue_mutex);
                sleep(1);
                continue;
            }
        }*/

        if (best == -1) {
            pthread_cond_wait(&scheduler_wakeup, &queue_mutex);  // nothing pending, sleep
            pthread_mutex_unlock(&queue_mutex);
            continue;
        }

        pthread_mutex_unlock(&queue_mutex);
//
//        // release exactly this one
        pthread_mutex_lock(&pending[best].mutex);
        pending[best].ready_to_launch = 1;
        pthread_cond_signal(&pending[best].cond);
        pthread_mutex_unlock(&pending[best].mutex);
    }
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
    pthread_t t;
    pthread_create(&t, NULL, scheduler_loop, NULL);
    
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
