// shared_state.h
//
// Shared by scheduler_daemon.c (the standalone scheduler process) and
// agent.c (the .so each vLLM process LD_PRELOADs). Note this header has NO
// CUDA dependency at all -- it only needs plain C types -- because the
// daemon never touches CUDA itself; see scheduler_daemon.c for why.

#ifndef SHARED_STATE_H
#define SHARED_STATE_H

#include <pthread.h>
#include <semaphore.h>
#include <stdatomic.h>
#include <stddef.h>
#include <sys/types.h>

#define SHM_NAME     "/kernel_sched_shm_v3"
#define MAX_PENDING  256
#define MAX_REGIONS  256
#define NAME_LEN     128

typedef struct {
    pid_t  pid;
    char   kernel_name[NAME_LEN];
    char   layer_tag[NAME_LEN];
    int    priority;     // filled in by the daemon's policy
    int    active;       // slot in use
    int    ready;        // daemon sets this to release the waiting agent

    // Filled in by the DAEMON before it sets ready=1. The releasing agent
    // reads this and -- because only its own process can validly touch its
    // own weight memory -- executes the actual prefetch launch itself.
    int    has_prefetch;
    char   prefetch_tag[NAME_LEN];

    sem_t  wake_sem;      // agent blocks here; daemon posts to release it
} PendingSlot;

// One registered weight-tensor region. ptr is a raw device pointer that is
// ONLY valid for dereferencing inside owner_pid's own process -- the daemon
// reads tag/nbytes to make decisions, it never dereferences ptr itself.
typedef struct {
    pid_t  owner_pid;
    char   tag[NAME_LEN];
    void*  ptr;
    size_t nbytes;
    int    active;
} SharedWeightRegion;

typedef struct {
    _Atomic int initialized;   // set by the daemon once setup is complete

    pthread_mutex_t mutex;          // guards slots[] and regions[]/region_count
    sem_t           new_request_sem; // daemon's arbitration loop wakes on this

    PendingSlot        slots[MAX_PENDING];
    SharedWeightRegion regions[MAX_REGIONS];
    int                region_count;
} SchedulerSharedState;

#endif
