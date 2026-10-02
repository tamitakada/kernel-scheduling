#ifndef SHARED_STATE_HPP
#define SHARED_STATE_HPP

#include <atomic>
#include <cstddef>
#include <pthread.h>
#include <semaphore.h>
#include <sys/types.h>

constexpr const char* SHM_NAME = "/kernel_sched_shm_v4";
constexpr int MAX_PENDING  = 256;
constexpr int MAX_INFLIGHT = 64;  // max kernels tracked as "currently executing" at once
constexpr int MAX_REGIONS  = 256;
constexpr int NAME_LEN     = 128;

struct PendingSlot {
    pid_t pid;
    char  kernel_name[NAME_LEN];
    char  layer_tag[NAME_LEN];
    int   priority;   // filled in by the daemon's policy
    int   active;     // slot in use (request/response queue lifecycle)
    int   ready;      // daemon sets this to release the waiting agent

    // Filled in by the DAEMON before it sets ready=1.
    int  has_prefetch;
    char prefetch_tag[NAME_LEN];

    // Index into SchedulerSharedState::inflight[] that the daemon reserved
    // for THIS kernel at release time, or -1. The agent reads this after
    // being released and carries it through to the kernel's actual GPU
    // completion, so the right inflight slot gets cleared -- not just a
    // generic counter decremented.
    int inflight_slot;

    sem_t wake_sem;  // agent blocks here; daemon posts to release it
};

// Tracks ONE currently-executing kernel. Unlike a single counter, this lets
// the daemon (or anything inspecting shared memory) see exactly WHICH
// kernel(s), from which process, are in flight right now.
struct InflightSlot {
    std::atomic<int> active; // 0 = free, 1 = a kernel is currently running here
    pid_t pid;
    char  kernel_name[NAME_LEN];
    char  layer_tag[NAME_LEN];
};

// One registered weight-tensor region. ptr is a raw device pointer that is
// ONLY valid for dereferencing inside owner_pid's own process.
struct SharedWeightRegion {
    pid_t  owner_pid;
    char   tag[NAME_LEN];
    void*  ptr;
    size_t nbytes;
    int    active;
};

struct SchedulerSharedState {
    std::atomic<int> initialized; // set by the daemon once setup is complete

    pthread_mutex_t mutex;          // guards slots[], inflight[], regions[]/region_count
    sem_t           new_request_sem; // daemon's arbitration loop wakes on this

    PendingSlot        slots[MAX_PENDING];
    InflightSlot       inflight[MAX_INFLIGHT];
    SharedWeightRegion regions[MAX_REGIONS];
    int                region_count;
};

#endif
