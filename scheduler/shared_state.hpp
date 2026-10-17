#ifndef SHARED_STATE_HPP
#define SHARED_STATE_HPP

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <pthread.h>
#include <semaphore.h>
#include <sys/types.h>

// v5: the structs below gained batching fields. A new name makes an agent built against the old layout fail at once
// ("scheduler_daemon never started") instead of reading a segment with a different layout.
constexpr const char* SHM_NAME = "/kernel_sched_shm_v5";
constexpr int MAX_PENDING  = 256;
constexpr int MAX_INFLIGHT = 64;  // max kernels tracked as "currently executing" at once
constexpr int MAX_REGIONS  = 1024;
constexpr int NAME_LEN     = 128;

struct PendingSlot {
    pid_t pid;
    char  kernel_name[NAME_LEN];
    char  layer_tag[NAME_LEN];
    int   priority;   // filled in by the daemon's policy
    int   active;     // slot in use (request/response queue lifecycle)
    int   ready;      // daemon sets this to release the waiting agent
    int   kernel_id;

    // Filled in by the DAEMON before it sets ready=1.
    int  has_prefetch;
    char prefetch_tag[NAME_LEN];
    size_t prefetch_max_bytes;

    int do_prefetch;

    // Written by the AGENT before it posts the request (they let the daemon log say exactly which kernel it is):
    uint64_t request_ns;       // CLOCK_MONOTONIC when the launch was intercepted (the clock behind start_ns / end_ns in the agent log)
    int      agent_kernel_id;  // the agent's own per-process launch counter; with the pid it names a row of agent_<pid>_timing.tsv

    // Written by the DAEMON before it sets ready=1: which batch this kernel was released in.
    uint64_t batch_id;         // 1, 2, 3, ...: kernels sharing a batch_id were released together
    int      batch_size;       // how many kernels that batch holds
    int      batch_pos;        // this kernel's 0-based place in the order the batch's wake-ups were posted

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
    std::atomic<uint64_t> end_ns; // written by the agent's completion callback just before it clears `active`; 0 = not known
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