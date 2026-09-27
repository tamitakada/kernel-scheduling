#include <semaphore.h>
#include <pthread.h>
#include <sys/types.h>

#define MAX_PENDING 256
#define SHM_NAME "/cupti_sched_shm"

typedef struct {
    pid_t pid;
    char kernel_name[128];
    int priority;
    int active;       // slot currently in use
    int ready;        // daemon sets this when it's this request's turn
    sem_t wake_sem;   // agent blocks on this; daemon posts to release it
} PendingSlot;

typedef struct {
    PendingSlot slots[MAX_PENDING];
    sem_t new_request_sem;    // agents post; daemon's loop waits on this
    pthread_mutex_t mutex;    // guards the slot table
} SchedulerSharedState;
