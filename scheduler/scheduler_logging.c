#include "scheduler_logging.h"

static KernelLog kernel_log[MAX_LOG_LEN];
static int kernel_log_len = 0;

static inline uint64_t now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

unsigned int log_kernel_arrival(const char* kernel_name, uint8_t qlen) {
    int len = strlen(kernel_name);
    memcpy(kernel_log[kernel_log_len].kernel_name, kernel_name, len);
    kernel_log[kernel_log_len].kernel_name[len] = '\0';
    
    kernel_log[kernel_log_len].launched_at_ns = now_ns();
    kernel_log[kernel_log_len].qlen_at_launch = qlen;
    kernel_log_len++;
    return kernel_log_len;
}

void log_kernel_exec_start(int log_id, uint8_t bsize) {
    kernel_log[log_id - 1].exec_start_at_ns = now_ns();
    kernel_log[log_id - 1].exec_kernel_bsize = bsize;
}

void log_kernel_exec_end(int log_id) {
    // kernel_log[log_id - 1].exec_end_at_ns = now_ns();
}

void write_log() {
    uint64_t start = kernel_log[0].launched_at_ns;

    FILE *f = fopen("kernel_exec_log.txt", "w");
    for (size_t i = 0; i < kernel_log_len; i++)
        fprintf(f, "%s,%lu,%lu,%d,%d\n", kernel_log[i].kernel_name,
                kernel_log[i].launched_at_ns - start, 
                kernel_log[i].exec_start_at_ns - start,
                // kernel_log[i].exec_end_at_ns - start,
                kernel_log[i].exec_kernel_bsize,
                kernel_log[i].qlen_at_launch);
    fclose(f);
}