#pragma once

#define MAX_NAME_LEN    128
#define MAX_LOG_LEN     5000

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

typedef struct {
    char        kernel_name[MAX_NAME_LEN];
    uint64_t    launched_at_ns;
    uint64_t    exec_start_at_ns;
    // uint64_t    exec_end_at_ns;
    uint8_t     exec_kernel_bsize;
    uint8_t     qlen_at_launch;
} KernelLog;

unsigned int log_kernel_arrival(const char* kernel_name, uint8_t qlen);
void log_kernel_exec_start(int log_id, uint8_t bsize);
void log_kernel_exec_end(int log_id);

void write_log();