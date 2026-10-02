#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

__global__ void sleep_kernel(int id, int ms) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Kernel %d beginning sleep for %d ms\n", id, ms);
        for (int i = 0; i < ms; i++) {
            __nanosleep(1000000);  // 1 ms
        }
        printf("Kernel %d finished sleep\n\n", id);
    }
}

int main() {
    const int NUM_KERNELS = 5;
    const int SLEEP_MS = 2000;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    for (int i = 0; i < NUM_KERNELS; ++i) {
        sleep_kernel<<<1, 1, 0, stream>>>(i, SLEEP_MS);
        printf("Kernel %d queued for exec...\n", i);
        CUDA_CHECK(cudaGetLastError());
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaStreamDestroy(stream));
    
    return 0;
}
