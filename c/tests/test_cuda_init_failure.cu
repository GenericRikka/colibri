/* Two logical devices mapped to one real GPU, with real stream lifetimes.
 * This tests initialization rollback, not physical multi-GPU execution. */
#include "../backend_gpu_compat.h"
#include <cstdio>
static int attempts, live_streams, fail_stream;
static cudaError_t real_count(int *n) { return cudaGetDeviceCount(n); }
static cudaError_t test_count(int *n) { *n=2;return cudaSuccess; }
static cudaError_t test_select(int) { return cudaSetDevice(0); }
static cudaError_t test_props(cudaDeviceProp *p,int) { return cudaGetDeviceProperties(p,0); }
static cudaError_t test_create(cudaStream_t *s,unsigned flags) {
    if(++attempts==fail_stream) return cudaErrorMemoryAllocation;
    cudaError_t e=cudaStreamCreateWithFlags(s,flags);if(e==cudaSuccess)live_streams++;return e;
}
static cudaError_t test_destroy(cudaStream_t s) {
    cudaError_t e=cudaStreamDestroy(s);if(e==cudaSuccess)live_streams--;return e;
}
#undef cudaGetDeviceCount
#undef cudaSetDevice
#undef cudaGetDeviceProperties
#undef cudaStreamCreateWithFlags
#undef cudaStreamDestroy
#define cudaGetDeviceCount test_count
#define cudaSetDevice test_select
#define cudaGetDeviceProperties test_props
#define cudaStreamCreateWithFlags test_create
#define cudaStreamDestroy test_destroy
#include "../backend_cuda.cu"
int main(void) {
    int count=0;if(real_count(&count)!=cudaSuccess||!count){puts("SKIP: no GPU");return 0;}
    int ids[]={0,1}, duplicate[]={0,0},invalid[]={0,2};
    for(int round=0;round<2;round++){
        attempts=0;
        if(coli_cuda_init(duplicate,2)||attempts||live_streams||coli_cuda_device_count()) return 1;
        if(coli_cuda_init(invalid,2)||attempts||live_streams||coli_cuda_device_count()) return 2;
        for(fail_stream=1;fail_stream<=2;fail_stream++){
            attempts=0;
            if(coli_cuda_init(ids,2)||live_streams||coli_cuda_device_count()) return 3;
        }
        fail_stream=0;attempts=0;
        if(!coli_cuda_init(ids,2)||live_streams!=2||coli_cuda_device_count()!=2) return 4;
        /* Rejected input must leave an existing backend reachable. */
        if(coli_cuda_init(invalid,2)||live_streams!=2||coli_cuda_device_count()!=2) return 5;
        if(coli_cuda_init(duplicate,2)||live_streams!=2||coli_cuda_device_count()!=2) return 6;
        coli_cuda_shutdown();if(live_streams||coli_cuda_device_count()) return 7;
    }
    puts("CUDA initialization rollback: PASS");return 0;
}
