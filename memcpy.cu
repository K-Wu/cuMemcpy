#include <cuda.h>
#include <cstdint>
#include <iostream>
#include "memcpy_tuned.h"
#define cuda_err_chk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=false)
{
    if (code != cudaSuccess)
    {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(1);
    }
}
__forceinline__ __device__ uint32_t lane_id()
{
    uint32_t ret;
    asm volatile ("mov.u32 %0, %laneid;" : "=r"(ret));
    return ret;
}

__forceinline__ __device__ uint64_t time()
{
    uint32_t ret;
    asm volatile ("mov.u64  %0,%globaltimer;" : "=r"(ret));
    return ret;
}




/*warp memcpy, assumes alignment at type T and num is a count in type T*/
template <typename T>
__device__
void warp_memcpy(T* __restrict__ dest, const T* __restrict__ src, size_t num);
 // {
//         uint32_t mask = __activemask();
//         uint32_t active_cnt = __popc(mask);
//         uint32_t lane = threadIdx.x & 0x1F;//lane_id();
//         uint32_t prior_mask = mask >> (32 - lane);
//         uint32_t prior_count = __popc(prior_mask);
//         _warp_memcpy<T, 8, 32>(dest, src, prior_count, num);
//         //uint64_t begin = time();
        
//         //uint64_t end = time();
//         //return (end-begin);
// }


template <typename T>
__global__
void memcpy(T* dest, const T* src, size_t num) {
    warp_memcpy(dest, src, num);
}

__global__
void verify(uint8_t* dest, uint8_t* src, size_t num, int* error_count) {
    size_t n_threads = blockDim.x*gridDim.x;
    for (size_t idx=threadIdx.x+blockDim.x*blockIdx.x; idx<num;idx+=n_threads){
        if(dest[idx]!=src[idx]){
            atomicAdd(error_count,1);
        }
        
    }
}
enum type {
uint8,
uint16,
uint32,
uint64,
uint128,
uint256
};
int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Please provide type and count!\n";
        return 1;
    }
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    const std::string type_str = argv[1];
    const std::string count_str = argv[2];
    const unsigned int type = std::stoul(type_str);
    const unsigned int count = std::stoul(count_str);
    void* h_src_ptr;
    void* d_src_ptr;
    void* d_dest_ptr;
    void* d_dest_aligned_ptr;
    int* d_error_count;
    int error_count;
    const unsigned long long int size = (1ULL << type) * (count+1);
    cuda_err_chk(cudaHostAlloc(&h_src_ptr, size, cudaHostAllocMapped));
    cuda_err_chk(cudaHostGetDevicePointer(&d_src_ptr, h_src_ptr, 0));
    d_src_ptr = (void*)(((uint64_t)d_src_ptr) + ((1ULL << type)));
    cuda_err_chk(cudaMalloc(&d_dest_ptr, size));
    cuda_err_chk(cudaMalloc(&d_error_count, sizeof(int)));
    cuda_err_chk(cudaMemset(d_error_count,0, sizeof(int)));
    d_dest_aligned_ptr = (void*)(((uint64_t)d_dest_ptr) + ((1ULL << type)));
    //uint64_t reg_time;
    switch (type) {
        case uint8:
            cudaEventRecord(start);
            memcpy<uint8_t><<<1, 32>>>((uint8_t*) d_dest_aligned_ptr, (const uint8_t*) d_src_ptr, count);
            cudaEventRecord(stop);
            verify<<<1, 32>>>((uint8_t*) d_dest_aligned_ptr,  (uint8_t*)d_src_ptr, count*1, d_error_count);
            break;
        case uint16:
            cudaEventRecord(start);
            memcpy<uint16_t><<<1, 32>>>((uint16_t*) d_dest_aligned_ptr, (const uint16_t*) d_src_ptr, count);
            cudaEventRecord(stop);
            verify<<<1, 32>>>((uint8_t*) d_dest_aligned_ptr,  (uint8_t*)d_src_ptr, count*2, d_error_count);
            break;
        case uint32:
            cudaEventRecord(start);
            memcpy<uint32_t><<<1, 32>>>((uint32_t*) d_dest_aligned_ptr, (const uint32_t*) d_src_ptr, count);
            cudaEventRecord(stop);
            verify<<<1, 32>>>((uint8_t*) d_dest_aligned_ptr,  (uint8_t*)d_src_ptr, count*4, d_error_count);
            break;
        case uint64:
            cudaEventRecord(start);
            memcpy<uint64_t><<<1, 32>>>((uint64_t*) d_dest_aligned_ptr, (const uint64_t*) d_src_ptr, count);
            cudaEventRecord(stop);
            verify<<<1, 32>>>((uint8_t*) d_dest_aligned_ptr,  (uint8_t*)d_src_ptr, count*8, d_error_count);
            break;
        case uint128:
            cudaEventRecord(start);
            memcpy<ulonglong2><<<1, 32>>>((ulonglong2*) d_dest_aligned_ptr, (const ulonglong2*) d_src_ptr, count);
            cudaEventRecord(stop);
            verify<<<1, 32>>>((uint8_t*) d_dest_aligned_ptr,  (uint8_t*)d_src_ptr, count*16, d_error_count);
            break;
        case uint256:
            cudaEventRecord(start);
            memcpy<ulonglong4><<<1, 32>>>((ulonglong4*) d_dest_aligned_ptr, (const ulonglong4*) d_src_ptr, count);
            cudaEventRecord(stop);
            verify<<<1, 32>>>((uint8_t*) d_dest_aligned_ptr,  (uint8_t*)d_src_ptr, count*32, d_error_count);
            break;
        default:
            cudaEventRecord(start);
            std::cerr << "Invalid type!\n";
            break;
    }
    cuda_err_chk(cudaEventSynchronize(stop));
    cuda_err_chk(cudaMemcpy(&error_count,d_error_count,sizeof(int),cudaMemcpyDeviceToHost));
    cuda_err_chk(cudaFreeHost(h_src_ptr));
    cuda_err_chk(cudaFree(d_dest_ptr));
    cuda_err_chk(cudaFree(d_error_count));
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Bytes: " << (size - ((1ULL << type) )) << "\tTime: " << milliseconds << " ms\n";
    std::cout << "Bandwidth: " << (size - ((1ULL << type)))/milliseconds * 1000ULL/(1024ULL*1024ULL*1024ULL) << " GBytes/sec\n";
    std::cout <<"Error count: "<<error_count<<std::endl;
    
    //std::cout << "Reg Time "<< reg_time<<"\n";
}