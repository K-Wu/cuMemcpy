#include <cuda.h>
#include <cstdint>
#include <iostream>
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
void warp_memcpy(T* __restrict__ dest, const T* __restrict__ src, size_t num) {
        uint32_t mask = __activemask();
        uint32_t active_cnt = __popc(mask);
        uint32_t lane = threadIdx.x & 0x1F;//lane_id();
        uint32_t prior_mask = mask >> (32 - lane);
        uint32_t prior_count = __popc(prior_mask);
        //uint64_t begin = time();
        #pragma unroll 8
        for(size_t i = prior_count; i < num; i+=active_cnt) {
                dest[i] = src[i];
                //printf("tid: %llu\ti: %llu\n", (unsigned long long) threadIdx.x, (unsigned long long) i);
        }
        //uint64_t end = time();
        //return (end-begin);
}
template <typename T>
__global__
void memcpy(T* dest, const T* src, size_t num) {
    warp_memcpy(dest, src, num);
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
    const unsigned long long int size = (1ULL << type) * (count+1);
    cuda_err_chk(cudaHostAlloc(&h_src_ptr, size, cudaHostAllocMapped));
    cuda_err_chk(cudaHostGetDevicePointer(&d_src_ptr, h_src_ptr, 0));
    d_src_ptr = (void*)(((uint64_t)d_src_ptr) + ((1ULL << type)));
    cuda_err_chk(cudaMalloc(&d_dest_ptr, size));
    d_dest_aligned_ptr = (void*)(((uint64_t)d_dest_ptr) + ((1ULL << type)));
    //uint64_t reg_time;
    switch (type) {
        case uint8:
            cudaEventRecord(start);
            memcpy<uint8_t><<<1, 32>>>((uint8_t*) d_dest_aligned_ptr, (const uint8_t*) d_src_ptr, count);
             cudaEventRecord(stop);
            break;
        case uint16:
            cudaEventRecord(start);
            memcpy<uint16_t><<<1, 32>>>((uint16_t*) d_dest_aligned_ptr, (const uint16_t*) d_src_ptr, count);
            cudaEventRecord(stop);
            break;
        case uint32:
            cudaEventRecord(start);
            memcpy<uint32_t><<<1, 32>>>((uint32_t*) d_dest_aligned_ptr, (const uint32_t*) d_src_ptr, count);
            cudaEventRecord(stop);
            break;
        case uint64:
            cudaEventRecord(start);
            memcpy<uint64_t><<<1, 32>>>((uint64_t*) d_dest_aligned_ptr, (const uint64_t*) d_src_ptr, count);
            cudaEventRecord(stop);
            break;
        case uint128:
            cudaEventRecord(start);
            memcpy<ulonglong2><<<1, 32>>>((ulonglong2*) d_dest_aligned_ptr, (const ulonglong2*) d_src_ptr, count);
            cudaEventRecord(stop);
            break;
        case uint256:
            cudaEventRecord(start);
            memcpy<ulonglong4><<<1, 32>>>((ulonglong4*) d_dest_aligned_ptr, (const ulonglong4*) d_src_ptr, count);
            cudaEventRecord(stop);
            break;
        default:
            cudaEventRecord(start);
            std::cerr << "Invalid type!\n";
            break;
    }
    cuda_err_chk(cudaEventSynchronize(stop));
    cuda_err_chk(cudaFreeHost(h_src_ptr));
    cuda_err_chk(cudaFree(d_dest_ptr));
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Bytes: " << (size - ((1ULL << type) )) << "\tTime: " << milliseconds << " ms\n";
    std::cout << "Bandwidth: " << (size - ((1ULL << type)))/milliseconds * 1000ULL/(1024ULL*1024ULL*1024ULL) << " GBytes/sec\n";
    //std::cout << "Reg Time "<< reg_time<<"\n";
}