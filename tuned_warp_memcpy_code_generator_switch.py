
TUNED_CODE_H = "memcpy_tuned.h"

WARP_MEMCPY_CALLING_FORMAT_STRING ="""
_warp_memcpy_{T_sz}_{active_cnt} ( ({T_sz_generic_type}*) dest, ({T_sz_generic_type}*) src, prior_count, num);
"""

DEFINE_WARP_MEMCPY_FORMAT_STRING = """
__device__  __forceinline__
void  _warp_memcpy_{T_sz}_{active_cnt} ({T_sz_generic_type}* __restrict__ dest, const {T_sz_generic_type}* __restrict__ src, uint32_t prior_count, size_t num) {{
    #pragma unroll {unroll_factor} 
    for(size_t i = prior_count; i < num; i+={active_cnt}) {{
            dest[i] = src[i]; 
    }}
}}
"""

HEADING_CONTENT="""
#pragma once
#include <cuda.h>
#include <assert.h>
template <typename T> 
__device__  __forceinline__
void  _warp_memcpy(T* __restrict__ dest, const T* __restrict__ src, uint32_t prior_count, size_t num, int active_cnt);
template <typename T> 
__device__  __forceinline__
void  _warp_memcpy(T* __restrict__ dest, const T* __restrict__ src, uint32_t prior_count, size_t num, int active_cnt) { //general fall back scheme 
    for(size_t i = prior_count; i < num; i+=active_cnt) { 
            dest[i] = src[i]; 
    } 
}
"""

WARP_MEMCPY_DEFINITION="""
template <typename T>
__device__
void warp_memcpy(T* __restrict__ dest, const T* __restrict__ src, size_t num) {
        uint32_t mask = __activemask();
        uint32_t active_cnt = __popc(mask);
        uint32_t lane = threadIdx.x & 0x1F;//lane_id();
        uint32_t prior_mask = mask >> (32 - lane);
        uint32_t prior_count = __popc(prior_mask);
        {switch_table}
}
"""

T_sz_generic_type={4:"uint32_t", 8:"uint64_t", 32:"ulonglong4"}

#defining T_sz_active_cnt_unroll_factor_tuples. For now constantly set unroll factor as 8
T_sz_active_cnt_unroll_factor_tuples = set()
for T_sz in [4, 8, 32]:
    for active_cnt in range(1,33):
        T_sz_active_cnt_unroll_factor_tuples.add((T_sz,active_cnt,""))#fully unrolled


def generate_switch_table(T_sz_active_cnt_unroll_factor_tuples):
    T_szs = set([T_sz for (T_sz, active_cnt, unroll_factor) in T_sz_active_cnt_unroll_factor_tuples])
    active_cnts = set([active_cnt for (T_sz, active_cnt, unroll_factor) in T_sz_active_cnt_unroll_factor_tuples])

    result_str="switch (sizeof(T)) {\n"
    for  T_sz in T_szs:
        result_str+="case {T_sz}:{{ // beginning of T_sze switch block\n".format(T_sz=T_sz) #beginning of T_sze switch block
        result_str+="switch (active_cnt) {// beginning of active_cnt switch block\n" #beginning of active_cnt switch block
        for active_cnt in active_cnts:
            result_str+="case {active_cnt}: {{".format(active_cnt=active_cnt)
            result_str+=WARP_MEMCPY_CALLING_FORMAT_STRING.format(T_sz=T_sz, active_cnt=active_cnt, T_sz_generic_type=T_sz_generic_type[T_sz])
            result_str+="\nreturn;\n}\n"
        result_str+="default:assert(0);} // end of active_cnt switch block \n"  #end of active_cnt switch block
        result_str+="} // end of T_sz case block\n" # end of T_sz case block
    result_str += "default: _warp_memcpy<T> ( dest, src, prior_count, num, active_cnt);return; //general fall back scheme\n" #general fall back scheme
    result_str+="} //end of switch T_sz \n" #end of switch T_sz
    return result_str
if __name__ == "__main__":
    with open(TUNED_CODE_H,'w') as fd:
        # write head content
        fd.write(HEADING_CONTENT)
        #write all definitions based upon T_sz_active_cnt_unroll_factor_tuples
        for (T_sz, active_cnt, unroll_factor) in T_sz_active_cnt_unroll_factor_tuples:
            fd.write(DEFINE_WARP_MEMCPY_FORMAT_STRING.format(T_sz=T_sz, active_cnt=active_cnt,unroll_factor=unroll_factor, T_sz_generic_type=T_sz_generic_type[T_sz]))
        fd.write(WARP_MEMCPY_DEFINITION.replace(r"{switch_table}",generate_switch_table(T_sz_active_cnt_unroll_factor_tuples)))
        #write the generic entry function warp_memcpy definition 