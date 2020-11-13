python tuned_warp_memcpy_code_generator.py
nvcc -expt-relaxed-constexpr -std=c++11 -O3 -arch=sm_70 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_70,code=compute_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_75,code=compute_75 -o memcpy-ref memcpy-ref.cu
nvcc -expt-relaxed-constexpr -std=c++11 -O3 -arch=sm_70 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_70,code=compute_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_75,code=compute_75 -o memcpy memcpy.cu
