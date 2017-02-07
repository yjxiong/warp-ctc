#include "helper.hpp"
#include "helper.cpp"

__global__ void permute_kernel(const size_t* dims, size_t ndim, size_t n,
                          float* dst_data, size_t dst_dim, float beta,
                          const float* src_data, size_t src_dim, float alpha){
  for (int src_idx = blockIdx.x * blockDim.x + threadIdx.x; \
       src_idx < (n); \
       src_idx += blockDim.x * gridDim.x){

    size_t src_dim_idx = 0;
    size_t dst_dim_idx = 0;
    for (int i = 0, p = src_idx; i < ndim; ++i){
      size_t d = dims[i];
      if (i == src_dim) src_dim_idx = p % d;
      if (i == dst_dim) dst_dim_idx = p % d;
      p /= d;
    }

    size_t dst_idx = 0;
    for (int i = ndim - 1, p = src_idx, q = n; i >= 0; --i){
      size_t d = dims[i];
      q /= d;
      dst_idx *= d;

      if (i == src_dim){
        dst_idx += dst_dim_idx;
      }else if(i == dst_dim){
        dst_idx += src_dim_idx;
      }else{
        dst_idx += p / q;
      }
      p %= q;
    }

    dst_data[dst_idx] = src_data[src_idx] * alpha + dst_data[dst_idx] * beta;
  }
}

#define MAX_THREAD 1024

void permute_dimension_gpu(const size_t* dims, size_t ndim, size_t num,
                           float* dst_data, size_t dst_dim, float beta,
                           const float* src_data, size_t src_dim, float alpha){

  size_t num_threads = (num + MAX_THREAD - 1) / MAX_THREAD;
  permute_kernel  // NOLINT_NEXT_LINE(whitespace/operators)
  <<<num_threads, MAX_THREAD>>>(dims, ndim, num, dst_data, dst_dim, beta,
                          src_data, src_dim, alpha);

}