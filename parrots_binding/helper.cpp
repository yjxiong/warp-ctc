//
// Created by yjxiong on 2/7/17.
//

#include "helper.hpp"
#include <iostream>

void permute_dimension(const size_t* dims, size_t ndim, size_t num,
                       float* dst_data, size_t dst_dim, float beta,
                       const float* src_data, size_t src_dim, float alpha,
                       ctcOptions opts){
  if (opts.loc == CTC_GPU){
    permute_dimension_gpu(dims, ndim, num, dst_data, dst_dim, beta, src_data, src_dim, alpha);
  }else{
    permute_dimension_cpu(dims, ndim, num, dst_data, dst_dim, beta, src_data, src_dim, alpha);
  }

}

void permute_dimension_cpu(const size_t* dims, size_t ndim, size_t num,
                           float* dst_data, size_t dst_dim, float beta,
                           const float* src_data, size_t src_dim, float alpha){

  if (dst_dim == src_dim) return;


  for (size_t src_idx = 0; src_idx < num; ++src_idx){
    size_t src_dim_idx = 0;
    size_t dst_dim_idx = 0;
    for (int i = 0, p = src_idx; i < ndim; ++i){
      size_t d = dims[i];
      if (i == src_dim) src_dim_idx = p % d;
      if (i == dst_dim) dst_dim_idx = p % d;
      p /= d;

    }
    size_t dst_idx = 0;
    for (int i = ndim - 1, p = src_idx, q = num; i >= 0; --i){

      size_t offset;
      size_t d;

      q /= dims[i];

      if (i == src_dim){
        d = dims[dst_dim];
        offset = dst_dim_idx;
      }else if(i == dst_dim){
        d = dims[src_dim];
        offset = src_dim_idx;
      }else{
        d = dims[i];
        offset = p / q;
      }
      dst_idx = dst_idx * d + offset;
      p %= q;
    }

    dst_data[dst_idx] = src_data[src_idx] * alpha + dst_data[dst_idx] * beta;
  }

}