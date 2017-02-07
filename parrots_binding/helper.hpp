#ifndef PARROTS_CTC_HELPER
#define PARROTS_CTC_HELPER

#include "ctc.h"

void permute_dimension(const size_t* dims, size_t ndim, size_t num,
                       float* dst_data, size_t dst_dim, float beta,
                       const float* src_data, size_t src_dim, float alpha,
                       ctcOptions opts);

void permute_dimension_cpu(const size_t* dims, size_t ndim, size_t num,
                           float* dst_data, size_t dst_dim, float beta,
                           const float* src_data, size_t src_dim, float alpha);

#ifdef __CUDACC__
void permute_dimension_gpu(const size_t* dims, size_t ndim, size_t num,
                           float* dst_data, size_t dst_dim, float beta,
                           const float* src_data, size_t src_dim, float alpha);
#else
void permute_dimension_gpu(const size_t* dims, size_t ndim, size_t num,
                           float* dst_data, size_t dst_dim, float beta,
                           const float* src_data, size_t src_dim, float alpha){}

#endif

#endif