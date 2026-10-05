#pragma once

#include "core/weight.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void q8_linear_swiglu_decode_pair_r16_launch(const Tensor& x, const Weight& w, Tensor& out,
                                             cudaStream_t stream);
void q8_linear_swiglu_splitk_exact_t_launch(const Tensor& x, const Weight& w, Tensor& out,
                                            cudaStream_t stream);
void q8_linear_swiglu_mma_r32_c64_launch(const Tensor& x, const Weight& w, Tensor& out,
                                         cudaStream_t stream);
void q8_linear_swiglu_mma_r32_c32_launch(const Tensor& x, const Weight& w, Tensor& out,
                                         cudaStream_t stream);
void q8_linear_swiglu_mma_r32_c48_launch(const Tensor& x, const Weight& w, Tensor& out,
                                         cudaStream_t stream);
void q8_linear_swiglu_mma_r32_c80_launch(const Tensor& x, const Weight& w, Tensor& out,
                                         cudaStream_t stream);
void q8_linear_swiglu_mma_r32_c96_launch(const Tensor& x, const Weight& w, Tensor& out,
                                         cudaStream_t stream);
void q8_linear_swiglu_mma_r32_c128_launch(const Tensor& x, const Weight& w, Tensor& out,
                                          cudaStream_t stream);
void q8_linear_swiglu_mma_r64_c64_launch(const Tensor& x, const Weight& w, Tensor& out,
                                         cudaStream_t stream);
void q8_linear_swiglu_mma_r64_c96_launch(const Tensor& x, const Weight& w, Tensor& out,
                                         cudaStream_t stream);
void q8_linear_swiglu_mma_r64_c128_launch(const Tensor& x, const Weight& w, Tensor& out,
                                          cudaStream_t stream);
void q8_linear_swiglu_mma_r128_c64_launch(const Tensor& x, const Weight& w, Tensor& out,
                                          cudaStream_t stream);
void q8_linear_swiglu_mma_r128_c80_launch(const Tensor& x, const Weight& w, Tensor& out,
                                          cudaStream_t stream);
void q8_dflash2_linear_swiglu_small_t_launch(const Tensor& x, const Weight& w, Tensor& out,
                                             cudaStream_t stream);

void q8_dflash2_linear_swiglu_mma_r32_c64_k128_launch(const Tensor&, const Weight&, Tensor&,
                                                      cudaStream_t);

void q8_dflash2_linear_swiglu_mma_r64_c64_k128_launch(const Tensor&, const Weight&, Tensor&,
                                                      cudaStream_t);

void q8_dflash2_linear_swiglu_mma_r64_c80_k128_launch(const Tensor&, const Weight&, Tensor&,
                                                      cudaStream_t);

void q8_dflash2_linear_swiglu_mma_r64_c96_k128_launch(const Tensor&, const Weight&, Tensor&,
                                                      cudaStream_t);

// Ada occupancy candidates for the K7 T8/T16/T32/T64 hot extents. Benchmark-only until the
// sm89 qualification demonstrates an end-to-end gain and the route table is changed explicitly.
void q8_dflash2_linear_swiglu_sm89_occ_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);

} // namespace ninfer::ops::detail
