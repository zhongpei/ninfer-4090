#pragma once

// ninfer::ops - split-KV causal small-T attention shared scaffolding. The BF16 and
// int8 partial kernels live in causal_attention_small_t_bf16.cuh and
// causal_attention_small_t_i8.cuh respectively; they are fully separate kernels (no
// shared body) so each KV format can be optimized independently. This header owns
// only what both share: layout constants, device helpers, and the split reducer.

#include "ops/common/math.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/warp.cuh"
#include "ops/softmax_attention/dense/causal_cache/geometry.cuh"
#include "ops/kernel/paged_kv_address.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kCausalHeadDim = 256;

struct CausalAppendInput {
    static constexpr bool writes_cache = true;
    const __nv_bfloat16* k;
    const __nv_bfloat16* v;
};

struct CausalCachedInput {
    static constexpr bool writes_cache = false;
};

template <typename Geometry>
__device__ __forceinline__ std::int64_t causal_cache_index(int physical_page, int kv_head, int d,
                                                           int page_offset) {
    return paged_kv_element_offset<kCausalHeadDim, Geometry::KVHeads>(physical_page, kv_head,
                                                                      page_offset, d);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t causal_q_index(int q_head, int d, int token = 0) {
    return static_cast<std::int64_t>(d) + static_cast<std::int64_t>(kCausalHeadDim) *
                                              (static_cast<std::int64_t>(q_head) +
                                               static_cast<std::int64_t>(Geometry::QHeads) * token);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t kv_cache_int8_new_index(int kv_head, int d, int token = 0) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(kCausalHeadDim) *
               (static_cast<std::int64_t>(kv_head) +
                static_cast<std::int64_t>(Geometry::KVHeads) * token);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t causal_partial_acc_index(int q_head, int d, int token,
                                                                 int split, int tokens) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(kCausalHeadDim) *
               (static_cast<std::int64_t>(q_head) +
                static_cast<std::int64_t>(Geometry::QHeads) *
                    (static_cast<std::int64_t>(token) + static_cast<std::int64_t>(tokens) * split));
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t causal_partial_stat_index(int q_head, int token, int split,
                                                                  int tokens) {
    return static_cast<std::int64_t>(q_head) +
           static_cast<std::int64_t>(Geometry::QHeads) *
               (static_cast<std::int64_t>(token) + static_cast<std::int64_t>(tokens) * split);
}

template <typename Geometry>
__device__ __forceinline__ bool causal_valid_q_head(int kv_head, int q_head) {
    return kv_head >= 0 && kv_head < Geometry::KVHeads && q_head >= kv_head * Geometry::GroupSize &&
           q_head < (kv_head + 1) * Geometry::GroupSize && q_head < Geometry::QHeads;
}

// sm89 FP8 uses absolute split boundaries. A query sees the same reduction partition
// regardless of TokenTile, compact-batch occupancy, graph envelope, or later columns.
// The largest range is 4096 keys == 64 physical KV pages, preserving the kernel's
// fixed page-id staging bound while covering the full 786432-token product envelope
// in at most 256 splits.
inline constexpr int kCausalSm89Fp8MaximumSplits = 256;

__host__ __device__ constexpr int causal_sm89_fp8_split_start(int split) {
    if (split < 16) return split * 128;
    if (split < 32) return 2048 + (split - 16) * 512;
    // Keep mid-context CTAs short enough to avoid a long final wave on Ada.
    // All starts remain absolute and page-aligned; reserve enough long ranges for
    // the maximum product context without increasing reducer/workspace bounds.
    if (split < 64) return 10240 + (split - 32) * 768;
    if (split < 76) return 34816 + (split - 64) * 2048;
    return 59392 + (split - 76) * 4096;
}

__host__ __device__ constexpr int causal_sm89_fp8_split_count(int window) {
    if (window <= 0) return 0;
    int splits = 0;
    if (window <= 2048) {
        splits = (window + 127) / 128;
    } else if (window <= 10240) {
        splits = 16 + (window - 2048 + 511) / 512;
    } else if (window <= 34816) {
        splits = 32 + (window - 10240 + 767) / 768;
    } else if (window <= 59392) {
        splits = 64 + (window - 34816 + 2047) / 2048;
    } else {
        splits = 76 + (window - 59392 + 4095) / 4096;
    }
    return splits < kCausalSm89Fp8MaximumSplits ? splits : kCausalSm89Fp8MaximumSplits;
}

static_assert(causal_sm89_fp8_split_start(kCausalSm89Fp8MaximumSplits) >= 786432);
static_assert(causal_sm89_fp8_split_start(kCausalSm89Fp8MaximumSplits) -
                  causal_sm89_fp8_split_start(kCausalSm89Fp8MaximumSplits - 1) <= 4096);

template <typename Geometry>
__device__ __forceinline__ int causal_small_t_default_splits(int window) {
    int target_keys_per_split = 480 / Geometry::SmallTSplitScale;
    if (window <= 4096) {
        target_keys_per_split = 64 / Geometry::SmallTSplitScale;
    } else if (window <= 8198) {
        target_keys_per_split = 128 / Geometry::SmallTSplitScale;
    } else if (window <= 16390) {
        target_keys_per_split = 256 / Geometry::SmallTSplitScale;
    }
    constexpr int kMinSplits = 4 * Geometry::SmallTSplitScale;
    int splits               = div_up(window, target_keys_per_split);
    splits                   = splits > kMinSplits ? splits : kMinSplits;
    return splits < Geometry::SmallTMaximumSplits ? splits : Geometry::SmallTMaximumSplits;
}

// Absolute key partitions keep each query's arithmetic independent of later columns,
// batch occupancy and graph envelopes. Coarser partitions only begin at fixed boundaries.
__host__ __device__ constexpr int causal_int8_split_start(int split) {
    if (split < 16) return split * 512;
    if (split < 32) return 8192 + (split - 16) * 1024;
    if (split < 48) return 24576 + (split - 32) * 4096;
    return 90112 + (split - 48) * 16384;
}
__host__ __device__ constexpr int causal_int8_split_count(int window) {
    if (window <= 8192) return (window + 511) / 512;
    if (window <= 24576) return 16 + (window - 8192 + 1023) / 1024;
    if (window <= 90112) return 32 + (window - 24576 + 4095) / 4096;
    return 48 + (window - 90112 + 16383) / 16384;
}

template <typename Geometry, bool Int8>
__device__ __forceinline__ int causal_small_t_active_splits(int window, int launch_capacity,
                                                            int tokens) {
    if (window <= 0) { return launch_capacity; }
    int splits = 0;
    if constexpr (Int8) {
        splits = causal_int8_split_count(window);
    } else {
        splits = causal_small_t_default_splits<Geometry>(window);
    }
    return splits < launch_capacity ? splits : launch_capacity;
}

template <typename Geometry>
__device__ __forceinline__ int
causal_small_t_quantized_active_splits(int window, int launch_capacity, int tokens) {
    int splits = causal_small_t_default_splits<Geometry>(window);
    if constexpr (Geometry::SmallTSplitScale == 1) {
        if (tokens == 1 && window > 8198) { splits = Geometry::SmallTMaximumSplits; }
    }
    return splits < launch_capacity ? splits : launch_capacity;
}

__device__ __forceinline__ int causal_small_t_tc_swz(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

template <typename Byte>
__device__ __forceinline__ void causal_small_t_store_byte_swizzled(Byte* tile, int row, int d,
                                                                   int d_b16_stride, Byte code) {
    const int col_b16 = d >> 1;
    const int byte    = d & 1;
    const int off     = (row * d_b16_stride + causal_small_t_tc_swz(row, col_b16)) * 2 + byte;
    tile[off]         = code;
}

__device__ __forceinline__ int causal_small_t_tc_swz32(int row, int col) {
    return (((col >> 3) ^ (row & 3)) << 3) | (col & 7);
}

// Signed int8 QK MMA, k=32 contraction. A = 16x32 s8 (4 regs/thread, 4 s8 each),
// B = 8x32 s8 col-major (2 regs/thread), D = 16x8 s32 (4 regs/thread). The A/B
// register byte layout is identical to the m16n8k16 bf16 fragments loaded by
// ldmatrix_x4/x2 over a d-contiguous int8 tile reinterpreted as
// b16 (two packed int8 per 16-bit lane), so the same ldmatrix helpers and XOR
// swizzle feed this MMA. The s32 accumulator layout matches the bf16 f32
// accumulator (c0/c1 -> row groupID, c2/c3 -> row groupID+8), so score
// consumption is unchanged; only per-64-group scale rescale differs.
template <typename Geometry>
__device__ __forceinline__ void causal_small_t_tc_row_to_qt(int row, int tokens, int kv_head,
                                                            int& q_head, int& token) {
    token             = row / Geometry::GroupSize;
    const int local_q = row - token * Geometry::GroupSize;
    q_head            = kv_head * Geometry::GroupSize + local_q;
}

// Merge one query/head's split statistics once per CTA. Published scalars are separate
// from the reduction/weight storage, so later writes cannot race another warp's scalar read.
template <class Geometry>
__device__ __forceinline__ float
causal_merge_split_statistics(const float* partial_m, const float* partial_l, int q_head, int token,
                              int tokens, int splits, float* weights, float* warp_sums,
                              float* scalars) {
    static_assert(Geometry::SmallTMaximumSplits <= 256);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const auto index   = causal_partial_stat_index<Geometry>(q_head, token, tid, tokens);
    const float m      = tid < splits ? partial_m[index] : -CUDART_INF_F;
    const float warp_m = warp_max(m);
    if (lane == 0) warp_sums[warp] = warp_m;
    __syncthreads();
    if (warp == 0) {
        const float maximum = warp_max(tid < 8 ? warp_sums[tid] : -CUDART_INF_F);
        if (tid == 0) scalars[0] = maximum;
    }
    __syncthreads();
    const float maximum     = scalars[0];
    const float l           = tid < splits ? partial_l[index] : 0.0f;
    const float weight      = l > 0.0f && maximum > -CUDART_INF_F ? expf(m - maximum) : 0.0f;
    const float denominator = block_reduce_sum<256>(l * weight, warp_sums);
    if (tid == 0) scalars[1] = denominator;
    __syncthreads();
    const float total = scalars[1];
    if (tid < splits) weights[tid] = total > 0.0f ? weight : 0.0f;
    __syncthreads();
    return total;
}

template <typename Geometry, int DChunk, bool Int8, bool MultiBatch, bool Masked, bool Offset>
__launch_bounds__(256) __global__ void causal_attention_small_t_reduce_output_kernel(
    const float* partial_acc, const float* partial_m, const float* partial_l,
    const std::int32_t* positions, const std::int32_t* valid_columns, std::int32_t tokens,
    std::int32_t full_width, std::int32_t column_begin, std::int32_t batch_size,
    std::int32_t split_count, __nv_bfloat16* out) {
    static_assert(DChunk > 0 && DChunk <= kCausalHeadDim);

    const int q_head      = static_cast<int>(blockIdx.x);
    const int d_start     = static_cast<int>(blockIdx.y) * DChunk;
    const int flat_column = static_cast<int>(blockIdx.z);
    int batch             = 0;
    int token             = flat_column;
    if constexpr (MultiBatch) {
        batch = flat_column / tokens;
        token = flat_column - batch * tokens;
    }
    const int tid = threadIdx.x;
    if (q_head >= Geometry::QHeads || token >= tokens) { return; }
    if constexpr (MultiBatch) {
        if (batch >= batch_size) { return; }
    }

    if constexpr (Offset) { positions += column_begin; }
    if constexpr (MultiBatch) { positions += batch * full_width; }
    // INT8-family rows use the width-one arithmetic even inside wide verification/prefill.
    const int last_pos = positions[Int8 ? token : tokens - 1];
    int output_column  = token;
    if constexpr (Offset) { output_column += column_begin; }
    if constexpr (MultiBatch) { output_column += batch * full_width; }
    if constexpr (Masked) {
        const int absolute_column = token + (Offset ? column_begin : 0);
        if (absolute_column >= valid_columns[batch]) {
            if (tid < DChunk && d_start + tid < kCausalHeadDim)
                out[causal_q_index<Geometry>(q_head, d_start + tid, output_column)] =
                    __float2bfloat16(0.0f);
            return;
        }
    }


    if constexpr (MultiBatch) {
        const std::int64_t partial_acc_row = static_cast<std::int64_t>(batch) * kCausalHeadDim *
                                             Geometry::QHeads * tokens * split_count;
        const std::int64_t partial_stat_row =
            static_cast<std::int64_t>(batch) * Geometry::QHeads * tokens * split_count;
        partial_acc += partial_acc_row;
        partial_m += partial_stat_row;
        partial_l += partial_stat_row;
    }

    const int window = last_pos + 1;
    const int active_split_count =
        causal_small_t_active_splits<Geometry, Int8>(window, split_count, Int8 ? 1 : tokens);

    __shared__ float weights[256], warp_sums[8], scalars[2];
    const float head_l =
        causal_merge_split_statistics<Geometry>(partial_m, partial_l, q_head, token, tokens,
                                                active_split_count, weights, warp_sums, scalars);
    const int d = d_start + tid;
    if (tid >= DChunk || d >= kCausalHeadDim) return;
    float numerator = 0.0f;
    for (int split = 0; split < active_split_count; ++split) {
        if (weights[split] != 0.0f)
            numerator +=
                partial_acc[causal_partial_acc_index<Geometry>(q_head, d, token, split, tokens)] *
                weights[split];
    }

    const float value = (head_l > 0.0f) ? numerator / head_l : 0.0f;
    out[causal_q_index<Geometry>(q_head, d, output_column)] = __float2bfloat16(value);
}

} // namespace ninfer::ops
