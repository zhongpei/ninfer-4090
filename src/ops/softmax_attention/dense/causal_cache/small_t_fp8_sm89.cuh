#pragma once

// RTX 4090 / sm_89 native E4M3 causal small-T attention.
//
// Persistent FP8 KV rows carry one FP16 scale per D256 row. Hadamard-rotated Q uses a main
// E4M3 term and an independently scaled E4M3 residual per query row. Both QK contractions
// run as E4M3 x E4M3 -> FP32 Tensor Core MMA. For PV, V has a
// different scale for every key row, so a plain P_fp8 x V_fp8 contraction would be incorrect.
// Instead this kernel forms A_k = P_k * VScale_k and represents A with a main E4M3 term
// plus an independently scaled E4M3 residual per query row/tile. Algebraically:
//
//   V_k = VScale_k * VCode_k
//   A_k ~= AScale * Acode_k + ResidualScale * ResidualCode_k
//   sum(P_k * V_k) ~= AScale * sum(Acode_k * VCode_k)
//                    + ResidualScale * sum(ResidualCode_k * VCode_k)
//
// Q and A residuals are formed from the original FP32 values minus the decoded main terms.
// Their native contractions preserve private arithmetic accuracy without changing persistent
// KV representation. Softmax statistics continue to use unquantized P.
//
// The legacy FP8 kernel remains available for sm_86 and as a source-level fallback. This path
// intentionally stages raw V twice (row-major load + shared transpose) rather than widening it to
// FP16; that keeps the native Tensor Core contract explicit and avoids the old decode bottleneck.

#include "ops/kv_cache/fp8_e4m3_row_codec.cuh"
#include "ops/kv_cache/hadamard_d256.cuh"
#include "ops/softmax_attention/dense/causal_cache/small_t.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include <cstdint>

namespace ninfer::ops {

__device__ __forceinline__ unsigned causal_fp8_swizzle_address(unsigned base, unsigned column,
                                                               unsigned matrix, unsigned row) {
    return base + ((column | matrix) ^ row);
}

// A 32-byte row has only two 16-byte blocks. The D256 swizzle's row&7 would
// move bytes outside this row and alias neighboring P / transposed-V rows.
__device__ __forceinline__ void causal_fp8_store_byte_swizzled_32(std::uint8_t* tile,
                                                                int row, int column,
                                                                std::uint8_t code) {
    tile[row * 32 + (column ^ ((row & 1) << 4))] = code;
}

__device__ __forceinline__ float causal_fp8_decode_code(std::uint8_t code) {
    __nv_fp8_e4m3 value;
    value.__x = code;
    return static_cast<float>(value);
}

template <typename Geometry, int TokenTile, int WarpsPerCta, int MinBlocksPerSm,
          bool MultiBatch, bool Masked>
__launch_bounds__(WarpsPerCta * 32, MinBlocksPerSm) __global__
void causal_attention_small_t_fp8_sm89_kernel(
    const __nv_bfloat16* q, const std::int32_t* positions,
    const std::uint8_t* cache_k, const std::uint8_t* cache_v,
    const __half* cache_k_scale, const __half* cache_v_scale,
    const std::int32_t* block_tables, const std::int32_t* valid_columns,
    const std::int32_t* table_rows, std::int32_t table_stride,
    std::int32_t full_width, std::int32_t column_begin, std::int32_t logical_capacity,
    float attention_scale, float* partial_acc, float* partial_m, float* partial_l) {
    constexpr int Wc              = WarpsPerCta;
    constexpr int RowCount        = TokenTile * Geometry::GroupSize;
    constexpr int RowTiles        = (RowCount + 15) / 16;
    constexpr int Br              = RowTiles * 16;
    constexpr int Bc              = 32;
    constexpr int D               = kCausalHeadDim;
    constexpr int Threads         = Wc * 32;
    constexpr int QKKs            = D / 32;
    constexpr int QKNt            = Bc / 8;
    constexpr int ConsumerPerTile = Wc / RowTiles;
    constexpr int PVNtPerWarp     = D / (ConsumerPerTile * 8);
    constexpr int PVKs            = Bc / 32;
    constexpr int PageIds         = 64;
    constexpr float Log2E         = 1.4426950408889634074F;
    constexpr unsigned FullMask   = 0xffffffffU;

    static_assert(TokenTile >= 1 && TokenTile * Geometry::GroupSize <= 48);
    static_assert(RowTiles >= 1 && RowTiles <= 3);
    static_assert(Wc % RowTiles == 0);
    static_assert(QKKs == 8 && PVKs == 1);
    static_assert(PVNtPerWarp == 4 || PVNtPerWarp == 8 || PVNtPerWarp == 16);

    // Static data belongs to the query/probability side; K/V tiles use dynamic shared memory.
    __shared__ __align__(16) std::uint8_t q_fp8[Br * D];
    __shared__ __align__(16) std::uint8_t q_res_fp8[Br * D];
    __shared__ __align__(16) std::uint8_t p_fp8[Br * Bc];
    __shared__ __align__(16) std::uint8_t p_res_fp8[Br * Bc];
    __shared__ float q_scale_s[Br];
    __shared__ float q_res_scale_s[Br];
    __shared__ float p_scale_s[Br];
    __shared__ float p_res_scale_s[Br];
    __shared__ float alpha_s[Br];
    __shared__ __align__(16) __half k_scale_s[Bc];
    __shared__ __align__(16) __half v_scale_s[Bc];
    __shared__ std::int32_t physical_pages_s[PageIds];

    extern __shared__ __align__(16) std::uint8_t dynamic_raw[];
    std::uint8_t* k_fp8 = dynamic_raw;
    std::uint8_t* v_raw = k_fp8 + Bc * D;
    std::uint8_t* v_t   = v_raw + Bc * D; // [D,Bc], b16-swizzled in the K dimension.

    const int kv_head     = static_cast<int>(blockIdx.x);
    const int split       = static_cast<int>(blockIdx.y);
    const int batch       = MultiBatch ? static_cast<int>(blockIdx.z) : 0;
    const int split_count = static_cast<int>(gridDim.y);
    const int tid         = static_cast<int>(threadIdx.x);
    const int warp        = tid >> 5;
    const int lane        = tid & 31;

    int valid_tokens = TokenTile;
    if constexpr (Masked) {
        const int remaining = valid_columns[batch] - column_begin;
        valid_tokens        = remaining <= 0 ? 0 : min(remaining, TokenTile);
    }
    std::int64_t column_base = column_begin;
    if constexpr (MultiBatch) column_base += static_cast<std::int64_t>(batch) * full_width;
    q += static_cast<std::int64_t>(D) * Geometry::QHeads * column_base;
    positions += column_base;

    const int table_row = table_rows == nullptr ? 0 : table_rows[batch];
    const std::int32_t* block_table =
        block_tables + static_cast<std::int64_t>(table_row) * table_stride;
    if constexpr (MultiBatch) {
        partial_acc += static_cast<std::int64_t>(batch) * D * Geometry::QHeads * TokenTile *
                       split_count;
        partial_m += static_cast<std::int64_t>(batch) * Geometry::QHeads * TokenTile * split_count;
        partial_l += static_cast<std::int64_t>(batch) * Geometry::QHeads * TokenTile * split_count;
    }

    auto write_neutral = [&]() {
        for (int row = tid; row < RowCount; row += Threads) {
            int q_head = 0, token = 0;
            causal_small_t_tc_row_to_qt<Geometry>(row, TokenTile, kv_head, q_head, token);
            if (causal_valid_q_head<Geometry>(kv_head, q_head)) {
                partial_m[causal_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] =
                    -CUDART_INF_F;
                partial_l[causal_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] =
                    0.0F;
            }
        }
        for (int index = tid; index < RowCount * D; index += Threads) {
            const int row = index / D;
            const int d   = index - row * D;
            int q_head = 0, token = 0;
            causal_small_t_tc_row_to_qt<Geometry>(row, TokenTile, kv_head, q_head, token);
            if (causal_valid_q_head<Geometry>(kv_head, q_head)) {
                partial_acc[causal_partial_acc_index<Geometry>(q_head, d, token, split,
                                                               TokenTile)] = 0.0F;
            }
        }
    };
    if (kv_head < 0 || kv_head >= Geometry::KVHeads || split_count <= 0) return;
    if (valid_tokens == 0) {
        write_neutral();
        return;
    }

    const int first_pos = positions[0];
    const int last_pos  = positions[TokenTile - 1];
    if (first_pos < 0 || last_pos < 0 || last_pos >= logical_capacity) {
        write_neutral();
        return;
    }
    const int window = last_pos + 1;
    const int active_split_count =
        causal_small_t_quantized_active_splits<Geometry>(window, split_count, TokenTile);
    if (split >= active_split_count) return;

    const int logical_tiles = div_up(window, Bc);
    const int first_owned_tile = split * logical_tiles / active_split_count;
    const int end_owned_tile   = (split + 1) * logical_tiles / active_split_count;
    const int split_start      = first_owned_tile * Bc;
    const int split_end        = min(end_owned_tile * Bc, window);
    if (split_start >= split_end) {
        write_neutral();
        return;
    }
    const int first_tile = split_start;
    const int key_blocks = div_up(split_end - first_tile, Bc);
    const int first_page = first_tile >> kPagedKVPageShift;
    const int page_count = ((split_end - 1) >> kPagedKVPageShift) - first_page + 1;
    for (int page = tid; page < page_count; page += Threads)
        physical_pages_s[page] = block_table[first_page + page];
    __syncthreads();

    // Hadamard + per-query-row E4M3 quantization. Rows padding the MMA tile stay zero.
    for (int row = warp; row < Br; row += Wc) {
        float values[8]{};
        float local_absmax = 0.0F;
        if (row < RowCount) {
            int q_head = 0, token = 0;
            causal_small_t_tc_row_to_qt<Geometry>(row, TokenTile, kv_head, q_head, token);
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int d = lane + 32 * r;
                values[r] = __bfloat162float(q[causal_q_index<Geometry>(q_head, d, token)]);
            }
            normalized_hadamard_d256_inplace(values, lane);
#pragma unroll
            for (float value : values) local_absmax = fmaxf(local_absmax, fabsf(value));
        }
        const float absmax = warp_max(local_absmax, FullMask);
        const float qs     = absmax > 0.0F ? absmax / kKVCacheFp8MaxFinite : 0.0F;
        const float inv    = qs > 0.0F ? 1.0F / qs : 0.0F;
        float residual_absmax = 0.0F;
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int d = lane + 32 * r;
            const std::uint8_t code = kv_cache_fp8_quant_code(values[r], inv);
            causal_small_t_store_byte_swizzled(q_fp8, row, d, D / 2,
                                                code);
            values[r] -= causal_fp8_decode_code(code) * qs;
            residual_absmax = fmaxf(residual_absmax, fabsf(values[r]));
        }
        residual_absmax = warp_max(residual_absmax, FullMask);
        const float qrs = residual_absmax > 0.0F
                              ? residual_absmax / kKVCacheFp8MaxFinite : 0.0F;
        const float rinv = qrs > 0.0F ? 1.0F / qrs : 0.0F;
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int d = lane + 32 * r;
            causal_small_t_store_byte_swizzled(q_res_fp8, row, d, D / 2,
                                               kv_cache_fp8_quant_code(values[r], rinv));
        }
        if (lane == 0) {
            q_scale_s[row] = qs;
            q_res_scale_s[row] = qrs;
        }
    }
    __syncthreads();

    auto issue_tile = [&](int tile_k0, int physical_page) {
        for (int key_l = tid; key_l < Bc; key_l += Threads) {
            const int key = tile_k0 + key_l;
            if (key >= split_start && key < split_end) {
                const auto off = kv_cache_fp8_scale_index<Geometry>(
                    physical_page, kv_head, key & kPagedKVPageMask);
                k_scale_s[key_l] = cache_k_scale[off];
                v_scale_s[key_l] = cache_v_scale[off];
            } else {
                k_scale_s[key_l] = __float2half_rn(0.0F);
                v_scale_s[key_l] = __float2half_rn(0.0F);
            }
        }
#pragma unroll 1
        for (int chunk = tid; chunk < Bc * (D / 16); chunk += Threads) {
            const int key_l = chunk / (D / 16);
            const int dc    = chunk - key_l * (D / 16);
            const int d     = dc * 16;
            const int key   = tile_k0 + key_l;
            const int col_b16 = d >> 1;
            std::uint8_t* kd =
                k_fp8 + (key_l * (D / 2) + causal_small_t_tc_swz(key_l, col_b16)) * 2;
            std::uint8_t* vd = v_raw + key_l * D + d;
            if (key >= split_start && key < split_end) {
                const auto off = kv_cache_fp8_code_index<Geometry>(
                    physical_page, kv_head, d, key & kPagedKVPageMask);
                cp_async<16, Cache::cg>(kd, cache_k + off);
                cp_async<16, Cache::cg>(vd, cache_v + off);
            } else {
                store_vec(kd, make_int4(0, 0, 0, 0));
                store_vec(vd, make_int4(0, 0, 0, 0));
            }
        }
        cp_commit();
    };

    int physical_page = physical_pages_s[0];
    issue_tile(first_tile, physical_page);
    cp_wait<0>();
    __syncthreads();

    float acc[PVNtPerWarp][4]{};
    float m0 = -CUDART_INF_F, m1 = -CUDART_INF_F;
    float l0 = 0.0F, l1 = 0.0F;

    const int gid      = lane >> 2;
    const int lid      = lane & 3;
    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int b_rin    = lane & 7;
    const int b_koff   = ((lane >> 3) & 1) << 3;

    for (int kb = 0; kb < key_blocks; ++kb) {
        const int k0 = first_tile + kb * Bc;

        // Transpose V code bytes into [D,Bc]. A b16-swizzled row then represents one output
        // dimension with the 32 contraction keys contiguous, which is exactly the B operand of
        // m16n8k32 row.col FP8 MMA.
        for (int item = tid; item < D * Bc; item += Threads) {
            const int d     = item / Bc;
            const int key_l = item - d * Bc;
            causal_fp8_store_byte_swizzled_32(v_t, d, key_l, v_raw[key_l * D + d]);
        }
        __syncthreads();

        if (warp < RowTiles) {
            const int row_base = warp * 16;
            float score[QKNt][4]{};
            float residual_score[QKNt][4]{};
            const unsigned q_lane_base =
                smem_addr(q_fp8) + static_cast<unsigned>((row_base + a_rowoff) * D);
            const unsigned q_as = static_cast<unsigned>((a_mat >> 1) << 4);
            const unsigned q_r  = static_cast<unsigned>(a_rin << 4);
            const unsigned k_lane_base =
                smem_addr(k_fp8) + static_cast<unsigned>(b_rin * D + (lane >> 4) * (8 * D));
            const unsigned k_as = static_cast<unsigned>((b_koff >> 3) << 4);
            const unsigned k_r  = static_cast<unsigned>(b_rin << 4);

#pragma unroll
            for (int kk = 0; kk < QKKs; ++kk) {
                const unsigned ck = static_cast<unsigned>(kk << 5);
                unsigned af[4];
                unsigned arf[4];
                ldmatrix_x4(af[0], af[1], af[2], af[3],
                            causal_fp8_swizzle_address(q_lane_base, ck, q_as, q_r));
                ldmatrix_x4(arf[0], arf[1], arf[2], arf[3],
                            causal_fp8_swizzle_address(
                                smem_addr(q_res_fp8) + static_cast<unsigned>((row_base + a_rowoff) * D),
                                ck, q_as, q_r));
#pragma unroll
                for (int nt = 0; nt < QKNt; ++nt) {
                    unsigned bf[2];
                    ldmatrix_x2(
                        bf[0], bf[1],
                        causal_fp8_swizzle_address(
                            k_lane_base + static_cast<unsigned>(nt * 8 * D), ck, k_as, k_r));
                    mma_fp8_e4m3(score[nt][0], score[nt][1], score[nt][2], score[nt][3],
                                 af[0], af[1], af[2], af[3], bf[0], bf[1]);
                    mma_fp8_e4m3(residual_score[nt][0], residual_score[nt][1],
                                 residual_score[nt][2], residual_score[nt][3],
                                 arf[0], arf[1], arf[2], arf[3], bf[0], bf[1]);
                }
            }

            const int row0 = row_base + gid;
            const int row1 = row0 + 8;
            const float qs0 = q_scale_s[row0];
            const float qs1 = q_scale_s[row1];
            const float qrs0 = q_res_scale_s[row0];
            const float qrs1 = q_res_scale_s[row1];
            int q_head0 = 0, token0 = 0, q_head1 = 0, token1 = 0;
            causal_small_t_tc_row_to_qt<Geometry>(row0, TokenTile, kv_head, q_head0, token0);
            causal_small_t_tc_row_to_qt<Geometry>(row1, TokenTile, kv_head, q_head1, token1);
            const int qabs0 = row0 < RowCount ? positions[token0] : -1;
            const int qabs1 = row1 < RowCount ? positions[token1] : -1;
            float bm0 = -CUDART_INF_F, bm1 = -CUDART_INF_F;
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const int key0 = nt * 8 + 2 * lid;
                const int key1 = key0 + 1;
                const float ks0 = __half2float(k_scale_s[key0]);
                const float ks1 = __half2float(k_scale_s[key1]);
                score[nt][0] = (score[nt][0] * qs0 + residual_score[nt][0] * qrs0) * ks0;
                score[nt][1] = (score[nt][1] * qs0 + residual_score[nt][1] * qrs0) * ks1;
                score[nt][2] = (score[nt][2] * qs1 + residual_score[nt][2] * qrs1) * ks0;
                score[nt][3] = (score[nt][3] * qs1 + residual_score[nt][3] * qrs1) * ks1;
                const int abs0 = k0 + key0;
                const int abs1 = k0 + key1;
                score[nt][0] = row0 < RowCount && abs0 < split_end && abs0 <= qabs0
                                   ? score[nt][0] * attention_scale
                                   : -CUDART_INF_F;
                score[nt][1] = row0 < RowCount && abs1 < split_end && abs1 <= qabs0
                                   ? score[nt][1] * attention_scale
                                   : -CUDART_INF_F;
                score[nt][2] = row1 < RowCount && abs0 < split_end && abs0 <= qabs1
                                   ? score[nt][2] * attention_scale
                                   : -CUDART_INF_F;
                score[nt][3] = row1 < RowCount && abs1 < split_end && abs1 <= qabs1
                                   ? score[nt][3] * attention_scale
                                   : -CUDART_INF_F;
                bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
                bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
            }
            bm0 = warp_max<4>(bm0, FullMask);
            bm1 = warp_max<4>(bm1, FullMask);
            const float nm0    = fmaxf(m0, bm0);
            const float nm1    = fmaxf(m1, bm1);
            const float alpha0 = m0 == -CUDART_INF_F ? 0.0F : exp2_approx((m0 - nm0) * Log2E);
            const float alpha1 = m1 == -CUDART_INF_F ? 0.0F : exp2_approx((m1 - nm1) * Log2E);

            float bl0 = 0.0F, bl1 = 0.0F;
            float scaled_abs0 = 0.0F, scaled_abs1 = 0.0F;
            float probs[QKNt][4];
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const int key0 = nt * 8 + 2 * lid;
                const int key1 = key0 + 1;
                probs[nt][0] = score[nt][0] > -CUDART_INF_F
                                   ? exp2_approx((score[nt][0] - nm0) * Log2E)
                                   : 0.0F;
                probs[nt][1] = score[nt][1] > -CUDART_INF_F
                                   ? exp2_approx((score[nt][1] - nm0) * Log2E)
                                   : 0.0F;
                probs[nt][2] = score[nt][2] > -CUDART_INF_F
                                   ? exp2_approx((score[nt][2] - nm1) * Log2E)
                                   : 0.0F;
                probs[nt][3] = score[nt][3] > -CUDART_INF_F
                                   ? exp2_approx((score[nt][3] - nm1) * Log2E)
                                   : 0.0F;
                bl0 += probs[nt][0] + probs[nt][1];
                bl1 += probs[nt][2] + probs[nt][3];
                const float vs0 = __half2float(v_scale_s[key0]);
                const float vs1 = __half2float(v_scale_s[key1]);
                scaled_abs0 = fmaxf(scaled_abs0, fabsf(probs[nt][0] * vs0));
                scaled_abs0 = fmaxf(scaled_abs0, fabsf(probs[nt][1] * vs1));
                scaled_abs1 = fmaxf(scaled_abs1, fabsf(probs[nt][2] * vs0));
                scaled_abs1 = fmaxf(scaled_abs1, fabsf(probs[nt][3] * vs1));
            }
            scaled_abs0 = warp_max<4>(scaled_abs0, FullMask);
            scaled_abs1 = warp_max<4>(scaled_abs1, FullMask);
            const float ps0 = scaled_abs0 > 0.0F ? scaled_abs0 / kKVCacheFp8MaxFinite : 0.0F;
            const float ps1 = scaled_abs1 > 0.0F ? scaled_abs1 / kKVCacheFp8MaxFinite : 0.0F;
            const float pinv0 = ps0 > 0.0F ? 1.0F / ps0 : 0.0F;
            const float pinv1 = ps1 > 0.0F ? 1.0F / ps1 : 0.0F;
            float residual_abs0 = 0.0F, residual_abs1 = 0.0F;
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const int key0 = nt * 8 + 2 * lid;
                const int key1 = key0 + 1;
                const float vs0 = __half2float(v_scale_s[key0]);
                const float vs1 = __half2float(v_scale_s[key1]);
                const float a0 = probs[nt][0] * vs0;
                const float a1 = probs[nt][1] * vs1;
                const float a2 = probs[nt][2] * vs0;
                const float a3 = probs[nt][3] * vs1;
                const std::uint8_t c0 = kv_cache_fp8_quant_code(a0, pinv0);
                const std::uint8_t c1 = kv_cache_fp8_quant_code(a1, pinv0);
                const std::uint8_t c2 = kv_cache_fp8_quant_code(a2, pinv1);
                const std::uint8_t c3 = kv_cache_fp8_quant_code(a3, pinv1);
                causal_fp8_store_byte_swizzled_32(
                    p_fp8, row0, key0, c0);
                causal_fp8_store_byte_swizzled_32(
                    p_fp8, row0, key1, c1);
                causal_fp8_store_byte_swizzled_32(
                    p_fp8, row1, key0, c2);
                causal_fp8_store_byte_swizzled_32(
                    p_fp8, row1, key1, c3);
                probs[nt][0] = a0 - causal_fp8_decode_code(c0) * ps0;
                probs[nt][1] = a1 - causal_fp8_decode_code(c1) * ps0;
                probs[nt][2] = a2 - causal_fp8_decode_code(c2) * ps1;
                probs[nt][3] = a3 - causal_fp8_decode_code(c3) * ps1;
                residual_abs0 = fmaxf(residual_abs0, fmaxf(fabsf(probs[nt][0]), fabsf(probs[nt][1])));
                residual_abs1 = fmaxf(residual_abs1, fmaxf(fabsf(probs[nt][2]), fabsf(probs[nt][3])));
            }
            residual_abs0 = warp_max<4>(residual_abs0, FullMask);
            residual_abs1 = warp_max<4>(residual_abs1, FullMask);
            const float prs0 = residual_abs0 > 0.0F
                                   ? residual_abs0 / kKVCacheFp8MaxFinite : 0.0F;
            const float prs1 = residual_abs1 > 0.0F
                                   ? residual_abs1 / kKVCacheFp8MaxFinite : 0.0F;
            const float prinv0 = prs0 > 0.0F ? 1.0F / prs0 : 0.0F;
            const float prinv1 = prs1 > 0.0F ? 1.0F / prs1 : 0.0F;
            if (lid == 0) {
                p_scale_s[row0] = ps0;
                p_scale_s[row1] = ps1;
                p_res_scale_s[row0] = prs0;
                p_res_scale_s[row1] = prs1;
                alpha_s[row0]   = alpha0;
                alpha_s[row1]   = alpha1;
            }
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const int key0 = nt * 8 + 2 * lid;
                const int key1 = key0 + 1;
                causal_fp8_store_byte_swizzled_32(
                    p_res_fp8, row0, key0, kv_cache_fp8_quant_code(probs[nt][0], prinv0));
                causal_fp8_store_byte_swizzled_32(
                    p_res_fp8, row0, key1, kv_cache_fp8_quant_code(probs[nt][1], prinv0));
                causal_fp8_store_byte_swizzled_32(
                    p_res_fp8, row1, key0, kv_cache_fp8_quant_code(probs[nt][2], prinv1));
                causal_fp8_store_byte_swizzled_32(
                    p_res_fp8, row1, key1, kv_cache_fp8_quant_code(probs[nt][3], prinv1));
            }
            bl0 = warp_sum<4>(bl0, FullMask);
            bl1 = warp_sum<4>(bl1, FullMask);
            l0  = __fmaf_rn(l0, alpha0, bl0);
            l1  = __fmaf_rn(l1, alpha1, bl1);
            m0  = nm0;
            m1  = nm1;
        }
        __syncthreads();

        const int consumer_tile  = warp % RowTiles;
        const int consumer_slice = warp / RowTiles;
        const int row_base       = consumer_tile * 16;
        const float alpha0       = alpha_s[row_base + gid];
        const float alpha1       = alpha_s[row_base + gid + 8];
#pragma unroll
        for (int n = 0; n < PVNtPerWarp; ++n) {
            acc[n][0] *= alpha0;
            acc[n][1] *= alpha0;
            acc[n][2] *= alpha1;
            acc[n][3] *= alpha1;
        }

        const unsigned p_lane_base =
            smem_addr(p_fp8) + static_cast<unsigned>((row_base + a_rowoff) * Bc);
        const unsigned p_as = static_cast<unsigned>((a_mat >> 1) << 4);
        const unsigned p_r  = static_cast<unsigned>((a_rin & 1) << 4);
        unsigned pf[4];
        unsigned prf[4];
        ldmatrix_x4(pf[0], pf[1], pf[2], pf[3],
                    causal_fp8_swizzle_address(p_lane_base, 0u, p_as, p_r));
        ldmatrix_x4(prf[0], prf[1], prf[2], prf[3],
                    causal_fp8_swizzle_address(
                        smem_addr(p_res_fp8) + static_cast<unsigned>((row_base + a_rowoff) * Bc),
                        0u, p_as, p_r));

#pragma unroll
        for (int n = 0; n < PVNtPerWarp; ++n) {
            const int global_n = consumer_slice * PVNtPerWarp + n;
            const unsigned v_lane_base =
                smem_addr(v_t) +
                static_cast<unsigned>(b_rin * Bc + (lane >> 4) * (8 * Bc) + global_n * 8 * Bc);
            const unsigned v_as = static_cast<unsigned>((b_koff >> 3) << 4);
            const unsigned v_r  = static_cast<unsigned>((b_rin & 1) << 4);
            unsigned vf[2];
            ldmatrix_x2(vf[0], vf[1],
                        causal_fp8_swizzle_address(v_lane_base, 0u, v_as, v_r));
            float tile0 = 0.0F, tile1 = 0.0F, tile2 = 0.0F, tile3 = 0.0F;
            mma_fp8_e4m3(tile0, tile1, tile2, tile3, pf[0], pf[1], pf[2], pf[3],
                         vf[0], vf[1]);
            float r_tile0 = 0.0F, r_tile1 = 0.0F, r_tile2 = 0.0F, r_tile3 = 0.0F;
            mma_fp8_e4m3(r_tile0, r_tile1, r_tile2, r_tile3,
                         prf[0], prf[1], prf[2], prf[3], vf[0], vf[1]);
            acc[n][0] += tile0 * p_scale_s[row_base + gid]
                            + r_tile0 * p_res_scale_s[row_base + gid];
            acc[n][1] += tile1 * p_scale_s[row_base + gid]
                            + r_tile1 * p_res_scale_s[row_base + gid];
            acc[n][2] += tile2 * p_scale_s[row_base + gid + 8]
                            + r_tile2 * p_res_scale_s[row_base + gid + 8];
            acc[n][3] += tile3 * p_scale_s[row_base + gid + 8]
                            + r_tile3 * p_res_scale_s[row_base + gid + 8];
        }

        const bool has_next = kb + 1 < key_blocks;
        if (has_next) {
            const int next_k0 = k0 + Bc;
            if ((next_k0 & kPagedKVPageMask) == 0)
                physical_page = physical_pages_s[(next_k0 >> kPagedKVPageShift) - first_page];
            issue_tile(next_k0, physical_page);
            cp_wait<0>();
        }
        __syncthreads();
    }

    if (warp < RowTiles && lid == 0) {
        const int row0 = warp * 16 + gid;
        const int row1 = row0 + 8;
        if (row0 < RowCount) {
            int q_head = 0, token = 0;
            causal_small_t_tc_row_to_qt<Geometry>(row0, TokenTile, kv_head, q_head, token);
            partial_m[causal_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] = m0;
            partial_l[causal_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] = l0;
        }
        if (row1 < RowCount) {
            int q_head = 0, token = 0;
            causal_small_t_tc_row_to_qt<Geometry>(row1, TokenTile, kv_head, q_head, token);
            partial_m[causal_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] = m1;
            partial_l[causal_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] = l1;
        }
    }
#pragma unroll
    for (int n = 0; n < PVNtPerWarp; ++n) {
        const int consumer_tile = warp % RowTiles;
        const int consumer_slice = warp / RowTiles;
        const int row_base = consumer_tile * 16;
        const int d0 = (consumer_slice * PVNtPerWarp + n) * 8 + 2 * lid;
        const int row0 = row_base + gid;
        const int row1 = row0 + 8;
        if (row0 < RowCount) {
            int q_head = 0, token = 0;
            causal_small_t_tc_row_to_qt<Geometry>(row0, TokenTile, kv_head, q_head, token);
            const auto dst =
                causal_partial_acc_index<Geometry>(q_head, d0, token, split, TokenTile);
            *reinterpret_cast<float2*>(&partial_acc[dst]) = make_float2(acc[n][0], acc[n][1]);
        }
        if (row1 < RowCount) {
            int q_head = 0, token = 0;
            causal_small_t_tc_row_to_qt<Geometry>(row1, TokenTile, kv_head, q_head, token);
            const auto dst =
                causal_partial_acc_index<Geometry>(q_head, d0, token, split, TokenTile);
            *reinterpret_cast<float2*>(&partial_acc[dst]) = make_float2(acc[n][2], acc[n][3]);
        }
    }
}

} // namespace ninfer::ops
