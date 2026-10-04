#include "core/weight.h"
#include "ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_kernels.h"
#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/hadamard_transform.cuh"

#include <cuda_bf16.h>
#include <mma.h>

namespace ninfer::ops::detail {
namespace {
constexpr int D = 5120, H = 48, Tile = 16, Splits = 10, Warps = 4;

// Fixed token and K tiles give every query the same arithmetic at every execution width.
// The control operand is the unrounded FP32 x*(1+w), represented as a BF16 high part plus
// its BF16 residual. Projecting both parts avoids making the public BF16 h a control boundary.
__global__ __launch_bounds__(128) void gdn_norm_control_partial(
    const __nv_bfloat16* x, const __nv_bfloat16* nw, const __nv_bfloat16* aw,
    const __nv_bfloat16* bw, float* partial, int tokens) {
    namespace mma = nvcuda::wmma;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int token0 = blockIdx.x * Tile, head0 = blockIdx.y * 16;
    const int split = blockIdx.z;
    __shared__ __align__(32) __nv_bfloat16 operands[Warps][4][16 * 32];
    __shared__ __align__(32) float products[Warps][2][16 * 16];
    auto* hi = operands[warp][0];
    auto* lo = operands[warp][1];
    auto* a = operands[warp][2];
    auto* b = operands[warp][3];
    mma::fragment<mma::accumulator, 16, 16, 16, float> ah, al, bh, bl;
    mma::fill_fragment(ah, 0.0f);
    mma::fill_fragment(al, 0.0f);
    mma::fill_fragment(bh, 0.0f);
    mma::fill_fragment(bl, 0.0f);
#pragma unroll
    for (int step = 0; step < 4; ++step) {
        const int k0 = split * 512 + warp * 128 + step * 32;
        for (int vec = lane; vec < 16 * 32 / 8; vec += 32) {
            const int i = vec * 8, row = i / 32, k = i % 32;
            int4 high{}, low{};
            if (token0 + row < tokens) {
                const int4 xv = load_vec<int4>(x + std::int64_t(token0 + row) * D + k0 + k);
                const int4 nv = load_vec<int4>(nw + k0 + k);
                const auto* xp = reinterpret_cast<const unsigned*>(&xv);
                const auto* np = reinterpret_cast<const unsigned*>(&nv);
                auto* hp = reinterpret_cast<__nv_bfloat162*>(&high);
                auto* lp = reinterpret_cast<__nv_bfloat162*>(&low);
#pragma unroll
                for (int p = 0; p < 4; ++p) {
                    const float2 v = bf16x2_bits_to_float2(xp[p]);
                    const float2 n = bf16x2_bits_to_float2(np[p]);
                    const float z0 = v.x * (1.0f + n.x), z1 = v.y * (1.0f + n.y);
                    hp[p] = __floats2bfloat162_rn(z0, z1);
                    const float2 rounded = __bfloat1622float2(hp[p]);
                    lp[p] = __floats2bfloat162_rn(z0 - rounded.x, z1 - rounded.y);
                }
            }
            store_vec(hi + i, high);
            store_vec(lo + i, low);
            store_vec(a + i, load_vec<int4>(aw + (head0 + row) * D + k0 + k));
            store_vec(b + i, load_vec<int4>(bw + (head0 + row) * D + k0 + k));
        }
        __syncwarp();
#pragma unroll
        for (int offset = 0; offset < 32; offset += 16) {
            mma::fragment<mma::matrix_a, 16, 16, 16, __nv_bfloat16, mma::row_major> af, bf;
            mma::fragment<mma::matrix_b, 16, 16, 16, __nv_bfloat16, mma::col_major> hf, lf;
            mma::load_matrix_sync(af, a + offset, 32);
            mma::load_matrix_sync(bf, b + offset, 32);
            mma::load_matrix_sync(hf, hi + offset, 32);
            mma::load_matrix_sync(lf, lo + offset, 32);
            mma::mma_sync(ah, af, hf, ah);
            mma::mma_sync(al, af, lf, al);
            mma::mma_sync(bh, bf, hf, bh);
            mma::mma_sync(bl, bf, lf, bl);
        }
        __syncwarp();
    }
    for (int i = 0; i < ah.num_elements; ++i) {
        ah.x[i] += al.x[i];
        bh.x[i] += bl.x[i];
    }
    mma::store_matrix_sync(products[warp][0], ah, 16, mma::mem_row_major);
    mma::store_matrix_sync(products[warp][1], bh, 16, mma::mem_row_major);
    __syncthreads();
    for (int i = threadIdx.x; i < 16 * 16; i += 128) {
        const int head = head0 + i / 16, token = token0 + i % 16;
        if (token >= tokens) { continue; }
        float av = 0.0f, bv = 0.0f;
#pragma unroll
        for (int w = 0; w < Warps; ++w) {
            av += products[w][0][i];
            bv += products[w][1][i];
        }
        const auto index = (std::int64_t(split) * tokens + token) * 2 * H + head;
        partial[index] = av;
        partial[index + H] = bv;
    }
}

template <bool Rotate>
__global__ __launch_bounds__(512) void gdn_norm_control_finish(
    const __nv_bfloat16* x, const __nv_bfloat16* nw, const float* partial,
    const float* alog, const float* bias, const __nv_bfloat16* signs,
    __nv_bfloat16* h, float* g, float* beta, int tokens, float eps) {
    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32, token = blockIdx.x;
    const auto row = std::int64_t(token) * D;
    float sum = 0.0f;
    for (int base = tid * 8; base < D; base += 512 * 8) {
        const int4 xv = load_vec<int4>(x + row + base);
        const auto* xp = reinterpret_cast<const unsigned*>(&xv);
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const float2 v = bf16x2_bits_to_float2(xp[p]);
            sum = fmaf(v.x, v.x, sum);
            sum = fmaf(v.y, v.y, sum);
        }
    }
    sum = warp_reduce_sum(sum);
    __shared__ float sums[16], inverse;
    if (lane == 0) { sums[warp] = sum; }
    __syncthreads();
    if (tid == 0) {
        sum = 0.0f;
#pragma unroll
        for (int w = 0; w < 16; ++w) { sum += sums[w]; }
        inverse = rsqrtf(sum / D + eps);
    }
    __syncthreads();
    if (tid < H) {
        float a = 0.0f, b = 0.0f;
#pragma unroll
        for (int split = 0; split < Splits; ++split) {
            const auto index = (std::int64_t(split) * tokens + token) * 2 * H + tid;
            a += partial[index];
            b += partial[index + H];
        }
        const auto index = std::int64_t(token) * H + tid;
        g[index] = -expf(alog[tid]) * softplus(a * inverse + bias[tid]);
        beta[index] = sigmoid(b * inverse);
    }
    if constexpr (Rotate) {
        constexpr int Groups = 16 / kHadamardQuarterWarps;
        __shared__ HadamardQuarterShared transform[Groups];
        const int group = warp / kHadamardQuarterWarps, quarter = warp % kHadamardQuarterWarps;
#pragma unroll
        for (int round = 0; round < 2; ++round) {
            const int block = round * Groups + group;
            if (block < D / kHadamardTransformBlock) {
                const int base = block * kHadamardTransformBlock;
                const int offset = base + quarter * 32 * 8 + lane * 8;
                float xv[8], nv[8], v[8];
                hadamard_unpack8(load_vec<uint4>(x + row + offset), xv);
                hadamard_unpack8(load_vec<uint4>(nw + offset), nv);
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    v[j] = __bfloat162float(__float2bfloat16_rn(xv[j] * inverse * (1 + nv[j])));
                }
                hadamard_quarter_forward_store(v, signs + base, h + row + base, transform[group],
                                               quarter, lane, [] { __syncthreads(); });
            } else {
                __syncthreads();
                __syncthreads();
            }
        }
    } else {
        for (int base = tid * 8; base < D; base += 512 * 8) {
            const int4 xv = load_vec<int4>(x + row + base);
            const int4 nv = load_vec<int4>(nw + base);
            const auto* xp = reinterpret_cast<const unsigned*>(&xv);
            const auto* np = reinterpret_cast<const unsigned*>(&nv);
            int4 output;
            auto* packed = reinterpret_cast<__nv_bfloat162*>(&output);
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                const float2 v = bf16x2_bits_to_float2(xp[p]);
                const float2 n = bf16x2_bits_to_float2(np[p]);
                packed[p] = __floats2bfloat162_rn(v.x * inverse * (1 + n.x),
                                                v.y * inverse * (1 + n.y));
            }
            store_vec(h + row + base, output);
        }
    }
}

} // namespace

void bf16_gdn_norm_gating_proj_27_launch(
    const Tensor& x, const Tensor& norm_weight, float eps, Tensor& h, const Weight& a_weight,
    const Weight& b_weight, const Tensor& alog, const Tensor& bias, Tensor& g, Tensor& beta,
    const Tensor* signs, void* workspace, cudaStream_t stream) {
    const int tokens = x.ne[1];
    auto* partial = static_cast<float*>(workspace);
    gdn_norm_control_partial<<<dim3((tokens + Tile - 1) / Tile, 3, Splits), 128, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const __nv_bfloat16*>(norm_weight.data),
        static_cast<const __nv_bfloat16*>(a_weight.qdata),
        static_cast<const __nv_bfloat16*>(b_weight.qdata), partial, tokens);
    const auto finish = [&]<bool Rotate>() {
        gdn_norm_control_finish<Rotate><<<tokens, 512, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const __nv_bfloat16*>(norm_weight.data), partial,
            static_cast<const float*>(alog.data), static_cast<const float*>(bias.data),
            Rotate ? static_cast<const __nv_bfloat16*>(signs->data) : nullptr,
            static_cast<__nv_bfloat16*>(h.data), static_cast<float*>(g.data),
            static_cast<float*>(beta.data), tokens, eps);
    };
    if (signs != nullptr) {
        finish.template operator()<true>();
    } else {
        finish.template operator()<false>();
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
