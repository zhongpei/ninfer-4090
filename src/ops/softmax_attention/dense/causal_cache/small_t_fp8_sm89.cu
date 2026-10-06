// RTX 4090 native E4M3 causal small-T launch ownership.
#include "ops/softmax_attention/dense/causal_cache/launch.h"

#if defined(NINFER_SM89)

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/kv_cache/plane_types.h"
#include "ops/softmax_attention/dense/causal_cache/small_t_fp8.cuh"
#include "ops/softmax_attention/dense/causal_cache/small_t_fp8_sm89.cuh"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

template <typename Geometry, int TokenTile, bool MultiBatch, bool Masked, bool PreparedQ>
void launch_sm89_partial(const Tensor& q, const std::uint8_t* prepared_q,
                         const std::uint8_t* prepared_q_res, const float* prepared_q_scale,
                         const float* prepared_q_res_scale, const Tensor& positions, float scale,
                         PagedKVBatchLayerView cache, const CausalSmallTInvocation& invocation,
                         CausalAttentionExecutionEnvelope envelope, std::int32_t splits,
                         Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                         cudaStream_t stream) {
    constexpr int RowCount = TokenTile * Geometry::GroupSize;
    constexpr int RowTiles = (RowCount + 15) / 16;
    // Narrow T1/T2 rows use four consumer warps. This doubles each warp's PV slice while
    // preserving the legal FP8 fragment geometry and leaves more block-level residency headroom.
    constexpr int Warps = RowTiles == 1 ? 4 : RowTiles == 3 ? 12 : 8;
    constexpr int MinBlocks = RowTiles == 1 ? 2 : 1;
    // K0/V0 + K1/V1 + one consumer V^T scratch tile.
    constexpr std::size_t DynamicBytes = 5u * 32u * kCausalHeadDim;

    constexpr auto kStorage = KvCacheStorage::Fp8E4M3Row256;
    const auto* cache_k_ptr = static_cast<const KvKeyCodeT<kStorage>*>(cache.k_pages.data);
    const auto* cache_v_ptr = static_cast<const KvValueCodeT<kStorage>*>(cache.v_pages.data);
    const auto* k_scale_ptr = static_cast<const KvKeyScaleT<kStorage>*>(cache.k_scale_pages.data);
    const auto* v_scale_ptr = static_cast<const KvValueScaleT<kStorage>*>(cache.v_scale_pages.data);
    assert_kv_planes<kStorage, decltype(cache_k_ptr), decltype(cache_v_ptr),
                     decltype(k_scale_ptr), decltype(v_scale_ptr)>();

    const auto kernel =
        causal_attention_small_t_fp8_sm89_kernel<Geometry, TokenTile, Warps, MinBlocks,
                                                  MultiBatch, Masked, PreparedQ>;
    configure_cuda_device_once([&] {
        return cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    static_cast<int>(DynamicBytes));
    });

    const dim3 grid(Geometry::KVHeads, splits, invocation.batch_size);
    kernel<<<grid, Warps * 32, DynamicBytes, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), prepared_q, prepared_q_res,
        prepared_q_scale, prepared_q_res_scale,
        static_cast<const std::int32_t*>(positions.data),
        cache_k_ptr, cache_v_ptr, k_scale_ptr, v_scale_ptr,
        static_cast<const std::int32_t*>(cache.block_tables.data),
        invocation.valid_columns == nullptr
            ? nullptr
            : static_cast<const std::int32_t*>(invocation.valid_columns->data),
        invocation.table_rows == nullptr
            ? nullptr
            : static_cast<const std::int32_t*>(invocation.table_rows->data),
        cache.block_tables.ne[0], invocation.full_width, invocation.column_begin,
        static_cast<std::int32_t>(envelope.max_visible_keys), scale,
        static_cast<float*>(partial_acc.data), static_cast<float*>(partial_m.data),
        static_cast<float*>(partial_l.data));
    CUDA_CHECK(cudaGetLastError());
}

template <typename Geometry, bool MultiBatch, bool Masked>
void launch_sm89_reduce(const Tensor& positions, const CausalSmallTInvocation& invocation,
                        std::int32_t splits, const Tensor& partial_acc, const Tensor& partial_m,
                        const Tensor& partial_l, Tensor& out, cudaStream_t stream) {
    constexpr int Block  = 256;
    constexpr int DChunk = Geometry::QHeads == 24 ? 256 : 64;
    const auto launch = [&]<bool Offset>() {
        const dim3 grid(Geometry::QHeads, div_up(kCausalHeadDim, DChunk),
                        invocation.width * invocation.batch_size);
        causal_attention_small_t_fp8_reduce_output_kernel<
            Geometry, DChunk, MultiBatch, Masked, Offset, true>
            <<<grid, Block, 0, stream>>>(
                static_cast<const float*>(partial_acc.data),
                static_cast<const float*>(partial_m.data),
                static_cast<const float*>(partial_l.data),
                static_cast<const std::int32_t*>(positions.data),
                invocation.valid_columns == nullptr
                    ? nullptr
                    : static_cast<const std::int32_t*>(invocation.valid_columns->data),
                invocation.width, invocation.full_width, invocation.column_begin,
                invocation.batch_size, splits, static_cast<__nv_bfloat16*>(out.data));
    };
    if (invocation.column_begin == 0)
        launch.template operator()<false>();
    else
        launch.template operator()<true>();
    CUDA_CHECK(cudaGetLastError());
}

struct Sm89PreparedQView {
    std::uint8_t* main = nullptr;
    std::uint8_t* residual = nullptr;
    float* scale = nullptr;
    float* residual_scale = nullptr;
};

template <typename Geometry, int TokenTile>
Sm89PreparedQView prepare_sm89_q(const Tensor& q, const CausalSmallTInvocation& invocation,
                                 Tensor& workspace, cudaStream_t stream) {
    constexpr std::size_t D = kCausalHeadDim;
    const std::size_t rows = static_cast<std::size_t>(Geometry::QHeads) * TokenTile *
                             static_cast<std::size_t>(invocation.batch_size);
    const std::size_t code_bytes = rows * D;
    const std::size_t required = 2u * code_bytes + 2u * rows * sizeof(float);
    if (workspace.data == nullptr || workspace.bytes() < required) {
        throw std::logic_error("sm89 native FP8 prepared-Q workspace is too small");
    }

    auto* base = static_cast<std::uint8_t*>(workspace.data);
    Sm89PreparedQView view;
    view.main = base;
    view.residual = base + code_bytes;
    view.scale = reinterpret_cast<float*>(base + 2u * code_bytes);
    view.residual_scale = view.scale + rows;

    causal_attention_prepare_q_fp8_sm89_kernel<Geometry, TokenTile>
        <<<static_cast<unsigned>(rows), 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(q.data), invocation.full_width,
            invocation.column_begin, invocation.batch_size, view.main, view.residual,
            view.scale, view.residual_scale);
    CUDA_CHECK(cudaGetLastError());
    return view;
}

template <typename Geometry>
void dispatch_sm89(const Tensor& q, const Tensor& positions, float scale,
                   PagedKVBatchLayerView cache, const CausalSmallTInvocation& invocation,
                   CausalAttentionExecutionEnvelope envelope, Tensor& partial_acc,
                   Tensor& partial_m, Tensor& partial_l, Tensor& prepared_q_workspace,
                   Tensor& out, cudaStream_t stream) {
    const std::int32_t splits = causal_attention_split_capacity(
        Geometry::QHeads, invocation.width, cache.storage, envelope, invocation.batch_size);

    const auto partial =
        [&]<int Tokens, bool MultiBatch, bool Masked, bool PreparedQ>(const Sm89PreparedQView& pq) {
            launch_sm89_partial<Geometry, Tokens, MultiBatch, Masked, PreparedQ>(
                q, pq.main, pq.residual, pq.scale, pq.residual_scale, positions, scale, cache,
                invocation, envelope, splits, partial_acc, partial_m, partial_l, stream);
        };
    const auto metadata = [&]<int Tokens>() {
        const bool use_prepared_q = causal_attention_fp8_sm89_uses_prepared_q(splits);
        Sm89PreparedQView pq;
        if (use_prepared_q) {
            pq = prepare_sm89_q<Geometry, Tokens>(q, invocation, prepared_q_workspace, stream);
        }
        const auto route = [&]<bool MultiBatch, bool Masked>() {
            if (use_prepared_q)
                partial.template operator()<Tokens, MultiBatch, Masked, true>(pq);
            else
                partial.template operator()<Tokens, MultiBatch, Masked, false>(pq);
        };

        const bool masked = invocation.valid_columns != nullptr;
        if (invocation.batch_size == 1) {
            if (masked)
                route.template operator()<false, true>();
            else
                route.template operator()<false, false>();
        } else if (masked) {
            route.template operator()<true, true>();
        } else {
            route.template operator()<true, false>();
        }
    };

    switch (invocation.width) {
    case 1: metadata.template operator()<1>(); break;
    case 2: metadata.template operator()<2>(); break;
    case 3: metadata.template operator()<3>(); break;
    case 4: metadata.template operator()<4>(); break;
    case 5: metadata.template operator()<5>(); break;
    case 6: metadata.template operator()<6>(); break;
    case 7:
        if constexpr (Geometry::QHeads == 24) metadata.template operator()<7>();
        else throw std::invalid_argument("sm89 native FP8: unsupported query-row tile");
        break;
    case 8:
        if constexpr (Geometry::QHeads == 24) metadata.template operator()<8>();
        else throw std::invalid_argument("sm89 native FP8: unsupported query-row tile");
        break;
    default:
        throw std::invalid_argument("sm89 native FP8 small-T requires T in [1,8]");
    }

    const bool masked = invocation.valid_columns != nullptr;
    if (invocation.batch_size == 1) {
        if (masked)
            launch_sm89_reduce<Geometry, false, true>(positions, invocation, splits, partial_acc,
                                                      partial_m, partial_l, out, stream);
        else
            launch_sm89_reduce<Geometry, false, false>(positions, invocation, splits, partial_acc,
                                                       partial_m, partial_l, out, stream);
    } else if (masked) {
        launch_sm89_reduce<Geometry, true, true>(positions, invocation, splits, partial_acc,
                                                 partial_m, partial_l, out, stream);
    } else {
        launch_sm89_reduce<Geometry, true, false>(positions, invocation, splits, partial_acc,
                                                  partial_m, partial_l, out, stream);
    }
}

} // namespace

void causal_attention_small_t_fp8_sm89_launch(
    const Tensor& q, const Tensor&, const Tensor&, const Tensor& positions,
    const Tensor& valid_columns, const Tensor& table_rows, float scale,
    PagedKVBatchLayerView cache, CausalAttentionExecutionEnvelope envelope,
    std::int32_t column_begin, std::int32_t width, Tensor& partial_acc, Tensor& partial_m,
    Tensor& partial_l, Tensor& prepared_q_workspace, Tensor& out, cudaStream_t stream) {
    const CausalSmallTInvocation invocation{
        .valid_columns = valid_columns.data == nullptr ? nullptr : &valid_columns,
        .table_rows    = &table_rows,
        .full_width    = q.ne[2],
        .column_begin  = column_begin,
        .width         = width,
        .batch_size    = q.ne[3],
    };
    if (q.ne[1] == CausalD256H24Kv4::QHeads) {
        dispatch_sm89<CausalD256H24Kv4>(q, positions, scale, cache, invocation, envelope,
                                         partial_acc, partial_m, partial_l, prepared_q_workspace,
                                         out, stream);
        return;
    }
    dispatch_sm89<CausalD256H16Kv2>(q, positions, scale, cache, invocation, envelope,
                                     partial_acc, partial_m, partial_l, prepared_q_workspace,
                                     out, stream);
}

void causal_attention_cached_small_t_fp8_sm89_launch(
    const Tensor& q, const Tensor& positions, float scale, const PagedKVLayerView& cache,
    CausalAttentionExecutionEnvelope envelope, Tensor& partial_acc, Tensor& partial_m,
    Tensor& partial_l, Tensor& prepared_q_workspace, Tensor& out, cudaStream_t stream) {
    const CausalSmallTInvocation invocation{
        .valid_columns = nullptr,
        .table_rows    = nullptr,
        .full_width    = q.ne[2],
        .column_begin  = 0,
        .width         = q.ne[2],
        .batch_size    = 1,
    };
    PagedKVBatchLayerView batch_cache = single_row_paged_kv_batch_view(cache);
    if (q.ne[1] == CausalD256H24Kv4::QHeads) {
        dispatch_sm89<CausalD256H24Kv4>(q, positions, scale, batch_cache, invocation, envelope,
                                         partial_acc, partial_m, partial_l, prepared_q_workspace,
                                         out, stream);
        return;
    }
    dispatch_sm89<CausalD256H16Kv2>(q, positions, scale, batch_cache, invocation, envelope,
                                     partial_acc, partial_m, partial_l, prepared_q_workspace,
                                     out, stream);
}

} // namespace ninfer::ops::detail

#endif // NINFER_SM89
