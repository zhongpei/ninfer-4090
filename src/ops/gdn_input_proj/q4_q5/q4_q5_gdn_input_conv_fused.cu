#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_kernels.h"

#include "core/device.h"
#include "ops/common/small_t_layout.cuh"
#include "ops/gdn_input_proj/gdn_conv.cuh"
#include "ops/linear/q4/q4_ksplit_mma.cuh"
#include "ops/linear/q5/q5_small_t_mma.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kHidden      = 5120;
constexpr std::int32_t kQkRows      = 4096;
constexpr std::int32_t kQueryRows   = 2048;
constexpr std::int32_t kKeyRows     = 2048;
constexpr std::int32_t kValueRows   = 6144;
constexpr std::int32_t kZRows       = 6144;
constexpr std::int32_t kValueZRows  = kValueRows + kZRows;
constexpr std::int32_t kChannels    = kQkRows + kValueRows;

struct GdnQkGeometry {
    static constexpr int kOutputRows   = kQkRows;
    static constexpr int kInputRows    = kHidden;
    static constexpr int kGroupsPerRow = kHidden / 64;
};

template <int Tokens, class Publish>
GdnConvEpilogue<Publish> make_conv_epilogue(
    const Tensor& conv_weight, const Tensor& conv_states, const Tensor& valid_columns,
    const Tensor& initial_state_slots, Tensor& query, Tensor& key, Tensor& value,
    std::int32_t global_row_offset, Publish publish) {
    return {
        static_cast<const __nv_bfloat16*>(conv_weight.data),
        static_cast<const __nv_bfloat16*>(conv_states.data),
        static_cast<const std::int32_t*>(initial_state_slots.data),
        valid_columns.data == nullptr ? nullptr
                                      : static_cast<const std::int32_t*>(valid_columns.data),
        static_cast<__nv_bfloat16*>(query.data),
        static_cast<__nv_bfloat16*>(key.data),
        static_cast<__nv_bfloat16*>(value.data),
        kChannels,
        kQueryRows,
        kKeyRows,
        kValueRows,
        global_row_offset,
        Tokens,
        0,
        publish,
    };
}

// Current small-T MMA kernels reduce K into FP32 registers, then historically round the projection
// through a global BF16 [channels,T] matrix before gdn_projected_conv reads it again. The tile
// epilogue consumes the reduced accumulator inside the same CTA instead. q4_ksplit_mma/q5_small_t
// reuse their retired K-reduction shared arena as [row,token] FP32 storage before calling this.
template <int Tokens, class Publish>
struct Q4GdnConvTileEpilogue {
    static constexpr bool kIsTileEpilogue = true;
    GdnConvEpilogue<Publish> conv;

    template <int ActiveCols, int TileCols, int RowsPerCta>
    __device__ __forceinline__ void store_tile(const float* dense, int global_row0, int local_row,
                                               int live_columns) const {
        static_assert(Tokens <= TileCols);
        static_assert(ActiveCols >= Tokens);
        if (local_row >= RowsPerCta) return;
        float projected[Tokens];
#pragma unroll
        for (int token = 0; token < Tokens; ++token) {
            projected[token] = token < live_columns ? dense[local_row * TileCols + token] : 0.0F;
        }
        conv.store(global_row0 + local_row, projected);
    }
};

template <int Tokens, class Publish>
struct Q5GdnConvTileEpilogue {
    static constexpr bool kIsTileEpilogue = true;
    GdnConvEpilogue<Publish> conv;
    __nv_bfloat16* z;

    template <int TileCols, int RowsPerCta>
    __device__ __forceinline__ void store_tile(const float* dense, int global_row0, int local_row,
                                               int live_columns) const {
        static_assert(Tokens <= TileCols);
        if (local_row >= RowsPerCta) return;
        const int parent_row = global_row0 + local_row;
        float projected[Tokens];
#pragma unroll
        for (int token = 0; token < Tokens; ++token) {
            projected[token] = token < live_columns ? dense[local_row * TileCols + token] : 0.0F;
        }
        if (parent_row < kValueRows) {
            // The Q5 parent starts after Q/K in the logical GDN channel space.
            conv.store(parent_row, projected);
            return;
        }
#pragma unroll
        for (int token = 0; token < Tokens; ++token) {
            z[static_cast<std::int64_t>(token) * kZRows + parent_row - kValueRows] =
                __float2bfloat16_rn(projected[token]);
        }
    }
};

template <int Tokens, class Publish>
void launch_q4_fused(const Tensor& x, const Weight& weight, GdnConvEpilogue<Publish> conv,
                     cudaStream_t stream) {
    using Epilogue = Q4GdnConvTileEpilogue<Tokens, Publish>;
    const Epilogue epilogue{conv};
    if constexpr (Tokens <= 8) {
        q4_ksplit_mma_launch<GdnQkGeometry, 8, 8, Epilogue, Q4KSplitIdentityRows, true, 8, 1>(
            kQkRows / SmallTLayout<8>::kRowsPerCta, stream,
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), nullptr, epilogue,
            Q4KSplitIdentityRows{}, Tokens);
    } else {
        q4_ksplit_mma_launch<GdnQkGeometry, 16, 16, Epilogue, Q4KSplitIdentityRows, true, 4, 2>(
            kQkRows / SmallTLayout<4>::kRowsPerCta, stream,
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), nullptr, epilogue,
            Q4KSplitIdentityRows{}, Tokens);
    }
}

template <int Tokens, class Publish>
void launch_q5_fused(const Tensor& x, const Weight& weight, GdnConvEpilogue<Publish> conv,
                     Tensor& z, cudaStream_t stream) {
    using Epilogue = Q5GdnConvTileEpilogue<Tokens, Publish>;
    const Epilogue epilogue{conv, static_cast<__nv_bfloat16*>(z.data)};
    if constexpr (Tokens <= 4) {
        q5_small_t_mma_launch<kValueZRows, kHidden, 4, 2, Epilogue, 8>(
            stream, static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.qhigh),
            static_cast<const std::uint8_t*>(weight.scales), epilogue, Tokens);
    } else if constexpr (Tokens <= 8) {
        q5_small_t_mma_launch<kValueZRows, kHidden, 8, 1, Epilogue, 8>(
            stream, static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.qhigh),
            static_cast<const std::uint8_t*>(weight.scales), epilogue, Tokens);
    } else {
        q5_small_t_mma_launch<kValueZRows, kHidden, 16, 3, Epilogue, 4>(
            stream, static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.qhigh),
            static_cast<const std::uint8_t*>(weight.scales), epilogue, Tokens);
    }
}

template <int Tokens, class Publish>
void launch_pair(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
                 const Tensor& conv_weight, const Tensor& conv_states,
                 const Tensor& valid_columns, const Tensor& initial_state_slots,
                 Tensor& query, Tensor& key, Tensor& value, Tensor& z, Publish publish,
                 cudaStream_t stream) {
    auto qk_conv = make_conv_epilogue<Tokens>(
        conv_weight, conv_states, valid_columns, initial_state_slots, query, key, value, 0, publish);
    auto value_conv = make_conv_epilogue<Tokens>(
        conv_weight, conv_states, valid_columns, initial_state_slots, query, key, value, kQkRows,
        publish);

    // Keep the same projection ordering as q4_q5_gdn_input_small_t_launch. The two parent kernels
    // touch disjoint weights/outputs but both read x; serial launch preserves the established
    // stream and CUDA-graph contract while eliminating the intermediate global matrix.
    launch_q5_fused<Tokens>(x, value_z_weight, value_conv, z, stream);
    launch_q4_fused<Tokens>(x, qk_weight, qk_conv, stream);
    CUDA_CHECK(cudaGetLastError());
}

template <class Launch>
void dispatch_width(int tokens, Launch&& launch) {
    switch (tokens) {
    case 1: launch.template operator()<1>(); return;
    case 2: launch.template operator()<2>(); return;
    case 3: launch.template operator()<3>(); return;
    case 4: launch.template operator()<4>(); return;
    case 5: launch.template operator()<5>(); return;
    case 6: launch.template operator()<6>(); return;
    case 7: launch.template operator()<7>(); return;
    case 8: launch.template operator()<8>(); return;
    case 9: launch.template operator()<9>(); return;
    case 10: launch.template operator()<10>(); return;
    case 11: launch.template operator()<11>(); return;
    case 12: launch.template operator()<12>(); return;
    case 13: launch.template operator()<13>(); return;
    case 14: launch.template operator()<14>(); return;
    case 15: launch.template operator()<15>(); return;
    case 16: launch.template operator()<16>(); return;
    default:
        throw std::invalid_argument("Q4/Q5 fused GDN conv requires batch=1 and T in [1,16]");
    }
}

} // namespace

void q4_q5_gdn_input_conv_snapshot_fused_launch(
    const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
    const Tensor& conv_weight, Tensor& conv_states, const Tensor& valid_columns,
    const Tensor& initial_state_slots, const Tensor& snapshot_base_slots,
    Tensor& query, Tensor& key, Tensor& value, Tensor& z, cudaStream_t stream) {
    const SnapshotHistoryPublish publish{
        static_cast<__nv_bfloat16*>(conv_states.data),
        static_cast<const std::int32_t*>(snapshot_base_slots.data),
        kChannels,
    };
    const auto launch = [&]<int Tokens>() {
        launch_pair<Tokens>(x, qk_weight, value_z_weight, conv_weight, conv_states, valid_columns,
                            initial_state_slots, query, key, value, z, publish, stream);
    };
    dispatch_width(x.ne[1], launch);
}

void q4_q5_gdn_input_conv_record_fused_launch(
    const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
    const Tensor& conv_weight, const Tensor& conv_states, const Tensor& valid_columns,
    const Tensor& initial_state_slots, Tensor& conv_record,
    Tensor& query, Tensor& key, Tensor& value, Tensor& z, cudaStream_t stream) {
    const RecordColumnPublish publish{
        static_cast<__nv_bfloat16*>(conv_record.data),
        kChannels,
        x.ne[1],
    };
    const auto launch = [&]<int Tokens>() {
        launch_pair<Tokens>(x, qk_weight, value_z_weight, conv_weight, conv_states, valid_columns,
                            initial_state_slots, query, key, value, z, publish, stream);
    };
    dispatch_width(x.ne[1], launch);
}

} // namespace ninfer::ops::detail
