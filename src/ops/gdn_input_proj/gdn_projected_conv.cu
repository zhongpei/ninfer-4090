#include "ops/gdn_input_proj/gdn_projected_conv.h"

#include "core/device.h"
#include "core/weight.h"
#include "ops/linear/t2/t2_small_t_v2.cuh"
#include "ops/gdn_input_proj/gdn_conv.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace ninfer::ops::detail {
namespace {

template <int Channels, int QueryRows, int KeyRows, int ValueRows, int StaticWidth,
          class Projected, class Publish>
__global__ void gdn_projected_conv_kernel(
    const Projected* __restrict__ projected, const __nv_bfloat16* __restrict__ conv_weight,
    const __nv_bfloat16* __restrict__ state_read, const std::int32_t* __restrict__ valid_columns,
    const std::int32_t* __restrict__ initial_state_slots, __nv_bfloat16* __restrict__ query,
    __nv_bfloat16* __restrict__ key, __nv_bfloat16* __restrict__ value, std::int32_t width,
    Publish publish) {
    static_assert(Channels == QueryRows + KeyRows + ValueRows);
    const std::int32_t row = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= Channels) { return; }
    const std::int32_t batch = static_cast<std::int32_t>(blockIdx.y);
    if constexpr (StaticWidth != 0) { width = StaticWidth; }

    std::int32_t valid                 = valid_columns == nullptr ? width : valid_columns[batch];
    valid                              = valid < 0 ? 0 : (valid > width ? width : valid);
    constexpr std::int64_t slot_stride = static_cast<std::int64_t>(Channels) * 3;
    const std::int64_t initial_base =
        static_cast<std::int64_t>(initial_state_slots[batch]) * slot_stride;
    float s0       = __bfloat162float(state_read[initial_base + row]);
    float s1       = __bfloat162float(state_read[initial_base + Channels + row]);
    float s2       = __bfloat162float(state_read[initial_base + 2LL * Channels + row]);
    const float w0 = __bfloat162float(conv_weight[row]);
    const float w1 = __bfloat162float(conv_weight[Channels + row]);
    const float w2 = __bfloat162float(conv_weight[2LL * Channels + row]);
    const float w3 = __bfloat162float(conv_weight[3LL * Channels + row]);

    for (std::int32_t token = 0; token < width; ++token) {
        const std::int64_t column = static_cast<std::int64_t>(batch) * width + token;
        if (token >= valid) {
            if (row < QueryRows) {
                query[column * QueryRows + row] = __float2bfloat16_rn(0.0F);
            } else if (row < QueryRows + KeyRows) {
                key[column * KeyRows + row - QueryRows] = __float2bfloat16_rn(0.0F);
            } else {
                value[column * ValueRows + row - QueryRows - KeyRows] = __float2bfloat16_rn(0.0F);
            }
            continue;
        }

        const float p = [&] {
            if constexpr (std::is_same_v<Projected, float>) {
                return projected[column * Channels + row];
            } else {
                return __bfloat162float(projected[column * Channels + row]);
            }
        }();
        float conv                 = fmaf(w0, s0, 0.0F);
        conv                       = fmaf(w1, s1, conv);
        conv                       = fmaf(w2, s2, conv);
        conv                       = fmaf(w3, p, conv);
        const __nv_bfloat16 output = __float2bfloat16_rn(silu(conv));
        if (row < QueryRows) {
            query[column * QueryRows + row] = output;
        } else if (row < QueryRows + KeyRows) {
            key[column * KeyRows + row - QueryRows] = output;
        } else {
            value[column * ValueRows + row - QueryRows - KeyRows] = output;
        }
        publish.publish(token, batch, row, s1, s2, p);
        s0 = s1;
        s1 = s2;
        // Earlier columns cross the same observable BF16 history boundary as restored calls.
        if constexpr (std::is_same_v<Projected, float>) {
            s2 = __bfloat162float(__float2bfloat16_rn(p));
        } else {
            s2 = p;
        }
    }
}

template <int Channels, int QueryRows, int KeyRows, int ValueRows, class Projected, class Publish>
void launch(const Tensor& projected, const Tensor& conv_weight, const Tensor& state_read,
            const Tensor& valid_columns, const Tensor& initial_state_slots, Tensor& query,
            Tensor& key, Tensor& value, Publish publish, cudaStream_t stream) {
    constexpr int kDefaultThreads = 256;
    const std::int32_t width      = projected.ne[1];
    const std::int32_t batch      = projected.ne[2];
    if constexpr (Channels == 10240) {
        if (width == 4 && batch == 1) {
            constexpr int kT4Threads = 64;
            gdn_projected_conv_kernel<Channels, QueryRows, KeyRows, ValueRows, 4, Projected>
                <<<(Channels + kT4Threads - 1) / kT4Threads, kT4Threads, 0, stream>>>(
                    static_cast<const Projected*>(projected.data),
                    static_cast<const __nv_bfloat16*>(conv_weight.data),
                    static_cast<const __nv_bfloat16*>(state_read.data),
                    valid_columns.data == nullptr
                        ? nullptr
                        : static_cast<const std::int32_t*>(valid_columns.data),
                    static_cast<const std::int32_t*>(initial_state_slots.data),
                    static_cast<__nv_bfloat16*>(query.data), static_cast<__nv_bfloat16*>(key.data),
                    static_cast<__nv_bfloat16*>(value.data), width, publish);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
    }
    const dim3 grid((Channels + kDefaultThreads - 1) / kDefaultThreads,
                    static_cast<unsigned>(batch));
    gdn_projected_conv_kernel<Channels, QueryRows, KeyRows, ValueRows, 0, Projected>
        <<<grid, kDefaultThreads, 0, stream>>>(
            static_cast<const Projected*>(projected.data),
            static_cast<const __nv_bfloat16*>(conv_weight.data),
            static_cast<const __nv_bfloat16*>(state_read.data),
            valid_columns.data == nullptr ? nullptr
                                          : static_cast<const std::int32_t*>(valid_columns.data),
            static_cast<const std::int32_t*>(initial_state_slots.data),
            static_cast<__nv_bfloat16*>(query.data), static_cast<__nv_bfloat16*>(key.data),
            static_cast<__nv_bfloat16*>(value.data), width, publish);
    CUDA_CHECK(cudaGetLastError());
}

template <int Channels, int QueryRows, int KeyRows, int ValueRows>
__global__ void gdn_projected_tree_conv_kernel(
    const __nv_bfloat16* __restrict__ projected, const __nv_bfloat16* __restrict__ conv_weight,
    const __nv_bfloat16* __restrict__ state_read,
    const std::int32_t* __restrict__ initial_state_slots,
    const std::int32_t* __restrict__ parents, __nv_bfloat16* __restrict__ conv_record,
    __nv_bfloat16* __restrict__ query, __nv_bfloat16* __restrict__ key,
    __nv_bfloat16* __restrict__ value, std::int32_t width) {
    static_assert(Channels == QueryRows + KeyRows + ValueRows);
    const std::int32_t row = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= Channels) return;

    constexpr std::int64_t slot_stride = static_cast<std::int64_t>(Channels) * 3;
    const std::int64_t initial_base =
        static_cast<std::int64_t>(initial_state_slots[0]) * slot_stride;
    const float s0 = __bfloat162float(state_read[initial_base + row]);
    const float s1 = __bfloat162float(state_read[initial_base + Channels + row]);
    const float s2 = __bfloat162float(state_read[initial_base + 2LL * Channels + row]);
    const float w0 = __bfloat162float(conv_weight[row]);
    const float w1 = __bfloat162float(conv_weight[Channels + row]);
    const float w2 = __bfloat162float(conv_weight[2LL * Channels + row]);
    const float w3 = __bfloat162float(conv_weight[3LL * Channels + row]);

    for (std::int32_t node = 0; node < width; ++node) {
        const std::int32_t p1 = parents[node];
        const std::int32_t p2 = p1 >= 0 ? parents[p1] : -1;
        const std::int32_t p3 = p2 >= 0 ? parents[p2] : -1;

        // The convolution history is tail_3(source_history || ancestors(current)).
        const float newest =
            p1 >= 0 ? __bfloat162float(
                          projected[static_cast<std::int64_t>(p1) * Channels + row])
                    : s2;
        const float middle =
            p2 >= 0 ? __bfloat162float(
                          projected[static_cast<std::int64_t>(p2) * Channels + row])
                    : (p1 >= 0 ? s2 : s1);
        const float oldest =
            p3 >= 0 ? __bfloat162float(
                          projected[static_cast<std::int64_t>(p3) * Channels + row])
                    : (p2 >= 0 ? s2 : (p1 >= 0 ? s1 : s0));
        const float current =
            __bfloat162float(projected[static_cast<std::int64_t>(node) * Channels + row]);

        float conv = fmaf(w0, oldest, 0.0F);
        conv = fmaf(w1, middle, conv);
        conv = fmaf(w2, newest, conv);
        conv = fmaf(w3, current, conv);
        const __nv_bfloat16 output = __float2bfloat16_rn(silu(conv));
        conv_record[static_cast<std::int64_t>(node) * Channels + row] =
            __float2bfloat16_rn(current);
        if (row < QueryRows) {
            query[static_cast<std::int64_t>(node) * QueryRows + row] = output;
        } else if (row < QueryRows + KeyRows) {
            key[static_cast<std::int64_t>(node) * KeyRows + row - QueryRows] = output;
        } else {
            value[static_cast<std::int64_t>(node) * ValueRows + row - QueryRows - KeyRows] =
                output;
        }
    }
}

template <int Channels, int QueryRows, int KeyRows, int ValueRows>
void launch_tree(const Tensor& projected, const Tensor& conv_weight, const Tensor& state_read,
                 const Tensor& initial_state_slots, const Tensor& parents, Tensor& conv_record,
                 Tensor& query, Tensor& key, Tensor& value, cudaStream_t stream) {
    constexpr int kThreads = 256;
    gdn_projected_tree_conv_kernel<Channels, QueryRows, KeyRows, ValueRows>
        <<<(Channels + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(projected.data),
            static_cast<const __nv_bfloat16*>(conv_weight.data),
            static_cast<const __nv_bfloat16*>(state_read.data),
            static_cast<const std::int32_t*>(initial_state_slots.data),
            static_cast<const std::int32_t*>(parents.data),
            static_cast<__nv_bfloat16*>(conv_record.data),
            static_cast<__nv_bfloat16*>(query.data), static_cast<__nv_bfloat16*>(key.data),
            static_cast<__nv_bfloat16*>(value.data), projected.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

template <class Projected, class Publish>
void dispatch(const Tensor& projected, const Tensor& conv_weight, const Tensor& state_read,
              const Tensor& valid_columns, const Tensor& initial_state_slots, Tensor& query,
              Tensor& key, Tensor& value, Publish publish, cudaStream_t stream) {
    if (projected.ne[0] == 10240 && query.ne[0] == 2048 && key.ne[0] == 2048 &&
        value.ne[0] == 6144) {
        launch<10240, 2048, 2048, 6144, Projected>(projected, conv_weight, state_read, valid_columns,
                                                      initial_state_slots, query, key, value, publish,
                                                      stream);
        return;
    }
    if (projected.ne[0] == 8192 && query.ne[0] == 2048 && key.ne[0] == 2048 &&
        value.ne[0] == 4096) {
        launch<8192, 2048, 2048, 4096, Projected>(projected, conv_weight, state_read, valid_columns,
                                                     initial_state_slots, query, key, value, publish,
                                                     stream);
        return;
    }
    throw std::invalid_argument("GDN projected-conv received an unregistered geometry");
}

template <class Publish>
void dispatch_dtype(const Tensor& projected, const Tensor& conv_weight, const Tensor& state_read,
                    const Tensor& valid_columns, const Tensor& initial_state_slots, Tensor& query,
                    Tensor& key, Tensor& value, Publish publish, cudaStream_t stream) {
    if (projected.dtype == DType::FP32) {
        dispatch<float>(projected, conv_weight, state_read, valid_columns, initial_state_slots,
                        query, key, value, publish, stream);
    } else {
        dispatch<__nv_bfloat16>(projected, conv_weight, state_read, valid_columns,
                               initial_state_slots, query, key, value, publish, stream);
    }
}

template <bool ValueParent>
struct T2GdnCurrentPublish {
    float* projected;
    __nv_bfloat16* record;
    __nv_bfloat16* z;

    __device__ __forceinline__ void operator()(__nv_bfloat16*, int column, int row, int,
                                              float value) const {
        if constexpr (ValueParent) {
            if (row >= 6144) {
                z[static_cast<std::int64_t>(column) * 6144 + row - 6144] =
                    __float2bfloat16_rn(value);
                return;
            }
            row += 4096;
        }
        const auto index = static_cast<std::int64_t>(column) * 10240 + row;
        projected[index] = value;
        if (record != nullptr) { record[index] = __float2bfloat16_rn(value); }
    }
};

template <bool ValueParent, int ColumnTiles>
void launch_t2_current(const Tensor& x, const Weight& weight, Tensor& projected, Tensor& z,
                       __nv_bfloat16* record, cudaStream_t stream) {
    using Schedule = T2SmallTv2Schedule<4, 2, ColumnTiles, 2, 4>;
    const dim3 grid(static_cast<unsigned>(weight.n / Schedule::kRows),
                    static_cast<unsigned>((x.ne[1] + Schedule::kColumns - 1) / Schedule::kColumns));
    const T2GdnCurrentPublish<ValueParent> publish{static_cast<float*>(projected.data), record,
                                                  static_cast<__nv_bfloat16*>(z.data)};
    t2_small_t_v2_kernel<Schedule><<<grid, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), nullptr, weight.n, weight.k, x.ne[1],
        publish);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void gdn_t2_current_projection_launch(const Tensor& x, const Weight& qk_weight,
                                      const Weight& value_z_weight, Tensor& projected, Tensor& z,
                                      Tensor* record, cudaStream_t stream) {
    auto* record_data = record == nullptr ? nullptr : static_cast<__nv_bfloat16*>(record->data);
    if (x.ne[1] <= 8) {
        launch_t2_current<false, 1>(x, qk_weight, projected, z, record_data, stream);
        launch_t2_current<true, 1>(x, value_z_weight, projected, z, record_data, stream);
    } else {
        launch_t2_current<false, 2>(x, qk_weight, projected, z, record_data, stream);
        launch_t2_current<true, 2>(x, value_z_weight, projected, z, record_data, stream);
    }
}

void gdn_projected_conv_snapshot_launch(const Tensor& projected, const Tensor& conv_weight,
                                        Tensor& conv_states, const Tensor& valid_columns,
                                        const Tensor& initial_state_slots,
                                        const Tensor& snapshot_base_slots, Tensor& query,
                                        Tensor& key, Tensor& value, cudaStream_t stream) {
    dispatch_dtype(projected, conv_weight, conv_states, valid_columns, initial_state_slots, query, key,
             value,
             SnapshotHistoryPublish{static_cast<__nv_bfloat16*>(conv_states.data),
                                    static_cast<const std::int32_t*>(snapshot_base_slots.data),
                                    projected.ne[0]},
             stream);
}

void gdn_projected_conv_record_launch(const Tensor& conv_record, const Tensor& conv_weight,
                                      const Tensor& conv_states, const Tensor& valid_columns,
                                      const Tensor& initial_state_slots, Tensor& query, Tensor& key,
                                      Tensor& value, cudaStream_t stream) {
    dispatch_dtype(conv_record, conv_weight, conv_states, valid_columns, initial_state_slots, query, key,
             value, NoHistoryPublish{}, stream);
}

void gdn_projected_tree_conv_record_launch(
    const Tensor& projected, const Tensor& conv_weight, const Tensor& conv_states,
    const Tensor& initial_state_slots, const Tensor& parents, Tensor& conv_record,
    Tensor& query, Tensor& key, Tensor& value, cudaStream_t stream) {
    if (projected.ne[0] == 10240 && query.ne[0] == 2048 && key.ne[0] == 2048 &&
        value.ne[0] == 6144) {
        launch_tree<10240, 2048, 2048, 6144>(projected, conv_weight, conv_states,
                                             initial_state_slots, parents, conv_record,
                                             query, key, value, stream);
        return;
    }
    if (projected.ne[0] == 8192 && query.ne[0] == 2048 && key.ne[0] == 2048 &&
        value.ne[0] == 4096) {
        launch_tree<8192, 2048, 2048, 4096>(projected, conv_weight, conv_states,
                                            initial_state_slots, parents, conv_record,
                                            query, key, value, stream);
        return;
    }
    throw std::invalid_argument("GDN projected tree conv received an unregistered geometry");
}

} // namespace ninfer::ops::detail
