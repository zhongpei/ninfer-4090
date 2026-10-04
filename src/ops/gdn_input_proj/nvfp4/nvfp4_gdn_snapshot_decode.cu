#include "core/weight.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.h"

#include "core/device.h"
#include "ops/gdn_input_proj/gdn_conv_output.cuh"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_gemv.cuh"

namespace ninfer::ops::detail {

namespace {
template <class Publish>
void launch_decode(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                   const Tensor& conv_states, const Tensor& valid_columns,
                   const Tensor& initial_slot, Tensor& query, Tensor& key,
                   Tensor& value, Tensor& z, Publish publish, cudaStream_t stream) {
    using Geometry = Nvfp4N16384K5120;
    using Schedule =
        Nvfp4GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 2>;

    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    const float inverse   = 1.0F / weight.weight_scale_divisor;
    nvfp4_gemv_kernel<Geometry, Schedule><<<kBlocks, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), inverse, Nvfp4IdentityEpilogue{},
        make_gdn_conv_output<1>(
            conv_weight, conv_states, valid_columns, initial_slot, query, key, value, z,
            publish));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void nvfp4_gdn_snapshot_decode_launch(
    const Tensor& x, const Weight& weight, const Tensor& conv_weight, Tensor& conv_states,
    const Tensor& valid_columns, const Tensor& initial_slot, const Tensor& snapshot_base_slot,
    Tensor& query, Tensor& key, Tensor& value, Tensor& z, cudaStream_t stream) {
    launch_decode(x, weight, conv_weight, conv_states, valid_columns, initial_slot, query, key, value, z,
                  SnapshotHistoryPublish{static_cast<__nv_bfloat16*>(conv_states.data),
                                         static_cast<const std::int32_t*>(snapshot_base_slot.data),
                                         kGdnChannels}, stream);
}

void nvfp4_gdn_record_decode_launch(
    const Tensor& x, const Weight& weight, const Tensor& conv_weight, const Tensor& conv_states,
    const Tensor& valid_columns, const Tensor& initial_slot, Tensor& conv_record,
    Tensor& query, Tensor& key, Tensor& value, Tensor& z, cudaStream_t stream) {
    launch_decode(x, weight, conv_weight, conv_states, valid_columns, initial_slot, query, key, value, z,
                  RecordColumnPublish{static_cast<__nv_bfloat16*>(conv_record.data), kGdnChannels, 1},
                  stream);
}

} // namespace ninfer::ops::detail
