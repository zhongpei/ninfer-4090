#pragma once

#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// Current projection remains FP32; public z/record and convolution history remain BF16.
void gdn_t2_current_projection_launch(const Tensor& x, const Weight& qk_weight,
                                      const Weight& value_z_weight, Tensor& projected, Tensor& z,
                                      Tensor* record, cudaStream_t stream);

void gdn_projected_conv_snapshot_launch(const Tensor& projected, const Tensor& conv_weight,
                                        Tensor& conv_states, const Tensor& valid_columns,
                                        const Tensor& initial_state_slots,
                                        const Tensor& snapshot_base_slots, Tensor& query,
                                        Tensor& key, Tensor& value, cudaStream_t stream);

void gdn_projected_conv_record_launch(const Tensor& conv_record, const Tensor& conv_weight,
                                      const Tensor& conv_states, const Tensor& valid_columns,
                                      const Tensor& initial_state_slots, Tensor& query, Tensor& key,
                                      Tensor& value, cudaStream_t stream);

void gdn_projected_tree_conv_record_launch(
    const Tensor& projected, const Tensor& conv_weight, const Tensor& conv_states,
    const Tensor& initial_state_slots, const Tensor& parents, Tensor& conv_record,
    Tensor& query, Tensor& key, Tensor& value, cudaStream_t stream);

} // namespace ninfer::ops::detail
