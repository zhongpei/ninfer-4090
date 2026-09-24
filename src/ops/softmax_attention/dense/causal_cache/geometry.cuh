#pragma once

#include "ops/softmax_attention/common/head_mapping.cuh"

namespace ninfer::ops {

template <int QHeadsValue, int KVHeadsValue, int SmallTSplitScaleValue>
struct CausalAttentionGeometry : AttentionHeadMapping<QHeadsValue, KVHeadsValue> {
    static_assert(SmallTSplitScaleValue > 0);

    static constexpr int SmallTSplitScale    = SmallTSplitScaleValue;
    // 256 splits keep each split below the 64-page staging limit through the supported
    // 786,432-token YaRN envelope while staying within the reducer's fixed 256-way contract.
    static constexpr int SmallTMaximumSplits = 256;
};

using CausalD256H24Kv4 = CausalAttentionGeometry<24, 4, 1>;
using CausalD256H16Kv2 = CausalAttentionGeometry<16, 2, 2>;

} // namespace ninfer::ops
