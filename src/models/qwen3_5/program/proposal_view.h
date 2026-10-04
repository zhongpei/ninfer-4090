#pragma once

#include "models/qwen3_5/program/round_buffers.h"

#include <stdexcept>

namespace ninfer::models::qwen3_5 {

// A packed view of the resident proposal storage, with outputs written directly into the
// selected target's prefix buffers. This changes physical lane strides, not merely validity
// masks. No device allocation/copy is performed here; all addresses remain stable for graphs.
// Context append storage is intentionally NOT narrowed: the preceding committed round may
// still have up to the startup K+1 feature columns to append before this proposal can run.
[[nodiscard]] inline DFlashDecodeState selected_dflash_proposal_view(
    const DFlashDecodeState& frame, std::uint32_t drafts) {
    if (drafts == 0 || drafts > static_cast<std::uint32_t>(frame.draft_tokens.ne[0])) {
        throw std::invalid_argument("selected DFlash proposal width is outside resident capacity");
    }
    const auto width = static_cast<std::int32_t>(drafts + 1U);
    const auto batch = frame.anchors.ne[0];
    if (batch <= 0 || !frame.proposal_ids.data || !frame.proposal_positions.data ||
        frame.proposal_ids.dtype != DType::I32 || frame.proposal_positions.dtype != DType::I32 ||
        frame.proposal_ids.ne[0] < width || frame.proposal_positions.ne[0] < width ||
        frame.proposal_ids.ne[1] < batch || frame.proposal_positions.ne[1] < batch) {
        throw std::logic_error("selected DFlash proposal has no planned input storage");
    }
    DFlashDecodeState result = frame.target_view(drafts);
    // A width slice would keep the original 16-column stride between requests. Rebinding a
    // contiguous prefix ensures C2..C8 use [K+1,B], [K,B] and [16,K,B] consistently end to end.
    result.proposal_ids = Tensor(frame.proposal_ids.data, DType::I32, {width, batch});
    result.proposal_positions = Tensor(frame.proposal_positions.data, DType::I32, {width, batch});
    return result;
}

} // namespace ninfer::models::qwen3_5
