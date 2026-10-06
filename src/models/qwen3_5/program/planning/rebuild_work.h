#pragma once

#include "runtime/contract/resources.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::models::qwen3_5::runtime_support {

inline void include_rebuild_boundary(std::uint32_t& tail_begin, std::uint32_t boundary,
                                     std::uint32_t frontier) noexcept {
    if (boundary <= frontier) { tail_begin = std::max(tail_begin, boundary); }
}

inline void advance_segmented_rebuild_work(runtime::PrefillWork& work, std::uint32_t tail_begin,
                                           std::uint32_t grid_begin, std::uint32_t previous_frontier,
                                           std::uint32_t frontier, std::uint32_t prefill_chunk) {
    if (frontier < previous_frontier || work.tokens != previous_frontier ||
        tail_begin > previous_frontier || grid_begin > tail_begin || prefill_chunk == 0) {
        throw std::logic_error("sequence rebuild work is not aligned with its frontier");
    }
    if (frontier == previous_frontier) { return; }

    const auto tail_chunks = [=](std::uint32_t end) -> std::uint64_t {
        if (end == tail_begin) { return 0; }
        return 1ULL + (end - grid_begin - 1ULL) / prefill_chunk -
               (tail_begin - grid_begin) / prefill_chunk;
    };
    const std::uint64_t old_tail_chunks = tail_chunks(previous_frontier);
    const std::uint64_t new_tail_chunks = tail_chunks(frontier);
    if (work.chunks < old_tail_chunks) {
        throw std::logic_error("sequence rebuild chunk accounting is invalid");
    }

    const runtime::PrefillWork total = runtime::make_prefill_work(
        0, frontier, work.vision_items, work.vision_patches, prefill_chunk);
    work.chunks          = work.chunks - old_tail_chunks + new_tail_chunks;
    work.tokens          = total.tokens;
    work.attention_pairs = total.attention_pairs;
}

} // namespace ninfer::models::qwen3_5::runtime_support
