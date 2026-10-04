#pragma once

#include "ninfer/types.h"
#include <array>

namespace ninfer::runtime {

inline constexpr std::array<std::uint32_t, 4> kCalibratedDraftActions{0, 7, 11, 15};

// Immutable after startup. Zero-initialized uncovered cells execute target-only.
struct CalibratedRoutingTable {
    std::array<std::array<std::uint32_t, 3>, kMaximumConcurrency> draft_tokens{};

    [[nodiscard]] std::uint32_t select(std::uint32_t active_batch,
                                        std::uint32_t execution_frontier) const noexcept {
        if (active_batch == 0 || active_batch > kMaximumConcurrency ||
            execution_frontier > 32768) { return 0; }
        const std::size_t interval = execution_frontier <= 1024 ? 0 :
                                     execution_frontier <= 8192 ? 1 : 2;
        return draft_tokens[active_batch - 1][interval];
    }
};

} // namespace ninfer::runtime
