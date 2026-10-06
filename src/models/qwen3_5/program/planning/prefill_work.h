#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <span>

namespace ninfer::models::qwen3_5::runtime_support {

// Captures and rewrite frontiers are sorted by token position. Equal chunks give
// the exact physical chunk count; a smaller service chunk gives an upper bound
// for automatic prefill, which can select either chunk size each scheduler round.
template <class Capture>
std::uint64_t segmented_prefill_units(std::uint32_t begin, std::uint32_t end,
                                      std::uint32_t execution_chunk, std::uint32_t service_chunk,
                                      std::span<const Capture> captures,
                                      std::span<const std::uint32_t> rewrite_frontiers) noexcept {
    if (begin >= end || execution_chunk == 0 || service_chunk == 0) { return 0; }
    const auto rounded_units = [service_chunk](std::uint64_t tokens) {
        return tokens == 0 ? 0ULL : 1ULL + (tokens - 1ULL) / service_chunk;
    };
    const auto nominal_units = [&](std::uint32_t first, std::uint32_t last,
                                    std::uint32_t origin) {
        const auto head = std::min(last - first,
                                   execution_chunk - (first - origin) % execution_chunk);
        const auto rest = last - first - head;
        return rounded_units(head) + (rest / execution_chunk) * rounded_units(execution_chunk) +
               rounded_units(rest % execution_chunk);
    };
    std::uint64_t units = 0;
    std::uint32_t segment_begin = begin;
    std::uint32_t grid_begin = begin;
    std::size_t capture_index = 0;
    std::size_t rewrite_index = 0;
    while (capture_index < captures.size() || rewrite_index < rewrite_frontiers.size()) {
        const auto capture_frontier = capture_index < captures.size()
            ? captures[capture_index].frontier : std::numeric_limits<std::uint32_t>::max();
        const auto rewrite_frontier = rewrite_index < rewrite_frontiers.size()
            ? rewrite_frontiers[rewrite_index] : std::numeric_limits<std::uint32_t>::max();
        const auto frontier = std::min(capture_frontier, rewrite_frontier);
        const bool capture = capture_frontier == frontier;
        if (capture) { ++capture_index; }
        if (rewrite_frontier == frontier) { ++rewrite_index; }
        if (frontier <= segment_begin || frontier >= end) { continue; }
        if (execution_chunk == service_chunk) {
            units += nominal_units(segment_begin, frontier, grid_begin);
        } else if (!capture) {
            // Automatic prefill can switch between execution_chunk and service_chunk
            // each round. Each interior rewrite split adds at most one rounded unit.
            ++units;
        }
        if (capture) {
            if (execution_chunk != service_chunk) {
                units += nominal_units(grid_begin, frontier, grid_begin);
            }
            // Capture offers return early, discarding the nominal remainder. Rewrite
            // splits remain inside the same nominal block and do not restart its grid.
            grid_begin = frontier;
        }
        segment_begin = frontier;
    }
    if (execution_chunk == service_chunk) {
        return units + nominal_units(segment_begin, end, grid_begin);
    }
    // All-large blocks maximize the unsplit automatic service charge: for n large
    // blocks the charge is n*ceil(C/Q)+ceil((T-n*C)/Q), nondecreasing in n since
    // C <= ceil(C/Q)*Q. Add the rewrite bound above within each capture interval.
    return units + nominal_units(grid_begin, end, grid_begin);
}

} // namespace ninfer::models::qwen3_5::runtime_support
