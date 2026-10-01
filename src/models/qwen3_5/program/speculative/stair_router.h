#pragma once

#include "ninfer/types.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace ninfer::models::qwen3_5::detail {

inline constexpr std::size_t kStairTrackedDraftPositions = 15;

struct StairRouterState {
    std::uint64_t rounds = 0;
    std::array<std::uint64_t, kStairTrackedDraftPositions> attempted{};
    std::array<std::uint64_t, kStairTrackedDraftPositions> accepted{};
    std::array<std::uint64_t, kSpeculativeStairLevels> selected{};
    std::uint32_t last_extent = 0;

    void observe(std::uint32_t extent, std::uint32_t accepted_count,
                 const SpeculativeRoutingOptions& options) noexcept {
        extent = std::min<std::uint32_t>(extent, kStairTrackedDraftPositions);
        accepted_count = std::min(accepted_count, extent);
        if (extent == 0) return;
        ++rounds;
        for (std::uint32_t i = 0; i < extent; ++i) {
            ++attempted[i];
            if (i < accepted_count) ++accepted[i];
        }
        for (std::size_t i = 0; i < options.widths.size(); ++i) {
            if (options.widths[i] == extent) {
                ++selected[i];
                break;
            }
        }
        last_extent = extent;
    }
};

struct TreeStairRouterState {
    std::uint64_t rounds = 0;
    std::array<std::uint64_t, kSpeculativeStairLevels> observed{};
    std::array<std::uint64_t, kSpeculativeStairLevels> committed_tokens{};
    std::array<std::uint64_t, kSpeculativeStairLevels> selected{};
    std::uint32_t last_extent = 0;

    void observe(std::uint32_t extent, std::uint32_t accepted_drafts,
                 const SpeculativeRoutingOptions& options) noexcept {
        if (extent == 0) return;
        ++rounds;
        for (std::size_t i = 0; i < options.widths.size(); ++i) {
            if (options.widths[i] == extent) {
                ++observed[i];
                committed_tokens[i] +=
                    static_cast<std::uint64_t>(std::min(accepted_drafts, extent)) + 1ULL;
                ++selected[i];
                break;
            }
        }
        last_extent = extent;
    }
};

[[nodiscard]] inline double stair_expected_commits(const SpeculativeRoutingOptions& options,
                                                   const StairRouterState& state,
                                                   std::uint32_t extent) noexcept {
    double expected = 1.0;
    const double prior_weight = static_cast<double>(options.prior_weight);
    const double prior_hits = static_cast<double>(options.prior_acceptance) * prior_weight;
    for (std::uint32_t i = 0;
         i < extent && i < static_cast<std::uint32_t>(kStairTrackedDraftPositions); ++i) {
        const double attempts = static_cast<double>(state.attempted[i]);
        const double accepts = static_cast<double>(state.accepted[i]);
        const double denom = attempts + prior_weight;
        const double survival =
            denom > 0.0 ? (accepts + prior_hits) / denom
                        : static_cast<double>(options.prior_acceptance);
        expected += std::clamp(survival, 0.0, 1.0);
    }
    return expected;
}

[[nodiscard]] inline double tree_stair_expected_commits(
    const SpeculativeRoutingOptions& options, const TreeStairRouterState& state,
    std::size_t rung) noexcept {
    const double prior_weight = static_cast<double>(options.prior_weight);
    const double nodes = static_cast<double>(options.widths[rung]);
    const double prior_expected =
        1.0 + nodes * static_cast<double>(options.prior_acceptance);
    const double observations = static_cast<double>(state.observed[rung]);
    const double committed = static_cast<double>(state.committed_tokens[rung]);
    const double denom = observations + prior_weight;
    return denom > 0.0 ? (committed + prior_weight * prior_expected) / denom
                       : prior_expected;
}

[[nodiscard]] inline std::uint32_t
choose_stair_extent(const SpeculativeRoutingOptions& options, const StairRouterState& state,
                    std::uint32_t maximum_extent) noexcept {
    if (options.mode != SpeculativeRoutingMode::Stair || maximum_extent == 0) {
        return maximum_extent;
    }

    std::size_t last_eligible = options.widths.size();
    for (std::size_t i = 0; i < options.widths.size(); ++i) {
        if (options.widths[i] <= maximum_extent) last_eligible = i;
    }
    if (last_eligible == options.widths.size()) return maximum_extent;

    if (state.rounds < options.warmup_rounds ||
        (options.probe_period != 0 && state.rounds != 0 &&
         state.rounds % options.probe_period == 0)) {
        return options.widths[last_eligible];
    }

    double best_score = -std::numeric_limits<double>::infinity();
    std::size_t best = last_eligible;
    std::size_t previous = options.widths.size();
    double previous_score = -std::numeric_limits<double>::infinity();

    for (std::size_t i = 0; i <= last_eligible; ++i) {
        const std::uint32_t extent = options.widths[i];
        const double cost = static_cast<double>(options.draft_cost) +
                            static_cast<double>(options.verify_costs[i]);
        const double score = stair_expected_commits(options, state, extent) / cost;
        if (score > best_score) {
            best_score = score;
            best = i;
        }
        if (extent == state.last_extent) {
            previous = i;
            previous_score = score;
        }
    }

    if (previous != options.widths.size() &&
        previous_score * (1.0 + static_cast<double>(options.switch_margin)) >= best_score) {
        return options.widths[previous];
    }
    return options.widths[best];
}

[[nodiscard]] inline std::uint32_t choose_tree_stair_extent(
    const SpeculativeRoutingOptions& options, const TreeStairRouterState& state,
    std::uint32_t maximum_extent) noexcept {
    if (maximum_extent == 0) return 0;
    if (options.mode != SpeculativeRoutingMode::Stair) return maximum_extent;

    std::size_t last_eligible = options.widths.size();
    for (std::size_t i = 0; i < options.widths.size(); ++i) {
        if (options.widths[i] <= maximum_extent) last_eligible = i;
    }
    if (last_eligible == options.widths.size()) return maximum_extent;

    if (state.rounds < options.warmup_rounds ||
        (options.probe_period != 0 && state.rounds != 0 &&
         state.rounds % options.probe_period == 0)) {
        return options.widths[last_eligible];
    }

    double best_score = -std::numeric_limits<double>::infinity();
    double previous_score = -std::numeric_limits<double>::infinity();
    std::size_t best = last_eligible;
    std::size_t previous = options.widths.size();

    for (std::size_t i = 0; i <= last_eligible; ++i) {
        const double cost = static_cast<double>(options.draft_cost) +
                            static_cast<double>(options.verify_costs[i]);
        const double score = tree_stair_expected_commits(options, state, i) / cost;
        if (score > best_score) {
            best_score = score;
            best = i;
        }
        if (options.widths[i] == state.last_extent) {
            previous = i;
            previous_score = score;
        }
    }

    if (previous != options.widths.size() &&
        previous_score * (1.0 + static_cast<double>(options.switch_margin)) >= best_score) {
        return options.widths[previous];
    }
    return options.widths[best];
}

} // namespace ninfer::models::qwen3_5::detail
