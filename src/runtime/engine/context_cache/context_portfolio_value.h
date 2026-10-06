#pragma once

#include "runtime/contract/resources.h"
#include "ninfer/types.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace ninfer::runtime {

struct ContextPortfolioOwnerPolicy {
    PlanningOwnerId owner;
    std::uint32_t private_retention_weight = 0;
    bool explicit_shared_credit            = false;
};

struct ContextPortfolioCheckpointValue {
    PlanningOwnerId owner;
    // One bit per request in the manager's demand window, which is 64 wide: see
    // kDemandWindowCapacity, sized to outlast the distance between two requests sharing a prefix.
    std::uint64_t demand_mask          = 0;
    std::uint64_t rebuild_ns           = 0;
    std::uint64_t baseline_recovery_ns = 0;
    std::uint64_t target_recovery_ns   = 0;
};

struct ContextPortfolioValueResult {
    std::uint64_t baseline_public_value   = 0;
    std::uint64_t target_public_value     = 0;
    std::uint64_t private_transition_loss = 0;
    bool saturated                        = false;
};

struct ContextPortfolioProjection {
    std::array<std::uint64_t, 64> target_demand{};
    std::uint64_t target_credit = 0;
};

struct SharedCaptureValue {
    std::uint32_t frontier       = 0;
    std::uint64_t demand_mask    = 0;
    std::uint64_t rebuild_ns     = 0;
    bool explicit_credit        = false;
    bool pressure_capable       = false;
};

// Folds one complete inactive checkpoint portfolio. Empirical demands and explicit shared credit
// form public portfolio values. A private owner contributes the largest loss of one concrete
// checkpoint capability, so a surviving later checkpoint cannot mask damage to an earlier one.
class ContextPortfolioValue {
public:
    ContextPortfolioValue() { owner_scratch_.reserve(32U); }

    [[nodiscard]] ContextPortfolioValueResult
    fold(std::span<const ContextPortfolioOwnerPolicy> owners,
         std::span<const ContextPortfolioCheckpointValue> checkpoints,
         ContextPortfolioProjection* projection = nullptr) {
        std::array<std::uint64_t, 64> baseline_demand{};
        std::array<std::uint64_t, 64> target_demand{};
        owner_scratch_.clear();
        for (const ContextPortfolioOwnerPolicy& policy : owners) {
            if (std::find_if(owner_scratch_.begin(), owner_scratch_.end(), [&](const auto& item) {
                    return item.owner == policy.owner;
                }) != owner_scratch_.end()) {
                throw std::logic_error("portfolio owner policy ID is duplicated");
            }
            owner_scratch_.push_back(
                OwnerValue{.owner                    = policy.owner,
                           .private_retention_weight = policy.private_retention_weight,
                           .explicit_shared_credit   = policy.explicit_shared_credit});
        }

        for (const ContextPortfolioCheckpointValue& checkpoint : checkpoints) {
            const auto owner = std::find_if(
                owner_scratch_.begin(), owner_scratch_.end(),
                [&](const OwnerValue& item) { return item.owner == checkpoint.owner; });
            if (owner == owner_scratch_.end()) {
                throw std::logic_error("portfolio checkpoint has no owner policy");
            }
            const std::uint64_t baseline_saving =
                checkpoint.rebuild_ns > checkpoint.baseline_recovery_ns
                    ? checkpoint.rebuild_ns - checkpoint.baseline_recovery_ns
                    : 0;
            const std::uint64_t target_saving =
                checkpoint.rebuild_ns > checkpoint.target_recovery_ns
                    ? checkpoint.rebuild_ns - checkpoint.target_recovery_ns
                    : 0;
            owner->baseline_best = std::max(owner->baseline_best, baseline_saving);
            owner->target_best   = std::max(owner->target_best, target_saving);
            owner->private_transition_loss =
                std::max(owner->private_transition_loss,
                         baseline_saving > target_saving ? baseline_saving - target_saving : 0);
            for (std::uint32_t bit = 0; bit < 64U; ++bit) {
                if ((checkpoint.demand_mask & (1ULL << bit)) == 0) { continue; }
                baseline_demand[bit] = std::max(baseline_demand[bit], baseline_saving);
                target_demand[bit]   = std::max(target_demand[bit], target_saving);
            }
        }

        if (projection) {
            projection->target_demand = target_demand;
            projection->target_credit = 0;
        }
        ContextPortfolioValueResult result;
        for (std::size_t bit = 0; bit < baseline_demand.size(); ++bit) {
            add(result.baseline_public_value, baseline_demand[bit], result.saturated);
            add(result.target_public_value, target_demand[bit], result.saturated);
        }
        for (const OwnerValue& owner : owner_scratch_) {
            if (owner.private_retention_weight != 0) {
                add_weighted(result.private_transition_loss, owner.private_transition_loss,
                             owner.private_retention_weight, result.saturated);
            }
            if (owner.explicit_shared_credit) {
                add(result.baseline_public_value, owner.baseline_best, result.saturated);
                add(result.target_public_value, owner.target_best, result.saturated);
                if (projection) {
                    add(projection->target_credit, owner.target_best, result.saturated);
                }
            }
        }
        return result;
    }

    // Candidates add independent owners with zero baseline saving and zero recovery cost.
    // Existing checkpoint values therefore remain fixed; only demand maxima and owner credit
    // change. The split oracle is nonnegative but need not be additive or monotone.
    template <class SplitCostFn>
    [[nodiscard]] static std::vector<std::uint32_t>
    select_shared_captures(const ContextPortfolioValueResult& baseline,
                           const ContextPortfolioProjection& projection,
                           std::span<const SharedCaptureValue> candidates,
                           std::uint32_t catalog_slots, std::uint32_t vacant_slots,
                           SplitCostFn&& split_cost) {
        if (candidates.size() > kMaximumSharedPrefixCandidates) {
            throw std::logic_error("prepared shared candidates exceed the frontend contract");
        }
        if (baseline.saturated || catalog_slots == 0) { return {}; }
        std::uint64_t threshold = baseline.baseline_public_value;
        bool saturated = false;
        add(threshold, baseline.private_transition_loss, saturated);
        if (saturated) { return {}; }

        struct Bound {
            std::array<std::uint64_t, 64> demands{};
            std::uint64_t credit = 0;
            bool saturated = false;
        };
        std::array<Bound, kMaximumSharedPrefixCandidates + 1> suffix{};
        for (std::size_t index = candidates.size(); index-- > 0;) {
            suffix[index] = suffix[index + 1];
            const auto& candidate = candidates[index];
            if (candidate.explicit_credit) {
                add(suffix[index].credit, candidate.rebuild_ns, suffix[index].saturated);
            }
            for (std::size_t bit = 0; bit < 64; ++bit) {
                if (candidate.demand_mask & (1ULL << bit)) {
                    suffix[index].demands[bit] =
                        std::max(suffix[index].demands[bit], candidate.rebuild_ns);
                }
            }
        }
        auto demands = projection.target_demand;
        std::vector<std::uint32_t> frontiers, best;
        frontiers.reserve(candidates.size());
        std::uint64_t best_gain = 0;
        const auto visit = [&](auto&& self, std::size_t index, std::uint32_t surplus,
                               std::uint64_t credit, bool credit_saturated) -> void {
            if (credit_saturated) { return; }
            std::uint64_t upper = credit;
            bool upper_saturated = credit_saturated || suffix[index].saturated;
            add(upper, suffix[index].credit, upper_saturated);
            for (std::size_t bit = 0; bit < 64; ++bit) {
                add(upper, std::max(demands[bit], suffix[index].demands[bit]), upper_saturated);
            }
            // Equality can still improve the fewer-frontiers / lexicographic tie break.
            if (!upper_saturated && (upper <= threshold || upper - threshold < best_gain)) {
                return;
            }
            if (index == candidates.size()) {
                if (frontiers.empty() || credit_saturated) { return; }
                std::uint64_t value = credit;
                bool value_saturated = false;
                for (const auto saving : demands) { add(value, saving, value_saturated); }
                if (value_saturated || value <= threshold || value - threshold < best_gain) {
                    return;
                }
                const std::uint64_t cost = split_cost(frontiers);
                if (cost >= value - threshold) { return; }
                const std::uint64_t gain = value - threshold - cost;
                if (gain > best_gain ||
                    (gain == best_gain && (best.empty() || frontiers.size() < best.size() ||
                     (frontiers.size() == best.size() && frontiers < best)))) {
                    best_gain = gain;
                    best = frontiers;
                }
                return;
            }
            const auto& candidate = candidates[index];
            if (frontiers.size() < catalog_slots &&
                (candidate.pressure_capable || surplus < vacant_slots)) {
                const auto saved = demands;
                for (std::size_t bit = 0; bit < 64; ++bit) {
                    if (candidate.demand_mask & (1ULL << bit)) {
                        demands[bit] = std::max(demands[bit], candidate.rebuild_ns);
                    }
                }
                auto next_credit = credit;
                bool next_saturated = credit_saturated;
                if (candidate.explicit_credit) {
                    add(next_credit, candidate.rebuild_ns, next_saturated);
                }
                // Keep traversal state ordered even when input candidates are unordered.
                const auto at =
                    std::lower_bound(frontiers.begin(), frontiers.end(), candidate.frontier);
                const auto offset = at - frontiers.begin();
                frontiers.insert(at, candidate.frontier);
                self(self, index + 1, surplus + !candidate.pressure_capable, next_credit,
                     next_saturated);
                frontiers.erase(frontiers.begin() + offset);
                demands = saved;
            }
            self(self, index + 1, surplus, credit, credit_saturated);
        };
        visit(visit, 0, 0, projection.target_credit, false);
        return best;
    }

private:
    struct OwnerValue {
        PlanningOwnerId owner;
        std::uint32_t private_retention_weight = 0;
        bool explicit_shared_credit            = false;
        std::uint64_t baseline_best            = 0;
        std::uint64_t target_best              = 0;
        std::uint64_t private_transition_loss  = 0;
    };

    static void add(std::uint64_t& value, std::uint64_t increment, bool& saturated) noexcept {
        if (increment > std::numeric_limits<std::uint64_t>::max() - value) {
            value     = std::numeric_limits<std::uint64_t>::max();
            saturated = true;
        } else {
            value += increment;
        }
    }

    static void add_weighted(std::uint64_t& value, std::uint64_t increment, std::uint32_t weight,
                             bool& saturated) noexcept {
        if (weight != 0 && increment > std::numeric_limits<std::uint64_t>::max() / weight) {
            add(value, std::numeric_limits<std::uint64_t>::max(), saturated);
            saturated = true;
            return;
        }
        add(value, increment * weight, saturated);
    }

    std::vector<OwnerValue> owner_scratch_;
};

} // namespace ninfer::runtime
