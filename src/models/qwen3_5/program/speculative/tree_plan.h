#pragma once

#include "ninfer/types.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <queue>
#include <stdexcept>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

inline constexpr std::uint32_t kMaximumVerifyTreeRows = 16;
inline constexpr std::uint32_t kMaximumVerifyTreeDraftNodes = kMaximumVerifyTreeRows - 1;

struct VerifyTreePlan {
    std::array<TokenId, kMaximumVerifyTreeRows> tokens{};
    std::array<std::int32_t, kMaximumVerifyTreeRows> parents{};
    std::array<std::int32_t, kMaximumVerifyTreeRows> depths{};
    std::uint32_t rows = 0;

    [[nodiscard]] bool valid() const noexcept {
        if (rows == 0 || rows > kMaximumVerifyTreeRows || parents[0] != -1 || depths[0] != 0) {
            return false;
        }
        for (std::uint32_t i = 1; i < rows; ++i) {
            if (parents[i] < 0 || static_cast<std::uint32_t>(parents[i]) >= i ||
                depths[i] != depths[static_cast<std::size_t>(parents[i])] + 1) {
                return false;
            }
        }
        return true;
    }
};

namespace tree_detail {

struct BuilderNode {
    TokenId token = 0;
    std::int32_t parent = -1;
    std::int32_t depth = 0;
    double score = 1.0;
    std::uint32_t step = 0;
    std::uint32_t candidate_rank = 0;
    std::vector<std::int32_t> children;
};

struct Frontier {
    double neg_log_probability = 0.0;
    std::uint64_t tie = 0;
    std::int32_t parent = 0;
    std::uint32_t step = 0;
    std::uint32_t candidate_rank = 0;
    std::uint32_t predecessor_rank = 0;

    bool operator<(const Frontier& other) const noexcept {
        if (neg_log_probability != other.neg_log_probability) {
            return neg_log_probability > other.neg_log_probability;
        }
        return tie > other.tie;
    }
};

inline double log_softmax_value(const float* row, std::uint32_t candidate,
                                std::uint32_t count = 16) {
    float maximum = -std::numeric_limits<float>::infinity();
    for (std::uint32_t i = 0; i < count; ++i) maximum = std::max(maximum, row[i]);
    double total = 0.0;
    for (std::uint32_t i = 0; i < count; ++i) total += std::exp(double(row[i] - maximum));
    return double(row[candidate] - maximum) - std::log(total);
}

inline std::uint32_t argmax(const float* row, std::uint32_t count = 16) {
    std::uint32_t best = 0;
    for (std::uint32_t i = 1; i < count; ++i) {
        if (row[i] > row[best]) best = i;
    }
    return best;
}

} // namespace tree_detail

// candidates layout is [candidate=16, step=K], lattice is [candidate=16, predecessor=16, step=K]
// with candidate as the fastest axis, exactly the candidate-selector device layout for B=1.
[[nodiscard]] inline VerifyTreePlan
build_verify_tree(TokenId anchor, const TokenId* candidates, const float* lattice,
                  std::uint32_t steps, const SpeculativeTreeOptions& options) {
    if (candidates == nullptr || lattice == nullptr || steps == 0 || steps > 15 ||
        options.nodes == 0 || options.nodes > kMaximumVerifyTreeDraftNodes ||
        options.spine == 0 || options.spine > 15) {
        throw std::invalid_argument("invalid DFlash2 verify-tree input");
    }

    std::vector<tree_detail::BuilderNode> nodes;
    nodes.reserve(options.nodes + 1);
    nodes.push_back(tree_detail::BuilderNode{.token = anchor, .parent = -1, .depth = 0});

    std::priority_queue<tree_detail::Frontier> frontier;
    std::uint64_t tie = 0;
    std::int32_t parent = 0;
    std::uint32_t predecessor = 0;
    double path_lp = 0.0;
    const std::uint32_t spine = std::min({steps, options.spine, options.nodes});

    for (std::uint32_t step = 0; step < spine; ++step) {
        const float* edge = lattice + (static_cast<std::size_t>(step) * 16 + predecessor) * 16;
        const std::uint32_t selected = tree_detail::argmax(edge);
        const double base_lp = path_lp;
        path_lp += tree_detail::log_softmax_value(edge, selected);

        const std::int32_t here = parent;
        const std::int32_t node = static_cast<std::int32_t>(nodes.size());
        nodes.push_back(tree_detail::BuilderNode{
            .token = candidates[static_cast<std::size_t>(step) * 16 + selected],
            .parent = here,
            .depth = nodes[static_cast<std::size_t>(here)].depth + 1,
            .score = std::exp(path_lp),
            .step = step,
            .candidate_rank = selected,
        });
        nodes[static_cast<std::size_t>(here)].children.push_back(node);
        parent = node;

        for (std::uint32_t alt = 0; alt < 16; ++alt) {
            if (alt == selected) continue;
            ++tie;
            frontier.push(tree_detail::Frontier{
                .neg_log_probability = -(base_lp + tree_detail::log_softmax_value(edge, alt)),
                .tie = tie,
                .parent = here,
                .step = step,
                .candidate_rank = alt,
                .predecessor_rank = predecessor,
            });
        }
        predecessor = selected;
    }

    while (!frontier.empty() && nodes.size() - 1 < options.nodes) {
        const auto item = frontier.top();
        frontier.pop();
        const double lp = -item.neg_log_probability;
        const std::int32_t node = static_cast<std::int32_t>(nodes.size());
        nodes.push_back(tree_detail::BuilderNode{
            .token = candidates[static_cast<std::size_t>(item.step) * 16 + item.candidate_rank],
            .parent = item.parent,
            .depth = nodes[static_cast<std::size_t>(item.parent)].depth + 1,
            .score = std::exp(lp),
            .step = item.step,
            .candidate_rank = item.candidate_rank,
        });
        nodes[static_cast<std::size_t>(item.parent)].children.push_back(node);

        if (item.step + 1 < steps) {
            const std::uint32_t next_step = item.step + 1;
            const float* edge =
                lattice + (static_cast<std::size_t>(next_step) * 16 + item.candidate_rank) * 16;
            for (std::uint32_t next = 0; next < 16; ++next) {
                ++tie;
                frontier.push(tree_detail::Frontier{
                    .neg_log_probability =
                        -(lp + tree_detail::log_softmax_value(edge, next)),
                    .tie = tie,
                    .parent = node,
                    .step = next_step,
                    .candidate_rank = next,
                    .predecessor_rank = item.candidate_rank,
                });
            }
        }
    }

    // Target tree kernels consume DFS preorder. Reorder the builder while keeping the root first.
    std::vector<std::int32_t> order;
    order.reserve(nodes.size());
    const auto visit = [&](auto&& self, std::int32_t node) -> void {
        order.push_back(node);
        auto children = nodes[static_cast<std::size_t>(node)].children;
        std::stable_sort(children.begin(), children.end(), [&](std::int32_t a, std::int32_t b) {
            return nodes[static_cast<std::size_t>(a)].score >
                   nodes[static_cast<std::size_t>(b)].score;
        });
        for (std::int32_t child : children) self(self, child);
    };
    visit(visit, 0);

    std::array<std::int32_t, kMaximumVerifyTreeRows> remap{};
    remap.fill(-1);
    for (std::size_t i = 0; i < order.size(); ++i) {
        remap[static_cast<std::size_t>(order[i])] = static_cast<std::int32_t>(i);
    }

    VerifyTreePlan out;
    out.rows = static_cast<std::uint32_t>(order.size());
    for (std::uint32_t i = 0; i < out.rows; ++i) {
        const auto& src = nodes[static_cast<std::size_t>(order[i])];
        out.tokens[i] = src.token;
        out.parents[i] = src.parent < 0 ? -1 : remap[static_cast<std::size_t>(src.parent)];
        out.depths[i] = src.depth;
    }
    if (!out.valid()) throw std::logic_error("DFlash2 tree builder produced an invalid DFS tree");
    return out;
}

struct VerifyTreeAcceptance {
    std::array<std::int32_t, kMaximumVerifyTreeRows> path_nodes{};
    std::array<TokenId, kMaximumVerifyTreeRows> licensed_tokens{};
    std::uint32_t count = 0;
    std::uint32_t accepted_drafts = 0;
};

[[nodiscard]] inline VerifyTreeAcceptance
accept_verify_tree(const VerifyTreePlan& tree, const TokenId* target_argmax) {
    if (!tree.valid() || target_argmax == nullptr) {
        throw std::invalid_argument("invalid verify-tree acceptance input");
    }
    VerifyTreeAcceptance out;
    std::int32_t current = 0;
    out.path_nodes[0] = 0;

    while (true) {
        const TokenId wanted = target_argmax[current];
        std::int32_t child = -1;
        for (std::uint32_t i = static_cast<std::uint32_t>(current) + 1; i < tree.rows; ++i) {
            if (tree.parents[i] == current && tree.tokens[i] == wanted) {
                child = static_cast<std::int32_t>(i);
                break;
            }
        }
        if (child < 0) {
            out.licensed_tokens[out.count] = wanted;
            ++out.count;
            break;
        }
        out.licensed_tokens[out.count] = wanted;
        ++out.count;
        ++out.accepted_drafts;
        current = child;
        out.path_nodes[out.count] = current;
        if (out.count >= kMaximumVerifyTreeRows) {
            throw std::logic_error("verify tree accepted path exceeds target width");
        }
    }
    // count licensed tokens equals number of target input rows that must be committed:
    // root plus each accepted draft node. path_nodes[0..count) is therefore the replay selector.
    return out;
}

} // namespace ninfer::models::qwen3_5::detail
