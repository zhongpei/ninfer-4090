#include "models/qwen3_5/program/speculative/tree_plan.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>

namespace {

int check(bool condition, const char* message) {
    if (condition) return 0;
    std::cerr << message << '\n';
    return 1;
}

} // namespace

int main() {
    using namespace ninfer;
    using namespace ninfer::models::qwen3_5::detail;

    constexpr std::uint32_t steps = 3;
    std::array<TokenId, 16 * steps> candidates{};
    for (std::uint32_t step = 0; step < steps; ++step) {
        for (std::uint32_t rank = 0; rank < 16; ++rank) {
            candidates[static_cast<std::size_t>(step) * 16 + rank] =
                static_cast<TokenId>(100 + step * 100 + rank);
        }
    }
    candidates[0] = 11;
    candidates[16] = 12;
    candidates[32] = 13;
    candidates[1] = 21;
    candidates[17] = 22;
    candidates[33] = 23;

    std::array<float, 16 * 16 * steps> lattice{};
    lattice.fill(-8.0F);
    for (std::uint32_t step = 0; step < steps; ++step) {
        for (std::uint32_t predecessor = 0; predecessor < 16; ++predecessor) {
            float* row = lattice.data() +
                         (static_cast<std::size_t>(step) * 16 + predecessor) * 16;
            row[0] = 4.0F;
            row[1] = predecessor == 1 ? 3.5F : 2.0F;
        }
    }

    SpeculativeTreeOptions options;
    options.mode = SpeculativeTreeMode::Lattice;
    options.nodes = 7;
    options.spine = 3;

    const VerifyTreePlan tree =
        build_verify_tree(10, candidates.data(), lattice.data(), steps, options);

    int failures = 0;
    failures += check(tree.valid(), "tree planner produced an invalid DFS tree");
    failures += check(tree.rows == options.nodes + 1,
                      "tree planner did not fill the requested node budget");
    failures += check(tree.tokens[0] == 10 && tree.parents[0] == -1 && tree.depths[0] == 0,
                      "tree planner changed the anchor contract");

    std::int32_t node11 = -1;
    std::int32_t node12 = -1;
    for (std::uint32_t i = 1; i < tree.rows; ++i) {
        if (tree.parents[i] == 0 && tree.tokens[i] == 11) node11 = static_cast<std::int32_t>(i);
    }
    if (node11 >= 0) {
        for (std::uint32_t i = 1; i < tree.rows; ++i) {
            if (tree.parents[i] == node11 && tree.tokens[i] == 12)
                node12 = static_cast<std::int32_t>(i);
        }
    }
    failures += check(node11 > 0 && node12 > node11,
                      "tree planner lost the greedy lattice spine");

    std::array<TokenId, kMaximumVerifyTreeRows> target{};
    target.fill(-1);
    target[0] = 11;
    if (node11 >= 0) target[static_cast<std::size_t>(node11)] = 12;
    if (node12 >= 0) target[static_cast<std::size_t>(node12)] = 777;

    const VerifyTreeAcceptance accepted = accept_verify_tree(tree, target.data());
    failures += check(accepted.count == 3 && accepted.accepted_drafts == 2,
                      "tree acceptance returned the wrong committed extent");
    failures += check(accepted.licensed_tokens[0] == 11 &&
                          accepted.licensed_tokens[1] == 12 &&
                          accepted.licensed_tokens[2] == 777,
                      "tree acceptance returned the wrong licensed token sequence");
    failures += check(accepted.path_nodes[0] == 0 &&
                          accepted.path_nodes[1] == node11 &&
                          accepted.path_nodes[2] == node12,
                      "tree acceptance did not preserve the target input path");

    // Missing branch: the target correction is licensed immediately at the current node.
    target.fill(-1);
    target[0] = 999;
    const VerifyTreeAcceptance rejected = accept_verify_tree(tree, target.data());
    failures += check(rejected.count == 1 && rejected.accepted_drafts == 0 &&
                          rejected.licensed_tokens[0] == 999 &&
                          rejected.path_nodes[0] == 0,
                      "tree root rejection did not preserve correction semantics");

    return failures == 0 ? 0 : 1;
}
