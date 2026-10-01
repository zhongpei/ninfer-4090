#include "models/qwen3_5/program/speculative/stair_router.h"

#include <iostream>
#include <chrono>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

} // namespace

int main() {
    using namespace ninfer;
    using namespace ninfer::models::qwen3_5::detail;

    int failures = 0;
    SpeculativeRoutingOptions options;
    options.mode = SpeculativeRoutingMode::Stair;
    options.widths = {3, 7, 11, 15};
    options.verify_costs = {1.0F, 1.02F, 1.05F, 1.10F};
    options.draft_cost = 0.25F;
    options.warmup_rounds = 4;
    options.probe_period = 16;
    options.switch_margin = 0.0F;
    options.prior_weight = 2.0F;
    options.prior_acceptance = 0.70F;

    StairRouterState state;
    failures += check(choose_stair_extent(options, state, 15) == 15,
                      "Stair warmup did not choose the widest rung");

    for (int round = 0; round < 12; ++round) {
        state.observe(15, 1, options);
    }
    // Twelve observations leave a 0.1 posterior survival for each later position, so the
    // widest rung still earns enough expected commits to outweigh its small extra cost.
    failures += check(choose_stair_extent(options, state, 15) == 15,
                      "limited rejection evidence prematurely narrowed the verify width");
    // After 257 rounds the later-position posterior is 1.4/259. This makes width 3
    // cheaper per expected commit. Avoid a multiple of 16, which intentionally triggers a probe.
    for (int round = 12; round < 257; ++round) {
        state.observe(15, 1, options);
    }
    failures += check(choose_stair_extent(options, state, 15) == 3,
                      "poor long-horizon acceptance did not cut the verify width");

    StairRouterState easy;
    options.warmup_rounds = 0;
    options.probe_period = 0;
    for (int round = 0; round < 16; ++round) {
        easy.observe(15, 15, options);
    }
    failures += check(choose_stair_extent(options, easy, 15) == 15,
                      "full acceptance did not keep the widest verify width");

    StairRouterState censored;
    options.probe_period = 4;
    for (int round = 0; round < 4; ++round) {
        censored.observe(3, 1, options);
    }
    failures += check(choose_stair_extent(options, censored, 15) == 15,
                      "periodic wide probe did not escape censored narrow observations");

    failures += check(choose_stair_extent(options, easy, 2) == 2,
                      "tail extent below the first rung was not preserved");

    options.mode = SpeculativeRoutingMode::Fixed;
    failures += check(choose_stair_extent(options, easy, 9) == 9,
                      "fixed mode changed the configured verify extent");

    TreeStairRouterState tree;
    options.mode = SpeculativeRoutingMode::Stair;
    options.warmup_rounds = 0;
    options.probe_period = 0;
    for (int i = 0; i < 12; ++i) {
        tree.observe(3, 3, options);
        tree.observe(15, 1, options);
    }
    failures += check(choose_tree_stair_extent(options, tree, 15) == 11,
                      "Tree-Stair ignored the prior for an unobserved eligible rung");
    for (int i = 0; i < 12; ++i) {
        tree.observe(7, 1, options);
        tree.observe(11, 1, options);
    }
    failures += check(choose_tree_stair_extent(options, tree, 15) == 3,
                      "Tree-Stair did not prefer the higher committed-token-per-cost rung");
    options.mode = SpeculativeRoutingMode::Fixed;
    failures += check(choose_tree_stair_extent(options, tree, 11) == 11,
                      "fixed tree mode changed the maximum tree budget");

    // Two requests admitted from the same snapshot accumulate independent evidence. Publishing
    // the first must neither rewrite the second's decision state nor lose either contribution.
    PersistentSpecRouterState aggregate;
    aggregate.chain.observe(3, 2, options);
    aggregate.tree.observe(7, 2, options);
    const auto original = aggregate;
    RequestSpecRouterState first, second;
    first.begin(SpeculativeRouterScope::Engine, aggregate);
    second.begin(SpeculativeRouterScope::Engine, aggregate);
    first.current.chain.observe(3, 1, options);
    second.current.chain.observe(7, 4, options);
    first.current.tree.observe(3, 2, options);
    second.current.tree.observe(11, 3, options);
    first.finish(SpeculativeRouterScope::Engine, aggregate, true);
    failures += check(second.current.chain.rounds == original.chain.rounds + 1 &&
                          second.current.chain.attempted[0] == original.chain.attempted[0] + 1 &&
                          second.current.tree.observed[0] == original.tree.observed[0],
                      "another request's publication changed the active router snapshot");
    second.finish(SpeculativeRouterScope::Engine, aggregate, true);
    failures += check(aggregate.chain.rounds == original.chain.rounds + 2 &&
                          aggregate.chain.attempted[0] == original.chain.attempted[0] + 2 &&
                          aggregate.chain.accepted[0] == original.chain.accepted[0] + 2 &&
                          aggregate.chain.accepted[3] == original.chain.accepted[3] + 1 &&
                          aggregate.chain.selected[0] == original.chain.selected[0] + 1 &&
                          aggregate.chain.selected[1] == original.chain.selected[1] + 1 &&
                          aggregate.chain.last_extent == 7 &&
                          aggregate.tree.rounds == original.tree.rounds + 2 &&
                          aggregate.tree.observed[0] == original.tree.observed[0] + 1 &&
                          aggregate.tree.observed[2] == original.tree.observed[2] + 1 &&
                          aggregate.tree.committed_tokens[0] == original.tree.committed_tokens[0] + 3 &&
                          aggregate.tree.committed_tokens[2] == original.tree.committed_tokens[2] + 4 &&
                          aggregate.tree.last_extent == 11,
                      "concurrent router completion lost or repeated snapshot evidence");
    const auto completed = aggregate;
    first.finish(SpeculativeRouterScope::Engine, aggregate, true);
    second.finish(SpeculativeRouterScope::Engine, aggregate, true);
    failures += check(aggregate.chain.rounds == completed.chain.rounds &&
                          aggregate.tree.rounds == completed.tree.rounds,
                      "a repeated completion republished router evidence");

    first.begin(SpeculativeRouterScope::Request, aggregate);
    failures += check(first.current.chain.rounds == 0 && first.current.tree.rounds == 0,
                      "reused request router retained the previous admission snapshot");
    RequestSpecRouterState request_only, failed, chain_only, tree_only;
    request_only.begin(SpeculativeRouterScope::Request, aggregate);
    failures += check(request_only.current.chain.rounds == 0 && request_only.current.tree.rounds == 0,
                      "request-scope admission inherited engine evidence");
    request_only.current.chain.observe(3, 2, options);
    request_only.current.tree.observe(3, 2, options);
    request_only.finish(SpeculativeRouterScope::Request, aggregate, true);
    failed.begin(SpeculativeRouterScope::Engine, aggregate);
    failed.current.chain.observe(3, 2, options);
    failed.current.tree.observe(3, 2, options);
    failed.finish(SpeculativeRouterScope::Engine, aggregate, false);
    failures += check(aggregate.chain.rounds == completed.chain.rounds &&
                          aggregate.tree.rounds == completed.tree.rounds,
                      "request-only or failed completion published engine router evidence");
    chain_only.begin(SpeculativeRouterScope::Engine, aggregate);
    chain_only.current.chain.observe(3, 0, options);
    chain_only.finish(SpeculativeRouterScope::Engine, aggregate, true);
    failures += check(aggregate.chain.rounds == completed.chain.rounds + 1 &&
                          aggregate.tree.rounds == completed.tree.rounds &&
                          aggregate.tree.last_extent == completed.tree.last_extent,
                      "chain-only evidence changed tree counters or last extent");
    tree_only.begin(SpeculativeRouterScope::Engine, aggregate);
    tree_only.current.tree.observe(15, 1, options);
    tree_only.finish(SpeculativeRouterScope::Engine, aggregate, true);
    failures += check(aggregate.tree.rounds == completed.tree.rounds + 1 &&
                          aggregate.chain.rounds == completed.chain.rounds + 1 &&
                          aggregate.chain.last_extent == 3,
                      "tree-only evidence changed chain counters or last extent");

    const auto state_path = std::filesystem::temp_directory_path() /
        ("ninfer_router_state_test_" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    save_persistent_spec_router_state(state_path, aggregate);
    PersistentSpecRouterState restored;
    failures += check(load_persistent_spec_router_state(state_path, restored) &&
                          restored.chain.rounds == aggregate.chain.rounds &&
                          restored.chain.attempted == aggregate.chain.attempted &&
                          restored.chain.accepted == aggregate.chain.accepted &&
                          restored.chain.selected == aggregate.chain.selected &&
                          restored.chain.last_extent == aggregate.chain.last_extent &&
                          restored.tree.observed == aggregate.tree.observed &&
                          restored.tree.committed_tokens == aggregate.tree.committed_tokens &&
                          restored.tree.selected == aggregate.tree.selected &&
                          restored.tree.rounds == aggregate.tree.rounds &&
                          restored.tree.last_extent == aggregate.tree.last_extent,
                      "combined chain/tree snapshot lost published evidence");
    std::filesystem::remove(state_path);

    return failures == 0 ? 0 : 1;
}
