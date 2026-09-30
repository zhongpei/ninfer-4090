#include "models/qwen3_5/program/speculative/stair_router.h"

#include <iostream>

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

    StairRouterState state;
    failures += check(choose_stair_extent(options, state, 15) == 15,
                      "Stair warmup did not choose the widest rung");

    for (int round = 0; round < 12; ++round) {
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

    return failures == 0 ? 0 : 1;
}
