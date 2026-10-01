#pragma once

#include "ninfer/types.h"

#include <nlohmann/json.hpp>

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace ninfer::product {

// Load only routing economics. Model identity / device metadata remain advisory so one profile can
// be inspected or copied, but malformed widths/costs are rejected by the ordinary option validator.
inline void apply_speculative_profile(const std::filesystem::path& path,
                                      SpeculativeRoutingOptions& routing) {
    std::ifstream input(path);
    if (!input) {
        throw std::invalid_argument("cannot open speculative profile: " + path.string());
    }
    nlohmann::json doc;
    input >> doc;
    if (!doc.is_object()) {
        throw std::invalid_argument("speculative profile must be a JSON object");
    }
    const auto widths = doc.at("widths").get<std::array<std::uint32_t, kSpeculativeStairLevels>>();
    const auto costs = doc.at("verify_costs").get<std::array<float, kSpeculativeStairLevels>>();
    const float draft_cost = doc.value("draft_cost", routing.draft_cost);
    for (std::size_t i = 0; i < widths.size(); ++i) {
        if (widths[i] == 0 || widths[i] > 15 ||
            (i != 0 && widths[i] <= widths[i - 1]) ||
            !std::isfinite(costs[i]) || !(costs[i] > 0.0F)) {
            throw std::invalid_argument("speculative profile has invalid width/cost table");
        }
    }
    if (!std::isfinite(draft_cost) || draft_cost < 0.0F) {
        throw std::invalid_argument("speculative profile has invalid draft_cost");
    }
    routing.widths = widths;
    routing.verify_costs = costs;
    routing.draft_cost = draft_cost;
    if (doc.contains("prior_acceptance")) {
        routing.prior_acceptance = doc.at("prior_acceptance").get<float>();
    }
    if (doc.contains("prior_weight")) {
        routing.prior_weight = doc.at("prior_weight").get<float>();
    }
    if (doc.contains("switch_margin")) {
        routing.switch_margin = doc.at("switch_margin").get<float>();
    }
}

} // namespace ninfer::product
