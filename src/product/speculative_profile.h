#pragma once

#include "ninfer/types.h"
#include "product/speculative_options.h"

#include <fstream>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

namespace ninfer::product {

inline void apply_speculative_profile(const std::filesystem::path& path,
                                      SpeculativeOptions& options) {
    std::ifstream input(path);
    if (!input) {
        throw std::invalid_argument("cannot open speculative profile: " + path.string());
    }
    nlohmann::json value;
    input >> value;
    if (!value.is_object() || value.value("version", 0) != 1) {
        throw std::invalid_argument("invalid speculative profile version: " + path.string());
    }
    if (value.contains("stair_widths")) {
        const auto widths = value.at("stair_widths").get<std::vector<std::uint32_t>>();
        if (widths.size() != kSpeculativeStairLevels) {
            throw std::invalid_argument("speculative profile stair_widths must contain four values");
        }
        for (std::size_t i = 0; i < widths.size(); ++i) options.routing.widths[i] = widths[i];
    }
    if (value.contains("stair_costs")) {
        const auto costs = value.at("stair_costs").get<std::vector<float>>();
        if (costs.size() != kSpeculativeStairLevels) {
            throw std::invalid_argument("speculative profile stair_costs must contain four values");
        }
        for (std::size_t i = 0; i < costs.size(); ++i) options.routing.verify_costs[i] = costs[i];
    }
    if (value.contains("draft_cost")) {
        options.routing.draft_cost = value.at("draft_cost").get<float>();
    }
    if (value.contains("prior_acceptance")) {
        options.routing.prior_acceptance = value.at("prior_acceptance").get<float>();
    }
    if (value.contains("prior_weight")) {
        options.routing.prior_weight = value.at("prior_weight").get<float>();
    }
    if (value.contains("warmup_rounds")) {
        options.routing.warmup_rounds = value.at("warmup_rounds").get<std::uint32_t>();
    }
    if (value.contains("probe_period")) {
        options.routing.probe_period = value.at("probe_period").get<std::uint32_t>();
    }
    if (value.contains("switch_margin")) {
        options.routing.switch_margin = value.at("switch_margin").get<float>();
    }
    options.routing.mode = SpeculativeRoutingMode::Stair;
}

} // namespace ninfer::product
