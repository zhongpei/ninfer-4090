#pragma once

#include "ninfer/types.h"

#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>

// Shared public argv behavior for both product parsers; no Engine/model loading.
template <class Parse>
int calibrated_product_options_cases(Parse parse, const std::vector<std::string>& base,
                                     const std::string& help) {
    int failures = 0;
    const auto defaults = parse(base);
    if (defaults.speculative.routing.mode != ninfer::SpeculativeRoutingMode::Fixed ||
        !defaults.speculative.routing.profile_path.empty()) {
        std::cerr << "fixed/profile-free default changed\n";
        ++failures;
    }
    const auto configured = [&] {
        auto args = base;
        args.insert(args.end(), {"--spec", "dflash2", "--draft-tokens", "15",
                                 "--spec-router", "calibrated", "--spec-router-profile", "route.json"});
        return args;
    };
    try {
        const auto options = parse(configured());
        if (options.speculative.routing.mode != ninfer::SpeculativeRoutingMode::Calibrated ||
            options.speculative.routing.profile_path != "route.json" ||
            options.speculative.routing.scope != ninfer::SpeculativeRouterScope::Request) {
            std::cerr << "calibrated options not preserved\n";
            ++failures;
        }
    } catch (const std::exception& error) {
        std::cerr << "valid calibrated argv rejected: " << error.what() << '\n';
        ++failures;
    }
    for (const std::vector<std::string>& sampling : {
        std::vector<std::string>{"--temperature", "0.7"},
        {"--presence-penalty", "0.5"}, {"--frequency-penalty", "0.5"}}) {
        auto args = configured();
        args.insert(args.end(), sampling.begin(), sampling.end());
        try { (void)parse(args); }
        catch (const std::exception& error) {
            std::cerr << "calibrated parser rejected runtime fallback sampling: " << error.what() << '\n';
            ++failures;
        }
    }
    if (help.find("fixed|stair|calibrated") == std::string::npos ||
        help.find("--spec-router-profile PATH") == std::string::npos) {
        std::cerr << "help omits calibrated/profile entry\n";
        ++failures;
    }
    if (help.find("greedy sampling with zero presence/frequency penalties") == std::string::npos ||
        help.find("other sampling uses existing fixed K15") == std::string::npos) {
        std::cerr << "help omits calibrated sampling eligibility/fallback\n";
        ++failures;
    }
    const auto rejects = [&](const std::vector<std::string>& args, const char* label) {
        try { (void)parse(args); }
        catch (const std::invalid_argument&) { return; }
        std::cerr << "accepted incompatible calibrated options: " << label << '\n';
        ++failures;
    };
    auto missing = configured();
    missing.resize(missing.size() - 2);
    rejects(missing, "missing profile");
    for (const char* mode : {"fixed", "stair"}) {
        auto args = base;
        args.insert(args.end(), {"--spec", "dflash2", "--draft-tokens", "15",
                                 "--spec-router", mode, "--spec-router-profile", "route.json"});
        rejects(args, "profile with another mode");
    }
    for (const char* backend : {"mtp", "dflash"}) {
        auto args = base;
        args.insert(args.end(), {"--spec", backend, "--draft-tokens", backend == std::string("mtp") ? "3" : "15",
                                 "--spec-router", "calibrated", "--spec-router-profile", "route.json"});
        rejects(args, "wrong backend");
    }
    auto no_backend = base;
    no_backend.insert(no_backend.end(), {"--spec-router", "calibrated", "--spec-router-profile", "route.json"});
    rejects(no_backend, "missing backend");
    auto small = base;
    small.insert(small.end(), {"--spec", "dflash2", "--draft-tokens", "7",
                              "--spec-router", "calibrated", "--spec-router-profile", "route.json"});
    rejects(small, "startup K below 15");
    for (const std::vector<std::string>& suffix : {
        std::vector<std::string>{"--spec-tree", "lattice"},
        {"--lookup-ngram", "5"}, {"--lookup-strategy", "vote", "--lookup-ngram", "5", "--lookup-dflash", "skip"},
        {"--lookup-persistent-tokens", "10"}, {"--lookup-corpus-prefix", "corpus"},
        {"--spec-router-scope", "engine"}, {"--spec-router-scope", "engine", "--spec-router-state", "state"},
        {"--spec-stair-widths", "1,7,11,15"}, {"--spec-stair-costs", "2,2,2,2"},
        {"--spec-stair-draft-cost", "0.5"}, {"--spec-stair-prior", "0.5"},
        {"--spec-stair-prior-weight", "3"}, {"--spec-stair-warmup", "5"},
        {"--spec-stair-probe-period", "0"}, {"--spec-stair-margin", "0.03"}}) {
        auto args = configured();
        args.insert(args.end(), suffix.begin(), suffix.end());
        rejects(args, suffix.front().c_str());
    }
    return failures;
}
