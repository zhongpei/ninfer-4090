#pragma once

#include "ninfer/types.h"

#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::product {

[[nodiscard]] inline SpeculativeBackend parse_speculative_backend(std::string_view value) {
    if (value == "mtp") { return SpeculativeBackend::Mtp; }
    if (value == "dflash") { return SpeculativeBackend::DFlash; }
    if (value == "dflash2") { return SpeculativeBackend::DFlash2; }
    throw std::invalid_argument("invalid speculative backend: " + std::string(value));
}

[[nodiscard]] inline SpeculativeTreeMode parse_speculative_tree_mode(std::string_view value) {
    if (value == "off") { return SpeculativeTreeMode::Off; }
    if (value == "lattice") { return SpeculativeTreeMode::Lattice; }
    throw std::invalid_argument("invalid speculative tree mode: " + std::string(value));
}

[[nodiscard]] inline LookupDraftStrategy parse_lookup_draft_strategy(std::string_view value) {
    if (value == "recent") { return LookupDraftStrategy::Recent; }
    if (value == "vote") { return LookupDraftStrategy::Vote; }
    throw std::invalid_argument("invalid lookup strategy: " + std::string(value));
}

[[nodiscard]] inline LookupDFlashMode parse_lookup_dflash_mode(std::string_view value) {
    if (value == "off") { return LookupDFlashMode::Off; }
    if (value == "replace") { return LookupDFlashMode::Replace; }
    if (value == "skip") { return LookupDFlashMode::HeadSkip; }
    throw std::invalid_argument("invalid lookup DFlash mode: " + std::string(value));
}

[[nodiscard]] inline SpeculativeRoutingMode
parse_speculative_routing_mode(std::string_view value) {
    if (value == "fixed") { return SpeculativeRoutingMode::Fixed; }
    if (value == "stair") { return SpeculativeRoutingMode::Stair; }
    throw std::invalid_argument("invalid speculative router: " + std::string(value));
}

inline void parse_speculative_stair_widths(std::string_view value,
                                           SpeculativeRoutingOptions& routing) {
    std::array<std::uint32_t, kSpeculativeStairLevels> parsed{};
    std::size_t start = 0;
    for (std::size_t i = 0; i < parsed.size(); ++i) {
        const std::size_t comma = value.find(',', start);
        const std::string_view piece =
            value.substr(start, comma == std::string_view::npos ? std::string_view::npos
                                                                : comma - start);
        if (piece.empty()) { throw std::invalid_argument("--spec-stair-widths has an empty entry"); }
        const std::string text(piece);
        char* end = nullptr;
        const unsigned long n = std::strtoul(text.c_str(), &end, 10);
        if (end == text.c_str() || *end != '\0' || n == 0 || n > 15) {
            throw std::invalid_argument("invalid --spec-stair-widths entry: " + text);
        }
        parsed[i] = static_cast<std::uint32_t>(n);
        if (i != 0 && parsed[i] <= parsed[i - 1]) {
            throw std::invalid_argument("--spec-stair-widths must be strictly increasing");
        }
        if (i + 1 == parsed.size()) {
            if (comma != std::string_view::npos) {
                throw std::invalid_argument("--spec-stair-widths requires exactly four entries");
            }
        } else {
            if (comma == std::string_view::npos) {
                throw std::invalid_argument("--spec-stair-widths requires exactly four entries");
            }
            start = comma + 1;
        }
    }
    routing.widths = parsed;
}

inline void parse_speculative_stair_costs(std::string_view value,
                                          SpeculativeRoutingOptions& routing) {
    std::array<float, kSpeculativeStairLevels> parsed{};
    std::size_t start = 0;
    for (std::size_t i = 0; i < parsed.size(); ++i) {
        const std::size_t comma = value.find(',', start);
        const std::string_view piece =
            value.substr(start, comma == std::string_view::npos ? std::string_view::npos
                                                                : comma - start);
        if (piece.empty()) { throw std::invalid_argument("--spec-stair-costs has an empty entry"); }
        const std::string text(piece);
        char* end = nullptr;
        const float v = std::strtof(text.c_str(), &end);
        if (end == text.c_str() || *end != '\0' || !std::isfinite(v) || !(v > 0.0F)) {
            throw std::invalid_argument("invalid --spec-stair-costs entry: " + text);
        }
        parsed[i] = v;
        if (i + 1 == parsed.size()) {
            if (comma != std::string_view::npos) {
                throw std::invalid_argument("--spec-stair-costs requires exactly four entries");
            }
        } else {
            if (comma == std::string_view::npos) {
                throw std::invalid_argument("--spec-stair-costs requires exactly four entries");
            }
            start = comma + 1;
        }
    }
    routing.verify_costs = parsed;
}

[[nodiscard]] inline const char* speculative_backend_name(SpeculativeBackend backend) noexcept {
    switch (backend) {
    case SpeculativeBackend::None:
        return "none";
    case SpeculativeBackend::Mtp:
        return "mtp";
    case SpeculativeBackend::DFlash:
        return "dflash";
    case SpeculativeBackend::DFlash2:
        return "dflash2";
    }
    return "unknown";
}

inline void validate_speculative_cli_options(const SpeculativeOptions& options) {
    const auto& router = options.routing;
    const auto& tree = options.tree;
    const auto& lookup = options.lookup;
    if (tree.nodes == 0 || tree.nodes > 15 || tree.spine == 0 || tree.spine > 15) {
        throw std::invalid_argument("invalid speculative tree node/spine policy");
    }
    if (tree.mode != SpeculativeTreeMode::Off) {
        if (options.backend != SpeculativeBackend::DFlash2) {
            throw std::invalid_argument("--spec-tree lattice requires --spec dflash2");
        }
        if (options.draft_tokens < 5) {
            throw std::invalid_argument("--spec-tree lattice requires --draft-tokens >= 5");
        }
        if (tree.nodes != options.draft_tokens) {
            throw std::invalid_argument(
                "first-version --spec-tree-nodes must equal --draft-tokens");
        }
        if (tree.spine > tree.nodes) {
            throw std::invalid_argument("--spec-tree-spine must not exceed --spec-tree-nodes");
        }
    }
    if (lookup.max_order == 0 || lookup.max_order > 16 || lookup.max_matches == 0 ||
        lookup.min_support == 0 || !std::isfinite(lookup.min_confidence) ||
        lookup.min_confidence < 0.0F || lookup.min_confidence > 1.0F ||
        lookup.base_drafts == 0 || lookup.base_drafts > 15 ||
        lookup.deep_drafts == 0 || lookup.deep_drafts > 15 ||
        lookup.deep_drafts < lookup.base_drafts || !std::isfinite(lookup.corpus_weight) ||
        lookup.corpus_weight < 0.0F || lookup.corpus_samples == 0) {
        throw std::invalid_argument("invalid context-lookup drafting parameter");
    }
    if (lookup.strategy == LookupDraftStrategy::Vote && options.lookup_ngram != 0 &&
        options.lookup_ngram > lookup.max_order) {
        throw std::invalid_argument(
            "--lookup-ngram must not exceed --lookup-max-order with --lookup-strategy vote");
    }
    if (lookup.dflash_mode != LookupDFlashMode::Off) {
        if (options.lookup_ngram == 0 || lookup.strategy != LookupDraftStrategy::Vote) {
            throw std::invalid_argument(
                "--lookup-dflash replace|skip requires --lookup-ngram and --lookup-strategy vote");
        }
        if (options.backend != SpeculativeBackend::DFlash &&
            options.backend != SpeculativeBackend::DFlash2) {
            throw std::invalid_argument(
                "--lookup-dflash replace|skip requires --spec dflash|dflash2");
        }
        // Copy widths are policy maxima. The runtime clamps both to the startup DFlash K, request
        // budget and remaining context so one A/B profile can compare K7 and K15 artifacts.
    }
    if (!std::isfinite(router.draft_cost) || router.draft_cost < 0.0F ||
        !std::isfinite(router.prior_acceptance) || router.prior_acceptance < 0.0F ||
        router.prior_acceptance > 1.0F || !std::isfinite(router.prior_weight) ||
        router.prior_weight < 0.0F || !std::isfinite(router.switch_margin) ||
        router.switch_margin < 0.0F) {
        throw std::invalid_argument("invalid speculative Stair router numeric parameter");
    }
    for (std::size_t i = 0; i < kSpeculativeStairLevels; ++i) {
        if (router.widths[i] == 0 || router.widths[i] > 15 ||
            (i != 0 && router.widths[i] <= router.widths[i - 1]) ||
            !std::isfinite(router.verify_costs[i]) || !(router.verify_costs[i] > 0.0F)) {
            throw std::invalid_argument("invalid speculative Stair router width/cost table");
        }
    }
    if (router.mode == SpeculativeRoutingMode::Stair) {
        if (options.backend != SpeculativeBackend::DFlash &&
            options.backend != SpeculativeBackend::DFlash2) {
            throw std::invalid_argument("--spec-router stair requires --spec dflash|dflash2");
        }
        if (router.widths.back() > options.draft_tokens) {
            throw std::invalid_argument(
                "--spec-stair-widths must not exceed the configured --draft-tokens maximum");
        }
    }

    switch (options.backend) {
    case SpeculativeBackend::None:
        if (options.draft_tokens != 0 || options.proposal_head != ProposalHead::Full) {
            throw std::invalid_argument(
                "--draft-tokens and --lm-head-draft require --spec mtp|dflash|dflash2");
        }
        return;
    case SpeculativeBackend::Mtp:
        if (options.draft_tokens == 0 || options.draft_tokens > 5) {
            throw std::invalid_argument("--spec mtp requires --draft-tokens in [1,5]");
        }
        return;
    case SpeculativeBackend::DFlash:
        if (options.draft_tokens == 0 || options.draft_tokens > 15) {
            throw std::invalid_argument("--spec dflash requires --draft-tokens in [1,15]");
        }
        return;
    case SpeculativeBackend::DFlash2:
        if (options.draft_tokens == 0 || options.draft_tokens > 15) {
            throw std::invalid_argument("--spec dflash2 requires --draft-tokens in [1,15]");
        }
        return;
    }
    throw std::invalid_argument("invalid speculative backend");
}

} // namespace ninfer::product
