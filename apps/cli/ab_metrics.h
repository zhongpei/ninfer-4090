#pragma once

// Opt-in machine evidence for the CLI harness. Never parse human-format SI suffixes
// for qualification, and never send this record through stdout (model output).
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <sstream>
#include <string>
#include <string_view>

namespace ninfer::cli {

// Kept independent of CUDA headers so the serializer itself can be checked on a CPU.
// Instantiated with GenerationResult / ResolvedSamplingParameters / MemorySummary.
template <class Result, class Sampling, class Memory>
std::string ab_metrics_record(const Result& result, const Sampling& sampling,
                              const Memory& memory) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::setprecision(std::numeric_limits<double>::max_digits10);
    out << "{\"schema\":1";
    const auto integer = [&](const char* key, auto value) {
        out << ",\"" << key << "\":" << value;
    };
    const auto number = [&](const char* key, double value) {
        out << ",\"" << key << "\":";
        if (std::isfinite(value)) out << value;
        else out << "null";
    };
    const auto ratio = [&](const char* key, double numerator, double denominator) {
        number(key, denominator > 0.0 ? numerator / denominator
                                     : std::numeric_limits<double>::quiet_NaN());
    };
    const auto generated = result.generated_token_ids.size();
    integer("generated", generated);
    integer("decoded", generated == 0 ? 0 : generated - 1);
    integer("prompt_tokens", result.prompt.prompt_tokens);
    integer("finish_reason", static_cast<int>(result.finish_reason));
    number("decode_seconds", result.timings.decode_seconds);
    number("prefill_seconds", result.timings.prefill_seconds);
    ratio("decode_tok_s", generated == 0 ? 0.0 : static_cast<double>(generated - 1),
          result.timings.decode_seconds);
    number("temperature", sampling.temperature);
    number("presence_penalty", sampling.presence_penalty);
    number("frequency_penalty", sampling.frequency_penalty);
    const auto& s = result.speculative;
    integer("rounds", s.rounds);
    integer("drafted", s.drafted_tokens);
    integer("accepted", s.accepted_tokens);
    ratio("acceptance_pct", 100.0 * static_cast<double>(s.accepted_tokens),
          static_cast<double>(s.drafted_tokens));
    ratio("tok_per_round", static_cast<double>(s.rounds) + s.accepted_tokens,
          static_cast<double>(s.rounds));
    integer("tree_rounds", s.tree_rounds);
    integer("tree_fallback", s.tree_fallback_rounds);
    integer("tree_nodes", s.tree_nodes);
    integer("tree_accepted", s.tree_accepted_drafts);
    integer("lookup_queries", s.lookup_queries);
    integer("lookup_hits", s.lookup_hits);
    integer("lookup_rounds", s.lookup_rounds);
    integer("lookup_drafted", s.lookup_drafted_tokens);
    integer("lookup_accepted", s.lookup_accepted_tokens);
    integer("head_skip_rounds", s.lookup_head_skip_rounds);
    integer("workspace_peak_bytes", memory.workspace.peak_used_bytes);
    integer("runtime_reservation_bytes", memory.runtime_reservation_bytes);
    out << ",\"token_ids\":[";
    for (std::size_t i = 0; i < generated; ++i) {
        if (i != 0) out << ',';
        out << result.generated_token_ids[i];
    }
    out << "]}";
    return out.str();
}

template <class Result, class Sampling, class Memory>
void emit_ab_metrics(const Result& result, const Sampling& sampling, const Memory& memory) {
    const char* enabled = std::getenv("NINFER_AB_METRICS");
    if (enabled != nullptr && std::string_view(enabled) == "1") {
        std::cerr << "\nNINFER_METRICS_JSON " << ab_metrics_record(result, sampling, memory) << '\n';
    }
}

} // namespace ninfer::cli
