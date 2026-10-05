#include "evaluation.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ninfer::perplexity {

std::vector<WindowPlan> plan_windows(std::size_t tokens, std::uint32_t context,
                                     std::uint32_t stride) {
    if (tokens < 2) { throw std::invalid_argument("perplexity stream must contain two tokens"); }
    if (context < 2 || stride == 0 || stride >= context) {
        throw std::invalid_argument("perplexity requires context>=2 and 1<=stride<context");
    }

    std::vector<WindowPlan> windows;
    std::size_t previous_end = std::min<std::size_t>(tokens, context);
    windows.push_back(WindowPlan{.input_begin  = 0,
                                 .input_end    = previous_end,
                                 .target_begin = 1,
                                 .target_end   = previous_end,
                                 .first_target = 1});
    while (previous_end < tokens) {
        const std::size_t end          = std::min(tokens, previous_end + stride);
        const std::size_t begin        = end > context ? end - context : 0;
        const std::size_t local_target = previous_end - begin;
        if (local_target == 0 || local_target >= end - begin ||
            local_target > std::numeric_limits<std::uint32_t>::max()) {
            throw std::logic_error("perplexity window has an invalid target suffix");
        }
        windows.push_back(WindowPlan{
            .input_begin  = begin,
            .input_end    = end,
            .target_begin = previous_end,
            .target_end   = end,
            .first_target = static_cast<std::uint32_t>(local_target),
        });
        previous_end = end;
    }
    return windows;
}

std::vector<WindowPlan> plan_depth_windows(std::size_t tokens,
                                                   std::span<const std::uint32_t> depths,
                                                   std::uint32_t tail_tokens) {
    if (tokens < 2) { throw std::invalid_argument("perplexity stream must contain two tokens"); }
    if (depths.empty()) { throw std::invalid_argument("long-context perplexity requires depths"); }
    if (tail_tokens == 0) { throw std::invalid_argument("long-context perplexity tail must be positive"); }

    std::vector<WindowPlan> windows;
    windows.reserve(depths.size());
    std::uint32_t previous = 0;
    bool first = true;
    for (const std::uint32_t depth : depths) {
        if (depth == 0) { throw std::invalid_argument("long-context perplexity depth must be positive"); }
        if (!first && depth <= previous) {
            throw std::invalid_argument("long-context perplexity depths must be strictly increasing");
        }
        first = false;
        previous = depth;
        if (static_cast<std::size_t>(depth) >= tokens) { continue; }
        const std::size_t target_end =
            std::min<std::size_t>(tokens, static_cast<std::size_t>(depth) + tail_tokens);
        windows.push_back(WindowPlan{
            .input_begin = 0,
            .input_end = target_end,
            .target_begin = depth,
            .target_end = target_end,
            .first_target = depth,
        });
    }
    return windows;
}

void ScoreAggregate::add(std::span<const float> logprobs) {
    for (const float logprob : logprobs) {
        if (!std::isfinite(logprob)) {
            throw std::runtime_error("causal scoring returned a non-finite logprob");
        }
        total_nll -= static_cast<double>(logprob);
    }
    if (logprobs.size() > std::numeric_limits<std::uint64_t>::max() - scored_tokens) {
        throw std::overflow_error("perplexity scored-token count overflowed");
    }
    scored_tokens += static_cast<std::uint64_t>(logprobs.size());
}

void ScoreAggregate::add(const ScoreAggregate& other) noexcept {
    scored_tokens += other.scored_tokens;
    total_nll += other.total_nll;
}

double ScoreAggregate::mean_nll() const {
    if (scored_tokens == 0) { throw std::logic_error("perplexity aggregate is empty"); }
    return total_nll / static_cast<double>(scored_tokens);
}

double ScoreAggregate::ppl() const { return std::exp(mean_nll()); }

} // namespace ninfer::perplexity
