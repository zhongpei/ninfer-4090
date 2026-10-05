#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::perplexity {

struct WindowPlan {
    std::size_t input_begin    = 0;
    std::size_t input_end      = 0;
    std::size_t target_begin   = 0;
    std::size_t target_end     = 0;
    std::uint32_t first_target = 0;
};

[[nodiscard]] std::vector<WindowPlan> plan_windows(std::size_t tokens, std::uint32_t context,
                                                   std::uint32_t stride);

// Long-history quality protocol. Each requested depth D builds the real prefix [0,D) into Main KV
// and scores only the following tail. Unlike plan_windows(), it never truncates history or resets
// the window near D, so KV quantization error accumulated across a long cache is observable in the
// measured suffix NLL. Depths that do not leave at least one target token in this stream are
// omitted; callers record per-depth coverage explicitly.
[[nodiscard]] std::vector<WindowPlan>
plan_depth_windows(std::size_t tokens, std::span<const std::uint32_t> depths,
                   std::uint32_t tail_tokens);

struct ScoreAggregate {
    std::uint64_t scored_tokens = 0;
    double total_nll            = 0.0;

    void add(std::span<const float> logprobs);
    void add(const ScoreAggregate& other) noexcept;
    [[nodiscard]] double mean_nll() const;
    [[nodiscard]] double ppl() const;
};

} // namespace ninfer::perplexity
