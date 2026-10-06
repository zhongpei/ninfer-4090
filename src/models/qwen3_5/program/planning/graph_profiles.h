#pragma once
#include "models/qwen3_5/program/program.h"

namespace ninfer::models::qwen3_5::detail {

[[nodiscard]] std::vector<GraphExecutionProfile> ordinary_graph_profiles(
    std::uint32_t capacity, std::int32_t query_heads, KvCacheStorage storage,
    std::uint32_t batch_size);
[[nodiscard]] std::vector<GraphExecutionProfile> mtp_graph_profiles(
    std::uint32_t capacity, std::uint32_t draft_window, std::int32_t query_heads,
    KvCacheStorage storage, std::uint32_t batch_size);
[[nodiscard]] std::vector<GraphExecutionProfile> dflash_graph_profiles(
    SpeculativeBackend backend, std::uint32_t capacity, std::uint32_t draft_window,
    std::uint32_t batch_size, std::int32_t query_heads, KvCacheStorage storage);

} // namespace ninfer::models::qwen3_5::detail
