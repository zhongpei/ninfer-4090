#include "models/qwen3_5/program/planning/graph_profiles.h"
#include "ops/softmax_attention/dense/causal_cache/launch.h"

#include <algorithm>
#include <map>
#include <stdexcept>

namespace ninfer::models::qwen3_5::detail {
namespace {
std::vector<GraphExecutionProfile>
graph_profiles_through(std::uint32_t max_frontier,
                       const std::vector<std::uint32_t>& preferred_ends) {
    std::vector<GraphExecutionProfile> out;
    std::uint32_t begin = 0;
    for (const std::uint32_t preferred_end : preferred_ends) {
        if (begin > max_frontier) { break; }
        const std::uint32_t end = std::min(preferred_end, max_frontier);
        out.push_back({begin, end});
        if (end == max_frontier) { return out; }
        begin = end + 1;
    }
    if (begin <= max_frontier) { out.push_back({begin, max_frontier}); }
    return out;
}

std::vector<GraphExecutionProfile> dflash_base_profiles(std::uint32_t capacity,
                                                        std::uint32_t draft_window) {
    if (draft_window == 0 || capacity == 0) { return {}; }
    const std::uint32_t block        = draft_window + 1;
    const std::uint32_t max_frontier = capacity - 1;
    std::vector<std::uint32_t> ends{
        96U, 127U, 511U, 1023U, 2047U, 4095U, 8191U, 16383U, 32767U, 65536U, 131072U, 196608U,
    };
    const auto add_target_boundary = [&](std::uint32_t visible_end) {
        if (visible_end >= block) { ends.push_back(visible_end - block); }
    };
    for (const std::uint32_t visible_end : {128U, 512U, 2048U, 4096U, 8198U, 16390U, 32768U}) {
        add_target_boundary(visible_end);
    }
    if (draft_window >= 6 && draft_window <= 15) {
        add_target_boundary(draft_window <= 11 ? 512U : 1024U);
    }
    std::sort(ends.begin(), ends.end());
    ends.erase(std::unique(ends.begin(), ends.end()), ends.end());
    return graph_profiles_through(max_frontier, ends);
}

using TopologySignature = std::vector<std::uint32_t>;

// Supported routes change once from prompt to small/chunked attention. Within each route,
// native FP8's prepared-Q selection is monotone in the upper visible-key bound. The finite
// preferred ranges remain split-policy ranges; refine them at node-topology changes as well.
template <class Signature>
std::vector<GraphExecutionProfile> refine_topologies(
    const std::vector<GraphExecutionProfile>& base, Signature&& signature) {
    std::vector<GraphExecutionProfile> out;
    std::map<TopologySignature, std::uint32_t> classes;
    for (const auto& range : base) {
        auto begin = range.min;
        while (begin <= range.max) {
            const auto key = signature(begin);
            auto end = range.max;
            if (key != signature(end)) {
                auto lo = begin;
                auto hi = end;
                while (hi - lo > 1U) {
                    const auto mid = lo + (hi - lo) / 2U;
                    if (signature(mid) == key) { lo = mid; } else { hi = mid; }
                }
                end = lo;
            }
            const auto [entry, inserted] = classes.emplace(key, classes.size());
            (void)inserted;
            out.push_back({begin, end, entry->second});
            if (end == range.max) { break; }
            begin = end + 1U;
        }
    }
    return out;
}

std::uint32_t attention_topology(std::uint32_t capacity, std::uint32_t frontier,
                                 std::uint32_t offset, std::int32_t width,
                                 std::int32_t query_heads, KvCacheStorage storage,
                                 std::uint32_t batch_size) {
    const auto visible = static_cast<std::uint32_t>(std::min<std::uint64_t>(
        capacity, static_cast<std::uint64_t>(frontier) + offset));
    return ops::detail::causal_attention_graph_topology(
        query_heads, width, batch_size, storage, {1, visible});
}

} // namespace

std::vector<GraphExecutionProfile> ordinary_graph_profiles(
    std::uint32_t capacity, std::int32_t query_heads, KvCacheStorage storage,
    std::uint32_t batch_size) {
    // E+1 is the one-token visible window. Early ranges limit empty producer CTAs; later ranges
    // follow measured split-policy transitions until the producer grid reaches its fixed cap.
    return refine_topologies(
        graph_profiles_through(capacity - 1, {127, 511, 2047, 4095, 8197, 16389, 32767}),
        [&](std::uint32_t frontier) {
            return TopologySignature{attention_topology(capacity, frontier, 1, 1,
                query_heads, storage, batch_size)};
        });
}

std::vector<GraphExecutionProfile> mtp_graph_profiles(
    std::uint32_t capacity, std::uint32_t draft_window, std::int32_t query_heads,
    KvCacheStorage storage, std::uint32_t batch_size) {
    if (draft_window == 0 || capacity == 0) { return {}; }
    // Bound the final AR window E+2K at split-policy transitions until the grid reaches its cap.
    std::vector<std::uint32_t> ends;
    const auto add_shifted = [&](std::uint32_t visible_end, std::uint32_t offset) {
        if (visible_end >= offset) { ends.push_back(visible_end - offset); }
    };
    for (const std::uint32_t visible_end : {128U, 512U, 2048U, 4096U, 8198U, 16390U, 32768U}) {
        add_shifted(visible_end, 2 * draft_window);
    }
    // Target verify and MTP batch both have T=K+1 and W=E+K+1. Preserve one concrete INT8
    // implementation per range at the T=4/5/6 launch boundaries.
    if (draft_window == 3) {
        add_shifted(1029, draft_window + 1);
    } else if (draft_window == 4) {
        for (const std::uint32_t visible_end : {128U, 512U, 1029U}) {
            add_shifted(visible_end, draft_window + 1);
        }
    } else if (draft_window == 5) {
        for (const std::uint32_t visible_end : {128U, 160U, 2054U, 8198U}) {
            add_shifted(visible_end, draft_window + 1);
        }
    }
    std::sort(ends.begin(), ends.end());
    ends.erase(std::unique(ends.begin(), ends.end()), ends.end());
    return refine_topologies(graph_profiles_through(capacity - 1, ends),
        [&](std::uint32_t frontier) {
            TopologySignature key{attention_topology(capacity, frontier, draft_window + 1,
                draft_window + 1, query_heads, storage, batch_size)};
            // Target verify and MTP batch use the same geometry/storage/envelope. Every AR
            // step has its own larger visible-key envelope and can add prepared-Q separately.
            for (std::uint32_t step = 0; step + 1 < draft_window; ++step) {
                key.push_back(attention_topology(capacity, frontier, draft_window + step + 2,
                    1, query_heads, storage, batch_size));
            }
            return key;
        });
}

std::vector<GraphExecutionProfile> dflash_graph_profiles(
    SpeculativeBackend backend, std::uint32_t capacity, std::uint32_t draft_window,
    std::uint32_t batch_size, std::int32_t query_heads, KvCacheStorage storage) {
    if (capacity == 0 || draft_window == 0 || draft_window > 15) {
        throw std::invalid_argument("invalid masked draft graph dimensions");
    }
    if (backend == SpeculativeBackend::DFlash2) {
        auto profiles = graph_profiles_through(capacity - 1, {96, 511, 2047, 8191, 32767});
        for (std::size_t i = 0; i < profiles.size(); ++i) {
            profiles[i].topology_class = static_cast<std::uint32_t>(i);
        }
        return profiles;
    }
    return refine_topologies(dflash_base_profiles(capacity, draft_window),
        [&](std::uint32_t frontier) {
            return TopologySignature{frontier > 96U ? 1U : 0U,
                attention_topology(capacity, frontier, draft_window + 1, draft_window + 1,
                    query_heads, storage, batch_size)};
        });
}

} // namespace ninfer::models::qwen3_5::detail
