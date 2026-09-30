#pragma once
#include "models/qwen3_5/program/speculative/lookup_draft.h"
#include "ninfer/types.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ninfer::qwen3_5 {

inline constexpr std::uint32_t kLookupMaximumDrafts = 15;
inline constexpr TokenId kLookupDocumentSeparator = static_cast<TokenId>(-1);

namespace lookup_detail {

inline std::uint64_t hash_tokens(std::span<const TokenId> tokens, std::uint32_t order) noexcept {
    std::uint64_t h = 1469598103934665603ULL ^ (static_cast<std::uint64_t>(order) << 48U);
    for (TokenId token : tokens) {
        const auto word = static_cast<std::uint32_t>(token);
        for (unsigned shift = 0; shift != 32; shift += 8) {
            h ^= static_cast<std::uint8_t>(word >> shift);
            h *= 1099511628211ULL;
        }
    }
    return h;
}

inline bool equal_at(std::span<const TokenId> a, std::size_t start,
                     std::span<const TokenId> b) noexcept {
    return start <= a.size() && b.size() <= a.size() - start &&
           std::equal(b.begin(), b.end(), a.begin() + static_cast<std::ptrdiff_t>(start));
}

struct MatchSet {
    std::uint32_t order = 0;
    std::vector<std::size_t> follow;
    float weight = 1.0F;
    std::uint8_t source = 0;
};

inline MatchSet local_matches(std::span<const TokenId> ledger, std::uint32_t min_order,
                              std::uint32_t max_order, std::uint32_t max_matches) {
    MatchSet out;
    out.source = 1;
    max_order = std::min<std::uint32_t>(max_order, static_cast<std::uint32_t>(ledger.size()));
    for (std::uint32_t order = max_order; order >= min_order && order != 0; --order) {
        if (ledger.size() <= order) { continue; }
        const auto pattern = ledger.last(order);
        const std::size_t last_start = ledger.size() - order;
        for (std::size_t start = last_start; start-- > 0;) {
            if (!equal_at(ledger, start, pattern)) { continue; }
            out.follow.push_back(start + order);
            if (out.follow.size() >= max_matches) { break; }
        }
        if (!out.follow.empty()) {
            out.order = order;
            return out;
        }
        if (order == min_order) { break; }
    }
    return out;
}

template <class T>
inline std::vector<T> read_binary(const std::filesystem::path& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { throw std::runtime_error("cannot open lookup corpus: " + path.string()); }
    const std::streamsize bytes = f.tellg();
    if (bytes < 0 || bytes % static_cast<std::streamsize>(sizeof(T)) != 0) {
        throw std::runtime_error("invalid lookup corpus array: " + path.string());
    }
    f.seekg(0);
    std::vector<T> out(static_cast<std::size_t>(bytes) / sizeof(T));
    if (!out.empty() &&
        !f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(bytes))) {
        throw std::runtime_error("cannot read lookup corpus: " + path.string());
    }
    return out;
}

} // namespace lookup_detail

class LookupPersistentStore {
public:
    LookupPersistentStore() = default;
    LookupPersistentStore(std::uint32_t max_tokens, std::uint32_t max_order,
                          std::uint32_t max_matches)
        : max_tokens_(max_tokens), max_order_(max_order), max_matches_(max_matches) {
        if (max_tokens_ != 0) { tokens_.reserve(static_cast<std::size_t>(max_tokens_) + 1U); }
    }

    bool enabled() const noexcept { return max_tokens_ != 0; }

    void append_sequence(std::span<const TokenId> seq) {
        if (!enabled() || seq.empty()) { return; }
        if (tokens_.size() + seq.size() + 1U > max_tokens_) {
            tokens_.clear();
            index_.clear();
        }
        std::size_t take = seq.size();
        if (take + 1U > max_tokens_) { take = max_tokens_ > 1U ? max_tokens_ - 1U : 0U; }
        if (take == 0) { return; }
        const std::size_t first = seq.size() - take;
        tokens_.push_back(kLookupDocumentSeparator);
        for (std::size_t i = first; i < seq.size(); ++i) {
            const std::size_t follow = tokens_.size();
            tokens_.push_back(seq[i]);
            record(follow);
        }
    }

    lookup_detail::MatchSet lookup(std::span<const TokenId> context, std::uint32_t min_order,
                                   std::uint32_t max_order,
                                   std::uint32_t max_matches) const {
        lookup_detail::MatchSet out;
        out.source = 2;
        max_order = std::min({max_order, max_order_,
                              static_cast<std::uint32_t>(context.size())});
        for (std::uint32_t order = max_order; order >= min_order && order != 0; --order) {
            const auto pattern = context.last(order);
            const auto it = index_.find(lookup_detail::hash_tokens(pattern, order));
            if (it != index_.end()) {
                for (auto p = it->second.rbegin();
                     p != it->second.rend() && out.follow.size() < max_matches; ++p) {
                    if (*p < order || *p >= tokens_.size()) { continue; }
                    const auto stored =
                        std::span<const TokenId>(tokens_).subspan(*p - order, order);
                    if (std::find(stored.begin(), stored.end(), kLookupDocumentSeparator) ==
                            stored.end() &&
                        std::equal(stored.begin(), stored.end(), pattern.begin()) &&
                        tokens_[*p] != kLookupDocumentSeparator) {
                        out.follow.push_back(*p);
                    }
                }
            }
            if (!out.follow.empty()) {
                out.order = order;
                return out;
            }
            if (order == min_order) { break; }
        }
        return out;
    }

    std::span<const TokenId> continuation(std::size_t pos,
                                          std::uint32_t maximum) const noexcept {
        if (pos >= tokens_.size()) { return {}; }
        std::size_t n = std::min<std::size_t>(maximum, tokens_.size() - pos);
        for (std::size_t i = 0; i < n; ++i) {
            if (tokens_[pos + i] == kLookupDocumentSeparator) {
                n = i;
                break;
            }
        }
        return std::span<const TokenId>(tokens_).subspan(pos, n);
    }

private:
    void record(std::size_t follow) {
        for (std::uint32_t order = 1; order <= max_order_ && follow >= order; ++order) {
            const auto gram =
                std::span<const TokenId>(tokens_).subspan(follow - order, order);
            if (std::find(gram.begin(), gram.end(), kLookupDocumentSeparator) != gram.end()) {
                continue;
            }
            auto& bucket = index_[lookup_detail::hash_tokens(gram, order)];
            bucket.push_back(follow);
            if (bucket.size() > max_matches_) {
                bucket.erase(bucket.begin(),
                             bucket.begin() + static_cast<std::ptrdiff_t>(bucket.size() -
                                                                          max_matches_));
            }
        }
    }

    std::uint32_t max_tokens_ = 0, max_order_ = 0, max_matches_ = 0;
    std::vector<TokenId> tokens_;
    std::unordered_map<std::uint64_t, std::vector<std::size_t>> index_;
};

class LookupCorpusStore {
public:
    LookupCorpusStore() = default;
    explicit LookupCorpusStore(const std::filesystem::path& prefix) {
        if (prefix.empty()) { return; }
        auto tp = prefix; tp += ".tokens.i32";
        auto sp = prefix; sp += ".suffix.u32";
        tokens_ = lookup_detail::read_binary<TokenId>(tp);
        suffix_ = lookup_detail::read_binary<std::uint32_t>(sp);
        if (tokens_.empty() || suffix_.empty()) {
            throw std::runtime_error("lookup corpus arrays are empty");
        }
        for (auto p : suffix_) {
            if (p >= tokens_.size()) { throw std::runtime_error("lookup suffix is out of range"); }
        }
    }

    bool enabled() const noexcept { return !tokens_.empty() && !suffix_.empty(); }

    lookup_detail::MatchSet lookup(std::span<const TokenId> context, std::uint32_t min_order,
                                   std::uint32_t max_order, std::uint32_t samples,
                                   float weight) const {
        lookup_detail::MatchSet out;
        out.source = 4;
        out.weight = weight;
        if (!enabled()) { return out; }
        max_order =
            std::min<std::uint32_t>(max_order, static_cast<std::uint32_t>(context.size()));
        for (std::uint32_t order = max_order; order >= min_order && order != 0; --order) {
            const auto [lo, hi] = range(context.last(order));
            const std::size_t width = hi - lo;
            if (width != 0) {
                const std::size_t count = std::min<std::size_t>(samples, width);
                out.follow.reserve(count);
                for (std::size_t i = 0; i < count; ++i) {
                    const std::size_t at = count == width ? i : (i * width) / count;
                    const std::size_t follow = static_cast<std::size_t>(suffix_[lo + at]) + order;
                    if (follow < tokens_.size() && tokens_[follow] >= 0) {
                        out.follow.push_back(follow);
                    }
                }
                if (!out.follow.empty()) {
                    out.order = order;
                    return out;
                }
            }
            if (order == min_order) { break; }
        }
        return out;
    }

    std::span<const TokenId> continuation(std::size_t pos,
                                          std::uint32_t maximum) const noexcept {
        if (pos >= tokens_.size()) { return {}; }
        std::size_t n = std::min<std::size_t>(maximum, tokens_.size() - pos);
        for (std::size_t i = 0; i < n; ++i) {
            if (tokens_[pos + i] < 0) { n = i; break; }
        }
        return std::span<const TokenId>(tokens_).subspan(pos, n);
    }

private:
    int compare_at(std::size_t index, std::span<const TokenId> pattern) const noexcept {
        std::size_t pos = suffix_[index];
        for (TokenId want : pattern) {
            if (pos >= tokens_.size()) { return -1; }
            const TokenId have = tokens_[pos++];
            if (have < want) { return -1; }
            if (have > want) { return 1; }
        }
        return 0;
    }

    std::pair<std::size_t, std::size_t>
    range(std::span<const TokenId> pattern) const noexcept {
        std::size_t lo = 0, hi = suffix_.size();
        while (lo < hi) {
            const std::size_t mid = lo + (hi - lo) / 2U;
            if (compare_at(mid, pattern) < 0) lo = mid + 1U; else hi = mid;
        }
        const std::size_t begin = lo;
        hi = suffix_.size();
        while (lo < hi) {
            const std::size_t mid = lo + (hi - lo) / 2U;
            if (compare_at(mid, pattern) <= 0) lo = mid + 1U; else hi = mid;
        }
        return {begin, lo};
    }

    std::vector<TokenId> tokens_;
    std::vector<std::uint32_t> suffix_;
};

struct LookupDraftProposal {
    std::array<TokenId, kLookupMaximumDrafts> tokens{};
    std::uint32_t count = 0, match_order = 0, support = 0;
    float confidence = 0.0F;
    std::uint8_t sources = 0;
    explicit operator bool() const noexcept { return count != 0; }
};

namespace lookup_detail {
struct Vote {
    std::array<TokenId, kLookupMaximumDrafts> tokens{};
    std::uint32_t count = 0, support = 0;
    float weight = 0.0F;
    std::uint8_t sources = 0;
};

inline void add_vote(std::vector<Vote>& votes, std::span<const TokenId> c, float weight,
                     std::uint8_t source) {
    if (c.empty() || !(weight > 0.0F)) { return; }
    const auto count = static_cast<std::uint32_t>(
        std::min<std::size_t>(c.size(), kLookupMaximumDrafts));
    auto same = [&](const Vote& v) {
        return v.count == count &&
               std::equal(v.tokens.begin(), v.tokens.begin() + count, c.begin());
    };
    auto it = std::find_if(votes.begin(), votes.end(), same);
    if (it == votes.end()) {
        Vote v; v.count = count; std::copy_n(c.begin(), count, v.tokens.begin());
        votes.push_back(v); it = std::prev(votes.end());
    }
    it->weight += weight; ++it->support; it->sources |= source;
}
} // namespace lookup_detail

// Count all continuations at the longest matched order. Local and persistent observations get
// weight 1; static-corpus observations use corpus_weight. This is the chain subset of TandemLLM's
// counted suffix-tree drafter, adapted to NInfer's exact chain verifier.
inline LookupDraftProposal lookup_draft_vote(
    std::span<const TokenId> ledger, std::uint32_t min_order, std::uint32_t max_drafts,
    const LookupDraftOptions& options, const LookupPersistentStore* persistent,
    const LookupCorpusStore* corpus) {
    if (ledger.empty() || min_order == 0 || max_drafts == 0) { return {}; }
    max_drafts = std::min(max_drafts, kLookupMaximumDrafts);

    std::array<lookup_detail::MatchSet, 3> m;
    m[0] = lookup_detail::local_matches(ledger, min_order, options.max_order, options.max_matches);
    m[1].source = 2;
    if (persistent && persistent->enabled()) {
        m[1] = persistent->lookup(ledger, min_order, options.max_order, options.max_matches);
    }
    m[2].source = 4;
    if (corpus && corpus->enabled()) {
        m[2] = corpus->lookup(ledger, min_order, options.max_order, options.corpus_samples,
                              options.corpus_weight);
    }

    std::uint32_t best_order = 0;
    for (const auto& x : m) best_order = std::max(best_order, x.order);
    if (best_order == 0) return {};

    std::vector<lookup_detail::Vote> votes;
    votes.reserve(options.max_matches * 2U + options.corpus_samples);
    float total = 0.0F;

    auto add_source = [&](const lookup_detail::MatchSet& x, auto get) {
        if (x.order != best_order) return;
        for (auto pos : x.follow) {
            const auto c = get(pos, max_drafts);
            if (c.empty()) continue;
            lookup_detail::add_vote(votes, c, x.weight, x.source);
            total += x.weight;
        }
    };
    add_source(m[0], [&](std::size_t pos, std::uint32_t k) {
        return ledger.subspan(pos, std::min<std::size_t>(k, ledger.size() - pos));
    });
    add_source(m[1], [&](std::size_t pos, std::uint32_t k) {
        return persistent ? persistent->continuation(pos, k) : std::span<const TokenId>{};
    });
    add_source(m[2], [&](std::size_t pos, std::uint32_t k) {
        return corpus ? corpus->continuation(pos, k) : std::span<const TokenId>{};
    });
    if (votes.empty() || !(total > 0.0F)) return {};

    const auto best = std::max_element(votes.begin(), votes.end(),
        [](const auto& a, const auto& b) { return a.weight < b.weight; });
    LookupDraftProposal out;
    out.tokens = best->tokens; out.count = best->count; out.match_order = best_order;
    out.support = best->support; out.confidence = best->weight / total; out.sources = best->sources;
    if (out.support < options.min_support || out.confidence < options.min_confidence) return {};
    return out;
}

} // namespace ninfer::qwen3_5
