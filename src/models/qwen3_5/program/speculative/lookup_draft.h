#pragma once

#include "ninfer/types.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ninfer::qwen3_5 {

inline constexpr std::uint32_t kLookupMaximumDrafts = 15;
inline constexpr TokenId kLookupDocumentSeparator = static_cast<TokenId>(-1);

// Historical nearest-occurrence lookup. Kept as the "recent" A/B arm.
[[nodiscard]] inline std::uint32_t lookup_draft(std::span<const TokenId> ledger,
                                                std::uint32_t ngram, std::uint32_t max_drafts,
                                                TokenId* out) {
    if (ngram == 0 || max_drafts == 0 || out == nullptr) { return 0; }
    const std::size_t size = ledger.size();
    if (size <= ngram) { return 0; }
    const TokenId* const tail = ledger.data() + (size - ngram);
    for (std::size_t start = size - ngram; start-- > 0;) {
        bool match = true;
        for (std::uint32_t j = 0; j < ngram; ++j) {
            if (ledger[start + j] != tail[j]) {
                match = false;
                break;
            }
        }
        if (!match) { continue; }
        const std::size_t follow = start + ngram;
        if (follow >= size) { continue; }
        const auto count = static_cast<std::uint32_t>(
            std::min<std::size_t>(size - follow, max_drafts));
        for (std::uint32_t j = 0; j < count; ++j) { out[j] = ledger[follow + j]; }
        return count;
    }
    return 0;
}

namespace lookup_detail {

[[nodiscard]] inline std::uint64_t hash_tokens(std::span<const TokenId> tokens,
                                               std::uint32_t order) noexcept {
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

[[nodiscard]] inline bool equal_at(std::span<const TokenId> haystack, std::size_t start,
                                   std::span<const TokenId> needle) noexcept {
    return start <= haystack.size() && needle.size() <= haystack.size() - start &&
           std::equal(needle.begin(), needle.end(),
                      haystack.begin() + static_cast<std::ptrdiff_t>(start));
}

struct MatchSet {
    std::uint32_t order = 0;
    std::vector<std::size_t> follow;
    float weight = 1.0F;
    std::uint8_t source = 0;
};

[[nodiscard]] inline MatchSet local_matches(std::span<const TokenId> ledger,
                                            std::uint32_t min_order,
                                            std::uint32_t max_order,
                                            std::uint32_t max_matches) {
    MatchSet result;
    result.source = 1;
    max_order = std::min<std::uint32_t>(max_order, static_cast<std::uint32_t>(ledger.size()));
    for (std::uint32_t order = max_order; order >= min_order && order != 0; --order) {
        if (ledger.size() <= order) {
            if (order == min_order) { break; }
            continue;
        }
        const auto pattern = ledger.last(order);
        const std::size_t last_start = ledger.size() - order;
        for (std::size_t start = last_start; start-- > 0;) {
            if (!equal_at(ledger, start, pattern)) { continue; }
            const std::size_t follow = start + order;
            if (follow < ledger.size()) {
                result.follow.push_back(follow);
                if (result.follow.size() >= max_matches) { break; }
            }
        }
        if (!result.follow.empty()) {
            result.order = order;
            return result;
        }
        if (order == min_order) { break; }
    }
    return result;
}

template <class T>
[[nodiscard]] inline std::vector<T> read_binary_vector(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) { throw std::runtime_error("cannot open lookup corpus file: " + path.string()); }
    const std::streamsize bytes = input.tellg();
    if (bytes < 0 || bytes % static_cast<std::streamsize>(sizeof(T)) != 0) {
        throw std::runtime_error("lookup corpus file has invalid element alignment: " +
                                 path.string());
    }
    input.seekg(0);
    std::vector<T> out(static_cast<std::size_t>(bytes) / sizeof(T));
    if (!out.empty() &&
        !input.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(bytes))) {
        throw std::runtime_error("failed reading lookup corpus file: " + path.string());
    }
    return out;
}

} // namespace lookup_detail

// Process-persistent suffix memory: completed ledgers are kept across requests in the same process.
// It is deliberately independent from NInfer's exact StateImage/KV cache: this store may only
// propose tokens and therefore never acquires model-state authority.
class LookupPersistentStore {
public:
    LookupPersistentStore() = default;
    LookupPersistentStore(std::uint32_t max_tokens, std::uint32_t max_order,
                          std::uint32_t max_matches,
                          std::filesystem::path persistent_path = {})
        : max_tokens_(max_tokens), max_order_(max_order), max_matches_(max_matches),
          persistent_path_(std::move(persistent_path)) {
        if (max_tokens_ != 0) {
            tokens_.reserve(static_cast<std::size_t>(max_tokens_) + 1U);
        }
        if (!persistent_path_.empty()) {
            if (!enabled()) {
                throw std::invalid_argument(
                    "lookup persistent path requires a nonzero persistent token budget");
            }
            load_persistent();
        }
    }

    [[nodiscard]] bool enabled() const noexcept { return max_tokens_ != 0; }

    void append_sequence(std::span<const TokenId> sequence) {
        if (!enabled() || sequence.empty() || max_tokens_ < 2) return;

        bool rewrote = false;
        if (tokens_.size() + sequence.size() + 1U > max_tokens_) {
            tokens_.clear();
            index_.clear();
            rewrote = true;
        }

        const std::size_t take =
            std::min<std::size_t>(sequence.size(), static_cast<std::size_t>(max_tokens_ - 1U));
        const std::size_t source_start = sequence.size() - take;
        const std::size_t append_begin = tokens_.size();
        tokens_.push_back(kLookupDocumentSeparator);
        for (std::size_t source = source_start; source < sequence.size(); ++source) {
            const std::size_t follow = tokens_.size();
            tokens_.push_back(sequence[source]);
            record_follow(follow);
        }

        if (!persistent_path_.empty()) {
            if (rewrote) {
                rewrite_persistent();
            } else {
                append_persistent(
                    std::span<const TokenId>(tokens_).subspan(append_begin));
            }
        }
    }

    [[nodiscard]] lookup_detail::MatchSet
    lookup(std::span<const TokenId> context, std::uint32_t min_order,
           std::uint32_t max_order, std::uint32_t max_matches) const {
        lookup_detail::MatchSet result;
        result.source = 2;
        max_order = std::min({max_order, max_order_,
                              static_cast<std::uint32_t>(context.size())});
        for (std::uint32_t order = max_order; order >= min_order && order != 0; --order) {
            const auto pattern = context.last(order);
            const auto found = index_.find(lookup_detail::hash_tokens(pattern, order));
            if (found != index_.end()) {
                for (auto it = found->second.rbegin();
                     it != found->second.rend() && result.follow.size() < max_matches; ++it) {
                    const std::size_t follow = *it;
                    if (follow < order || follow >= tokens_.size() ||
                        tokens_[follow] == kLookupDocumentSeparator) {
                        continue;
                    }
                    const std::size_t begin = follow - order;
                    const auto stored = std::span<const TokenId>(tokens_).subspan(begin, order);
                    if (std::find(stored.begin(), stored.end(), kLookupDocumentSeparator) !=
                            stored.end() ||
                        !std::equal(stored.begin(), stored.end(), pattern.begin())) {
                        continue;
                    }
                    result.follow.push_back(follow);
                }
            }
            if (!result.follow.empty()) {
                result.order = order;
                return result;
            }
            if (order == min_order) break;
        }
        return result;
    }

    [[nodiscard]] std::span<const TokenId> continuation(std::size_t pos,
                                                        std::uint32_t max_drafts) const noexcept {
        if (pos >= tokens_.size()) return {};
        std::size_t count = std::min<std::size_t>(max_drafts, tokens_.size() - pos);
        for (std::size_t i = 0; i < count; ++i) {
            if (tokens_[pos + i] == kLookupDocumentSeparator) {
                count = i;
                break;
            }
        }
        return std::span<const TokenId>(tokens_).subspan(pos, count);
    }

private:
    void record_follow(std::size_t follow) {
        if (follow == 0 || follow >= tokens_.size() ||
            tokens_[follow] == kLookupDocumentSeparator) {
            return;
        }
        for (std::uint32_t order = 1; order <= max_order_ && follow >= order; ++order) {
            const std::size_t begin = follow - order;
            const auto gram = std::span<const TokenId>(tokens_).subspan(begin, order);
            if (std::find(gram.begin(), gram.end(), kLookupDocumentSeparator) != gram.end()) {
                continue;
            }
            auto& bucket = index_[lookup_detail::hash_tokens(gram, order)];
            bucket.push_back(follow);
            if (bucket.size() > max_matches_) {
                bucket.erase(bucket.begin(),
                             bucket.begin() + static_cast<std::ptrdiff_t>(
                                                  bucket.size() - max_matches_));
            }
        }
    }

    void rebuild_index() {
        index_.clear();
        for (std::size_t follow = 1; follow < tokens_.size(); ++follow) {
            record_follow(follow);
        }
    }

    void load_persistent() {
        std::error_code ec;
        if (!std::filesystem::exists(persistent_path_, ec)) {
            if (ec) {
                throw std::runtime_error(
                    "cannot inspect lookup persistent path: " + persistent_path_.string());
            }
            return;
        }
        tokens_ = lookup_detail::read_binary_vector<TokenId>(persistent_path_);
        if (tokens_.size() > max_tokens_) {
            const std::size_t begin = tokens_.size() - max_tokens_;
            auto first = std::find(tokens_.begin() + static_cast<std::ptrdiff_t>(begin),
                                   tokens_.end(), kLookupDocumentSeparator);
            if (first == tokens_.end()) {
                tokens_.clear();
            } else {
                tokens_.erase(tokens_.begin(), first);
            }
            rewrite_persistent();
        }
        rebuild_index();
    }

    void ensure_parent_directory() const {
        const auto parent = persistent_path_.parent_path();
        if (parent.empty()) return;
        std::error_code ec;
        std::filesystem::create_directories(parent, ec);
        if (ec) {
            throw std::runtime_error(
                "cannot create lookup persistent directory: " + parent.string());
        }
    }

    void append_persistent(std::span<const TokenId> values) const {
        if (values.empty()) return;
        ensure_parent_directory();
        std::ofstream out(persistent_path_, std::ios::binary | std::ios::app);
        if (!out) {
            throw std::runtime_error(
                "cannot append lookup persistent history: " + persistent_path_.string());
        }
        out.write(reinterpret_cast<const char*>(values.data()),
                  static_cast<std::streamsize>(values.size() * sizeof(TokenId)));
        if (!out) {
            throw std::runtime_error(
                "failed writing lookup persistent history: " + persistent_path_.string());
        }
    }

    void rewrite_persistent() const {
        ensure_parent_directory();
        auto temporary = persistent_path_;
        temporary += ".tmp";
        {
            std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
            if (!out) {
                throw std::runtime_error(
                    "cannot rewrite lookup persistent history: " + temporary.string());
            }
            if (!tokens_.empty()) {
                out.write(reinterpret_cast<const char*>(tokens_.data()),
                          static_cast<std::streamsize>(tokens_.size() * sizeof(TokenId)));
            }
            if (!out) {
                throw std::runtime_error(
                    "failed rewriting lookup persistent history: " + temporary.string());
            }
        }
        std::error_code ec;
        std::filesystem::remove(persistent_path_, ec);
        ec.clear();
        std::filesystem::rename(temporary, persistent_path_, ec);
        if (ec) {
            throw std::runtime_error(
                "cannot publish lookup persistent history: " + persistent_path_.string());
        }
    }

    std::uint32_t max_tokens_ = 0;
    std::uint32_t max_order_ = 0;
    std::uint32_t max_matches_ = 0;
    std::filesystem::path persistent_path_;
    std::vector<TokenId> tokens_;
    std::unordered_map<std::uint64_t, std::vector<std::size_t>> index_;
};

// Static token corpus plus suffix array. PREFIX names PREFIX.tokens.i32 and PREFIX.suffix.u32.
class LookupCorpusStore {
public:
    LookupCorpusStore() = default;

    explicit LookupCorpusStore(const std::filesystem::path& prefix) {
        if (prefix.empty()) { return; }
        auto token_path = prefix;
        token_path += ".tokens.i32";
        auto suffix_path = prefix;
        suffix_path += ".suffix.u32";
        tokens_ = lookup_detail::read_binary_vector<TokenId>(token_path);
        suffix_ = lookup_detail::read_binary_vector<std::uint32_t>(suffix_path);
        if (tokens_.empty() || suffix_.empty()) {
            throw std::runtime_error("lookup corpus arrays must not be empty");
        }
        for (std::uint32_t pos : suffix_) {
            if (pos >= tokens_.size()) {
                throw std::runtime_error("lookup corpus suffix position is out of range");
            }
        }
    }

    [[nodiscard]] bool enabled() const noexcept { return !tokens_.empty() && !suffix_.empty(); }

    [[nodiscard]] lookup_detail::MatchSet
    lookup(std::span<const TokenId> context, std::uint32_t min_order,
           std::uint32_t max_order, std::uint32_t max_samples, float weight) const {
        lookup_detail::MatchSet result;
        result.source = 4;
        result.weight = weight;
        if (!enabled()) { return result; }
        max_order =
            std::min<std::uint32_t>(max_order, static_cast<std::uint32_t>(context.size()));
        for (std::uint32_t order = max_order; order >= min_order && order != 0; --order) {
            const auto pattern = context.last(order);
            const auto [lo, hi] = range(pattern);
            const std::size_t width = hi - lo;
            if (width != 0) {
                const std::size_t count = std::min<std::size_t>(max_samples, width);
                result.follow.reserve(count);
                for (std::size_t sample = 0; sample < count; ++sample) {
                    const std::size_t offset =
                        count == width ? sample : (sample * width) / count;
                    const std::size_t follow =
                        static_cast<std::size_t>(suffix_[lo + offset]) + order;
                    if (follow < tokens_.size() &&
                        tokens_[follow] != kLookupDocumentSeparator) {
                        result.follow.push_back(follow);
                    }
                }
                if (!result.follow.empty()) {
                    result.order = order;
                    return result;
                }
            }
            if (order == min_order) { break; }
        }
        return result;
    }

    [[nodiscard]] std::span<const TokenId> continuation(std::size_t pos,
                                                        std::uint32_t max_drafts) const noexcept {
        if (pos >= tokens_.size()) { return {}; }
        std::size_t count = std::min<std::size_t>(max_drafts, tokens_.size() - pos);
        for (std::size_t i = 0; i < count; ++i) {
            if (tokens_[pos + i] < 0) {
                count = i;
                break;
            }
        }
        return std::span<const TokenId>(tokens_).subspan(pos, count);
    }

private:
    [[nodiscard]] int compare_at(std::size_t suffix_index,
                                 std::span<const TokenId> pattern) const noexcept {
        std::size_t pos = suffix_[suffix_index];
        for (TokenId want : pattern) {
            if (pos >= tokens_.size()) { return -1; }
            const TokenId have = tokens_[pos++];
            if (have < want) { return -1; }
            if (have > want) { return 1; }
        }
        return 0;
    }

    [[nodiscard]] std::pair<std::size_t, std::size_t>
    range(std::span<const TokenId> pattern) const noexcept {
        std::size_t lo = 0, hi = suffix_.size();
        while (lo < hi) {
            const std::size_t mid = lo + (hi - lo) / 2U;
            if (compare_at(mid, pattern) < 0) {
                lo = mid + 1U;
            } else {
                hi = mid;
            }
        }
        const std::size_t begin = lo;
        hi = suffix_.size();
        while (lo < hi) {
            const std::size_t mid = lo + (hi - lo) / 2U;
            if (compare_at(mid, pattern) <= 0) {
                lo = mid + 1U;
            } else {
                hi = mid;
            }
        }
        return {begin, lo};
    }

    std::vector<TokenId> tokens_;
    std::vector<std::uint32_t> suffix_;
};

struct LookupDraftProposal {
    std::array<TokenId, kLookupMaximumDrafts> tokens{};
    std::uint32_t count = 0;
    std::uint32_t match_order = 0;
    std::uint32_t support = 0;
    float confidence = 0.0F;
    std::uint8_t sources = 0;

    [[nodiscard]] explicit operator bool() const noexcept { return count != 0; }
};

namespace lookup_detail {

struct ContinuationVote {
    std::array<TokenId, kLookupMaximumDrafts> tokens{};
    std::uint32_t count = 0;
    float weight = 0.0F;
    std::uint32_t support = 0;
    std::uint8_t sources = 0;
};

inline void add_vote(std::vector<ContinuationVote>& votes, std::span<const TokenId> continuation,
                     float weight, std::uint8_t source) {
    if (continuation.empty() || !(weight > 0.0F)) { return; }
    const auto count = static_cast<std::uint32_t>(
        std::min<std::size_t>(continuation.size(), kLookupMaximumDrafts));
    auto same = [&](const ContinuationVote& vote) {
        return vote.count == count &&
               std::equal(vote.tokens.begin(), vote.tokens.begin() + count, continuation.begin());
    };
    auto it = std::find_if(votes.begin(), votes.end(), same);
    if (it == votes.end()) {
        ContinuationVote vote;
        vote.count = count;
        std::copy_n(continuation.begin(), count, vote.tokens.begin());
        votes.push_back(vote);
        it = std::prev(votes.end());
    }
    it->weight += weight;
    ++it->support;
    it->sources |= source;
}

} // namespace lookup_detail

[[nodiscard]] inline LookupDraftProposal
lookup_draft_vote(std::span<const TokenId> ledger, std::uint32_t min_order,
                  std::uint32_t max_drafts, const LookupDraftOptions& options,
                  const LookupPersistentStore* persistent,
                  const LookupCorpusStore* corpus) {
    LookupDraftProposal out;
    if (ledger.empty() || min_order == 0 || max_drafts == 0) { return out; }
    max_drafts = std::min(max_drafts, kLookupMaximumDrafts);

    std::array<lookup_detail::MatchSet, 3> matches;
    matches[0] = lookup_detail::local_matches(
        ledger, min_order, options.max_order, options.max_matches);
    if (persistent != nullptr && persistent->enabled()) {
        matches[1] = persistent->lookup(ledger, min_order, options.max_order, options.max_matches);
    } else {
        matches[1].source = 2;
    }
    if (corpus != nullptr && corpus->enabled()) {
        matches[2] = corpus->lookup(ledger, min_order, options.max_order,
                                    options.corpus_samples, options.corpus_weight);
    } else {
        matches[2].source = 4;
    }

    for (const auto& match : matches) { out.match_order = std::max(out.match_order, match.order); }
    if (out.match_order == 0) { return out; }

    std::vector<lookup_detail::ContinuationVote> votes;
    votes.reserve(static_cast<std::size_t>(options.max_matches) * 2U + options.corpus_samples);
    float total_weight = 0.0F;
    const auto consume = [&](const lookup_detail::MatchSet& match, auto continuation_fn) {
        if (match.order != out.match_order || match.follow.empty()) { return; }
        for (std::size_t pos : match.follow) {
            const auto continuation = continuation_fn(pos, max_drafts);
            if (continuation.empty()) { continue; }
            lookup_detail::add_vote(votes, continuation, match.weight, match.source);
            total_weight += match.weight;
        }
    };
    consume(matches[0], [&](std::size_t pos, std::uint32_t k) {
        const std::size_t count = std::min<std::size_t>(k, ledger.size() - pos);
        return ledger.subspan(pos, count);
    });
    consume(matches[1], [&](std::size_t pos, std::uint32_t k) {
        return persistent != nullptr ? persistent->continuation(pos, k)
                                     : std::span<const TokenId>{};
    });
    consume(matches[2], [&](std::size_t pos, std::uint32_t k) {
        return corpus != nullptr ? corpus->continuation(pos, k)
                                 : std::span<const TokenId>{};
    });

    if (votes.empty() || !(total_weight > 0.0F)) { return {}; }
    const auto best = std::max_element(
        votes.begin(), votes.end(),
        [](const auto& a, const auto& b) { return a.weight < b.weight; });
    out.count = best->count;
    out.tokens = best->tokens;
    out.support = best->support;
    out.confidence = best->weight / total_weight;
    out.sources = best->sources;
    if (out.support < options.min_support || out.confidence < options.min_confidence) {
        return {};
    }
    return out;
}

} // namespace ninfer::qwen3_5
