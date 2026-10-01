#include "models/qwen3_5/program/speculative/lookup_draft.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <vector>

namespace {

int check(bool ok, const char* message) {
    if (ok) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

template <class T>
void write_vector(const std::filesystem::path& path, const std::vector<T>& values) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) { throw std::runtime_error("failed to create lookup test corpus"); }
    out.write(reinterpret_cast<const char*>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(T)));
    if (!out) { throw std::runtime_error("failed to write lookup test corpus"); }
}

} // namespace

int main() {
    using namespace ninfer;
    using namespace ninfer::qwen3_5;

    int failures = 0;
    LookupDraftOptions options;
    options.max_order      = 4;
    options.max_matches    = 16;
    options.min_support    = 1;
    options.min_confidence = 0.5F;

    std::vector<TokenId> ledger{1, 2, 3, 4, 8, 1, 2, 3, 4, 8, 1, 2, 3, 4};
    const auto local = lookup_draft_vote(ledger, 4, 2, options, nullptr, nullptr);
    failures += check(local && local.match_order == 4 && local.tokens[0] == 8 &&
                          local.support == 2,
                      "local voting did not combine repeated continuation");

    LookupPersistentStore persistent(256, 4, 16);
    std::vector<TokenId> prior{4, 5, 6, 7, 42, 43};
    persistent.append_sequence(prior);
    std::vector<TokenId> query{10, 4, 5, 6, 7};
    const auto remembered = lookup_draft_vote(query, 4, 2, options, &persistent, nullptr);
    failures += check(remembered && remembered.tokens[0] == 42 && remembered.tokens[1] == 43 &&
                          (remembered.sources & 2) != 0,
                      "persistent suffix history did not propose continuation");

    const std::filesystem::path prefix = std::filesystem::temp_directory_path() /
        ("ninfer_lookup_vote_test_" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    auto snapshot_path = prefix;
    snapshot_path += ".snapshot";
    {
        LookupPersistentStore original(256, 4, 16, snapshot_path);
        original.append_sequence(prior);
    }
    {
        LookupPersistentStore restored(256, 4, 16, snapshot_path);
        const auto replay = lookup_draft_vote(query, 4, 2, options, &restored, nullptr);
        failures += check(replay && replay.count == 2 && replay.tokens[0] == 42 &&
                              replay.tokens[1] == 43 && replay.support == 1 &&
                              (replay.sources & 2) != 0,
                          "lookup snapshot did not preserve indexed continuation after restart");
    }
    std::filesystem::remove(snapshot_path);
    std::filesystem::path token_path = prefix;
    token_path += ".tokens.i32";
    std::filesystem::path suffix_path = prefix;
    suffix_path += ".suffix.u32";
    std::error_code ignored;
    std::filesystem::remove(token_path, ignored);
    std::filesystem::remove(suffix_path, ignored);

    // Two documents deliberately carry the same 4-token key and continuation. The -1 separators
    // make any attempted cross-document continuation terminate before it can vote.
    std::vector<TokenId> corpus_tokens{4, 5, 6, 7, 77, 78, -1,
                                       4, 5, 6, 7, 77, 78, -1};
    std::vector<std::uint32_t> suffix(corpus_tokens.size());
    for (std::uint32_t i = 0; i < suffix.size(); ++i) { suffix[i] = i; }
    std::sort(suffix.begin(), suffix.end(), [&](std::uint32_t left, std::uint32_t right) {
        std::size_t a = left;
        std::size_t b = right;
        while (a < corpus_tokens.size() && b < corpus_tokens.size()) {
            if (corpus_tokens[a] != corpus_tokens[b]) {
                return corpus_tokens[a] < corpus_tokens[b];
            }
            ++a;
            ++b;
        }
        return a == corpus_tokens.size() && b != corpus_tokens.size();
    });
    write_vector(token_path, corpus_tokens);
    write_vector(suffix_path, suffix);

    try {
        LookupCorpusStore corpus(prefix);
        options.corpus_weight = 1.0F;
        options.corpus_samples = 16;
        const auto from_corpus =
            lookup_draft_vote(query, 4, 2, options, nullptr, &corpus);
        failures += check(from_corpus && from_corpus.tokens[0] == 77 &&
                              from_corpus.tokens[1] == 78 &&
                              (from_corpus.sources & 4) != 0 &&
                              from_corpus.support == 2,
                          "static suffix corpus did not contribute counted continuation votes");
    } catch (const std::exception& error) {
        std::cerr << "static corpus test failed: " << error.what() << '\n';
        ++failures;
    }
    std::filesystem::remove(token_path, ignored);
    std::filesystem::remove(suffix_path, ignored);

    std::array<TokenId, 4> recent{};
    const auto recent_count =
        lookup_draft(std::span<const TokenId>(ledger), 4, 4, recent.data());
    failures += check(recent_count != 0 && recent[0] == 8,
                      "historical recent lookup changed semantics");

    return failures == 0 ? 0 : 1;
}
