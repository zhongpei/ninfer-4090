#include "ninfer/engine.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <thread>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

struct Observations {
    std::mutex mutex;
    std::condition_variable ready;
    std::vector<ninfer::DecodeRoundEvent> events;
    std::atomic<bool> fail_callback{false};
    std::function<void()> hook;

    void record(const ninfer::DecodeRoundEvent& event) {
        std::function<void()> current_hook;
        {
            std::lock_guard lock(mutex);
            events.push_back(event);
            current_hook = hook;
        }
        ready.notify_all();
        if (current_hook) { current_hook(); }
        if (fail_callback) { throw std::runtime_error("intentional observer failure"); }
    }
    std::vector<ninfer::DecodeRoundEvent> take(std::size_t count) {
        std::unique_lock lock(mutex);
        require(ready.wait_for(lock, std::chrono::seconds(10), [&] {
            return events.size() >= count;
        }), "decode observer published no completed round");
        return events;
    }
    void set_hook(std::function<void()> value) {
        std::lock_guard lock(mutex);
        hook = std::move(value);
    }
    void clear() {
        std::lock_guard lock(mutex);
        events.clear();
    }
};

ninfer::RequestOptions request(unsigned outputs) {
    ninfer::RequestOptions result;
    result.execution.requested_output_tokens = outputs;
    result.execution.sampling.temperature = 0.0F;
    result.execution.allow_prefix_reuse = false;
    result.stop.include_model_defaults = false;
    return result;
}

void run(const char* artifact, unsigned k) {
    Observations observed;
    ninfer::EngineOptions options;
    options.artifact_path = artifact;
    options.max_context = 256;
    options.prefill_chunk = 256;
    options.max_concurrency = 2;
    options.max_pending_requests = 2;
    options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(512);
    options.kv_cache = ninfer::KvCacheStorage::Int8Group64;
    options.context_cache.device_state_slots = 1;
    if (k != 0) {
        options.speculative.backend = ninfer::SpeculativeBackend::DFlash2;
        options.speculative.draft_tokens = k;
        options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    }
    ninfer::GenerationResult disabled_reference;
    {
        require(!options.decode_round_observer.callback, "decode observer is not disabled by default");
        ninfer::Engine disabled(options);
        const auto tokens = disabled.tokenize_text("Count from one to twenty: one, two, three,");
        disabled_reference = disabled.generate(disabled.prepare_tokens(tokens), request(16));
    }
    options.decode_round_observer.callback = [&](const auto& event) { observed.record(event); };
    ninfer::Engine engine(options);
    const auto prompt = engine.tokenize_text("Count from one to twenty: one, two, three,");
    const auto before = engine.runtime_stats();
    const auto first = engine.generate(engine.prepare_tokens(prompt), request(16));
    require(first.generated_token_ids.size() == 16, "round observation changed output budget");
    require(first.generated_token_ids == disabled_reference.generated_token_ids &&
                first.finish_reason == disabled_reference.finish_reason,
            "enabled observer changed disabled-route semantics");
    const auto rounds = k == 0 ? 15 : first.speculative.rounds;
    const auto events = observed.take(rounds);
    std::uint64_t commits = 0;
    std::uint32_t frontier = static_cast<std::uint32_t>(prompt.size());
    for (const auto& event : events) {
        require(event.active_batch == 1, "observer inferred batch from startup capacity");
        require(event.max_execution_frontier == frontier, "observer reported wrong execution frontier");
        require(event.verify_width == k + 1 && event.draft_tokens == k,
                "observer confused budget extent with physical verification width");
        require(event.backend == (k ? ninfer::SpeculativeBackend::DFlash2 : ninfer::SpeculativeBackend::None),
                "observer reported wrong backend");
        require(event.neural_drafter_executed == (k != 0) &&
                    event.proposal_width == (k ? k + 1 : 0),
                "observer reported wrong actual neural proposal shape");
        require(event.elapsed_ns > 0 && event.committed_tokens > 0,
                "observer lost completed round duration or commit");
        commits += event.committed_tokens;
        frontier += static_cast<std::uint32_t>(event.committed_tokens);
    }
    require(commits == 15, "observer counted licensed candidates instead of actual commits");
    const auto after = engine.runtime_stats();
    require(after.committed_decode_tokens - before.committed_decode_tokens == commits,
            "round commit count differs from Engine publication");

    // A deliberately slow callback must remain outside the measured round.
    observed.clear();
    std::atomic<bool> delayed{false};
    observed.set_hook([&] {
        if (!delayed.exchange(true)) { std::this_thread::sleep_for(std::chrono::milliseconds(500)); }
    });
    const auto slow_started = std::chrono::steady_clock::now();
    const auto slow = engine.generate(engine.prepare_tokens(prompt), request(16));
    const auto slow_elapsed = std::chrono::steady_clock::now() - slow_started;
    const auto slow_events = observed.take(k == 0 ? 15 : slow.speculative.rounds);
    require(slow_elapsed >= std::chrono::milliseconds(500) &&
                slow_events.front().elapsed_ns < 500000000ULL,
            "round duration included the observer callback");
    observed.set_hook({});

    // Stop truncates a licensed block to the exact published prefix.
    require(first.generated_token_ids[1] != first.generated_token_ids[0],
            "observer stop fixture needs distinct leading tokens");
    observed.clear();
    auto stopping = request(16);
    stopping.stop.token_ids.push_back(first.generated_token_ids[1]);
    const auto stopped = engine.generate(engine.prepare_tokens(prompt), stopping);
    require(stopped.finish_reason == ninfer::FinishReason::StopToken &&
                stopped.generated_token_ids.size() == 2, "observer stop fixture did not truncate");
    const auto stop_events = observed.take(1);
    require(stop_events.size() == 1 && stop_events.front().committed_tokens == 1,
            "observer counted uncommitted candidates after stop");

    // Pause after the first completed round, enqueue another member, then let the compact
    // batch join and later shrink. No request waits for work on the paused worker here.
    observed.clear();
    std::mutex gate_mutex;
    std::condition_variable gate_ready;
    bool released = false;
    std::atomic<bool> gated{false};
    observed.set_hook([&] {
        if (gated.exchange(true)) { return; }
        std::unique_lock lock(gate_mutex);
        require(gate_ready.wait_for(lock, std::chrono::seconds(10), [&] { return released; }),
                "observer batch fixture gate timed out");
    });
    auto second_prompt = engine.prepare_tokens(prompt);
    auto a = engine.submit(engine.prepare_tokens(prompt), request(48));
    observed.take(1);
    auto b = engine.submit(std::move(second_prompt), request(4));
    {
        std::lock_guard lock(gate_mutex);
        released = true;
    }
    gate_ready.notify_all();
    const auto ar = a.wait();
    const auto br = b.wait();
    const auto batch_rounds = k == 0 ? 47 + 3 : ar.speculative.rounds + br.speculative.rounds;
    // row-rounds may share one batch event. The last long-request round is recognizable by
    // the final total commit count, so wait for settlement rather than assuming client C.
    std::vector<ninfer::DecodeRoundEvent> batch_events;
    {
        std::unique_lock lock(observed.mutex);
        require(observed.ready.wait_for(lock, std::chrono::seconds(10), [&] {
            std::uint64_t total = 0;
            for (const auto& event : observed.events) { total += event.committed_tokens; }
            return total == 50;
        }), "observer did not publish the full joined batch");
        batch_events = observed.events;
    }
    std::uint64_t rows = 0;
    bool joined = false;
    bool shrank = false;
    for (const auto& event : batch_events) {
        joined = joined || event.active_batch == 2;
        shrank = shrank || (joined && event.active_batch == 1);
        rows += event.active_batch;
        require(event.active_batch >= 1 && event.active_batch <= 2,
                "observer published invalid actual membership");
    }
    require(joined && shrank && rows == batch_rounds,
            "observer did not track actual compact membership transitions");
    observed.set_hook({});

    observed.clear();
    std::atomic<bool> cancel{false};
    observed.set_hook([&] { cancel.store(true); });
    const auto cancel_before = engine.runtime_stats();
    auto cancelled_handle = engine.submit(engine.prepare_tokens(prompt), request(96));
    const auto cancelled = cancelled_handle.wait(nullptr,
        ninfer::CancellationView([&] { return cancel.load(); }));
    require(cancelled.finish_reason == ninfer::FinishReason::Cancelled,
            "observer cancellation fixture did not cancel");
    // A prefill-only request is a public FIFO barrier: all earlier decode callbacks have
    // returned before it completes, while its first token contributes no decode commit.
    const auto barrier = engine.generate(engine.prepare_tokens(prompt), request(1));
    require(barrier.generated_token_ids.size() == 1, "observer cancellation barrier failed");
    const auto cancel_after = engine.runtime_stats();
    std::uint64_t cancel_commits = 0;
    {
        std::lock_guard lock(observed.mutex);
        for (const auto& event : observed.events) { cancel_commits += event.committed_tokens; }
    }
    require(cancel_commits == cancel_after.committed_decode_tokens - cancel_before.committed_decode_tokens &&
                cancel_commits + 1 == cancelled.generated_token_ids.size(),
            "observer cancellation count included unsettled output");
    observed.set_hook({});

    observed.clear();
    observed.fail_callback = true;
    const auto repeated = engine.generate(engine.prepare_tokens(prompt), request(16));
    observed.take(k == 0 ? 15 : repeated.speculative.rounds);
    require(repeated.generated_token_ids == first.generated_token_ids &&
                repeated.finish_reason == first.finish_reason,
            "observer exception changed generation semantics");
    std::cout << "ok decode observer K=" << k << "\n";
}

struct LookupCorpusFixture {
    std::filesystem::path directory;
    std::filesystem::path prefix;

    explicit LookupCorpusFixture(const std::vector<ninfer::TokenId>& tokens) {
        directory = std::filesystem::temp_directory_path() /
            ("ninfer-observer-lookup-" + std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()));
        require(std::filesystem::create_directory(directory), "cannot create observer lookup fixture");
        prefix = directory / "corpus";
        std::vector<std::uint32_t> suffix(tokens.size());
        std::iota(suffix.begin(), suffix.end(), 0);
        std::sort(suffix.begin(), suffix.end(), [&](auto left, auto right) {
            return std::lexicographical_compare(tokens.begin() + left, tokens.end(),
                                                tokens.begin() + right, tokens.end());
        });
        const auto write = [](const auto& path, const auto& values) {
            std::ofstream file(path, std::ios::binary);
            file.write(reinterpret_cast<const char*>(values.data()),
                       static_cast<std::streamsize>(values.size() * sizeof(values[0])));
            require(static_cast<bool>(file), "cannot write observer lookup fixture");
        };
        write(prefix.string() + ".tokens.i32", tokens);
        write(prefix.string() + ".suffix.u32", suffix);
    }
    ~LookupCorpusFixture() {
        std::error_code ignored;
        std::filesystem::remove_all(directory, ignored);
    }
};

void run_lookup(const char* artifact, ninfer::LookupDFlashMode mode) {
    ninfer::EngineOptions options;
    options.artifact_path = artifact;
    options.max_context = 128;
    options.prefill_chunk = 128;
    options.max_concurrency = 1;
    options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(128);
    options.kv_cache = ninfer::KvCacheStorage::Int8Group64;
    options.speculative.backend = ninfer::SpeculativeBackend::DFlash2;
    options.speculative.draft_tokens = 7;
    options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    std::vector<ninfer::TokenId> prompt;
    ninfer::GenerationResult reference;
    {
        ninfer::Engine baseline(options);
        prompt = baseline.tokenize_text("Count from one to twenty: one, two, three,");
        reference = baseline.generate(baseline.prepare_tokens(prompt), request(16));
    }
    auto corpus_tokens = prompt;
    corpus_tokens.insert(corpus_tokens.end(), reference.generated_token_ids.begin(),
                         reference.generated_token_ids.end());
    LookupCorpusFixture corpus(corpus_tokens);
    options.speculative.lookup_ngram = 3;
    options.speculative.lookup.strategy = ninfer::LookupDraftStrategy::Vote;
    options.speculative.lookup.dflash_mode = mode;
    options.speculative.lookup.corpus_prefix = corpus.prefix;
    options.speculative.lookup.corpus_weight = 1.0F;
    options.speculative.lookup.corpus_samples = 64;
    Observations observed;
    options.decode_round_observer.callback = [&](const auto& event) { observed.record(event); };
    ninfer::Engine engine(options);
    const auto result = engine.generate(engine.prepare_tokens(prompt), request(16));
    const auto events = observed.take(result.speculative.rounds);
    require(result.generated_token_ids == reference.generated_token_ids &&
                result.finish_reason == reference.finish_reason,
            "lookup observation changed the exact target output");
    require(result.speculative.lookup_hits > 0 && result.speculative.lookup_rounds > 0,
            "observer lookup fixture did not actually hit its corpus");
    std::uint64_t skipped = 0;
    for (const auto& event : events) {
        require(event.verify_width == 8 && event.draft_tokens == 7 && event.active_batch == 1,
                "lookup observation changed physical target shape");
        require(event.proposal_width == (event.neural_drafter_executed ? 8U : 0U),
                "lookup observation confused proposal width and execution");
        skipped += !event.neural_drafter_executed;
    }
    if (mode == ninfer::LookupDFlashMode::Replace) {
        require(result.speculative.lookup_replace_rounds > 0 && skipped == 0,
                "lookup Replace observer falsely reports skipping the neural drafter");
    } else {
        require(result.speculative.lookup_head_skip_rounds > 0 &&
                    skipped == result.speculative.lookup_head_skip_rounds,
                "lookup HeadSkip observer did not report its actual skipped neural rounds");
    }
    std::cout << "ok lookup observer mode=" << (mode == ninfer::LookupDFlashMode::Replace ? "replace" : "head-skip")
              << " hits=" << result.speculative.lookup_hits << " skipped=" << skipped << '\n';
}

}

int main(int argc, char** argv) {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') { return 77; }
    try {
        if (argc > 2) {
            const std::string mode = argv[2];
            require(mode == "lookup-replace" || mode == "lookup-head-skip", "unknown observer test scenario");
            run_lookup(artifact, mode == "lookup-replace" ? ninfer::LookupDFlashMode::Replace
                                                          : ninfer::LookupDFlashMode::HeadSkip);
        } else {
            run(artifact, argc > 1 ? static_cast<unsigned>(std::stoul(argv[1])) : 0);
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
