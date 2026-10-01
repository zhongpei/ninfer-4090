#include "ninfer/engine.h"
#include "speculative_page_boundary.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

ninfer::RequestOptions request(std::uint32_t outputs, bool reuse = false) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = outputs;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = reuse;
    options.stop.include_model_defaults       = false;
    return options;
}

void valid(const ninfer::GenerationResult& result, std::size_t outputs) {
    require(result.generated_token_ids.size() == outputs &&
                result.finish_reason == ninfer::FinishReason::OutputLimit,
            "DFlash2 did not honor the requested output budget");
    require(
        result.speculative.backend == ninfer::SpeculativeBackend::DFlash2 &&
            (outputs == 1 || result.speculative.rounds + result.speculative.fallback_steps != 0),
        "generation bypassed DFlash2");
}

ninfer::PromptInput media_prompt(ninfer::MediaKind kind) {
    const std::string header = "P6\n64 64\n255\n";
    ninfer::MessagePart media;
    media.kind              = ninfer::MessagePartKind::Media;
    media.media.kind        = kind;
    media.media.media_type  = "image/x-portable-pixmap";
    media.media.source_name = "pattern.ppm";
    media.media.bytes.assign(header.begin(), header.end());
    for (int i = 0; i < 64 * 64; ++i) {
        media.media.bytes.push_back(i & 255);
        media.media.bytes.push_back((i * 3) & 255);
        media.media.bytes.push_back((i * 7) & 255);
    }
    ninfer::ChatMessage user;
    user.role = ninfer::ChatRole::User;
    user.parts.push_back(std::move(media));
    user.parts.push_back({.kind  = ninfer::MessagePartKind::Text,
                          .text  = "Describe the pattern briefly.",
                          .media = {}});
    ninfer::PromptInput input;
    input.messages.push_back(std::move(user));
    input.options.enable_thinking = false;
    return input;
}

void tree_scenario(ninfer::Engine& engine, const std::vector<ninfer::TokenId>& prompt,
                   unsigned nodes, unsigned concurrency, bool stair) {
    // This shared fixture checks exact retained E=64 after a cross-page partial commit.
    ninfer::test::speculative_page_boundary(engine);
    const auto first = engine.generate(engine.prepare_tokens(prompt), request(24));
    valid(first, 24);
    require(first.speculative.tree_rounds > 0 && first.speculative.tree_nodes > 0,
            "tree scenario executed no actual tree round");
    require(first.speculative.tree_accepted_drafts > 0,
            "real tree fixture accepted no draft");
    std::uint64_t observed_tree_rounds = first.speculative.tree_rounds;
    if (stair) {
        for (int repetition = 0; repetition < 3; ++repetition) {
            const auto observed = engine.generate(engine.prepare_tokens(prompt), request(16));
            valid(observed, 16);
            observed_tree_rounds += observed.speculative.tree_rounds;
        }
        require(observed_tree_rounds > 4,
                "engine-scope tree Stair did not execute beyond its four warmup rounds");
    }
    for (unsigned outputs : {1U, 2U, nodes + 1U}) {
        const auto bounded = engine.generate(engine.prepare_tokens(prompt), request(outputs));
        valid(bounded, outputs);
        if (outputs <= 2) {
            require(bounded.speculative.tree_rounds == 0,
                    "tree verification exceeded the remaining output budget");
        }
    }

    const auto retained = engine.generate(engine.prepare_tokens(prompt), request(12, true));
    valid(retained, 12);
    auto follow = prompt;
    follow.insert(follow.end(), retained.generated_token_ids.begin(), retained.generated_token_ids.end());
    follow.push_back(198);
    const auto reused = engine.generate(engine.prepare_tokens(follow), request(8, true));
    const auto fresh = engine.generate(engine.prepare_tokens(follow), request(8));
    const auto fresh_repeat = engine.generate(engine.prepare_tokens(follow), request(8));
    valid(reused, 8); valid(fresh, 8); valid(fresh_repeat, 8);
    require(reused.reused_prompt_tokens > 0 &&
                reused.reused_prompt_tokens <= prompt.size() + retained.generated_token_ids.size() &&
                fresh.reused_prompt_tokens == 0,
            "tree continuation restored an uncommitted frontier");
    // Compare the same execution route. Incremental restore and full prefill have different
    // floating-point paths; the resource oracle above is the published retained frontier.
    require(stair || fresh.generated_token_ids == fresh_repeat.generated_token_ids,
            "fresh tree continuation did not reproduce its greedy route");

    bool checked_partial = false;
    const auto& reference = first.generated_token_ids;
    for (std::size_t i = 1; i < reference.size(); ++i) {
        if (std::find(reference.begin(), reference.begin() + i, reference[i]) != reference.begin() + i)
            continue;
        auto stopped_options = request(24, true);
        stopped_options.stop.token_ids.push_back(reference[i]);
        const auto stopped = engine.generate(engine.prepare_tokens(prompt), stopped_options);
        const auto licensed = 1 + stopped.speculative.rounds + stopped.speculative.accepted_tokens +
                              stopped.speculative.fallback_steps;
        if (stopped.generated_token_ids.size() >= licensed) continue;
        require(stopped.finish_reason == ninfer::FinishReason::StopToken &&
                    stopped.speculative.tree_rounds > 0 &&
                    (stair || stopped.generated_token_ids.size() <= i + 1),
                "tree partial terminal did not honor its stop token");
        auto continuation = prompt;
        continuation.insert(continuation.end(), stopped.generated_token_ids.begin(), stopped.generated_token_ids.end());
        continuation.push_back(198);
        const auto stop_reused = engine.generate(engine.prepare_tokens(continuation), request(8, true));
        const auto stop_fresh = engine.generate(engine.prepare_tokens(continuation), request(8));
        valid(stop_reused, 8); valid(stop_fresh, 8);
        require(stop_reused.reused_prompt_tokens > 0 &&
                    stop_reused.reused_prompt_tokens <= prompt.size() + stopped.generated_token_ids.size() &&
                    stop_fresh.reused_prompt_tokens == 0,
                "tree stop retained an uncommitted branch suffix");
        checked_partial = true;
        break;
    }
    require(checked_partial, "real tree fixture did not exercise a partial terminal commit");

    auto sampled = request(16);
    sampled.execution.sampling.temperature = 0.8F;
    sampled.execution.sampling.top_p = 0.9F;
    sampled.execution.sampling.top_k = 20;
    sampled.execution.sampling.seed = 42;
    const auto sample1 = engine.generate(engine.prepare_tokens(prompt), sampled);
    const auto sample2 = engine.generate(engine.prepare_tokens(prompt), sampled);
    valid(sample1, 16); valid(sample2, 16);
    require(sample1.speculative.tree_rounds > 0 && sample2.speculative.tree_rounds > 0 &&
                (stair || sample1.generated_token_ids == sample2.generated_token_ids),
            "same tree route and seed did not reproduce sampled tokens");

    auto penalty = request(16);
    penalty.execution.sampling.presence_penalty = 0.5F;
    penalty.execution.sampling.frequency_penalty = 0.25F;
    const auto penalized = engine.generate(engine.prepare_tokens(prompt), penalty);
    valid(penalized, 16);
    require(penalized.speculative.tree_rounds == 0 && penalized.speculative.tree_fallback_rounds > 0,
            "target penalties did not select the tree chain fallback");

    std::uint64_t batched_fallbacks = 0;
    if (concurrency > 1) {
        std::vector<ninfer::GenerationHandle> handles;
        for (unsigned row = 0; row < concurrency; ++row)
            handles.push_back(engine.submit(engine.prepare_tokens(prompt), request(24 + row)));
        for (unsigned row = 0; row < concurrency; ++row) {
            const auto result = handles[row].wait();
            valid(result, 24 + row);
            batched_fallbacks += result.speculative.tree_fallback_rounds;
        }
        require(batched_fallbacks > 0, "multiple active rows did not select tree chain fallback");
    }
    const auto stats = engine.runtime_stats();
    require(stats.device_backend_kv_occupied_pages == 0 && stats.backend_kv_d2h_bytes == 0 &&
                stats.backend_kv_h2d_bytes == 0,
            "tree DFlash2 allocated or transferred a full backend KV pool");
    std::cout << "ok scenario=" << (stair ? "tree-stair" : "tree-" + std::to_string(nodes))
              << " C=" << concurrency << " observed_tree_rounds=" << observed_tree_rounds
              << " tree_rounds=" << first.speculative.tree_rounds
              << " tree_accepted=" << first.speculative.tree_accepted_drafts
              << " partial_terminal=" << checked_partial
              << " penalty_fallbacks=" << penalized.speculative.tree_fallback_rounds
              << " batch_fallbacks=" << batched_fallbacks << '\n';
}

} // namespace

// Optional K, Graph, optimized-head, B and KV codec arguments select representative integration
// routes without multiplying test binaries. The artifact supplies the actual weight
// representations. Optional argument 8 selects tree-3/tree-7/tree-11/tree-15 or tree-stair.
// Tree scenarios use engine router scope, default to C1, and retain startup K independently.
int main(int argc, char** argv) {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (!artifact || !*artifact) {
        std::cout << "skip: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }
    try {
        const std::string scenario = argc > 8 ? argv[8] : "chain";
        const bool stair = scenario == "tree-stair";
        unsigned tree_nodes = 0;
        if (scenario != "chain" && !stair) {
            require(scenario == "tree-3" || scenario == "tree-7" ||
                        scenario == "tree-11" || scenario == "tree-15",
                    "scenario must be chain, tree-3/tree-7/tree-11/tree-15 or tree-stair");
            tree_nodes = static_cast<unsigned>(std::stoul(scenario.substr(5)));
        }
        const auto k         = argc > 1 ? static_cast<unsigned>(std::stoul(argv[1])) : 15U;
        if (stair) tree_nodes = std::min(15U, k);
        const bool graph     = argc > 2 ? std::stoi(argv[2]) != 0 : true;
        const bool optimized = argc > 3 ? std::stoi(argv[3]) != 0 : true;
        const auto batch     = argc > 4 ? static_cast<unsigned>(std::stoul(argv[4])) : (tree_nodes ? 1U : 8U);
        require(tree_nodes <= k, "tree node budget exceeds startup draft K");
        ninfer::EngineOptions options;
        options.artifact_path   = artifact;
        options.max_context     = 2304;
        options.kv_capacity     = ninfer::KvCapacityPolicy::explicit_capacity(2304 * batch);
        options.prefill_chunk   = 2304;
        options.max_concurrency = batch;
        options.context_cache.device_state_slots     = argc > 7 ? std::stoul(argv[7]) : 3U;
        options.use_cuda_graph                       = graph;
        options.enable_vision                        = argc > 6 && std::stoi(argv[6]) != 0;
        options.kv_cache                             = argc > 5 && std::string(argv[5]) == "int8"
                                                           ? ninfer::KvCacheStorage::Int8Group64
                                                           : ninfer::KvCacheStorage::BFloat16;
        auto penalty                                 = request(24);
        penalty.execution.sampling.presence_penalty  = 0.5F;
        penalty.execution.sampling.frequency_penalty = 0.25F;
        options.speculative.backend                  = ninfer::SpeculativeBackend::DFlash2;
        options.speculative.draft_tokens             = k;
        options.speculative.proposal_head =
            optimized ? ninfer::ProposalHead::Optimized : ninfer::ProposalHead::Full;
        if (tree_nodes) {
            options.speculative.tree.mode = ninfer::SpeculativeTreeMode::Lattice;
            options.speculative.tree.nodes = tree_nodes;
            options.speculative.tree.spine = std::min(7U, tree_nodes);
            options.speculative.routing.mode = stair ? ninfer::SpeculativeRoutingMode::Stair
                                                       : ninfer::SpeculativeRoutingMode::Fixed;
            options.speculative.routing.scope = ninfer::SpeculativeRouterScope::Engine;
        }
        ninfer::Engine engine(options);
        const auto prompt = engine.tokenize_text("Count from one to twenty: one, two, three,");
        if (tree_nodes) {
            tree_scenario(engine, prompt, tree_nodes, batch, stair);
            return 0;
        }
        ninfer::test::speculative_page_boundary(engine);
        const auto first = engine.generate(engine.prepare_tokens(prompt), request(24));
        valid(first, 24);
        const auto& reference = first.generated_token_ids;
        require(first.speculative.accepted_tokens != 0, "real draft fixture accepted no proposal");
        const auto penalized = engine.generate(engine.prepare_tokens(prompt), penalty);
        valid(penalized, 24);


        // All rows share a known target prefix, while their budgets force P=0, partial and full W.
        std::vector<ninfer::GenerationHandle> handles;
        for (unsigned row = 0; row < batch; ++row) {
            handles.push_back(engine.submit(engine.prepare_tokens(prompt), request(2 + row * 3)));
        }
        for (unsigned row = 0; row < batch; ++row) {
            const auto result = handles[row].wait();
            valid(result, 2 + row * 3);
        }
        // Rejection RNG uses the same logical positions and seed for repeat requests.
        auto sampled                                 = request(16);
        sampled.execution.sampling.temperature       = 0.8F;
        sampled.execution.sampling.top_p             = 0.9F;
        sampled.execution.sampling.top_k             = 20;
        sampled.execution.sampling.presence_penalty  = 0.3F;
        sampled.execution.sampling.frequency_penalty = 0.2F;
        sampled.execution.sampling.seed              = 42;
        const auto sample1 = engine.generate(engine.prepare_tokens(prompt), sampled);
        const auto sample2 = engine.generate(engine.prepare_tokens(prompt), sampled);
        valid(sample1, 16);
        require(sample1.generated_token_ids == sample2.generated_token_ids,
                "same DFlash2 seed and inputs did not reproduce the conditional path");

        // Terminal flush and fork must retain the consumed frontier for a later request.
        const auto retained = engine.generate(engine.prepare_tokens(prompt), request(12, true));
        auto continuation   = prompt;
        continuation.insert(continuation.end(), retained.generated_token_ids.begin(),
                            retained.generated_token_ids.end());
        continuation.push_back(198);
        const auto reused = engine.generate(engine.prepare_tokens(continuation), request(8, true));
        const auto fresh  = engine.generate(engine.prepare_tokens(continuation), request(8, false));
        valid(reused, 8);
        valid(fresh, 8);
        require(reused.reused_prompt_tokens != 0 && fresh.reused_prompt_tokens == 0,
                "DFlash2 prefix restore did not reuse its retained frontier");

        if (k >= 7) {
            bool checked_partial = false;
            for (std::size_t i = 1; i < std::min<std::size_t>(k, reference.size()); ++i) {
                if (std::find(reference.begin(), reference.begin() + i, reference[i]) !=
                    reference.begin() + i) {
                    continue;
                }
                auto stopped_options = request(24, true);
                stopped_options.stop.token_ids.push_back(reference[i]);
                const auto stopped =
                    engine.generate(engine.prepare_tokens(prompt), stopped_options);
                const auto licensed = 1 + stopped.speculative.rounds +
                                      stopped.speculative.accepted_tokens +
                                      stopped.speculative.fallback_steps;
                if (stopped.generated_token_ids.size() >= licensed) { continue; }
                require(stopped.finish_reason == ninfer::FinishReason::StopToken,
                        "partial terminal did not stop at its token");
                auto follow = prompt;
                follow.insert(follow.end(), stopped.generated_token_ids.begin(),
                              stopped.generated_token_ids.end());
                follow.push_back(198);
                const auto reused_stop =
                    engine.generate(engine.prepare_tokens(follow), request(8, true));
                const auto fresh_stop =
                    engine.generate(engine.prepare_tokens(follow), request(8, false));
                valid(reused_stop, 8);
                valid(fresh_stop, 8);
                require(reused_stop.reused_prompt_tokens != 0 &&
                            reused_stop.reused_prompt_tokens <=
                                prompt.size() + stopped.generated_token_ids.size() &&
                            fresh_stop.reused_prompt_tokens == 0,
                        "partial terminal exposed an uncommitted prefix");
                checked_partial = true;
                break;
            }
            require(checked_partial, "fixture did not exercise a stop within a licensed block");
        }
        if (options.enable_vision) {
            for (const auto kind : {ninfer::MediaKind::Image, ninfer::MediaKind::Video}) {
                const auto image =
                    engine.generate(engine.prepare(media_prompt(kind)), request(8, true));
                const auto reused_image =
                    engine.generate(engine.prepare(media_prompt(kind)), request(8, true));
                valid(image, 8);
                valid(reused_image, 8);
                require(image.prompt.has_media && image.timings.vision_seconds > 0 &&
                            reused_image.reused_prompt_tokens != 0 &&
                            reused_image.timings.vision_seconds == 0.0,
                        "Vision DFlash2 capture/restore recomputed retained media");
            }
        }
        const auto stats = engine.runtime_stats();
        require(stats.device_backend_kv_occupied_pages == 0 && stats.backend_kv_d2h_bytes == 0 &&
                    stats.backend_kv_h2d_bytes == 0,
                "DFlash2 allocated or transferred a full backend KV pool");
        if (k == 15) {
            // One oversized prefill replaces the ring, then decode appends across its wrap point.
            auto long_prompt = std::vector<ninfer::TokenId>(2100, 198);
            long_prompt.insert(long_prompt.end(), prompt.begin(), prompt.end());
            const auto long_run =
                engine.generate(engine.prepare_tokens(long_prompt), request(12, true));
            valid(long_run, 12);
            long_prompt.insert(long_prompt.end(), long_run.generated_token_ids.begin(),
                               long_run.generated_token_ids.end());
            const auto long_reuse =
                engine.generate(engine.prepare_tokens(long_prompt), request(6, true));
            const auto long_fresh =
                engine.generate(engine.prepare_tokens(long_prompt), request(6, false));
            valid(long_reuse, 6);
            valid(long_fresh, 6);
            require(long_reuse.reused_prompt_tokens > 2048 && long_fresh.reused_prompt_tokens == 0,
                    "DFlash2 ring wrap lost its retained frontier");
        }
        if (k == 15) {
            auto tail_prompt   = std::vector<ninfer::TokenId>(options.max_context - 4, 198);
            tail_prompt.back() = prompt.back();
            const auto tail    = engine.generate(engine.prepare_tokens(tail_prompt), request(9));
            require(tail.finish_reason == ninfer::FinishReason::ContextCapacity &&
                        tail.generated_token_ids.size() == 5,
                    "full proposal window escaped the target context capacity tail");
        }
        std::cout << "ok K=" << k << " B=" << batch << " graph=" << graph
                  << " optimized=" << optimized << " accepted=" << first.speculative.accepted_tokens
                  << "/" << first.speculative.drafted_tokens
                  << " state_d2h=" << stats.state_d2h_count
                  << " state_h2d=" << stats.state_h2d_count << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
