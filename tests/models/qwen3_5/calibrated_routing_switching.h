#pragma once

#include <atomic>
#include <functional>
#include <set>
#include <thread>

namespace {
namespace calibrated_switching {

struct Case {
    std::vector<ninfer::TokenId> prompt;
    ninfer::RequestOptions options;
    ninfer::GenerationResult reference;
    bool compare_none = true;
};

ninfer::RequestOptions options_for(unsigned budget, bool reuse = false) {
    auto options = request();
    options.execution.requested_output_tokens = budget;
    options.execution.allow_prefix_reuse = reuse;
    return options;
}

void same_response(const ninfer::GenerationResult& actual,
                   const ninfer::GenerationResult& reference,
                   const std::string& label = "response") {
    const auto mismatch = std::mismatch(actual.generated_token_ids.begin(),
        actual.generated_token_ids.end(), reference.generated_token_ids.begin(),
        reference.generated_token_ids.end());
    if (mismatch.first != actual.generated_token_ids.end() ||
        mismatch.second != reference.generated_token_ids.end()) {
        const auto index = mismatch.first - actual.generated_token_ids.begin();
        throw std::runtime_error(label + " token mismatch index=" + std::to_string(index) +
            " actual=" + (mismatch.first == actual.generated_token_ids.end() ? "end" : std::to_string(*mismatch.first)) +
            " reference=" + (mismatch.second == reference.generated_token_ids.end() ? "end" : std::to_string(*mismatch.second)) +
            " lengths=" + std::to_string(actual.generated_token_ids.size()) + "/" +
            std::to_string(reference.generated_token_ids.size()));
    }
    const auto field = [&](bool equal, const char* name) {
        if (!equal) { throw std::runtime_error(label + " response mismatch field=" + name); }
    };
    field(actual.finish_reason == reference.finish_reason, "finish_reason");
    field(actual.content == reference.content, "content");
    field(actual.reasoning == reference.reasoning, "reasoning");
    field(actual.reasoning_tokens == reference.reasoning_tokens, "reasoning_tokens");
    field(actual.matched_stop_string == reference.matched_stop_string, "matched_stop_string");
    field(actual.tool_calls.size() == reference.tool_calls.size(), "tool_calls size");
    for (std::size_t i = 0; i < actual.tool_calls.size(); ++i) {
        field(actual.tool_calls[i].name == reference.tool_calls[i].name, "tool name");
        field(actual.tool_calls[i].arguments_json == reference.tool_calls[i].arguments_json, "tool arguments");
    }
}

// Token lengths are public prompt inputs, not mutations of Program frontiers. Different padding
// and suffixes make both compact rows carry distinct Conv/GDN/context state.
std::vector<ninfer::TokenId> prompt_at(ninfer::Engine& engine, unsigned length, bool other = false) {
    const auto padding = engine.tokenize_text(other ? " world" : " hello");
    const auto tail = engine.tokenize_text(other
        ? "\nRepeat world: world, world, world,"
        : "\nCount from one to twenty: one, two, three,");
    require(!padding.empty() && tail.size() < length, "cannot construct public boundary prompt");
    std::vector<ninfer::TokenId> result(length - tail.size(), padding.front());
    result.insert(result.end(), tail.begin(), tail.end());
    return result;
}

ninfer::PromptInput chat_at(ninfer::Engine& engine, unsigned wanted) {
    int padding = static_cast<int>(wanted);
    for (unsigned attempt = 0; attempt < 8; ++attempt) {
        ninfer::PromptInput input;
        ninfer::ChatMessage user;
        user.role = ninfer::ChatRole::User;
        std::string text;
        for (int i = 0; i < padding; ++i) { text += " hello"; }
        text += "\nCount from one to twenty: one, two, three,";
        user.parts.push_back({.kind = ninfer::MessagePartKind::Text, .text = std::move(text), .media = {}});
        input.messages.push_back(std::move(user));
        input.options.enable_thinking = false;
        const auto count = engine.prepare(input).summary().prompt_tokens;
        if (count == wanted) { return input; }
        padding += static_cast<int>(wanted) - static_cast<int>(count);
        require(padding > 0, "cannot calibrate public chat boundary prompt");
    }
    throw std::runtime_error("public chat template did not reach requested token count");
}

struct Observations {
    std::mutex mutex;
    std::condition_variable ready;
    std::vector<ninfer::DecodeRoundEvent> events;
    std::function<void()> hook;

    void record(const ninfer::DecodeRoundEvent& event) {
        std::function<void()> current;
        {
            std::lock_guard lock(mutex);
            events.push_back(event);
            current = hook;
        }
        ready.notify_all();
        if (current) { current(); }
    }
    std::size_t size() {
        std::lock_guard lock(mutex);
        return events.size();
    }
    std::vector<ninfer::DecodeRoundEvent> snapshot() {
        std::lock_guard lock(mutex);
        return events;
    }
    void set_hook(std::function<void()> next) {
        std::lock_guard lock(mutex);
        hook = std::move(next);
    }
    void wait_for_size(std::size_t size) {
        std::unique_lock lock(mutex);
        require(ready.wait_for(lock, std::chrono::seconds(30), [&] { return events.size() >= size; }),
                "switching fixture did not observe a completed decode round");
    }
};

constexpr std::uint32_t actions[2][3]{{0, 7, 11}, {15, 0, 7}};
std::uint32_t selected(const ninfer::DecodeRoundEvent& event) {
    require(event.active_batch >= 1 && event.active_batch <= 2,
            "switching fixture did not observe its actual compact batch");
    if (event.max_execution_frontier > 32768) { return 0; }
    const auto interval = event.max_execution_frontier <= 1024 ? 0 :
                          event.max_execution_frontier <= 8192 ? 1 : 2;
    return actions[event.active_batch - 1][interval];
}

void audit(ninfer::Engine& engine, Observations& observations,
           const std::set<std::size_t>& fallback_events = {}) {
    const auto events = observations.snapshot();
    std::uint64_t counts[4]{};
    std::uint64_t switches = 0, commits = 0;
    std::optional<std::uint32_t> previous;
    for (std::size_t i = 0; i < events.size(); ++i) {
        const auto& event = events[i];
        const auto action = fallback_events.contains(i) ? 15U : selected(event);
        require(event.backend == ninfer::SpeculativeBackend::DFlash2 &&
                    event.draft_tokens == action && event.verify_width == action + 1 &&
                    event.proposal_width == (action ? 16U : 0U) &&
                    event.neural_drafter_executed == (action != 0) && event.elapsed_ns > 0,
                "switching fixture observed the wrong physical action or drafter execution");
        ++counts[action == 0 ? 0 : action == 7 ? 1 : action == 11 ? 2 : 3];
        switches += previous && *previous != action ? 1U : 0U;
        previous = action;
        commits += event.committed_tokens;
    }
    const auto stats = engine.runtime_stats();
    require(stats.calibrated_target_only_rounds == counts[0] &&
                stats.calibrated_k7_rounds == counts[1] && stats.calibrated_k11_rounds == counts[2] &&
                stats.calibrated_k15_rounds == counts[3] && stats.calibrated_route_switches == switches &&
                stats.calibrated_fixed_fallback_rounds == fallback_events.size() &&
                stats.committed_decode_tokens == commits,
            "switching counters do not describe the successfully settled rounds");
}

void barrier(ninfer::Engine& engine, const std::vector<ninfer::TokenId>& prompt) {
    // Public FIFO completion includes every earlier callback. This request adds no decode round.
    const auto result = engine.generate(engine.prepare_tokens(prompt), options_for(1));
    require(result.generated_token_ids.size() == 1, "switching FIFO barrier did not complete");
}

void run(const char* artifact, const std::string& scenario) {
    const bool membership = scenario == "membership";
    const bool restore_control = scenario == "restore-control";
    const bool ptc_only = scenario == "ptc-boundary";
    const bool boundary = scenario == "boundaries" || scenario == "boundary-short" || restore_control || ptc_only;
    const bool terminal = scenario == "terminal";
    require(membership || boundary || terminal, "unknown calibrated switching scenario");
    const bool short_boundary = scenario == "boundary-short";
    Observations observations;
    ninfer::EngineOptions options;
    options.artifact_path = artifact;
    options.max_concurrency = membership ? 2 : 1;
    options.max_pending_requests = 2;
    options.max_context = short_boundary ? 2304 : 9216;
    options.prefill_chunk = 1024;
    options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(
        options.max_context * options.max_concurrency);
    options.kv_cache = ninfer::KvCacheStorage::Int8Group64;
    options.speculative.backend = ninfer::SpeculativeBackend::DFlash2;
    options.speculative.draft_tokens = 15;
    options.speculative.proposal_head = ninfer::ProposalHead::Full;
    ProfileFile profile;
    std::vector<Case> cases;
    std::vector<ninfer::TokenId> newline;
    std::optional<ninfer::PromptInput> ptc_input;
    ninfer::GenerationResult ptc_reference;
    const auto generate_chat = [&](ninfer::Engine& engine, const ninfer::RequestOptions& execution,
                                   const std::string& label) {
        auto prepared = engine.prepare(*ptc_input);
        const auto count = prepared.summary().prompt_tokens;
        require(count == 8187, "PTC chat did not use its measured pre-boundary input");
        auto handle = engine.submit(std::move(prepared), execution);
        const auto sampling = handle.resolved_sampling();
        require(sampling.temperature == 0 && sampling.presence_penalty == 0 && sampling.frequency_penalty == 0,
                "PTC chat did not resolve greedy zero penalties");
        std::cout << nlohmann::json{{"case", label}, {"prompt_tokens", count},
            {"allow_prefix_reuse", execution.execution.allow_prefix_reuse},
            {"temperature", sampling.temperature}, {"presence_penalty", sampling.presence_penalty},
            {"frequency_penalty", sampling.frequency_penalty}} << std::endl;
        auto result = handle.wait();
        std::cout << nlohmann::json{{"case", label}, {"generated_tokens", result.generated_token_ids.size()},
            {"prefix_reuse_path", static_cast<unsigned>(result.prefix_reuse_path)},
            {"reused_prompt_tokens", result.reused_prompt_tokens}} << std::endl;
        return result;
    };
    const auto generate = [&](ninfer::Engine& engine, const std::vector<ninfer::TokenId>& prompt,
                              const ninfer::RequestOptions& execution, const std::string& label) {
        auto handle = engine.submit(engine.prepare_tokens(prompt), execution);
        const auto sampling = handle.resolved_sampling();
        std::cout << nlohmann::json{{"case", label}, {"prompt_tokens", prompt.size()},
            {"output_budget", execution.execution.requested_output_tokens},
            {"allow_prefix_reuse", execution.execution.allow_prefix_reuse},
            {"temperature", sampling.temperature}, {"presence_penalty", sampling.presence_penalty},
            {"frequency_penalty", sampling.frequency_penalty}} << std::endl;
        const auto result = handle.wait();
        std::cout << nlohmann::json{{"case", label}, {"generated_tokens", result.generated_token_ids.size()},
            {"prefix_reuse_path", static_cast<unsigned>(result.prefix_reuse_path)},
            {"reused_prompt_tokens", result.reused_prompt_tokens},
            {"finish_reason", static_cast<unsigned>(result.finish_reason)}} << std::endl;
        return result;
    };
    {
        ninfer::Engine fixed(options);
        newline = fixed.tokenize_text("\n");
        if (membership) {
            auto a = prompt_at(fixed, 63), b = prompt_at(fixed, 65, true);
            cases.push_back({a, options_for(144)});
            cases.push_back({b, options_for(8)});
            cases.push_back({a, options_for(8)});
            cases.push_back({b, options_for(144)});
            cases.push_back({prompt_at(fixed, 1100), options_for(128)});
            auto unsupported = options_for(8);
            unsupported.execution.sampling.presence_penalty = 0.3F;
            cases.push_back({prompt_at(fixed, 1150, true), unsupported, {}, false});
        } else if (boundary) {
            if (!restore_control && !ptc_only) { cases.push_back({prompt_at(fixed, 1019), options_for(64)}); }
            if (!short_boundary) { cases.push_back({prompt_at(fixed, 8187), options_for(64)}); }
        } else {
            cases.push_back({prompt_at(fixed, 1100), options_for(128)});
            cases.push_back({cases.front().prompt, options_for(1)});
            cases.push_back({cases.front().prompt, options_for(5)});
        }
        for (auto& item : cases) {
            if (ptc_only) { continue; }
            item.reference = generate(fixed, item.prompt, item.options, "fixed K15 prompt=" + std::to_string(item.prompt.size()));
            require(item.reference.generated_token_ids.size() == item.options.execution.requested_output_tokens,
                    "switching reference stopped before its requested boundary");
        }
        if (boundary && !short_boundary && !ptc_only) {
            auto follow = cases.back().prompt;
            follow.insert(follow.end(), cases.back().reference.generated_token_ids.begin(),
                          cases.back().reference.generated_token_ids.end());
            follow.insert(follow.end(), newline.begin(), newline.end());
            Case item{follow, options_for(16)};
            item.reference = generate(fixed, item.prompt, item.options, "fixed K15 prompt=" + std::to_string(item.prompt.size()));
            cases.push_back(std::move(item));
        }
        if (boundary && !short_boundary && !restore_control) {
            ptc_input = chat_at(fixed, 8187);
            ptc_reference = generate_chat(fixed, options_for(64), "fixed15 Chat8187 fresh");
            require(ptc_reference.generated_token_ids.size() == 64,
                    "PTC reference did not generate across 8192");
        }
        if (terminal) {
            const auto first = cases.front();
            std::size_t stop_index = 1;
            while (stop_index < 7 && std::find(first.reference.generated_token_ids.begin(),
                    first.reference.generated_token_ids.begin() + stop_index,
                    first.reference.generated_token_ids[stop_index]) !=
                    first.reference.generated_token_ids.begin() + stop_index) { ++stop_index; }
            require(stop_index < 7, "terminal fixture needs a distinct early stop token");
            auto token_stop = options_for(64);
            token_stop.stop.token_ids.push_back(first.reference.generated_token_ids[stop_index]);
            Case item{first.prompt, token_stop};
            item.reference = generate(fixed, item.prompt, item.options, "fixed K15 prompt=" + std::to_string(item.prompt.size()));
            require(item.reference.finish_reason == ninfer::FinishReason::StopToken,
                    "terminal reference did not stop at the requested token");
            cases.push_back(std::move(item));
            require(first.reference.content.size() >= 12, "terminal fixture needs nonempty text output");
            auto text_stop = options_for(64);
            text_stop.stop.strings.push_back(ninfer::StopString{.text = first.reference.content.substr(0, 12)});
            Case text{first.prompt, text_stop};
            text.reference = fixed.generate(fixed.prepare_tokens(text.prompt), text.options);
            require(text.reference.finish_reason == ninfer::FinishReason::StopString,
                    "terminal reference did not exercise text stop");
            cases.push_back(std::move(text));
        }
        const auto identity = fixed.load_summary().speculative_routing_identity;
        require(identity.has_value(), "fixed reference omitted routing profile identity");
        nlohmann::json cells = nlohmann::json::array();
        for (unsigned batch = 1; batch <= options.max_concurrency; ++batch) {
            for (unsigned interval = 0; interval < 3; ++interval) {
                cells.push_back({{"active_batch", batch},
                    {"frontier_upper", interval == 0 ? 1024 : interval == 1 ? 8192 : 32768},
                    {"draft_tokens", actions[batch - 1][interval]}});
            }
        }
        std::ofstream stream(profile.path);
        stream << nlohmann::json{{"schema_version", 1}, {"artifact_type", "ninfer_spec_router_profile"},
            {"identity", ninfer::runtime::routing_profile_identity_json(*identity)}, {"cells", cells}};
    }
    {
        auto baseline_options = options;
        baseline_options.speculative = {};
        ninfer::Engine baseline(baseline_options);
        for (const auto& item : cases) {
            if (item.compare_none && !ptc_only) {
                same_response(generate(baseline, item.prompt, item.options, "None prompt=" + std::to_string(item.prompt.size())), item.reference, "None vs fixed15 prompt=" + std::to_string(item.prompt.size()));
            }
        }
        if (ptc_input) {
            same_response(generate_chat(baseline, options_for(64), "None Chat8187 fresh"),
                          ptc_reference, "None Chat8187 vs fixed15");
        }
        if (restore_control) {
            const auto& root = cases.front();
            const auto& follow = cases.back();
            same_response(generate(baseline, root.prompt, options_for(64, true), "None retained root8187"),
                          root.reference, "None retained root8187");
            same_response(generate(baseline, follow.prompt, options_for(16, true), "None endpoint follow8252"),
                          follow.reference, "None endpoint follow8252 vs fixedfresh");
        }
    }
    if (restore_control) {
        ninfer::Engine fixed(options);
        const auto& root = cases.front();
        const auto& follow = cases.back();
        same_response(generate(fixed, root.prompt, options_for(64, true), "fixed15 retained root8187"),
                      root.reference, "fixed15 retained root8187");
        same_response(generate(fixed, follow.prompt, options_for(16, true), "fixed15 endpoint follow8252"),
                      follow.reference, "fixed15 endpoint follow8252 vs fixedfresh");
        std::cout << "ok public endpoint restore controls None/fixed15\n";
        return;
    }
    options.speculative.routing.mode = ninfer::SpeculativeRoutingMode::Calibrated;
    options.speculative.routing.profile_path = profile.path;
    options.decode_round_observer.callback = [&](const auto& event) { observations.record(event); };
    ninfer::Engine engine(options);
    std::set<std::size_t> fallback_events;
    if (membership) {
        bool bonus = false;
        for (unsigned wave = 0; wave < 3; ++wave) {
            const auto begin = observations.size();
            std::mutex gate_mutex;
            std::condition_variable gate_ready;
            bool release = false;
            std::atomic<bool> gated{false}, timeout{false};
            observations.set_hook([&] {
                if (gated.exchange(true)) { return; }
                std::unique_lock lock(gate_mutex);
                if (!gate_ready.wait_for(lock, std::chrono::seconds(30), [&] { return release; })) {
                    timeout = true;
                }
            });
            const auto& a = cases[wave * 2];
            const auto& b = cases[wave * 2 + 1];
            auto prepared_b = engine.prepare_tokens(b.prompt);
            auto handle_a = engine.submit(engine.prepare_tokens(a.prompt), a.options);
            observations.wait_for_size(begin + 1);
            auto handle_b = engine.submit(std::move(prepared_b), b.options);
            { std::lock_guard lock(gate_mutex); release = true; }
            gate_ready.notify_all();
            const auto result_a = handle_a.wait();
            const auto result_b = handle_b.wait();
            same_response(result_a, a.reference);
            same_response(result_b, b.reference);
            for (const auto* result : {&result_a, &result_b}) {
                bonus = bonus || (result->speculative.accepted_per_position.size() > 14 &&
                                  result->speculative.accepted_per_position[14] != 0);
            }
            observations.set_hook({});
            barrier(engine, a.prompt);
            require(!timeout, "compact batch admission gate timed out");
            const auto events = observations.snapshot();
            bool joined = false, shrank = false;
            std::uint64_t prior_a_commits = 0;
            for (std::size_t i = begin; i < events.size(); ++i) {
                if (events[i].active_batch == 2) {
                    if (!joined) {
                        require(events[i].max_execution_frontier == std::max<unsigned>(
                                    a.prompt.size() + prior_a_commits, b.prompt.size()),
                                "compact batch did not use its maximum actual execution frontier");
                    }
                    joined = true;
                    if (wave == 2) { fallback_events.insert(i); }
                }
                if (joined && events[i].active_batch == 1) { shrank = true; }
                if (!joined) { prior_a_commits += events[i].committed_tokens; }
            }
            require(joined && shrank, "switching fixture did not exercise C1 -> C2 -> C1");
        }
        require(bonus, "membership fixture did not observe a fully accepted K15 bonus round");
    } else if (boundary) {
        const unsigned boundary_cases = ptc_only ? 0 : short_boundary ? 1 : 2;
        for (unsigned i = 0; i < boundary_cases; ++i) {
            for (unsigned repeat = 0; repeat < 2; ++repeat) {
                const auto label = "Cal boundary prompt=" + std::to_string(cases[i].prompt.size()) +
                    " repeat=" + std::to_string(repeat);
                const auto before = observations.size();
                const auto result = generate(engine, cases[i].prompt, cases[i].options, label);
                const auto events = observations.snapshot();
                for (std::size_t j = before; j < events.size(); ++j) {
                    if (j == before || j + 1 == events.size() ||
                        events[j].draft_tokens != events[j - 1].draft_tokens) {
                        std::cout << nlohmann::json{{"case", label}, {"round", j},
                            {"actual_batch", events[j].active_batch},
                            {"frontier", events[j].max_execution_frontier},
                            {"draft_tokens", events[j].draft_tokens}, {"verify_width", events[j].verify_width},
                            {"proposal_width", events[j].proposal_width},
                            {"committed_tokens", events[j].committed_tokens}} << std::endl;
                    }
                }
                same_response(result, cases[i].reference, label + " vs fixed15");
            }
        }
        if (!short_boundary && !ptc_only) {
            const auto& root = cases[1];
            const auto retained = generate(engine, root.prompt, options_for(64, true), "Cal retained root8187");
            same_response(retained, root.reference, "Cal retained root8187 vs fixed15");
            require(retained.prefix_reuse_path == ninfer::PrefixReusePath::Root,
                    "retained boundary request did not begin at root");
            const auto& follow = cases.back();
            for (unsigned repeat = 0; repeat < 2; ++repeat) {
                if (repeat != 0) {
                    same_response(generate(engine, root.prompt, options_for(64, true),
                                           "Cal reestablished root8187"),
                                  root.reference, "Cal reestablished root8187 vs fixed15");
                }
                const auto reused = generate(engine, follow.prompt, options_for(16, true), "Cal endpoint follow repeat=" + std::to_string(repeat));
                same_response(reused, follow.reference, "Cal endpoint follow vs fixed15");
                require(reused.prefix_reuse_path == ninfer::PrefixReusePath::PrivateEndpoint &&
                            reused.reused_prompt_tokens == root.prompt.size() + root.reference.generated_token_ids.size() - 1,
                        "raw boundary continuation did not restore its committed private endpoint");
                same_response(generate(engine, follow.prompt, follow.options, "Cal fresh follow"), follow.reference, "Cal fresh follow vs fixed15");
            }
        }
        if (ptc_input) {
            const auto cold = generate_chat(engine, options_for(64, true), "Cal Chat8187 cold root");
            const auto ptc_begin = observations.size();
            const auto restored = generate_chat(engine, options_for(64, true), "Cal Chat8187 PTC restore");
            const auto ptc_events = observations.snapshot();
            bool selected_k7 = false, selected_k11_after_k7 = false;
            for (std::size_t j = ptc_begin; j < ptc_events.size(); ++j) {
                const auto& event = ptc_events[j];
                selected_k7 = selected_k7 || (event.draft_tokens == 7 && event.max_execution_frontier <= 8192);
                selected_k11_after_k7 = selected_k11_after_k7 ||
                    (selected_k7 && event.draft_tokens == 11 && event.max_execution_frontier > 8192);
            }
            require(selected_k7 && selected_k11_after_k7,
                    "actual PTC-restored decode did not switch K7 -> K11 across 8192");
            const auto fresh = generate_chat(engine, options_for(64), "Cal Chat8187 reuse-off");
            require(cold.prefix_reuse_path == ninfer::PrefixReusePath::Root && cold.reused_prompt_tokens == 0 &&
                        restored.prefix_reuse_path == ninfer::PrefixReusePath::PrivateTurnClosure &&
                        restored.reused_prompt_tokens > 0 && restored.reused_prompt_tokens < 8187 &&
                        fresh.reused_prompt_tokens == 0,
                    "structured chat did not exercise cold Root -> PrivateTurnClosure -> reuse-off");
            same_response(cold, ptc_reference, "Cal Chat8187 cold root vs fixed15");
            same_response(restored, ptc_reference, "Cal Chat8187 PTC vs fixed15");
            same_response(fresh, ptc_reference, "Cal Chat8187 reuse-off vs fixed15");
        }
        barrier(engine, cases.front().prompt);
        const auto events = observations.snapshot();
        bool crossed1024 = false, crossed8192 = false, returned_to_zero = false;
        std::optional<unsigned> previous;
        for (const auto& event : events) {
            crossed1024 = crossed1024 || (previous == 0 && event.draft_tokens == 7);
            crossed8192 = crossed8192 || (previous == 7 && event.draft_tokens == 11);
            returned_to_zero = returned_to_zero || (previous && *previous != 0 && event.draft_tokens == 0);
            previous = event.draft_tokens;
        }
        require((ptc_only || (crossed1024 && returned_to_zero)) && (short_boundary || crossed8192),
                "public committed frontiers did not cross the required routing intervals");
    } else {
        bool rejected = false, partial = false;
        for (std::size_t i = 0; i < cases.size(); ++i) {
            const auto before = observations.size();
            const auto result = engine.generate(engine.prepare_tokens(cases[i].prompt), cases[i].options);
            same_response(result, cases[i].reference);
            barrier(engine, cases[i].prompt);
            if (i == 1) { require(observations.size() == before, "output budget1 executed decode"); }
            rejected = rejected || result.speculative.drafted_tokens > result.speculative.accepted_tokens;
            const auto licensed = 1 + result.speculative.rounds + result.speculative.accepted_tokens +
                                  result.speculative.fallback_steps;
            partial = partial || result.generated_token_ids.size() < licensed;
        }
        require(rejected && partial, "terminal fixture did not exercise rejection and truncated commit");
        const auto& root = cases.front();
        std::atomic<bool> cancel{false};
        observations.set_hook([&] { cancel = true; });
        const auto cancellation_begin = observations.size();
        auto cancellation_options = options_for(128, true);
        auto handle = engine.submit(engine.prepare_tokens(root.prompt), cancellation_options);
        const auto cancelled = handle.wait(nullptr, ninfer::CancellationView([&] { return cancel.load(); }));
        observations.set_hook({});
        barrier(engine, root.prompt);
        require(cancelled.finish_reason == ninfer::FinishReason::Cancelled &&
                    cancelled.generated_token_ids.size() + 16 < root.reference.generated_token_ids.size() &&
                    std::equal(cancelled.generated_token_ids.begin(), cancelled.generated_token_ids.end(),
                               root.reference.generated_token_ids.begin()),
                "cancelled output was not an exact committed reference prefix");
        const auto cancellation_events = observations.snapshot();
        require(cancellation_events.size() > cancellation_begin,
                "cancellation fixture did not execute a completed decode round");
        auto follow = root.prompt;
        follow.insert(follow.end(), cancelled.generated_token_ids.begin(), cancelled.generated_token_ids.end());
        const auto reused = engine.generate(engine.prepare_tokens(follow), options_for(16, true));
        const auto fresh = engine.generate(engine.prepare_tokens(follow), options_for(16));
        same_response(reused, fresh);
        require(fresh.reused_prompt_tokens == 0 &&
                    reused.reused_prompt_tokens <= follow.size() &&
                    std::equal(fresh.generated_token_ids.begin(), fresh.generated_token_ids.end(),
                               root.reference.generated_token_ids.begin() + cancelled.generated_token_ids.size()),
                "continuation after cancellation exposed an uncommitted prefix");
        barrier(engine, root.prompt);
    }
    audit(engine, observations, fallback_events);
    std::cout << "ok calibrated switching " << scenario << ": public frontiers, exact response, actual shapes/counters\n";
}

} // namespace calibrated_switching
} // namespace
