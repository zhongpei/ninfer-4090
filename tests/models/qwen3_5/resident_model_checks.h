#pragma once

#include "runtime/engine/resident_model.h"

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ninfer::test {
namespace resident_checks_detail {
inline void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template <class Error, class Action>
void rejected(Action&& action, const char* message) {
    try { std::forward<Action>(action)(); }
    catch (const Error&) { return; }
    throw std::runtime_error(message);
}
inline EngineOptions target_options(EngineOptions options) {
    options.speculative = {};
    options.max_context = 512;
    options.kv_capacity = KvCapacityPolicy::explicit_capacity(512);
    options.max_concurrency = 1;
    options.max_pending_requests = 1;
    options.prefill_chunk = 256;
    options.use_cuda_graph = false;
    options.context_cache = {};
    options.context_cache.device_state_slots = 1;
    options.context_cache.host_state_slots = 1;
    options.context_cache.host_kv_capacity_bytes = 16ULL << 20;
    options.context_cache.max_private_continuations = 2;
    options.context_cache.max_shared_prefixes = 1;
    options.context_cache.max_long_anchors_per_continuation = 1;
    return options;
}
inline RequestOptions request() {
    RequestOptions options;
    options.execution.requested_output_tokens = 4;
    options.execution.sampling.temperature = 0.0F;
    options.execution.sampling.presence_penalty = 0.0F;
    options.execution.sampling.frequency_penalty = 0.0F;
    options.execution.allow_prefix_reuse = true;
    options.stop.include_model_defaults = false;
    return options;
}
inline void valid_target(const GenerationResult& result) {
    require(result.generated_token_ids.size() == 4 &&
                result.finish_reason == FinishReason::OutputLimit,
            "resident target-only Engine did not honor the output budget");
    require(result.speculative.backend == SpeculativeBackend::None &&
                result.speculative.rounds == 0,
            "resident target-only Engine used a speculative scheduler");
}
}

// These checks share the campaign owner and create only one Program at a time.
inline std::size_t resident_model_checks(runtime::ResidentModelSession& session,
                                        EngineOptions source) {
    using namespace resident_checks_detail;
    auto options = target_options(std::move(source));
    std::size_t forbidden_load_events = 0;
    options.startup_observer.callback = [&](const StartupEvent& event) {
        if (event.status == StartupStatus::Begin &&
            (event.phase == StartupPhase::ArtifactInspect ||
             event.phase == StartupPhase::TargetPlan ||
             event.phase == StartupPhase::WeightsMaterialize)) ++forbidden_load_events;
    };
    auto bad = options;
    bad.lm_head_q4 = !options.lm_head_q4;
    rejected<std::invalid_argument>([&] { (void)session.make_engine(bad); },
                                     "resident owner admitted different target storage");
    bad = options;
    bad.rope_scaling_factor = 0.5F;
    rejected<std::invalid_argument>([&] { (void)session.make_engine(bad); },
                                     "resident owner admitted invalid Program arithmetic");
    {
        DeviceContext device(options.devices.empty() ? options.device : options.devices.front());
        auto a16 = options;
        a16.prefill_a8 = false;
        auto instance = session.make_instance(a16, device);
        auto a8 = options;
        a8.prefill_a8 = true;
        rejected<std::invalid_argument>([&] {
            (void)models::qwen3_5::make_sequence_planner(instance.instance->parameters, device, a8);
        }, "A16 native Parameters admitted an A8 Program plan");
    }
    std::vector<TokenId> prompt;
    GenerationHandle outstanding;
    {
        auto engine = session.make_engine(options);
        require(engine.memory_summary().weights.capacity_bytes == session.resident_weight_bytes(),
                "resident target-only Engine hid the shared draft weight allocation");
        const auto load = engine.load_summary();
        require(load.artifact_bytes_read == 0 && load.host_to_device_bytes == 0,
                "resident Program replayed artifact materialization");
        rejected<std::logic_error>([&] { (void)session.make_engine(options); },
                                   "resident owner admitted concurrent Engines");
        prompt = engine.tokenize_text("Explain why an inference engine keeps weights resident while giving each request its own mutable context.");
        outstanding = engine.submit(engine.prepare_tokens(prompt), request());
    }
    rejected<std::logic_error>([&] { (void)session.make_engine(options); },
                               "live generation handle released its resident lease early");
    const auto first = outstanding.wait();
    valid_target(first);
    {
        auto engine = session.make_engine(options);
        require(engine.runtime_stats().computed_prefill_tokens == 0 &&
                    engine.runtime_stats().committed_decode_tokens == 0,
                "resident Program inherited previous Engine counters");
        const auto second = engine.generate(engine.prepare_tokens(prompt), request());
        valid_target(second);
        require(second.reused_prompt_tokens == 0 &&
                    second.generated_token_ids == first.generated_token_ids,
                "resident Program inherited mutable prompt state");
    }
    require(forbidden_load_events == 0 && session.model_load_count() == 1,
            "resident checks loaded weights more than once");
    return 5; // failed creation recovery, exclusive lease, None route, isolation, no reload
}

// Run after all jobs. A generation handle must retain the backing and upload context after
// both the session and the original Engine are destroyed.
inline std::size_t resident_model_teardown_checks(
    std::unique_ptr<runtime::ResidentModelSession>& session, EngineOptions source) {
    using namespace resident_checks_detail;
    auto options = target_options(std::move(source));
    GenerationHandle outstanding;
    {
        auto engine = session->make_engine(options);
        const auto prompt = engine.tokenize_text("Describe immutable model ownership.");
        outstanding = engine.submit(engine.prepare_tokens(prompt), request());
    }
    session.reset();
    valid_target(outstanding.wait());
    return 1;
}

} // namespace ninfer::test
