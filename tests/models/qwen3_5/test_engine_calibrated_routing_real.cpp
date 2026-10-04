#include "ninfer/engine.h"
#include "runtime/engine/speculative_routing_profile.h"
#include "runtime/engine/model_instance.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <condition_variable>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <mutex>
#include <stdexcept>

namespace {
void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}
struct ProfileFile {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("ninfer-calibrated-real-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".json");
    ~ProfileFile() { std::error_code error; std::filesystem::remove(path, error); }
};
ninfer::RequestOptions request() {
    ninfer::RequestOptions result;
    result.execution.requested_output_tokens = 16;
    result.execution.sampling.temperature = 0.0F;
    result.execution.sampling.presence_penalty = 0.0F;
    result.execution.sampling.frequency_penalty = 0.0F;
    result.execution.allow_prefix_reuse = false;
    result.stop.include_model_defaults = false;
    return result;
}
struct GraphMemoryTelemetry {
    std::size_t free_after_weights = 0;
    std::size_t graph_begin_free = 0;
    std::size_t graph_complete_free = 0;
    void observe(const ninfer::StartupEvent& event) {
        const bool after_weights = event.phase == ninfer::StartupPhase::TargetFinalize &&
            event.status == ninfer::StartupStatus::Begin;
        const bool graph = event.phase == ninfer::StartupPhase::CudaGraphPrepare;
        if (!after_weights && !graph) { return; }
        std::size_t free = 0, total = 0;
        require(cudaMemGetInfo(&free, &total) == cudaSuccess, "resource probe cudaMemGetInfo failed");
        if (after_weights) { free_after_weights = free; }
        if (graph && event.status == ninfer::StartupStatus::Begin) { graph_begin_free = free; }
        if (graph && event.status == ninfer::StartupStatus::Complete) { graph_complete_free = free; }
    }
};

void resource_probe(const char* artifact, std::uint32_t kv_capacity) {
    ninfer::EngineOptions options;
    options.artifact_path = artifact;
    options.max_context = 32768;
    options.prefill_chunk = 1024;
    options.max_concurrency = 8;
    options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(kv_capacity);
    options.kv_cache = ninfer::KvCacheStorage::Int8Group64;
    options.speculative.backend = ninfer::SpeculativeBackend::DFlash2;
    options.speculative.draft_tokens = 15;
    options.speculative.proposal_head = ninfer::ProposalHead::Full;
    ProfileFile profile;
    GraphMemoryTelemetry telemetry;
    options.startup_observer.callback = [&](const auto& event) { telemetry.observe(event); };
    {
        ninfer::Engine fixed(options);
        const auto memory = fixed.memory_summary();
        const auto identity = fixed.load_summary().speculative_routing_identity;
        require(identity.has_value(), "resource probe has no actual identity");
        std::cout << nlohmann::json{{"mode", "fixed"}, {"active_requests", 0}, {"capacity", 8},
            {"context", 32768}, {"kv_capacity", memory.kv_capacity},
            {"weights_bytes", memory.weights.capacity_bytes},
            {"runtime_reservation_bytes", memory.runtime_reservation_bytes},
            {"sequence_capacity_bytes", memory.sequence.capacity_bytes},
            {"workspace_capacity_bytes", memory.workspace.capacity_bytes},
            {"cuda_graph_definition_count", memory.cuda_graph_definition_count},
            {"cuda_graph_executable_count", memory.cuda_graph_executable_count},
            {"cuda_graph_prepare_peak_device_delta_bytes", memory.cuda_graph_prepare_peak_device_delta_bytes},
            {"cuda_graph_prepare_device_delta_bytes", memory.cuda_graph_prepare_device_delta_bytes},
            {"graph_allowance_bytes", memory.cuda_graph_allowance_bytes},
            {"available_after_weights_bytes", memory.available_after_weights_bytes},
            {"available_after_startup_bytes", memory.available_after_startup_bytes},
            {"graph_prepare_device_delta_bytes", telemetry.graph_begin_free - telemetry.graph_complete_free}} << std::endl;
        nlohmann::json cells = nlohmann::json::array();
        for (unsigned batch = 1; batch <= 8; ++batch) {
            for (const auto upper : {1024, 8192, 32768}) {
                cells.push_back({{"active_batch", batch}, {"frontier_upper", upper}, {"draft_tokens", 0}});
            }
        }
        std::ofstream stream(profile.path);
        stream << nlohmann::json{{"schema_version", 1}, {"artifact_type", "ninfer_spec_router_profile"},
            {"identity", ninfer::runtime::routing_profile_identity_json(*identity)}, {"cells", cells}};
    }
    telemetry = {};
    options.speculative.routing.mode = ninfer::SpeculativeRoutingMode::Calibrated;
    options.speculative.routing.profile_path = profile.path;
    try {
        ninfer::Engine calibrated(options);
        const auto memory = calibrated.memory_summary();
        require(memory.cuda_graph_prepare_peak_device_delta_bytes < memory.cuda_graph_allowance_bytes,
                "calibrated graph device peak exceeded reserved allowance");
        std::cout << nlohmann::json{{"mode", "calibrated"}, {"startup", "pass"},
            {"capacity", 8}, {"context", 32768}, {"kv_capacity", memory.kv_capacity},
            {"actions", {0, 7, 11, 15}}, {"physical_target_widths", {1, 8, 12, 16}},
            {"resident_proposal_width", 16}, {"warmed_active_batch_range", {1, 8}},
            {"runtime_reservation_bytes", memory.runtime_reservation_bytes},
            {"sequence_capacity_bytes", memory.sequence.capacity_bytes},
            {"workspace_capacity_bytes", memory.workspace.capacity_bytes},
            {"cuda_graph_definition_count", memory.cuda_graph_definition_count},
            {"cuda_graph_executable_count", memory.cuda_graph_executable_count},
            {"cuda_graph_prepare_peak_device_delta_bytes", memory.cuda_graph_prepare_peak_device_delta_bytes},
            {"cuda_graph_prepare_device_delta_bytes", memory.cuda_graph_prepare_device_delta_bytes},
            {"graph_allowance_bytes", memory.cuda_graph_allowance_bytes},
            {"available_after_weights_bytes", memory.available_after_weights_bytes},
            {"available_after_startup_bytes", memory.available_after_startup_bytes},
            {"graph_prepare_device_delta_bytes", telemetry.graph_begin_free - telemetry.graph_complete_free}} << std::endl;
    } catch (const std::exception& error) {
        std::cout << nlohmann::json{{"mode", "calibrated"}, {"startup", "fail"}, {"capacity", 8},
            {"context", 32768}, {"kv_capacity", kv_capacity},
            {"available_after_weights_bytes", telemetry.free_after_weights}, {"error", error.what()}} << std::endl;
        throw;
    }
}

void run(const char* artifact, std::uint32_t action = 0) {
    require(action == 0 || action == 7 || action == 11 || action == 15, "invalid forced action");
    ninfer::EngineOptions options;
    options.artifact_path = artifact;
    options.max_context = 256;
    options.prefill_chunk = 256;
    options.max_concurrency = 1;
    options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(256);
    options.kv_cache = ninfer::KvCacheStorage::Int8Group64;
    options.speculative.backend = ninfer::SpeculativeBackend::DFlash2;
    options.speculative.draft_tokens = 15;
    options.speculative.proposal_head = ninfer::ProposalHead::Full;
    std::vector<ninfer::TokenId> prompt;
    ninfer::GenerationResult reference;
    ProfileFile profile;
    {
        ninfer::Engine fixed(options);
        const auto summary = fixed.load_summary();
        require(summary.speculative_routing_identity.has_value(), "fixed DFlash2 K15 omitted calibration identity");
        prompt = fixed.tokenize_text("Count from one to twenty: one, two, three,");
        reference = fixed.generate(fixed.prepare_tokens(prompt), request());
        nlohmann::json cells = nlohmann::json::array();
        for (const auto upper : {1024, 8192, 32768}) {
            cells.push_back({{"active_batch", 1}, {"frontier_upper", upper}, {"draft_tokens", action}});
        }
        std::ofstream stream(profile.path);
        stream << nlohmann::json{{"schema_version", 1}, {"artifact_type", "ninfer_spec_router_profile"},
            {"identity", ninfer::runtime::routing_profile_identity_json(*summary.speculative_routing_identity)},
            {"cells", cells}, {"provenance", {{"purpose", "measurement_control"}, {"qualified", false}}}};
    }
    options.speculative.routing.mode = ninfer::SpeculativeRoutingMode::Calibrated;
    options.speculative.routing.profile_path = profile.path;
    std::mutex mutex;
    std::condition_variable ready;
    std::vector<ninfer::DecodeRoundEvent> events;
    options.decode_round_observer.callback = [&](const auto& event) {
        { std::lock_guard lock(mutex); events.push_back(event); }
        ready.notify_all();
    };
    ninfer::Engine engine(options);
    const auto result = engine.generate(engine.prepare_tokens(prompt), request());
    require(result.generated_token_ids == reference.generated_token_ids &&
            result.finish_reason == reference.finish_reason, "calibrated target-only changed tokens or completion");
    {
        std::unique_lock lock(mutex);
        require(ready.wait_for(lock, std::chrono::seconds(10), [&] { return !events.empty(); }), "no successful calibrated round");
        // Check the selected physical action before waiting for its target-only round count. The
        // missing implementation must fail here instead of merely timing out waiting for 15 events.
        for (const auto& event : events) {
            require(event.active_batch == 1 && event.draft_tokens == action && event.verify_width == action + 1 &&
                    event.backend == ninfer::SpeculativeBackend::DFlash2 &&
                    event.proposal_width == (action == 0 ? 0U : 16U) &&
                    event.neural_drafter_executed == (action != 0),
                    "forced profile did not execute its physical target and resident proposal widths");
        }
        require(ready.wait_for(lock, std::chrono::seconds(10), [&] {
            std::uint64_t committed = 0;
            for (const auto& event : events) { committed += event.committed_tokens; }
            return committed == 15;
        }), "forced action committed rounds incomplete");
    }
    const auto stats = engine.runtime_stats();
    const auto rounds = events.size();
    require(stats.calibrated_target_only_rounds == (action == 0 ? rounds : 0) &&
            stats.calibrated_k7_rounds == (action == 7 ? rounds : 0) &&
            stats.calibrated_k11_rounds == (action == 11 ? rounds : 0) &&
            stats.calibrated_k15_rounds == (action == 15 ? rounds : 0) &&
            stats.calibrated_fixed_fallback_rounds == 0 && stats.calibrated_route_switches == 0,
            "calibrated settled action counters differ");
    std::cout << "ok calibrated forced K" << action
              << ": exact output, physical width " << action + 1
              << ", resident proposal " << (action == 0 ? 0 : 16)
              << ", settled counters\n";
}
}
#include "calibrated_routing_switching.h"

namespace {
void a16_consistency(const char* artifact, const std::string& route = "both") {
    require(route == "both" || route == "fixed15" || route == "cal0", "unknown A16 consistency route");
    ninfer::EngineOptions options;
    options.artifact_path = artifact;
    options.max_context = 8192;
    options.prefill_chunk = 1024;
    options.max_concurrency = 1;
    options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(8192);
    options.kv_cache = ninfer::KvCacheStorage::Int8Group64;
    options.use_cuda_graph = true;
    options.prefill_a8 = false;
    ninfer::PromptInput input;
    ninfer::ChatMessage user;
    user.role = ninfer::ChatRole::User;
    user.parts.push_back({.kind = ninfer::MessagePartKind::Text,
        .text = "A user says: 'My API becomes unstable only when I raise traffic above the configured rate limit.' Reply as a systems engineer: explain retry storms, exponential backoff, jitter, and per-client rate distribution.",
        .media = {}});
    input.messages.push_back(std::move(user));
    input.options.enable_thinking = false;
    auto generation = request();
    generation.execution.requested_output_tokens = 512;
    generation.execution.sampling.presence_penalty = 0.0F;
    generation.execution.sampling.frequency_penalty = 0.0F;
    std::cout << "A16 qualification prefill_a8=false graph=on cache=on INT8 C1 context8192 prefill1024 output512 route="
              << route << std::endl;
    const auto generate = [&](ninfer::Engine& engine, const ninfer::RequestOptions& execution,
                              const char* label) {
        auto handle = engine.submit(engine.prepare(input), execution);
        const auto sampling = handle.resolved_sampling();
        std::cout << nlohmann::json{{"request", label}, {"temperature", sampling.temperature},
            {"top_k", sampling.top_k}, {"top_p", sampling.top_p}, {"min_p", sampling.min_p},
            {"presence_penalty", sampling.presence_penalty}, {"frequency_penalty", sampling.frequency_penalty},
            {"seed", sampling.seed}} << std::endl;
        require(sampling.temperature == 0.0F && sampling.presence_penalty == 0.0F &&
                    sampling.frequency_penalty == 0.0F,
                "A16 request did not resolve explicit greedy zero penalties");
        auto result = handle.wait();
        std::cout << nlohmann::json{{"request", label},
            {"prefix_reuse_path", static_cast<unsigned>(result.prefix_reuse_path)},
            {"reused_prompt_tokens", result.reused_prompt_tokens},
            {"generated_tokens", result.generated_token_ids.size()}} << std::endl;
        return result;
    };
    ninfer::GenerationResult baseline;
    const auto compare = [&](const ninfer::GenerationResult& result, const char* label) {
        require(result.generated_token_ids.size() == baseline.generated_token_ids.size(),
                "A16 routes returned different output budgets");
        for (std::size_t i = 0; i < baseline.generated_token_ids.size(); ++i) {
            if (baseline.generated_token_ids[i] != result.generated_token_ids[i]) {
                throw std::runtime_error(std::string(label) + " A16 baseline mismatch at generated index " +
                    std::to_string(i) + ": baseline=" + std::to_string(baseline.generated_token_ids[i]) +
                    " candidate=" + std::to_string(result.generated_token_ids[i]));
            }
        }
        calibrated_switching::same_response(result, baseline);
    };
    const auto exercise = [&](ninfer::Engine& engine, const char* label) {
        auto first = generate(engine, generation, "first fresh");
        if (baseline.generated_token_ids.empty()) {
            baseline = first;
            require(baseline.generated_token_ids.size() == generation.execution.requested_output_tokens,
                    "A16 baseline stopped before original output budget");
        }
        compare(first, label);
        auto reuse = generation;
        reuse.execution.allow_prefix_reuse = true;
        const auto root = generate(engine, reuse, "retained root");
        compare(root, label);
        const auto restored = generate(engine, reuse, "restored private closure");
        compare(restored, label);
        const auto fresh = generate(engine, generation, "final fresh");
        compare(fresh, label);
        require(root.prefix_reuse_path == ninfer::PrefixReusePath::Root &&
                    restored.prefix_reuse_path == ninfer::PrefixReusePath::PrivateTurnClosure &&
                    restored.reused_prompt_tokens != 0 && fresh.reused_prompt_tokens == 0,
                "A16 consistency did not exercise Root -> PrivateTurnClosure and reuse-off");
        std::cout << "ok A16 " << label << ": original chat512 exact response, Root -> PrivateTurnClosure/reuse-off\n"
                  << std::flush;
    };
    {
        ninfer::Engine target(options);
        exercise(target, "None");
    }
    options.speculative.backend = ninfer::SpeculativeBackend::DFlash2;
    options.speculative.draft_tokens = 15;
    options.speculative.proposal_head = ninfer::ProposalHead::Full;
    ProfileFile profile;
    {
        ninfer::Engine fixed(options);
        const auto identity = fixed.load_summary().speculative_routing_identity;
        require(identity.has_value() && !identity->execution_options.prefill_a8,
                "A16 routing identity omitted effective prefill setting");
        if (route != "cal0") { exercise(fixed, "fixed K15"); }
        nlohmann::json cells = nlohmann::json::array();
        for (unsigned upper : {1024, 8192, 32768}) {
            cells.push_back({{"active_batch", 1}, {"frontier_upper", upper}, {"draft_tokens", 0}});
        }
        std::ofstream stream(profile.path);
        stream << nlohmann::json{{"schema_version", 1}, {"artifact_type", "ninfer_spec_router_profile"},
            {"identity", ninfer::runtime::routing_profile_identity_json(*identity)}, {"cells", cells}};
    }
    if (route == "fixed15") { return; }
    options.speculative.routing.mode = ninfer::SpeculativeRoutingMode::Calibrated;
    options.speculative.routing.profile_path = profile.path;
    std::mutex mutex;
    std::condition_variable ready;
    std::uint64_t committed = 0;
    bool metadata_valid = true;
    ninfer::Engine* observed = nullptr;
    options.decode_round_observer.callback = [&](const auto& event) {
        std::lock_guard lock(mutex);
        const auto stats = observed->runtime_stats();
        if (committed == 0) {
            std::cout << nlohmann::json{{"first_cal_round", true}, {"draft_tokens", event.draft_tokens},
                {"verify_width", event.verify_width}, {"proposal_width", event.proposal_width},
                {"neural_drafter_executed", event.neural_drafter_executed},
                {"fixed_fallback_rounds", stats.calibrated_fixed_fallback_rounds}} << std::endl;
        }
        metadata_valid = metadata_valid && event.active_batch == 1 && event.verify_width == 1 &&
            event.draft_tokens == 0 && event.proposal_width == 0 && !event.neural_drafter_executed &&
            stats.calibrated_fixed_fallback_rounds == 0;
        committed += event.committed_tokens;
        ready.notify_all();
    };
    ninfer::Engine calibrated(options);
    observed = &calibrated;
    exercise(calibrated, "Cal forced0");
    std::unique_lock lock(mutex);
    require(ready.wait_for(lock, std::chrono::seconds(10), [&] { return committed == 4 * 511; }) && metadata_valid,
            "A16 Cal forced0 did not publish its physical action or actual commits");
}
}

#include "calibrated_program_zero_commit.h"

int main(int argc, char** argv) {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (!artifact || !*artifact) { std::cout << "skip: set NINFER_TEST_ARTIFACT\n"; return 77; }
    try {
        if (argc == 2 && std::string(argv[1]) == "--program-zero-commit") {
            program_zero_commit(artifact);
        } else if ((argc == 2 || argc == 3) && std::string(argv[1]) == "--a16-consistency") {
            a16_consistency(artifact, argc == 3 ? argv[2] : "both");
        } else if (argc == 3 && std::string(argv[1]) == "--resource-probe") {
            resource_probe(artifact, static_cast<std::uint32_t>(std::stoul(argv[2])));
        } else if (argc == 3 && std::string(argv[1]) == "--action") {
            run(artifact, static_cast<std::uint32_t>(std::stoul(argv[2])));
        } else if (argc == 3 && std::string(argv[1]) == "--switching") {
            calibrated_switching::run(artifact, argv[2]);
        } else { run(artifact); }
        return 0;
    }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
