// Public Engine generation measurement. No inference implementation lives here.
#include "ninfer/engine.h"
#include "ninfer_bench_support.h"
#include "runtime/engine/speculative_routing_profile.h"
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <iterator>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
std::uint64_t ns(Clock::duration d) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(d).count();
}
std::string quote(std::string_view s) {
    return "\"" + ninfer::bench::json_escape(s) + "\"";
}
unsigned number(const std::string& s, unsigned low, unsigned high) {
    std::size_t end = 0;
    const auto value = std::stoul(s, &end);
    if (end != s.size() || value < low || value > high)
        throw std::invalid_argument("out of range integer: " + s);
    return static_cast<unsigned>(value);
}
struct Measurement {
    unsigned repeat, slot;
    std::uint64_t latency_ns;
    ninfer::GenerationResult result;
};

using Json = nlohmann::json;
Json measurement_json(const Measurement& measured) {
    const auto& result = measured.result;
    Json tools = Json::array();
    for (const auto& tool : result.tool_calls) {
        tools.push_back({{"name", tool.name}, {"arguments_json", tool.arguments_json}});
    }
    Json response = {
        {"repeat", measured.repeat}, {"slot", measured.slot}, {"latency_ns", measured.latency_ns},
        {"prompt_tokens", result.prompt.prompt_tokens},
        {"finish_reason", static_cast<int>(result.finish_reason)},
        {"prefix_reuse_path", static_cast<int>(result.prefix_reuse_path)},
        {"reused_prompt_tokens", result.reused_prompt_tokens},
        {"content", result.content}, {"reasoning", result.reasoning},
        {"matched_stop_string", nullptr}, {"tool_calls", std::move(tools)},
        {"generated_token_ids", result.generated_token_ids},
    };
    if (result.matched_stop_string) { response["matched_stop_string"] = *result.matched_stop_string; }
    return response;
}
Json rounds_json(const std::vector<ninfer::DecodeRoundEvent>& rounds) {
    Json result = Json::array();
    for (std::size_t index = 0; index < rounds.size(); ++index) {
        const auto& event = rounds[index];
        result.push_back({
            {"round_index", index}, {"active_batch", event.active_batch},
            {"max_execution_frontier", event.max_execution_frontier},
            {"verify_width", event.verify_width}, {"draft_tokens", event.draft_tokens},
            {"proposal_width", event.proposal_width}, {"backend", static_cast<int>(event.backend)},
            {"neural_drafter_executed", event.neural_drafter_executed},
            {"committed_tokens", event.committed_tokens}, {"elapsed_ns", event.elapsed_ns},
        });
    }
    return result;
}
}

int main(int argc, char** argv) try {
    std::map<std::string, std::string> args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help") {
            std::cout << "ninfer_spec_router_calibration_bench --model PATH --prompt-file PATH --output PATH "
                         "[--draft-tokens 0|7|11|15] [--client-concurrency 1..8] "
                         "[--engine-concurrency 1..8] [--repeats 2] [--max-tokens 512] "
                         "[--max-context 32768] [--kv-capacity 32768] [--device 0]\n"
                         "INT8 KV, greedy zero penalties, thinking off, cache on, graph on. "
                         "Client concurrency controls submissions; events report actual batches.\n"
                         "--identity-only loads Fixed DFlash2 K15 and exports the public identity without generation.\n"
                         "--spec-router-profile PATH measures calibrated resident K15; --draft-tokens is the expected action.\n"
                         "--prime-prefix explicitly runs a separate 16-token prefix primer before measurement.\n"
                         "--allow-route-switching with a profile measures actual table-selected actions; omit --draft-tokens.\n";
            return 0;
        }
        if (key == "--identity-only" || key == "--prime-prefix" || key == "--allow-route-switching") {
            if (!args.emplace(key, "1").second) { throw std::invalid_argument("duplicate flag: " + key); }
            continue;
        }
        const std::vector<std::string> allowed = {"--model", "--prompt-file", "--output",
            "--draft-tokens", "--client-concurrency", "--engine-concurrency", "--repeats",
            "--max-tokens", "--max-context", "--kv-capacity", "--device", "--spec-router-profile"};
        if (std::find(allowed.begin(), allowed.end(), key) == allowed.end() || i + 1 == argc
            || !args.emplace(key, argv[++i]).second)
            throw std::invalid_argument("unknown, duplicate or incomplete option: " + key);
    }
    auto get = [&](const std::string& key, const std::string& fallback) {
        const auto found = args.find(key);
        return found == args.end() ? fallback : found->second;
    };
    const bool identity_only = args.contains("--identity-only");
    const bool resident = args.contains("--spec-router-profile");
    const bool prime_prefix = args.contains("--prime-prefix");
    const bool route_switching = args.contains("--allow-route-switching");
    if (route_switching && (!resident || identity_only || args.contains("--draft-tokens"))) {
        throw std::invalid_argument("route switching requires a profile, excludes identity-only and explicit draft tokens");
    }
    if ((identity_only && (resident || prime_prefix)) || (prime_prefix && !resident)) {
        throw std::invalid_argument("identity-only requires Fixed startup; prime-prefix requires a profile");
    }
    for (const auto key : {"--model", "--output"})
        if (!args.contains(key)) throw std::invalid_argument(std::string("required: ") + key);
    if (!identity_only && !args.contains("--prompt-file")) {
        throw std::invalid_argument("required: --prompt-file");
    }
    const auto clients = number(get("--client-concurrency", "1"), 1, 8);
    const auto concurrency = number(get("--engine-concurrency", "8"), clients, 8);
    const auto repeats = number(get("--repeats", "2"), 1, 1000);
    const auto budget = number(get("--max-tokens", "512"), 2, 32768);
    const auto k = number(get("--draft-tokens", identity_only ? "15" : "0"), 0, 15);
    if (k != 0 && k != 7 && k != 11 && k != 15)
        throw std::invalid_argument("draft tokens must be 0,7,11,15");
    if (identity_only && k != 15) {
        throw std::invalid_argument("identity-only requires Fixed DFlash2 K15");
    }
    if (resident && !std::filesystem::is_regular_file(args.at("--spec-router-profile"))) {
        throw std::invalid_argument("profile must be an explicit existing file");
    }
    if (!std::filesystem::is_regular_file(args.at("--model")))
        throw std::invalid_argument("model must be an explicit existing artifact");
    if (std::filesystem::exists(args.at("--output")))
        throw std::invalid_argument("output already exists");
    std::string prompt;
    if (!identity_only) {
        std::ifstream input(args.at("--prompt-file"), std::ios::binary);
        if (!input) throw std::invalid_argument("cannot read prompt");
        prompt.assign(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
        if (prompt.empty()) throw std::invalid_argument("empty prompt");
    }

    std::vector<ninfer::DecodeRoundEvent> rounds;
    std::mutex round_mutex;
    std::condition_variable round_settled;
    std::uint64_t observed_commits = 0;
    Json routing_identity;
    ninfer::runtime::CalibratedRoutingTable routing_table;
    Json normalized_routing_cells = Json::array();
    Json priming = {{"enabled", false}};
    ninfer::RuntimeStats runtime_snapshot;
    std::vector<Measurement> requests;
    std::vector<std::uint64_t> wave_wall_ns;
    ninfer::MemorySummary memory;
    std::string model_name;
    const auto device = number(get("--device", "0"), 0, 255);
    const auto context = number(get("--max-context", "32768"), 2, 32768);
    const auto capacity = number(get("--kv-capacity", "32768"), 2, 262144);
    {
        ninfer::EngineOptions options;
        options.artifact_path = args.at("--model");
        options.device = static_cast<int>(device);
        options.max_context = context;
        options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(capacity);
        options.max_concurrency = concurrency;
        options.kv_cache = ninfer::KvCacheStorage::Int8Group64;
        options.use_cuda_graph = true;
        options.context_cache.enabled = true;
        if (identity_only || resident || k) {
            options.speculative.backend = ninfer::SpeculativeBackend::DFlash2;
            options.speculative.draft_tokens = identity_only || resident ? 15 : k;
        }
        if (resident) {
            options.speculative.routing.mode = ninfer::SpeculativeRoutingMode::Calibrated;
            options.speculative.routing.profile_path = args.at("--spec-router-profile");
        }
        options.decode_round_observer.callback = [&](const ninfer::DecodeRoundEvent& event) {
            std::lock_guard lock(round_mutex);
            rounds.push_back(event);
            if (prime_prefix) {
                observed_commits += event.committed_tokens;
                round_settled.notify_all();
            }
        };
        ninfer::Engine engine(std::move(options));
        const auto load = engine.load_summary();
        model_name = load.model_name;
        if (identity_only || resident) {
            if (!load.speculative_routing_identity) {
                throw std::runtime_error("Engine did not publish its routing identity");
            }
            routing_identity = ninfer::runtime::routing_profile_identity_json(*load.speculative_routing_identity);
        }
        if (route_switching) {
            routing_table = ninfer::runtime::load_calibrated_routing_profile(
                args.at("--spec-router-profile"), *load.speculative_routing_identity);
            for (std::uint32_t batch = 1; batch <= 8; ++batch) {
                for (const std::uint32_t upper : {1024U, 8192U, 32768U}) {
                    normalized_routing_cells.push_back({{"active_batch", batch}, {"frontier_upper", upper},
                                                        {"draft_tokens", routing_table.select(batch, upper)}});
                }
            }
        }
        if (prime_prefix) {
            ninfer::PromptInput prefix;
            ninfer::ChatMessage message;
            ninfer::MessagePart part;
            part.text = prompt;
            message.parts.push_back(std::move(part));
            prefix.messages.push_back(std::move(message));
            prefix.options.enable_thinking = false;
            ninfer::RequestOptions request;
            request.execution.requested_output_tokens = 16;
            request.execution.allow_prefix_reuse = true;
            request.execution.sampling.temperature = 0.0F;
            request.execution.sampling.presence_penalty = 0.0F;
            request.execution.sampling.frequency_penalty = 0.0F;
            request.execution.sampling.seed = 0;
            request.stop.include_model_defaults = false;
            const auto start = Clock::now();
            auto result = engine.generate(engine.prepare(std::move(prefix)), request);
            const auto waiter_elapsed = ns(Clock::now() - start);
            if (result.finish_reason != ninfer::FinishReason::OutputLimit ||
                result.generated_token_ids.size() != 16) {
                throw std::runtime_error("primer did not complete its 16-token output budget");
            }
            // Output readiness can precede observer publication. Wait for settled real commits
            // without polling, sleeping or adding a CUDA synchronization.
            const auto expected_commits = result.generated_token_ids.size() - 1;
            std::unique_lock lock(round_mutex);
            round_settled.wait(lock, [&] { return observed_commits >= expected_commits; });
            if (observed_commits != expected_commits) {
                throw std::runtime_error("primer settled commit accounting mismatch");
            }
            priming = {{"enabled", true}, {"requested_output_tokens", 16},
                       {"request", measurement_json(Measurement{0, 0, waiter_elapsed, std::move(result)})},
                       {"rounds", rounds_json(rounds)}, {"wave_wall_ns", ns(Clock::now() - start)},
                       {"settled_committed_tokens", observed_commits}};
            rounds.clear();
            observed_commits = 0;
        }
        engine.reset_memory_peaks();
        if (!identity_only) for (unsigned rep = 0; rep < repeats; ++rep) {
            std::vector<ninfer::PreparedPrompt> prepared;
            for (unsigned slot = 0; slot < clients; ++slot) {
                ninfer::PromptInput p;
                ninfer::ChatMessage message;
                ninfer::MessagePart part;
                part.text = prompt;
                message.parts.push_back(std::move(part));
                p.messages.push_back(std::move(message));
                p.options.enable_thinking = false;
                prepared.push_back(engine.prepare(std::move(p)));
            }
            ninfer::RequestOptions request;
            request.execution.requested_output_tokens = budget;
            request.execution.allow_prefix_reuse = true;
            request.execution.sampling.temperature = 0.0F;
            request.execution.sampling.presence_penalty = 0.0F;
            request.execution.sampling.frequency_penalty = 0.0F;
            request.execution.sampling.seed = 0;
            std::vector<std::future<Measurement>> futures;
            const auto wave_start = Clock::now();
            for (unsigned slot = 0; slot < clients; ++slot) {
                const auto start = Clock::now();
                auto handle = engine.submit(std::move(prepared[slot]), request);
                futures.push_back(std::async(std::launch::async,
                    [handle = std::move(handle), start, rep, slot]() mutable {
                        auto result = handle.wait();
                        return Measurement{rep, slot, ns(Clock::now() - start), std::move(result)};
                    }));
            }
            for (auto& future : futures) requests.push_back(future.get());
            wave_wall_ns.push_back(ns(Clock::now() - wave_start));
        }
        memory = engine.memory_summary();
        runtime_snapshot = engine.runtime_stats();
    } // Join worker before reading the observer's final publication.
    std::ofstream destination(args.at("--output"), std::ios::binary);
    if (!destination) throw std::runtime_error("cannot create output");
    if (identity_only) {
        destination << Json{{"schema_version", 1}, {"artifact_type", "ninfer_routing_identity"},
                            {"identity", routing_identity}}.dump() << '\n';
        if (!destination) throw std::runtime_error("identity report write failed");
        return 0;
    }
    std::ostringstream output;
    output << "{\"schema_version\":1,\"artifact_type\":\"ninfer_calibration_measurement\","
           << "\"model\":" << quote(args.at("--model")) << ",\"model_name\":" << quote(model_name)
           << ",\"device\":" << device << ",\"client_concurrency\":" << clients
           << ",\"engine_concurrency\":" << concurrency << ",\"draft_tokens\":" << k
           << ",\"repeats\":" << repeats << ",\"proposal_head\":\"full\""
           << ",\"max_context\":" << context << ",\"kv_capacity\":" << capacity
           << ",\"max_tokens\":" << budget << ",\"kv\":\"int8\",\"graph\":true,\"cache\":true,"
           << "\"greedy\":true,\"presence_penalty\":0,\"frequency_penalty\":0,"
           << "\"wave_wall_ns\":[";
    for (std::size_t i = 0; i < wave_wall_ns.size(); ++i)
        output << (i ? "," : "") << wave_wall_ns[i];
    output << "],\"requests\":[";
    for (std::size_t i = 0; i < requests.size(); ++i) {
        const auto& r = requests[i]; const auto& g = r.result;
        output << (i ? "," : "") << "{\"repeat\":" << r.repeat << ",\"slot\":" << r.slot
               << ",\"latency_ns\":" << r.latency_ns << ",\"prompt_tokens\":" << g.prompt.prompt_tokens
               << ",\"finish_reason\":" << static_cast<int>(g.finish_reason)
               << ",\"prefix_reuse_path\":" << static_cast<int>(g.prefix_reuse_path)
               << ",\"reused_prompt_tokens\":" << g.reused_prompt_tokens
               << ",\"content\":" << quote(g.content) << ",\"reasoning\":" << quote(g.reasoning)
               << ",\"matched_stop_string\":"
               << (g.matched_stop_string ? quote(*g.matched_stop_string) : "null")
               << ",\"tool_calls\":[";
        for (std::size_t t = 0; t < g.tool_calls.size(); ++t)
            output << (t ? "," : "") << "{\"name\":" << quote(g.tool_calls[t].name)
                   << ",\"arguments_json\":" << quote(g.tool_calls[t].arguments_json) << "}";
        output << "],\"generated_token_ids\":[";
        for (std::size_t j = 0; j < g.generated_token_ids.size(); ++j)
            output << (j ? "," : "") << g.generated_token_ids[j];
        output << "]}";
    }
    output << "],\"rounds\":[";
    for (std::size_t i = 0; i < rounds.size(); ++i) {
        const auto& e = rounds[i];
        output << (i ? "," : "") << "{\"active_batch\":" << e.active_batch
               << ",\"max_execution_frontier\":" << e.max_execution_frontier
               << ",\"verify_width\":" << e.verify_width << ",\"draft_tokens\":" << e.draft_tokens
               << ",\"proposal_width\":" << e.proposal_width
               << ",\"backend\":" << static_cast<int>(e.backend)
               << ",\"neural_drafter_executed\":" << (e.neural_drafter_executed ? "true" : "false")
               << ",\"committed_tokens\":" << e.committed_tokens
               << ",\"elapsed_ns\":" << e.elapsed_ns << "}";
    }
    output << "],\"memory\":{\"workspace_logical_peak_bytes\":" << memory.workspace_logical_peak_bytes
           << ",\"workspace_allocator_peak_bytes\":" << memory.workspace.peak_used_bytes
           << ",\"runtime_reservation_bytes\":" << memory.runtime_reservation_bytes << "}}\n";
    if (resident) {
        const auto legacy = Json::parse(output.str());
        Json graph_memory = legacy.at("memory");
        graph_memory["cuda_graph_allowance_bytes"] = memory.cuda_graph_allowance_bytes;
        graph_memory["cuda_graph_definition_count"] = memory.cuda_graph_definition_count;
        graph_memory["cuda_graph_executable_count"] = memory.cuda_graph_executable_count;
        graph_memory["cuda_graph_prepare_peak_device_delta_bytes"] = memory.cuda_graph_prepare_peak_device_delta_bytes;
        graph_memory["cuda_graph_prepare_device_delta_bytes"] = memory.cuda_graph_prepare_device_delta_bytes;
        Json counters = {
            {"calibrated_target_only_rounds", runtime_snapshot.calibrated_target_only_rounds},
            {"calibrated_k7_rounds", runtime_snapshot.calibrated_k7_rounds},
            {"calibrated_k11_rounds", runtime_snapshot.calibrated_k11_rounds},
            {"calibrated_k15_rounds", runtime_snapshot.calibrated_k15_rounds},
            {"calibrated_route_switches", runtime_snapshot.calibrated_route_switches},
            {"calibrated_fixed_fallback_rounds", runtime_snapshot.calibrated_fixed_fallback_rounds},
        };
        Json record = {
            {"identity", routing_identity}, {"requested_action", k},
            {"configuration", {{"prompt", prompt}, {"max_tokens", budget},
                               {"client_concurrency", clients}, {"repeats", repeats},
                               {"sampling", {{"temperature", 0}, {"presence_penalty", 0}, {"frequency_penalty", 0}}}}},
            {"requests", legacy.at("requests")}, {"wave_wall_ns", wave_wall_ns},
            {"rounds", rounds_json(rounds)},
        };
        if (route_switching) { record.erase("requested_action"); }
        Json wrapper{
            {"schema_version", 1}, {"artifact_type", route_switching ? "ninfer_auto_router_measurement" :
                                                                  "ninfer_resident_router_measurement"},
            {"record", std::move(record)},
            {"resources", {{"memory", std::move(graph_memory)},
                           {"routing_counters_scope", "published_engine_snapshot_including_priming"},
                           {"runtime_stats", std::move(counters)}}},
            {"priming", std::move(priming)},
        };
        if (route_switching) {
            wrapper["routing_profile"] = {{"path", args.at("--spec-router-profile")},
                                          {"cells", normalized_routing_cells}};
        }
        destination << wrapper.dump() << '\n';
    } else {
        destination << output.str();
    }
    if (!destination) throw std::runtime_error("report write failed");
    if (resident) for (const auto& event : rounds) {
        const auto action = route_switching ? routing_table.select(event.active_batch, event.max_execution_frontier) : k;
        if (event.active_batch == 0 || event.active_batch > clients ||
            event.max_execution_frontier == 0 || event.max_execution_frontier > context ||
            event.backend != ninfer::SpeculativeBackend::DFlash2 || event.draft_tokens != action ||
            event.verify_width != action + 1 || event.proposal_width != (action == 0 ? 0 : 16) ||
            event.neural_drafter_executed != (action != 0)) {
            throw std::runtime_error(route_switching ? "actual auto action differs from profile/physical contract" :
                                                       "actual resident action differs from --draft-tokens expectation");
        }
    }
    if (rounds.empty()) throw std::runtime_error("no decode events; no batch calibration evidence");
    return 0;
} catch (const std::exception& error) {
    std::cerr << "calibration benchmark: " << error.what() << '\n';
    return 1;
}
