#include "ninfer_bench_support.h"

#include "ninfer/engine.h"
#include "runtime/engine/resident_model.h"
#include <nlohmann/json.hpp>

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <exception>
#include <algorithm>
#include <memory>
#include <array>
#include <chrono>
#include <cstdlib>
#include <map>
#include <optional>
#include <set>
#include <string_view>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string command_line(int argc, char** argv) {
    std::ostringstream out;
    for (int i = 0; i < argc; ++i) {
        if (i != 0) { out << ' '; }
        out << argv[i];
    }
    return out.str();
}

std::string cuda_version_string(int version) {
    if (version <= 0) { return {}; }
    return std::to_string(version / 1000) + "." + std::to_string((version % 1000) / 10);
}

void fill_cuda_environment(ninfer::bench::BenchEnvironment& env, int device) {
    env.device_id       = device;
    int runtime_version = 0;
    if (cudaRuntimeGetVersion(&runtime_version) == cudaSuccess) {
        env.cuda_runtime_version = cuda_version_string(runtime_version);
    }
    int driver_version = 0;
    if (cudaDriverGetVersion(&driver_version) == cudaSuccess) {
        env.cuda_driver_version = cuda_version_string(driver_version);
    }
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, device) == cudaSuccess) {
        env.gpu_name = properties.name;
    }
}

void require_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

bool has_decode_tests(const std::vector<ninfer::bench::BenchTest>& tests) {
    for (const auto& test : tests) {
        if (test.has_decode()) { return true; }
    }
    return false;
}

ninfer::RequestOptions benchmark_request(const ninfer::bench::BenchTest& test) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = test.requested_output_tokens();
    options.execution.allow_prefix_reuse      = false;
    options.execution.sampling.temperature    = 0.0F;
    options.stop.include_model_defaults       = false;
    options.output.raw                        = true;
    options.output.preserve_special_tokens    = true;
    return options;
}

ninfer::bench::RepTiming run_repetition(ninfer::Engine& engine,
                                        const ninfer::bench::BenchTest& test,
                                        const std::vector<ninfer::TokenId>& corpus) {
    const int prompt_tokens = test.kind == ninfer::bench::TestKind::Decode
                                  ? ninfer::bench::kDecodeSeedTokens
                                  : test.n_prompt;
    auto prompt = engine.prepare_tokens(ninfer::bench::prompt_slice(corpus, prompt_tokens), false);
    ninfer::GenerationResult generated =
        engine.generate(std::move(prompt), benchmark_request(test));

    const std::uint32_t expected = test.requested_output_tokens();
    if (generated.generated_token_ids.size() != expected) {
        throw std::runtime_error(test.label + " generated " +
                                 std::to_string(generated.generated_token_ids.size()) +
                                 " tokens; expected " + std::to_string(expected));
    }
    if (generated.finish_reason != ninfer::FinishReason::OutputLimit) {
        throw std::runtime_error(test.label + " did not finish at the requested output limit");
    }

    ninfer::bench::RepTiming timing;
    timing.timings                 = generated.timings;
    timing.speculative             = std::move(generated.speculative);
    timing.generated_output_tokens = expected;
    return timing;
}

void prime_decode_graph(ninfer::Engine& engine, ninfer::bench::BenchEnvironment& env,
                        const std::vector<ninfer::TokenId>& corpus) {
    if (!env.use_cuda_graph || env.decode_graph_prime_output_tokens == 0) { return; }
    const int decode_tokens = static_cast<int>(env.decode_graph_prime_output_tokens - 1);
    const ninfer::bench::BenchTest prime{ninfer::bench::TestKind::Decode, 0, decode_tokens,
                                         "decode-graph-prime"};
    (void)run_repetition(engine, prime, corpus);
    env.decode_graph_primed = true;
}

void write_output(const ninfer::bench::BenchOptions& options, const std::string& text) {
    if (options.output_file.empty()) {
        std::cout << text;
        return;
    }
    const std::filesystem::path path(options.output_file);
    if (!path.parent_path().empty()) { std::filesystem::create_directories(path.parent_path()); }
    std::ofstream output(path);
    if (!output) { throw std::runtime_error("failed to open output file: " + options.output_file); }
    output << text;
    std::cout << "wrote " << options.output_file << '\n';
}

// Resident A/B protocol: Engine/Program is recreated after each arm, but the materialized
// model weights stay on the selected GPU for the entire JSONL session. Never mutate a route
// while any Program, CUDA Graph, or kernel from the prior arm is still live.
using Json = nlohmann::json;

class ScopedStdoutToStderr {
public:
    ScopedStdoutToStderr() : saved_(std::cout.rdbuf(std::cerr.rdbuf())) {}
    ~ScopedStdoutToStderr() { std::cout.rdbuf(saved_); }
    ScopedStdoutToStderr(const ScopedStdoutToStderr&) = delete;
    ScopedStdoutToStderr& operator=(const ScopedStdoutToStderr&) = delete;
private:
    std::streambuf* saved_;
};

constexpr std::array<std::string_view, 10> kRouteEnvironment{
    "NINFER_DEVICE_ROUTE_MODE",
    "NINFER_DEVICE_PROFILE_PATH",
    "NINFER_DEVICE_PROFILES",
    "NINFER_DEVICE_ROUTE_ONLY",
    "NINFER_DEVICE_ROUTE_OVERRIDES",
    "NINFER_PROMPT_FAST",
    "NINFER_GDN_TWO_STAGE",
    "NINFER_GDN_TWO_STAGE_NUMERICS",
    "NINFER_PREFILL_ALIGN",
    "NINFER_DEVICE_ROUTE_TRACE",
};

void set_process_environment(const std::string& key, const std::optional<std::string>& value) {
#if defined(_WIN32)
    const int result = _putenv_s(key.c_str(), value ? value->c_str() : "");
#else
    const int result = value ? setenv(key.c_str(), value->c_str(), 1) : unsetenv(key.c_str());
#endif
    if (result != 0) throw std::runtime_error("cannot update route environment: " + key);
}

// Snapshot and restore *every* known key. Clearing unknown inherited settings before each
// arm avoids accidentally measuring two routes (including the required baseline) together.
class ScopedRouteEnvironment {
public:
    ScopedRouteEnvironment() {
        for (const auto key : kRouteEnvironment) {
            const char* current = std::getenv(std::string(key).c_str());
            previous_.emplace(std::string(key),
                              current ? std::optional<std::string>(current) : std::nullopt);
        }
    }
    ~ScopedRouteEnvironment() {
        for (const auto& [name, value] : previous_) {
            try { set_process_environment(name, value); } catch (...) {}
        }
    }
    void apply(const Json& config) {
        if (!config.is_object()) throw std::invalid_argument("resident arm env must be an object");
        for (const auto key : kRouteEnvironment)
            set_process_environment(std::string(key), std::nullopt);
        for (auto it = config.begin(); it != config.end(); ++it) {
            if (std::find(kRouteEnvironment.begin(), kRouteEnvironment.end(), it.key()) ==
                    kRouteEnvironment.end() ||
                !it.value().is_string() || it.value().get_ref<const std::string&>().size() > 4096)
                throw std::invalid_argument("unsupported resident route environment: " + it.key());
            set_process_environment(it.key(), it.value().get<std::string>());
        }
    }
private:
    std::map<std::string, std::optional<std::string>> previous_;
};

Json measure_resident_arm(ninfer::runtime::ResidentModelSession& resident,
                          const ninfer::EngineOptions& engine_options,
                          const ninfer::bench::BenchOptions& options,
                          ninfer::bench::BenchEnvironment base_env,
                          const std::vector<ninfer::bench::BenchTest>& tests,
                          const std::vector<ninfer::TokenId>& corpus,
                          const std::string& invocation) {
    ScopedStdoutToStderr redirect;
    const auto began = std::chrono::steady_clock::now();
    // Each arm gets its own CUDA graph, route choice and prefill workspace plan, but does
    // NOT repeat artifact reads, conversions, or weight upload.
    ninfer::Engine engine = resident.make_engine(engine_options);
    const double program_create_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count();
    base_env.prefill_chunk = engine.options().prefill_chunk;
    base_env.load = engine.load_summary();
    base_env.memory = engine.memory_summary();
    fill_cuda_environment(base_env, options.device);
    prime_decode_graph(engine, base_env, corpus);
    std::vector<ninfer::bench::TestResult> results;
    results.reserve(tests.size());
    for (const auto& test : tests) {
        ninfer::bench::TestResult result;
        result.test = test;
        engine.reset_memory_peaks();
        for (int warm = 0; warm < options.warmup; ++warm)
            (void)run_repetition(engine, test, corpus);
        result.reps.reserve(static_cast<std::size_t>(options.repetitions));
        for (int run = 0; run < options.repetitions; ++run)
            result.reps.push_back(run_repetition(engine, test, corpus));
        const auto memory = engine.memory_summary();
        result.workspace_peak_bytes = memory.workspace_logical_peak_bytes;
        result.workspace_allocator_peak_bytes = memory.workspace.peak_used_bytes;
        results.push_back(std::move(result));
    }
    Json report = Json::parse(ninfer::bench::format_json(base_env, invocation, results));
    report["residency"] = {
        {"scope", "single_process_weight_residency_fresh_program_per_arm"},
        {"model_load_count", resident.model_load_count()},
        {"resident_weight_bytes", resident.resident_weight_bytes()},
        {"program_create_seconds", program_create_seconds},
        {"program_recreated", true},
    };
    return report;
}

int resident_session(const ninfer::bench::BenchOptions& options,
                     const ninfer::EngineOptions& engine_options,
                     ninfer::bench::BenchEnvironment env,
                     const std::vector<ninfer::bench::BenchTest>& tests,
                     const std::vector<ninfer::TokenId>& corpus, std::string invocation) {
    if (options.speculative.backend != ninfer::SpeculativeBackend::DFlash2 ||
        options.speculative.proposal_head != ninfer::ProposalHead::Full ||
        options.profile_measured || !options.output_file.empty()) {
        throw std::invalid_argument(
            "--resident-session requires DFlash2 Full, no profiler and no --output-file");
    }
    std::unique_ptr<ninfer::runtime::ResidentModelSession> resident;
    {
        ScopedStdoutToStderr redirect;
        resident = std::make_unique<ninfer::runtime::ResidentModelSession>(engine_options);
    }
    if (resident->model_load_count() != 1)
        throw std::runtime_error("resident session must materialize weights exactly once");
    std::cout << Json{{"event", "ready"},
                      {"ok", true},
                      {"model_load_count", resident->model_load_count()},
                      {"resident_weight_bytes", resident->resident_weight_bytes()}}.dump()
              << std::endl;
    ScopedRouteEnvironment saved_environment;
    std::set<std::string> ids;
    std::string line;
    while (std::getline(std::cin, line)) {
        Json id = nullptr;
        try {
            if (line.size() > 16384) throw std::invalid_argument("resident packet too large");
            const Json packet = Json::parse(line);
            if (!packet.is_object() || !packet.contains("id") || !packet.at("id").is_string() ||
                packet.at("id").get<std::string>().empty() ||
                packet.at("id").get<std::string>().size() > 128)
                throw std::invalid_argument("resident request needs a bounded string id");
            id = packet.at("id");
            if (!ids.insert(id.get<std::string>()).second)
                throw std::invalid_argument("duplicate resident request id");
            if (packet.contains("stop")) {
                if (packet.size() != 2 || !packet.at("stop").is_boolean() ||
                    !packet.at("stop").get<bool>())
                    throw std::invalid_argument("resident stop accepts id and stop:true");
                std::cout << Json{{"event", "bye"}, {"id", id}, {"ok", true},
                                  {"model_load_count", resident->model_load_count()}}.dump()
                          << std::endl;
                return 0;
            }
            if (packet.size() != 2 || !packet.contains("route_env"))
                throw std::invalid_argument("resident arm requires id and route_env");
            saved_environment.apply(packet.at("route_env"));
            Json report = measure_resident_arm(*resident, engine_options, options, env, tests,
                                               corpus, invocation);
            if (resident->model_load_count() != 1)
                throw std::logic_error("resident route comparison rematerialized model weights");
            std::cout << Json{{"event", "measurement"}, {"id", id}, {"ok", true},
                              {"model_load_count", resident->model_load_count()},
                              {"report", std::move(report)}}.dump()
                      << std::endl;
        } catch (const std::exception& error) {
            std::cout << Json{{"event", "error"}, {"id", id}, {"ok", false},
                              {"error", error.what()}}.dump() << std::endl;
            std::cerr << "resident route matrix: " << error.what() << '\n';
            return 1; // Device state may be poisoned by a failing CUDA candidate.
        }
    }
    if (std::cin.bad()) throw std::runtime_error("resident session stdin read failure");
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    ninfer::bench::BenchOptions options;
    try {
        options = ninfer::bench::parse_args(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "ninfer_bench: " << error.what() << '\n';
        return 2;
    }
    if (options.help_requested) {
        std::cout << ninfer::bench::usage_text(argc > 0 ? argv[0] : "ninfer_bench");
        return 0;
    }

    try {
        const std::vector<ninfer::TokenId> corpus =
            ninfer::bench::load_corpus_ids(options.corpus_path);
        const std::vector<ninfer::bench::BenchTest> tests = ninfer::bench::expand_tests(options);
        if (options.profile_measured && (tests.size() != 1 || options.repetitions != 1)) {
            throw std::invalid_argument(
                "--profile-measured requires exactly one benchmark test and -r 1");
        }
        ninfer::bench::validate_prompt_lengths(tests, corpus.size());
        const std::uint32_t max_context = ninfer::bench::resolve_max_context(
            tests, options.max_context, options.speculative, options.use_cuda_graph);

        ninfer::EngineOptions engine_options;
        engine_options.artifact_path = options.artifact_path;
        engine_options.device        = options.device;
        engine_options.max_context   = max_context;
        engine_options.kv_capacity   = ninfer::KvCapacityPolicy::explicit_capacity(max_context);
        engine_options.prefill_chunk      = options.prefill_chunk;
        engine_options.prefill_chunk_auto = options.prefill_chunk_auto;
        engine_options.kv_cache      = options.kv_cache;
        engine_options.context_cache.enabled = false;
        engine_options.speculative           = options.speculative;
        engine_options.use_cuda_graph        = options.use_cuda_graph;
        engine_options.prefill_a8            = options.prefill_a8;
        engine_options.prefill_cublas        = options.prefill_cublas;
        engine_options.prefill_cublas_projections = options.prefill_cublas_projections;

        ninfer::bench::BenchEnvironment env;
        env.artifact_path            = options.artifact_path;
        env.artifact_file_size_bytes = ninfer::bench::file_size_or_zero(options.artifact_path);
        env.max_context              = max_context;
        env.requested_prefill_chunk  = options.prefill_chunk;
        env.prefill_chunk_auto       = options.prefill_chunk_auto;
        env.prefill_chunk            = options.prefill_chunk;
        env.kv_cache                 = options.kv_cache;
        env.speculative              = options.speculative;
        env.use_cuda_graph           = options.use_cuda_graph;
        env.repetitions              = options.repetitions;
        env.warmup                   = options.warmup;
        env.corpus_path              = options.corpus_path;
        env.corpus_tokens            = corpus.size();
        if (options.use_cuda_graph && has_decode_tests(tests)) {
            env.decode_graph_prime_output_tokens =
                ninfer::bench::decode_graph_prime_output_tokens(options.speculative);
        }

        std::cerr << "[ninfer_bench] loading " << options.artifact_path
                  << " (max_context=" << max_context
                  << ", kv_cache=" << ninfer::bench::kv_cache_name(options.kv_cache) << ")\n";
        if (options.resident_session) {
            return resident_session(options, engine_options, env, tests, corpus,
                                    command_line(argc, argv));
        }
        ninfer::Engine engine(std::move(engine_options));
        fill_cuda_environment(env, options.device);
        env.prefill_chunk = engine.options().prefill_chunk;
        env.load   = engine.load_summary();
        env.memory = engine.memory_summary();

        prime_decode_graph(engine, env, corpus);

        std::vector<ninfer::bench::TestResult> results;
        results.reserve(tests.size());
        for (std::size_t i = 0; i < tests.size(); ++i) {
            const auto& test = tests[i];
            std::cerr << "[ninfer_bench] test " << (i + 1) << '/' << tests.size() << ' '
                      << test.label << ": warmup=" << options.warmup
                      << " reps=" << options.repetitions << '\n';

            ninfer::bench::TestResult result;
            result.test = test;
            engine.reset_memory_peaks();
            for (int warmup = 0; warmup < options.warmup; ++warmup) {
                (void)run_repetition(engine, test, corpus);
            }
            result.reps.reserve(static_cast<std::size_t>(options.repetitions));
            if (options.profile_measured) {
                require_cuda(cudaDeviceSynchronize(), "profile pre-boundary synchronize");
                require_cuda(cudaProfilerStart(), "cudaProfilerStart");
            }
            for (int repetition = 0; repetition < options.repetitions; ++repetition) {
                result.reps.push_back(run_repetition(engine, test, corpus));
            }
            if (options.profile_measured) {
                require_cuda(cudaDeviceSynchronize(), "profile post-boundary synchronize");
                require_cuda(cudaProfilerStop(), "cudaProfilerStop");
            }
            const ninfer::MemorySummary memory    = engine.memory_summary();
            result.workspace_peak_bytes           = memory.workspace_logical_peak_bytes;
            result.workspace_allocator_peak_bytes = memory.workspace.peak_used_bytes;
            results.push_back(std::move(result));
        }

        std::string report;
        switch (options.output) {
        case ninfer::bench::OutputFormat::Table:
            report = ninfer::bench::format_table(env, results);
            break;
        case ninfer::bench::OutputFormat::Json:
            report = ninfer::bench::format_json(env, command_line(argc, argv), results);
            break;
        case ninfer::bench::OutputFormat::Csv:
            report = ninfer::bench::format_csv(env, results);
            break;
        }
        write_output(options, report);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ninfer_bench: " << error.what() << '\n';
        return 1;
    }
}
