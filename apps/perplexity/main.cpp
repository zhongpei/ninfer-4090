#include "corpus.h"
#include "evaluation.h"

#include "ninfer/engine.h"
#include "product/logging/logging.h"
#include "product/logging/pretty_format.h"
#include "product/logging/startup_log.h"

#include <nlohmann/json.hpp>
#include <spdlog/logger.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using json  = nlohmann::json;
using ninfer::perplexity::CorpusSelection;
using ninfer::perplexity::ScoreAggregate;
using ninfer::perplexity::WindowPlan;

struct Options {
    bool help_requested = false;
    std::filesystem::path artifact;
    std::optional<std::filesystem::path> corpus;
    std::optional<std::filesystem::path> text;
    std::optional<std::filesystem::path> output;
    std::uint32_t context     = 4096;
    std::uint32_t stride      = 2048;
    std::vector<std::uint32_t> depths;
    std::uint32_t tail_tokens = 2048;
    int device                = 0;
#if defined(NINFER_SM8X_COMPAT)
    // Keep INT8 as the compatibility/default baseline. sm_89 now has a native E4M3 QK/PV path,
    // but changing the product default requires the long-history quality and throughput evidence
    // this evaluator is designed to collect; sm_86 continues to use the legacy supported set.
    ninfer::KvCacheStorage kv = ninfer::KvCacheStorage::Int8Group64;
#else
    ninfer::KvCacheStorage kv = ninfer::KvCacheStorage::Fp8E4M3Row256;
#endif
    bool quick                          = false;
    bool lm_head_q4                     = false;
    bool lm_head_q6                     = false;
    bool embedding_q4                   = false;
    bool embedding_q6                   = false;
    bool mtp_experts_q4                 = false;
    bool gdn_state_fp16                 = false;
    bool mlp_a8_decode                  = false;
    bool prefill_a8                     = true;
    bool prefill_cublas                 = false;
    bool prefill_cublas_projections     = true;
    ninfer::product::LogLevel log_level = ninfer::product::LogLevel::Info;
};

std::string usage_text() {
    return "usage: ninfer-perplexity <model.ninfer> "
           "(--corpus <manifest.json> [--quick] | --text <utf8-file>)\n"
           "       [--context N] [--stride N] [--depths D1,D2,... --tail N] [--device N]\n"
           "       [--kv-dtype bf16|int8|fp8|rk8v4|rk4v4|rk4v4-e8|rk2v4-e8|nvfp4|k8v4] [--output <directory>]\n"
           "       (--depths switches to long-history quality mode: build prefix [0,D), score only\n"
           "        the following --tail tokens, and never truncate history before D)\n"
           "       [--lm-head-q4|--lm-head-q6] [--embedding-q4|--embedding-q6] [--mtp-experts-q4] [--gdn-state-fp16]\n"
           "       [--mlp-a8-decode] [--no-prefill-a8]\n"
           "       (--mlp-a8-decode is inert here: the route it enables is verify-phase"
           "        only, and scoring runs the prefill phase)\n"
           "       (--no-prefill-a8 is the opposite: scoring runs the prefill phase, so this is\n"
           "        how the integer prefill routes' perplexity cost is measured)\n"
           "       [--prefill-cublas [--no-prefill-cublas-projections]]\n"
           "       (--prefill-cublas scores through the cuBLAS prefill route, which is how its\n"
           "        perplexity cost is measured; the projections flag keeps the attention and GDN\n"
           "        input projections off it)\n"
           "       [--log-level trace|debug|info|warning|error|critical|off]\n";
}

[[noreturn]] void usage_error(std::string_view message) {
    throw std::invalid_argument(std::string(message));
}

template <class Integer>
Integer parse_integer(std::string_view text, const char* label) {
    Integer value{};
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size()) {
        usage_error(std::string("invalid ") + label + ": " + std::string(text));
    }
    return value;
}

std::vector<std::uint32_t> parse_depths(std::string_view text) {
    std::vector<std::uint32_t> out;
    std::size_t begin = 0;
    while (begin <= text.size()) {
        const std::size_t comma = text.find(',', begin);
        const std::string_view piece =
            text.substr(begin, comma == std::string_view::npos ? std::string_view::npos
                                                               : comma - begin);
        if (piece.empty()) usage_error("--depths entries must not be empty");
        const auto value = parse_integer<std::uint32_t>(piece, "depth");
        if (value == 0 || (!out.empty() && value <= out.back())) {
            usage_error("--depths must be positive and strictly increasing");
        }
        out.push_back(value);
        if (comma == std::string_view::npos) break;
        begin = comma + 1;
    }
    if (out.empty()) usage_error("--depths requires at least one depth");
    return out;
}

Options parse_options(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        return Options{.help_requested = true};
    }
    if (argc < 2 || std::string_view(argv[1]).starts_with("--")) {
        usage_error("artifact path is required");
    }
    Options out;
    out.artifact = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string_view option = argv[i];
        const auto value              = [&](const char* label) -> std::string_view {
            if (++i >= argc) { usage_error(std::string(label) + " requires a value"); }
            return argv[i];
        };
        if (option == "--corpus") {
            out.corpus = std::filesystem::path(value("--corpus"));
        } else if (option == "--text") {
            out.text = std::filesystem::path(value("--text"));
        } else if (option == "--quick") {
            out.quick = true;
        } else if (option == "--context") {
            out.context = parse_integer<std::uint32_t>(value("--context"), "context");
        } else if (option == "--stride") {
            out.stride = parse_integer<std::uint32_t>(value("--stride"), "stride");
        } else if (option == "--depths") {
            out.depths = parse_depths(value("--depths"));
        } else if (option == "--tail") {
            out.tail_tokens = parse_integer<std::uint32_t>(value("--tail"), "tail");
        } else if (option == "--device") {
            out.device = parse_integer<int>(value("--device"), "device");
        } else if (option == "--kv-dtype") {
            const std::string_view dtype = value("--kv-dtype");
            if (dtype == "bf16") {
                out.kv = ninfer::KvCacheStorage::BFloat16;
            } else if (dtype == "int8") {
                out.kv = ninfer::KvCacheStorage::Int8Group64;
            } else if (dtype == "fp8") {
                out.kv = ninfer::KvCacheStorage::Fp8E4M3Row256;
            } else if (dtype == "rk8v4") {
                out.kv = ninfer::KvCacheStorage::RotatedInt8KeyInt4ValueGroup64;
            } else if (dtype == "rk4v4") {
                out.kv = ninfer::KvCacheStorage::RotatedInt4KeyInt4ValueGroup64;
            } else if (dtype == "rk4v4-e8") {
                out.kv = ninfer::KvCacheStorage::RK4V4E8;
            } else if (dtype == "rk2v4-e8") {
                out.kv = ninfer::KvCacheStorage::RK2V4E8;
            } else if (dtype == "nvfp4") {
                out.kv = ninfer::KvCacheStorage::Nvfp4Group16;
            } else if (dtype == "k8v4") {
                out.kv = ninfer::KvCacheStorage::Fp8KeyNvfp4Value;
            } else {
                usage_error("--kv-dtype must be bf16, int8, fp8, rk8v4, rk4v4, rk4v4-e8, rk2v4-e8, nvfp4, or k8v4");
            }
        } else if (option == "--output") {
            out.output = std::filesystem::path(value("--output"));
        } else if (option == "--lm-head-q4") {
            out.lm_head_q4 = true;
        } else if (option == "--lm-head-q6") {
            out.lm_head_q6 = true;
        } else if (option == "--embedding-q4") {
            out.embedding_q4 = true;
        } else if (option == "--embedding-q6") {
            out.embedding_q6 = true;
        } else if (option == "--mtp-experts-q4") {
            out.mtp_experts_q4 = true;
        } else if (option == "--gdn-state-fp16") {
            out.gdn_state_fp16 = true;
        } else if (option == "--mlp-a8-decode") {
            out.mlp_a8_decode = true;
        } else if (option == "--no-prefill-a8") {
            out.prefill_a8 = false;
        } else if (option == "--prefill-cublas") {
            out.prefill_cublas = true;
        } else if (option == "--no-prefill-cublas-projections") {
            out.prefill_cublas_projections = false;
        } else if (option == "--log-level") {
            out.log_level = ninfer::product::parse_log_level(value("--log-level"));
        } else {
            usage_error("unknown option: " + std::string(option));
        }
    }
    if (out.corpus.has_value() == out.text.has_value()) {
        usage_error("exactly one of --corpus and --text is required");
    }
    if (out.quick && !out.corpus) { usage_error("--quick requires --corpus"); }
    if (out.context < 2 || out.stride == 0 || out.stride >= out.context) {
        usage_error("context/stride must satisfy context>=2 and 1<=stride<context");
    }
    if (out.tail_tokens == 0) usage_error("--tail must be positive");
    return out;
}

std::string kv_name(ninfer::KvCacheStorage value) {
    switch (value) {
    case ninfer::KvCacheStorage::BFloat16:
        return "bf16";
    case ninfer::KvCacheStorage::Int8Group64:
        return "int8";
    case ninfer::KvCacheStorage::Fp8E4M3Row256:
        return "fp8";
    case ninfer::KvCacheStorage::RotatedInt8KeyInt4ValueGroup64:
        return "rk8v4";
    case ninfer::KvCacheStorage::RotatedInt4KeyInt4ValueGroup64:
        return "rk4v4";
    case ninfer::KvCacheStorage::RK4V4E8:
        return "rk4v4-e8";
    case ninfer::KvCacheStorage::RK2V4E8:
        return "rk2v4-e8";
    case ninfer::KvCacheStorage::Nvfp4Group16:
        return "nvfp4";
    case ninfer::KvCacheStorage::Fp8KeyNvfp4Value:
        return "k8v4";
    }
    throw std::logic_error("unknown KV dtype");
}

std::string safe_component(std::string_view value) {
    std::string out;
    out.reserve(value.size());
    for (const unsigned char c : value) {
        out.push_back(std::isalnum(c) || c == '-' || c == '_' || c == '.' ? static_cast<char>(c)
                                                                          : '-');
    }
    return out.empty() ? "unknown" : out;
}

// gmtime_r is POSIX. MSVC provides gmtime_s with the destination first, the reverse of the
// POSIX argument order, so the two cannot be swapped by macro alone.
bool to_utc(const std::time_t& source, std::tm& out) {
#ifdef _WIN32
    return ::gmtime_s(&out, &source) == 0;
#else
    return ::gmtime_r(&source, &out) != nullptr;
#endif
}

std::string timestamp() {
    const std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm utc{};
    if (!to_utc(now, utc)) { throw std::runtime_error("failed to convert timestamp to UTC"); }
    std::ostringstream out;
    out << std::put_time(&utc, "%Y%m%d-%H%M%S");
    return out.str();
}

std::filesystem::path prepare_output_directory(const Options& options,
                                               const ninfer::LoadSummary& load,
                                               const CorpusSelection& corpus) {
    std::filesystem::path output = options.output.value_or(
        std::filesystem::path("profiles/perplexity") / safe_component(load.model_name) /
        safe_component(load.prefill_signature) / kv_name(options.kv) /
        safe_component(corpus.corpus_id) / safe_component(corpus.mode) / timestamp());
    if (std::filesystem::exists(output)) {
        if (!std::filesystem::is_directory(output) ||
            std::filesystem::directory_iterator(output) != std::filesystem::directory_iterator()) {
            throw std::runtime_error("output directory exists and is not empty: " +
                                     output.string());
        }
    } else if (!std::filesystem::create_directories(output)) {
        throw std::runtime_error("cannot create output directory: " + output.string());
    }
    return std::filesystem::absolute(output).lexically_normal();
}

double seconds_since(Clock::time_point begin) {
    return std::chrono::duration<double>(Clock::now() - begin).count();
}

json aggregate_json(const ScoreAggregate& value) {
    return json{{"scored_tokens", value.scored_tokens},
                {"total_nll", value.total_nll},
                {"mean_nll", value.mean_nll()},
                {"perplexity", value.ppl()}};
}

struct EvaluationStream {
    ninfer::perplexity::CorpusStream source;
    std::vector<ninfer::TokenId> tokens;
    std::vector<WindowPlan> windows;
};

int run(const Options& options, const std::shared_ptr<spdlog::logger>& logger,
        ninfer::product::StartupLogRenderer& startup_log,
        const std::shared_ptr<ninfer::product::TerminalProgress>& progress) {
    const Clock::time_point total_started = Clock::now();
    const bool depth_mode = !options.depths.empty();
    std::uint32_t effective_context = options.context;
    if (depth_mode) {
        const std::uint64_t required =
            static_cast<std::uint64_t>(options.depths.back()) + options.tail_tokens;
        if (required > std::numeric_limits<std::uint32_t>::max()) {
            throw std::invalid_argument("long-context perplexity depth+tail exceeds uint32");
        }
        effective_context =
            std::max(effective_context, static_cast<std::uint32_t>(required));
    }

    ninfer::EngineOptions engine_options;
    engine_options.artifact_path    = options.artifact;
    engine_options.purpose          = ninfer::EnginePurpose::CausalScoring;
    engine_options.device           = options.device;
    engine_options.max_context      = effective_context;
    engine_options.kv_capacity      = ninfer::KvCapacityPolicy::explicit_capacity(effective_context);
    engine_options.kv_cache         = options.kv;
    engine_options.lm_head_q4       = options.lm_head_q4;
    engine_options.lm_head_q6       = options.lm_head_q6;
    engine_options.embedding_q4     = options.embedding_q4;
    engine_options.embedding_q6     = options.embedding_q6;
    engine_options.mtp_experts_q4   = options.mtp_experts_q4;
    engine_options.gdn_state_fp16   = options.gdn_state_fp16;
    engine_options.mlp_a8_decode    = options.mlp_a8_decode;
    engine_options.prefill_a8       = options.prefill_a8;
    engine_options.prefill_cublas   = options.prefill_cublas;
    engine_options.prefill_cublas_projections = options.prefill_cublas_projections;
    engine_options.startup_observer = startup_log.observer();
    ninfer::Engine engine(std::move(engine_options));
    const ninfer::LoadSummary load = engine.load_summary();
    startup_log.engine_ready(load);

    const Clock::time_point preflight_started = Clock::now();
    logger->info("preparing corpus");
    CorpusSelection corpus = options.corpus
                                 ? ninfer::perplexity::load_corpus(*options.corpus, options.quick)
                                 : ninfer::perplexity::load_custom_text(*options.text);
    std::vector<EvaluationStream> streams;
    streams.reserve(corpus.streams.size());
    std::uint64_t total_scored_tokens = 0;
    std::uint64_t total_input_tokens  = 0;
    std::uint64_t total_windows       = 0;
    for (auto& source : corpus.streams) {
        std::vector<ninfer::TokenId> tokens = engine.tokenize_text(source.text);
        if (tokens.size() < 2) {
            throw std::runtime_error("stream tokenized to fewer than two tokens: " + source.id);
        }
        std::vector<WindowPlan> windows =
            depth_mode
                ? ninfer::perplexity::plan_depth_windows(tokens.size(), options.depths,
                                                         options.tail_tokens)
                : ninfer::perplexity::plan_windows(tokens.size(), options.context, options.stride);
        total_input_tokens += static_cast<std::uint64_t>(tokens.size());
        for (const WindowPlan& window : windows) {
            total_scored_tokens += static_cast<std::uint64_t>(window.target_end - window.target_begin);
        }
        total_windows += static_cast<std::uint64_t>(windows.size());
        if (!windows.empty()) {
            streams.push_back(EvaluationStream{.source  = std::move(source),
                                               .tokens  = std::move(tokens),
                                               .windows = std::move(windows)});
        }
    }
    if (streams.empty()) {
        throw std::runtime_error("no corpus stream reaches the requested long-context depth");
    }
    const double preflight_seconds = seconds_since(preflight_started);
    logger->info("corpus ready | {} streams | {} input tokens | {} scored tokens | {} windows | {}",
                 ninfer::product::format_pretty_count(streams.size()),
                 ninfer::product::format_pretty_count(total_input_tokens),
                 ninfer::product::format_pretty_count(total_scored_tokens),
                 ninfer::product::format_pretty_count(total_windows),
                 ninfer::product::format_pretty_duration(preflight_seconds));

    const std::filesystem::path output_directory = prepare_output_directory(options, load, corpus);
    const Clock::time_point scoring_started      = Clock::now();
    logger->info("scoring | {} streams | {} tokens | {} windows",
                 ninfer::product::format_pretty_count(streams.size()),
                 ninfer::product::format_pretty_count(total_scored_tokens),
                 ninfer::product::format_pretty_count(total_windows));
    Clock::time_point next_progress = scoring_started + std::chrono::seconds(10);
    ScoreAggregate overall;
    std::map<std::string, ScoreAggregate> domains;
    std::map<std::uint32_t, ScoreAggregate> depth_scores;
    std::map<std::uint32_t, std::uint32_t> depth_streams;
    json stream_reports             = json::array();
    std::uint64_t completed_windows = 0;

    for (std::size_t stream_index = 0; stream_index < streams.size(); ++stream_index) {
        EvaluationStream& stream = streams[stream_index];
        std::ostringstream stream_status;
        stream_status << "  scoring [" << stream_index + 1 << '/' << streams.size() << "] "
                      << ninfer::product::format_pretty_text(stream.source.id) << " | "
                      << ninfer::product::format_pretty_count(stream.tokens.size()) << " tokens | "
                      << ninfer::product::format_pretty_count(stream.windows.size()) << " windows";
        if (progress->enabled()) {
            progress->update(stream_status.str());
        } else {
            logger->debug("{}", stream_status.str());
        }
        const Clock::time_point stream_started = Clock::now();
        ScoreAggregate stream_score;
        json window_reports = json::array();
        for (std::size_t window_index = 0; window_index < stream.windows.size(); ++window_index) {
            const WindowPlan& window = stream.windows[window_index];
            std::vector<ninfer::TokenId> input(
                stream.tokens.begin() + static_cast<std::ptrdiff_t>(window.input_begin),
                stream.tokens.begin() + static_cast<std::ptrdiff_t>(window.input_end));
            const Clock::time_point window_started = Clock::now();
            std::vector<float> logprobs;
            try {
                logprobs = engine.score_tokens(std::move(input), window.first_target);
            } catch (const std::exception& error) {
                throw std::runtime_error("scoring " + stream.source.id + " window " +
                                         std::to_string(window_index) + " failed: " + error.what());
            }
            const std::size_t expected = window.target_end - window.target_begin;
            if (logprobs.size() != expected) {
                throw std::runtime_error("scoring returned an invalid target count for " +
                                         stream.source.id);
            }
            ScoreAggregate window_score;
            window_score.add(logprobs);
            stream_score.add(window_score);
            overall.add(window_score);
            domains[stream.source.domain].add(window_score);
            if (depth_mode) {
                const auto depth = static_cast<std::uint32_t>(window.target_begin);
                depth_scores[depth].add(window_score);
                ++depth_streams[depth];
            }
            ++completed_windows;
            json window_report            = aggregate_json(window_score);
            window_report["index"]        = window_index;
            window_report["input_begin"]  = window.input_begin;
            window_report["input_end"]    = window.input_end;
            window_report["target_begin"] = window.target_begin;
            window_report["target_end"]   = window.target_end;
            window_report["first_target"] = window.first_target;
            if (depth_mode) window_report["prefix_depth"] = window.target_begin;
            window_report["seconds"]      = seconds_since(window_started);
            window_reports.push_back(std::move(window_report));

            if (Clock::now() >= next_progress) {
                const double elapsed = seconds_since(scoring_started);
                const double rate    = static_cast<double>(overall.scored_tokens) / elapsed;
                const std::uint64_t remaining = total_scored_tokens - overall.scored_tokens;
                const double eta = rate > 0 ? static_cast<double>(remaining) / rate : 0.0;
                std::ostringstream line;
                line << "scoring | " << ninfer::product::format_pretty_count(overall.scored_tokens)
                     << '/' << ninfer::product::format_pretty_count(total_scored_tokens)
                     << " tokens | " << completed_windows << '/' << total_windows
                     << " windows | PPL " << std::fixed << std::setprecision(4) << overall.ppl()
                     << " | " << ninfer::product::format_pretty_rate(rate, "tok") << " | elapsed "
                     << ninfer::product::format_pretty_duration(elapsed) << " | ETA "
                     << ninfer::product::format_pretty_duration(eta);
                if (progress->enabled()) {
                    progress->update("  " + line.str());
                } else {
                    logger->info("{}", line.str());
                }
                next_progress = Clock::now() + std::chrono::seconds(10);
            }
        }
        const double stream_seconds = seconds_since(stream_started);
        progress->clear();
        logger->info("[{}/{}] {} | {} scored tokens | PPL {:.6g} | {}", stream_index + 1,
                     streams.size(), ninfer::product::format_pretty_text(stream.source.id),
                     ninfer::product::format_pretty_count(stream_score.scored_tokens),
                     stream_score.ppl(), ninfer::product::format_pretty_duration(stream_seconds));
        json stream_report               = aggregate_json(stream_score);
        stream_report["id"]              = stream.source.id;
        stream_report["domain"]          = stream.source.domain;
        stream_report["path"]            = stream.source.path.string();
        stream_report["input_tokens"]    = stream.tokens.size();
        if (!depth_mode) stream_report["unscored_tokens"] = 1;
        stream_report["seconds"] = stream_seconds;
        stream_report["windows"]         = std::move(window_reports);
        stream_reports.push_back(std::move(stream_report));
    }

    const double scoring_seconds = seconds_since(scoring_started);
    progress->clear();
    logger->info("scoring complete | {} tokens | {} windows | PPL {:.6g} | {} | {}",
                 ninfer::product::format_pretty_count(overall.scored_tokens), completed_windows,
                 overall.ppl(), ninfer::product::format_pretty_duration(scoring_seconds),
                 ninfer::product::format_pretty_rate(
                     static_cast<double>(overall.scored_tokens) / scoring_seconds, "tok"));
    json domain_reports = json::array();
    for (const auto& [domain, aggregate] : domains) {
        json item      = aggregate_json(aggregate);
        item["domain"] = domain;
        domain_reports.push_back(std::move(item));
    }
    json depth_reports = json::array();
    if (depth_mode) {
        for (const auto& [depth, aggregate] : depth_scores) {
            json item            = aggregate_json(aggregate);
            item["prefix_depth"] = depth;
            item["stream_count"] = depth_streams.at(depth);
            item["tail_tokens_requested"] = options.tail_tokens;
            depth_reports.push_back(std::move(item));
        }
    }
    json execution{
        {"purpose", "causal_scoring"},
        {"device", options.device},
        {"context_tokens", effective_context},
        {"prefill_chunk_tokens", 1024},
        {"score_tile_tokens", 1024},
        {"kv_dtype", kv_name(options.kv)},
        {"prefill_a8", options.prefill_a8},
        {"prefill_cublas", options.prefill_cublas},
        {"prefill_cublas_projections", options.prefill_cublas_projections},
    };
    if (depth_mode) {
        execution["protocol"] = "fixed-depth-long-history";
        execution["prefix_depths"] = options.depths;
        execution["tail_tokens"] = options.tail_tokens;
    } else {
        execution["protocol"] = "fixed-window-truncated-context";
        execution["stride_tokens"] = options.stride;
    }

    const ninfer::MemorySummary memory = engine.memory_summary();
    json report{
        {"schema_version", 3},
        {"metric",
         {{"name",
           depth_mode ? "fixed-depth long-history causal perplexity"
                      : "fixed-window truncated-context causal perplexity"},
          {"log_base", "natural"}}},
        {"artifact",
         {{"path", std::filesystem::absolute(options.artifact).lexically_normal().string()},
          {"architecture", load.architecture},
          {"name", load.model_name},
          {"prefill_signature", load.prefill_signature},
          {"formats", load.weight_formats}}},
        {"corpus",
         {{"id", corpus.corpus_id},
          {"mode", corpus.mode},
          {"source", corpus.source.string()},
          {"stream_count", streams.size()}}},
        {"execution", std::move(execution)},
        {"resources",
         {{"weights_capacity_bytes", memory.weights.capacity_bytes},
          {"sequence_capacity_bytes", memory.sequence.capacity_bytes},
          {"workspace_capacity_bytes", memory.workspace.capacity_bytes},
          {"kv_capacity_tokens", memory.kv_capacity},
          {"kv_payload_bytes", memory.kv_payload_bytes},
          {"runtime_reservation_bytes", memory.runtime_reservation_bytes},
          {"available_after_startup_bytes", memory.available_after_startup_bytes}}},
        {"timing",
         {{"load_seconds", load.load_seconds},
          {"read_and_tokenize_seconds", preflight_seconds},
          {"score_seconds", scoring_seconds},
          {"total_seconds", seconds_since(total_started)},
          {"scored_tokens_per_second",
           static_cast<double>(overall.scored_tokens) / scoring_seconds}}},
        {"streams", std::move(stream_reports)},
        {"domains", std::move(domain_reports)},
        {"depths", std::move(depth_reports)},
        {"overall", aggregate_json(overall)},
    };

    const std::filesystem::path temporary = output_directory / "report.json.tmp";
    const std::filesystem::path final     = output_directory / "report.json";
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) { throw std::runtime_error("cannot create report: " + temporary.string()); }
        output << std::setw(2) << report << '\n';
        output.flush();
        if (!output) { throw std::runtime_error("cannot write report: " + temporary.string()); }
    }
    std::filesystem::rename(temporary, final);

    std::cout << "Perplexity result\n"
              << "artifact: " << load.model_name << '\n'
              << "kv: " << kv_name(options.kv) << ", corpus: " << corpus.corpus_id << " / "
              << corpus.mode;
    if (depth_mode) {
        std::cout << ", long-history tail=" << options.tail_tokens << ", max-context="
                  << effective_context << "\n\n";
        std::cout << std::left << std::setw(16) << "prefix_depth" << std::right << std::setw(12)
                  << "streams" << std::setw(16) << "tokens" << std::setw(16) << "mean_nll"
                  << std::setw(16) << "ppl" << '\n';
        for (const auto& [depth, aggregate] : depth_scores) {
            std::cout << std::left << std::setw(16) << depth << std::right << std::setw(12)
                      << depth_streams.at(depth) << std::setw(16) << aggregate.scored_tokens
                      << std::setw(16) << std::fixed << std::setprecision(6)
                      << aggregate.mean_nll() << std::setw(16) << aggregate.ppl() << '\n';
        }
        std::cout << '\n';
    } else {
        std::cout << ", context/stride: " << options.context << '/' << options.stride << "\n\n";
    }
    std::cout << std::left << std::setw(24) << "domain" << std::right << std::setw(16) << "tokens"
              << std::setw(16) << "mean_nll" << std::setw(16) << "ppl" << '\n';
    for (const auto& [domain, aggregate] : domains) {
        std::cout << std::left << std::setw(24) << domain << std::right << std::setw(16)
                  << aggregate.scored_tokens << std::setw(16) << std::fixed << std::setprecision(6)
                  << aggregate.mean_nll() << std::setw(16) << aggregate.ppl() << '\n';
    }
    std::cout << std::left << std::setw(24) << "overall" << std::right << std::setw(16)
              << overall.scored_tokens << std::setw(16) << std::fixed << std::setprecision(6)
              << overall.mean_nll() << std::setw(16) << overall.ppl() << "\n\n"
              << "score rate: " << std::setprecision(1)
              << static_cast<double>(overall.scored_tokens) / scoring_seconds << " tok/s\n"
              << "report: " << final << '\n';
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    Options options;
    try {
        options = parse_options(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "ninfer-perplexity: " << error.what() << '\n';
        std::cerr << usage_text();
        return 1;
    }
    if (options.help_requested) {
        std::cout << usage_text();
        return 0;
    }

    ninfer::product::LoggingRuntime logging(
        {.logger_name  = "ninfer-perplexity",
         .level        = options.log_level,
         .presentation = ninfer::product::LogPresentation::Tool});
    const std::shared_ptr<spdlog::logger> logger = logging.logger();
    ninfer::product::StartupLogRenderer startup_log(logging);
    try {
        return run(options, logger, startup_log, logging.terminal_progress());
    } catch (const std::exception& error) {
        logging.terminal_progress()->clear();
        logger->error("{}", ninfer::product::format_pretty_text(error.what()));
        return 1;
    }
}
