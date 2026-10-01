#include "ninfer/engine.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <charconv>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace {

using json = nlohmann::json;

struct Options {
    std::filesystem::path model;
    std::filesystem::path input;
    std::filesystem::path output;
    std::uint32_t max_context = 8192;
    int device = 0;
    std::vector<int> devices;
    ninfer::KvCacheStorage kv = ninfer::KvCacheStorage::Int8Group64;
    bool gdn_state_fp16 = false;
    bool prefill_a8 = true;
    bool prefill_cublas = false;
    bool prefill_cublas_projections = true;
    ninfer::TeacherTraceOptions trace;
};

template <class Integer>
Integer parse_integer(std::string_view text, const char* label) {
    Integer value{};
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size()) {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + std::string(text));
    }
    return value;
}

std::vector<int> parse_devices(std::string_view text) {
    std::vector<int> out;
    while (!text.empty()) {
        const std::size_t comma = text.find(',');
        const std::string_view piece = text.substr(0, comma);
        out.push_back(parse_integer<int>(piece, "device"));
        if (comma == std::string_view::npos) break;
        text.remove_prefix(comma + 1);
    }
    if (out.empty() || out.size() > 2) {
        throw std::invalid_argument("--devices accepts one or two device ids");
    }
    return out;
}

std::vector<std::uint32_t> parse_layers(std::string_view text) {
    std::vector<std::uint32_t> out;
    while (!text.empty()) {
        const std::size_t comma = text.find(',');
        const std::string_view piece = text.substr(0, comma);
        out.push_back(parse_integer<std::uint32_t>(piece, "layer"));
        if (comma == std::string_view::npos) break;
        text.remove_prefix(comma + 1);
    }
    if (out.empty()) throw std::invalid_argument("--layers must not be empty");
    return out;
}

Options parse_options(int argc, char** argv) {
    if (argc < 2) {
        throw std::invalid_argument(
            "usage: ninfer-teacher MODEL.ninfer --input records.jsonl --out DIR "
            "[--max-context N] [--device N|--devices N,M] "
            "[--kv-dtype bf16|int8|fp8|rk8v4|rk4v4|rk4v4-e8|rk2v4-e8|nvfp4|k8v4] "
            "[--layers 5,19,33,47,61] [--top-k 16] "
            "[--gdn-state-fp16] [--no-prefill-a8] "
            "[--prefill-cublas] [--no-prefill-cublas-projections]");
    }
    Options out;
    out.model = argv[1];
    bool device_explicit = false;
    for (int i = 2; i < argc; ++i) {
        const std::string_view arg = argv[i];
        const auto value = [&](const char* label) -> std::string_view {
            if (++i >= argc) throw std::invalid_argument(std::string(label) + " needs a value");
            return argv[i];
        };
        if (arg == "--input") {
            out.input = std::string(value("--input"));
        } else if (arg == "--out") {
            out.output = std::string(value("--out"));
        } else if (arg == "--max-context") {
            out.max_context = parse_integer<std::uint32_t>(value("--max-context"), "max-context");
        } else if (arg == "--device") {
            out.device = parse_integer<int>(value("--device"), "device");
            device_explicit = true;
        } else if (arg == "--devices") {
            out.devices = parse_devices(value("--devices"));
        } else if (arg == "--layers") {
            out.trace.target_layer_ids = parse_layers(value("--layers"));
        } else if (arg == "--top-k") {
            out.trace.top_k = parse_integer<std::uint32_t>(value("--top-k"), "top-k");
        } else if (arg == "--kv-dtype") {
            const std::string_view dtype = value("--kv-dtype");
            if (dtype == "bf16") out.kv = ninfer::KvCacheStorage::BFloat16;
            else if (dtype == "int8") out.kv = ninfer::KvCacheStorage::Int8Group64;
            else if (dtype == "fp8") out.kv = ninfer::KvCacheStorage::Fp8E4M3Row256;
            else if (dtype == "rk8v4") out.kv = ninfer::KvCacheStorage::RotatedInt8KeyInt4ValueGroup64;
            else if (dtype == "rk4v4") out.kv = ninfer::KvCacheStorage::RotatedInt4KeyInt4ValueGroup64;
            else if (dtype == "rk4v4-e8") out.kv = ninfer::KvCacheStorage::RK4V4E8;
            else if (dtype == "rk2v4-e8") out.kv = ninfer::KvCacheStorage::RK2V4E8;
            else if (dtype == "nvfp4") out.kv = ninfer::KvCacheStorage::Nvfp4Group16;
            else if (dtype == "k8v4") out.kv = ninfer::KvCacheStorage::Fp8KeyNvfp4Value;
            else throw std::invalid_argument("invalid --kv-dtype");
        } else if (arg == "--gdn-state-fp16") {
            out.gdn_state_fp16 = true;
        } else if (arg == "--no-prefill-a8") {
            out.prefill_a8 = false;
        } else if (arg == "--prefill-cublas") {
            out.prefill_cublas = true;
        } else if (arg == "--no-prefill-cublas-projections") {
            out.prefill_cublas_projections = false;
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(arg));
        }
    }
    if (out.input.empty() || out.output.empty() || out.max_context < 2) {
        throw std::invalid_argument("--input, --out and --max-context>=2 are required");
    }
    if (!out.devices.empty() && device_explicit) {
        throw std::invalid_argument("--device and --devices are mutually exclusive");
    }
    if (out.trace.top_k == 0 || out.trace.top_k > 16) {
        throw std::invalid_argument("--top-k must be in [1,16]");
    }
    return out;
}

std::string safe_name(std::string value) {
    for (char& c : value) {
        const unsigned char u = static_cast<unsigned char>(c);
        if (!std::isalnum(u) && c != '-' && c != '_' && c != '.') c = '-';
    }
    if (value.empty()) value = "sample";
    return value;
}

template <class T>
void write_vector(const std::filesystem::path& path, const std::vector<T>& values) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot create " + path.string());
    if (!values.empty()) {
        out.write(reinterpret_cast<const char*>(values.data()),
                  static_cast<std::streamsize>(values.size() * sizeof(T)));
    }
    if (!out) throw std::runtime_error("cannot write " + path.string());
}

std::vector<ninfer::TokenId> record_tokens(const json& row, ninfer::Engine& engine) {
    if (row.contains("tokens")) {
        return row.at("tokens").get<std::vector<ninfer::TokenId>>();
    }
    if (!row.contains("text") || !row.at("text").is_string()) {
        throw std::invalid_argument("teacher JSONL row needs text or tokens");
    }
    return engine.tokenize_text(row.at("text").get<std::string>());
}

int run(const Options& options) {
    std::filesystem::create_directories(options.output);

    ninfer::EngineOptions engine_options;
    engine_options.artifact_path = options.model;
    engine_options.purpose = ninfer::EnginePurpose::CausalScoring;
    engine_options.enable_teacher_trace = true;
    engine_options.device = options.device;
    engine_options.devices = options.devices;
    engine_options.max_context = options.max_context;
    engine_options.kv_cache = options.kv;
    engine_options.gdn_state_fp16 = options.gdn_state_fp16;
    engine_options.prefill_a8 = options.prefill_a8;
    engine_options.prefill_cublas = options.prefill_cublas;
    engine_options.prefill_cublas_projections = options.prefill_cublas_projections;
    ninfer::Engine engine(std::move(engine_options));

    std::ifstream input(options.input);
    if (!input) throw std::runtime_error("cannot open " + options.input.string());

    json manifest{
        {"version", 1},
        {"format", "ninfer-native-teacher-v1"},
        {"artifact", std::filesystem::absolute(options.model).lexically_normal().string()},
        {"target_layer_ids", options.trace.target_layer_ids},
        {"top_k", options.trace.top_k},
        {"sequences", json::array()},
    };

    std::string line;
    std::uint64_t index = 0;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        json row = json::parse(line);
        if (row.is_string()) row = json{{"text", row.get<std::string>()}};
        if (!row.is_object()) throw std::invalid_argument("teacher JSONL row must be an object");

        std::vector<ninfer::TokenId> tokens = record_tokens(row, engine);
        if (tokens.size() < 2) {
            ++index;
            continue;
        }
        if (tokens.size() > options.max_context) {
            tokens.resize(options.max_context);
        }

        const std::string name =
            safe_name(row.value("name", "seq-" + std::to_string(index)));
        ninfer::TeacherTrace trace = engine.trace_tokens(tokens, options.trace);
        const std::string stem = name + "-" + std::to_string(index);

        write_vector(options.output / (stem + ".ids.i32"), trace.input_ids);
        write_vector(options.output / (stem + ".fused.bf16"), trace.fused_bf16);
        write_vector(options.output / (stem + ".argmax.i32"), trace.argmax);
        write_vector(options.output / (stem + ".top_ids.i32"), trace.top_ids);
        write_vector(options.output / (stem + ".top_lp.f32"), trace.top_logprobs);

        manifest["hidden_size"] = trace.hidden_size;
        manifest["sequences"].push_back({
            {"name", name},
            {"stem", stem},
            {"kind", row.value("kind", std::string("corpus"))},
            {"topic", row.value("topic", std::string("prose"))},
            {"split", row.value("split", std::string("train"))},
            {"rows", trace.input_ids.size()},
            {"input_tokens", tokens.size()},
        });
        std::cerr << '[' << (index + 1) << "] " << name << " | "
                  << trace.input_ids.size() << " predictor rows\n";
        ++index;
    }

    std::ofstream out(options.output / "manifest.json", std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot create teacher manifest");
    out << manifest.dump(2) << '\n';
    if (!out) throw std::runtime_error("cannot write teacher manifest");
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        return run(parse_options(argc, argv));
    } catch (const std::exception& error) {
        std::cerr << "ninfer-teacher: " << error.what() << '\n';
        return 1;
    }
}
