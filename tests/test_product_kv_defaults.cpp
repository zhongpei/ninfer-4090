#include "ninfer/types.h"
#include "options.h"
#include "ninfer_bench_support.h"
#include "serve/generation_service.h"
#include "serve/serve_options.h"

#include <iostream>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

template <class Parser>
auto parse(Parser parser, std::vector<std::string> args) {
    std::vector<char*> argv;
    for (auto& arg : args) argv.push_back(arg.data());
    return parser(static_cast<int>(argv.size()), argv.data());
}

void check_entry_points(const char* explicit_kv, ninfer::KvCacheStorage expected) {
    std::vector<std::string> cli_args{"ninfer", "model.ninfer", "--prompt", "hello"};
    std::vector<std::string> serve_args{"ninfer-serve", "model.ninfer"};
    std::vector<std::string> bench_args{"ninfer-bench", "--weights", "model.ninfer"};
    if (explicit_kv != nullptr) {
        for (auto* args : {&cli_args, &serve_args, &bench_args}) {
            args->insert(args->end(), {"--kv-dtype", explicit_kv});
        }
    }
    const auto cli = parse(ninfer::cli::parse_options, std::move(cli_args));
    const auto serve = parse(ninfer::serve::parse_serve_options, std::move(serve_args));
    const auto bench = parse(ninfer::bench::parse_args, std::move(bench_args));
    check(cli.kv_cache == expected, "CLI KV default or explicit selection");
    check(serve.kv_cache == expected, "serve KV default or explicit selection");
    check(bench.kv_cache == expected, "benchmark KV default or explicit selection");
    if (explicit_kv != nullptr && std::string(explicit_kv) == "rk4v4-e8") {
        check(ninfer::bench::kv_cache_name(bench.kv_cache) == "rk4v4-e8",
              "benchmark reports selected RK4V4-E8 storage");
    }
    check(ninfer::serve::make_engine_options(serve).kv_cache == expected,
          "serve KV selection reaches Engine options");
}

} // namespace

int main() try {
#if defined(NINFER_SM89)
    constexpr auto expected = ninfer::KvCacheStorage::Int8Group64;
#else
    constexpr auto expected = ninfer::KvCacheStorage::BFloat16;
#endif
    check(ninfer::EngineOptions{}.kv_cache == expected, "public Engine KV default");
    check_entry_points(nullptr, expected);
    check_entry_points("bf16", ninfer::KvCacheStorage::BFloat16);
    check_entry_points("fp8", ninfer::KvCacheStorage::Fp8E4M3Row256);
    check_entry_points("rk4v4-e8", ninfer::KvCacheStorage::RK4V4E8);
    const std::string expected_help = expected == ninfer::KvCacheStorage::Int8Group64
                                          ? "KV cache storage (default: int8)"
                                          : "KV cache storage (default: bf16)";
    check(ninfer::bench::usage_text("ninfer-bench").find(expected_help) != std::string::npos,
          "benchmark help reports the product KV default");
    std::cout << (failures ? "FAIL" : "OK") << " product_kv_defaults\n";
    return failures ? 1 : 0;
} catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
}
