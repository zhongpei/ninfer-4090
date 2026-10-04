#include "calibrated_product_options_cases.h"
#include "options.h"

#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

ninfer::cli::Options parse(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (std::string& argument : arguments) { argv.push_back(argument.data()); }
    return ninfer::cli::parse_options(static_cast<int>(argv.size()), argv.data());
}

bool rejects(const std::function<void()>& operation) {
    try {
        operation();
    } catch (const std::invalid_argument&) { return true; }
    return false;
}

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

} // namespace

int main() {
    int failures = calibrated_product_options_cases(parse, {"ninfer-cli", "model.ninfer", "--prompt", "hello"}, ninfer::cli::usage_text("ninfer-cli"));
    const ninfer::cli::Options configured =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--thinking-budget", "37"});
    failures += check(configured.thinking_budget == 37,
                      "--thinking-budget did not preserve its positive value");
    failures +=
        check(ninfer::cli::usage_text("ninfer-cli").find("--thinking-budget") != std::string::npos,
              "CLI help omits --thinking-budget");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                       "--thinking-budget", "0"});
                      }),
                      "zero --thinking-budget was accepted");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                       "--thinking-budget", "8", "--no-thinking"});
                      }),
                      "--thinking-budget was accepted with --no-thinking");
    const ninfer::cli::Options with_effort =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--thinking-budget", "8",
               "--reasoning-effort", "medium"});
    failures += check(with_effort.thinking_budget == 8 && with_effort.reasoning_effort,
                      "thinking budget did not coexist with reasoning effort");
    const ninfer::cli::Options dflash_vision =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--vision", "--spec", "dflash",
               "--draft-tokens", "7"});
    failures += check(dflash_vision.enable_vision &&
                          dflash_vision.speculative.backend == ninfer::SpeculativeBackend::DFlash &&
                          dflash_vision.speculative.draft_tokens == 7,
                      "CLI did not preserve the combined DFlash and Vision startup features");
    for (const auto k : {1U, 2U, 7U, 15U}) {
        const auto dflash2 = parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--spec",
                                    "dflash2", "--draft-tokens", std::to_string(k)});
        failures += check(dflash2.speculative.backend == ninfer::SpeculativeBackend::DFlash2 &&
                              dflash2.speculative.draft_tokens == k,
                          "CLI did not preserve the DFlash2 draft count");
    }
    const auto stair = parse(
        {"ninfer-cli", "model.ninfer", "--prompt", "hello", "--spec", "dflash2",
         "--draft-tokens", "15", "--spec-router", "stair", "--spec-stair-widths", "3,7,11,15",
         "--spec-stair-costs", "1,1.02,1.05,1.10", "--spec-stair-draft-cost", "0.25",
         "--spec-stair-prior", "0.65", "--spec-stair-prior-weight", "3",
         "--spec-stair-warmup", "6", "--spec-stair-probe-period", "12",
         "--spec-stair-margin", "0.03"});
    failures += check(stair.speculative.routing.mode == ninfer::SpeculativeRoutingMode::Stair &&
                          stair.speculative.routing.widths[2] == 11 &&
                          stair.speculative.routing.verify_costs[3] == 1.10F &&
                          stair.speculative.routing.warmup_rounds == 6 &&
                          stair.speculative.routing.probe_period == 12,
                      "CLI did not preserve Stair router A/B parameters");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                       "--spec", "mtp", "--draft-tokens", "3",
                                       "--spec-router", "stair",
                                       "--spec-stair-widths", "1,2,3,3"});
                      }),
                      "CLI accepted Stair routing on MTP or invalid width ordering");
    const auto lookup = parse(
        {"ninfer-cli", "model.ninfer", "--prompt", "hello", "--spec", "dflash2",
         "--draft-tokens", "15", "--lookup-ngram", "8", "--lookup-strategy", "vote",
         "--lookup-dflash", "skip", "--lookup-max-order", "12", "--lookup-max-matches", "48",
         "--lookup-min-support", "2", "--lookup-min-confidence", "0.7",
         "--lookup-base-drafts", "7", "--lookup-deep-after", "2", "--lookup-deep-drafts", "15",
         "--lookup-persistent-tokens", "262144", "--lookup-persistent-path", "state/lookup.bin",
         "--lookup-corpus-prefix", "corpus/qwen",
         "--lookup-corpus-weight", "0.4", "--lookup-corpus-samples", "32"});
    failures += check(lookup.speculative.lookup.strategy == ninfer::LookupDraftStrategy::Vote &&
                          lookup.speculative.lookup.dflash_mode == ninfer::LookupDFlashMode::HeadSkip &&
                          lookup.speculative.lookup.max_order == 12 &&
                          lookup.speculative.lookup.persistent_tokens == 262144 &&
                          lookup.speculative.lookup.persistent_path == "state/lookup.bin" &&
                          lookup.speculative.lookup.corpus_prefix == "corpus/qwen",
                      "CLI did not preserve multi-source lookup controls");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                       "--spec", "dflash2", "--draft-tokens", "15",
                                       "--lookup-ngram", "8", "--lookup-dflash", "skip"});
                      }),
                      "CLI accepted DFlash lookup takeover without vote strategy");
    const auto recent16 =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--spec", "mtp",
               "--draft-tokens", "3", "--lookup-ngram", "16"});
    failures += check(recent16.speculative.lookup_ngram == 16 &&
                          recent16.speculative.lookup.strategy == ninfer::LookupDraftStrategy::Recent,
                      "historical recent --lookup-ngram 16 no longer parses");
    const auto clamped_deep =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--spec", "dflash2",
               "--draft-tokens", "7", "--lookup-ngram", "5", "--lookup-strategy", "vote",
               "--lookup-dflash", "skip", "--lookup-deep-drafts", "15"});
    failures += check(clamped_deep.speculative.lookup.deep_drafts == 15,
                      "lookup deep policy could not exceed startup K for runtime clamping");
    for (const auto k : {0U, 16U}) {
        failures +=
            check(rejects([&] {
                      (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--spec",
                                   "dflash2", "--draft-tokens", std::to_string(k)});
                  }),
                  "CLI accepted an unsupported DFlash2 draft count");
    }
    const ninfer::cli::Options nvfp4 =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--kv-dtype", "nvfp4"});
    failures += check(nvfp4.kv_cache == ninfer::KvCacheStorage::Nvfp4Group16,
                      "--kv-dtype nvfp4 did not select group-16 NVFP4 KV");
    const ninfer::cli::Options k8v4 =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--kv-dtype", "k8v4"});
    failures += check(k8v4.kv_cache == ninfer::KvCacheStorage::Fp8KeyNvfp4Value,
                      "--kv-dtype k8v4 did not select asymmetric K8V4 KV");
    const std::string help = ninfer::cli::usage_text("ninfer-cli");
    failures +=
        check(help.find("nvfp4") != std::string::npos && help.find("k8v4") != std::string::npos,
              "CLI help omits a production KV storage mode");
    const ninfer::cli::Options route_defaults =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello"});
    failures += check(!route_defaults.prefill_cublas && route_defaults.prefill_cublas_projections &&
                          route_defaults.speculative.lookup_ngram == 0,
                      "the cuBLAS prefill route or context lookup is on by default");
    failures += check(route_defaults.speculative.routing.scope == ninfer::SpeculativeRouterScope::Request,
                      "CLI router state is shared across requests by default");
    const auto persisted_router = parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                        "--spec-router-scope", "engine", "--spec-router-state", "state/router.bin"});
    failures += check(persisted_router.speculative.routing.scope == ninfer::SpeculativeRouterScope::Engine &&
                          persisted_router.speculative.routing.state_path == "state/router.bin",
                      "CLI did not preserve explicit engine router persistence");
    failures += check(rejects([] {
                          parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                 "--spec-router-state", "state/router.bin"});
                      }), "CLI accepted router persistence with request scope");
    for (const char* removed : {"--spec-stair-profile", "--lookup-history-path"}) {
        failures += check(rejects([&] {
                              parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", removed, "state.bin"});
                          }), "CLI accepted a superseded persistence interface");
    }
    const ninfer::cli::Options route =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--spec", "mtp", "--draft-tokens",
               "3", "--lookup-ngram", "5", "--prefill-cublas", "--no-prefill-cublas-projections"});
    failures += check(route.prefill_cublas && !route.prefill_cublas_projections &&
                          route.speculative.lookup_ngram == 5 &&
                          route.speculative.backend == ninfer::SpeculativeBackend::Mtp &&
                          route.speculative.draft_tokens == 3,
                      "CLI did not parse the cuBLAS prefill and context-lookup controls");
    for (const char* flag : {"--prefill-cublas", "--no-prefill-cublas-projections", "--lookup-ngram",
                             "--spec-router", "--spec-stair-widths", "--spec-stair-costs",
                             "--lookup-strategy", "--lookup-dflash", "--lookup-persistent-path",
                             "--lookup-corpus-prefix"}) {
        failures += check(help.find(flag) != std::string::npos,
                          "CLI help omits an accepted prefill or drafting control");
    }
    const ninfer::cli::Options logging =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--log-level", "debug"});
    failures += check(logging.log_level == ninfer::product::LogLevel::Debug,
                      "CLI log level was not parsed");
    failures += check(help.find("--log-level") != std::string::npos,
                      "CLI help omits the log-level control");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                       "--log-level", "verbose"});
                      }),
                      "CLI accepted an unknown log level");
    failures +=
        check(rejects([] {
                  (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--top-k", "21"});
              }),
              "CLI accepted top_k beyond the executable candidate domain");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--device",
                                       "1", "--devices", "0,1"});
                      }),
                      "--device and --devices were accepted together");
    failures +=
        check(rejects([] {
                  (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--devices",
                               "4294967296"});
              }),
              "--devices silently narrowed an out-of-range id instead of rejecting it");
    const ninfer::cli::Options same_card = parse(
        {"ninfer-cli", "model.ninfer", "--prompt", "hello", "--devices", "0,0"});
    failures += check(same_card.devices.size() == 2 && same_card.devices[0] == 0 &&
                           same_card.devices[1] == 0,
                      "--devices 0,0 was not accepted for single-card split coverage");
    return failures == 0 ? 0 : 1;
}
