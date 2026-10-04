// Standalone CPU serializer check, not a CUDA/runtime/model test.
#include "apps/cli/ab_metrics.h"
#include <cstdint>
#include <vector>
#include <string_view>
struct Stats {
    std::uint64_t rounds=0, drafted_tokens=0, accepted_tokens=0;
    std::uint64_t tree_rounds=0, tree_fallback_rounds=0, tree_nodes=0, tree_accepted_drafts=0;
    std::uint64_t lookup_queries=0, lookup_hits=0, lookup_rounds=0;
    std::uint64_t lookup_drafted_tokens=0, lookup_accepted_tokens=0, lookup_head_skip_rounds=0;
};
struct Result {
    std::vector<int> generated_token_ids{10,20,30};
    struct {int prompt_tokens=20;} prompt;
    struct {double decode_seconds=0.00123456789, prefill_seconds=0.01;} timings;
    int finish_reason=1;
    Stats speculative;
};
struct Sampling {double temperature=0, presence_penalty=0, frequency_penalty=0;};
struct Memory {
    struct {std::uint64_t peak_used_bytes=1024;} workspace;
    std::uint64_t runtime_reservation_bytes=2048;
};
struct Runtime {
    std::uint64_t calibrated_target_only_rounds=0, calibrated_k7_rounds=0;
    std::uint64_t calibrated_k11_rounds=0, calibrated_k15_rounds=0;
    std::uint64_t calibrated_route_switches=0, calibrated_fixed_fallback_rounds=0;
};
int main(int argc, char** argv) {
    Runtime runtime{11,12,13,14,15,16};
    if (argc > 2 && std::string_view(argv[2]) == "zero") { runtime = Runtime{}; }
    const bool calibrated = argc > 1 && std::string_view(argv[1]) == "calibrated";
    ninfer::cli::emit_calibrated_snapshot_metrics(runtime, calibrated,
        [](const char* key, const std::string& value) {
            std::cerr << key << '=' << value << '\n';
        });
    std::cout << "NINFER_METRICS_JSON "
              << ninfer::cli::ab_metrics_record(Result{},Sampling{},Memory{},runtime) << '\n';
}
