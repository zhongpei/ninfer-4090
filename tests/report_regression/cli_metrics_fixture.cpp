// Standalone CPU serializer check, not a CUDA/runtime/model test.
#include "apps/cli/ab_metrics.h"
#include <cstdint>
#include <vector>
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
int main() {
    std::cout << "NINFER_METRICS_JSON "
              << ninfer::cli::ab_metrics_record(Result{},Sampling{},Memory{}) << '\n';
}
