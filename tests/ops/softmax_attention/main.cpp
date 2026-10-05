#include "ops/softmax_attention/dense/causal_cache/launch.h"

#include <exception>
#include <initializer_list>
#include <iostream>
#include <string_view>

int run_softmax_attention_causal_cache_tests();
int run_softmax_attention_width_consistency_tests(const char* family = nullptr);
int run_softmax_attention_represented_input_tests(const char* directory);
int run_softmax_attention_representative_range_tests();
int run_softmax_attention_dflash2_tests();
int run_softmax_attention_nvfp4_tests();
int run_softmax_attention_k8v4_tests();
int run_softmax_attention_plain_and_packed_tests();
int run_softmax_attention_context_tests();

namespace {

// Without this, any throw out of a suite leaves MSVC to terminate the process with
// STATUS_STACK_BUFFER_OVERRUN (0xC0000409) and no output at all -- which reads exactly like memory
// corruption in a CUDA kernel and is not. That misdiagnosis cost real time on the DFlash2 sweep:
// the actual fault was a std::vector index out of range in the host-side reference, and it took
// compute-sanitizer reporting zero device errors to rule the GPU out.
template <typename Suite>
int run_guarded(const char* what, Suite suite) {
    try {
        return suite();
    } catch (const std::exception& error) {
        std::cerr << "softmax_attention: " << what << " threw: " << error.what() << '\n';
        return 1;
    } catch (...) {
        std::cerr << "softmax_attention: " << what << " threw a non-std exception\n";
        return 1;
    }
}

// Host-only policy regression: no device allocations or kernels. This is not a
// substitute for a GPU row-reorder / full-model exactness test.
int run_split_capacity_batch_invariance_tests() {
    using ninfer::KvCacheStorage;
    using ninfer::ops::CausalAttentionExecutionEnvelope;
    using ninfer::ops::detail::causal_attention_split_capacity;
    constexpr KvCacheStorage formats[] = {
        KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64,
        KvCacheStorage::Fp8E4M3Row256, KvCacheStorage::RotatedInt8KeyInt4ValueGroup64,
    };
    constexpr CausalAttentionExecutionEnvelope envelopes[] = {
        {1, 128}, {129, 160}, {1, 512}, {1, 4096}, {4097, 5000},
        {5001, 8198}, {1, 8198}, {8199, 16390}, {16391, 32768},
        {1, 32768}, {1, 262144}, {1, 786432},
    };
    int failures = 0;
    for (const int heads : {24, 16}) {
        for (const auto format : formats) {
            for (const auto envelope : envelopes) {
                for (int width = 1; width <= (heads == 24 ? 8 : 6); ++width) {
                    const int single = causal_attention_split_capacity(
                        heads, width, format, envelope, 1);
                    for (int batch = 2; batch <= 8; ++batch) {
                        const int batched = causal_attention_split_capacity(
                            heads, width, format, envelope, batch);
                        if (single != batched) {
                            ++failures;
                            std::cerr << "split partition depends on batch: heads=" << heads
                                      << " width=" << width << " batch=" << batch
                                      << " keys=" << envelope.max_visible_keys
                                      << " single=" << single << " batched=" << batched << '\n';
                        }
                    }
                }
            }
        }
    }
#if defined(NINFER_SM89)
    struct Sm89Expected {
        int heads;
        int width;
        std::uint32_t visible;
        int expected;
    };
    constexpr Sm89Expected sm89_cases[]{
        {24, 1, 5631, 30}, {24, 5, 5631, 30}, {24, 6, 5631, 30},
        {24, 1, 8198, 32}, {24, 6, 8198, 32},
        {16, 1, 5631, 59}, {16, 5, 5631, 59}, {16, 6, 5631, 59},
        {16, 1, 8198, 86}, {16, 6, 8198, 86},
        // H24 width 7 is outside the sm89 one-wave specialization and keeps the
        // generic partitioning. This protects the exact T1..T6 scope.
        {24, 7, 5631, 64},
    };
    for (const auto& test : sm89_cases) {
        const int actual = causal_attention_split_capacity(
            test.heads, test.width, KvCacheStorage::Int8Group64,
            CausalAttentionExecutionEnvelope{1, test.visible}, 1);
        if (actual != test.expected) {
            ++failures;
            std::cerr << "sm89 split policy mismatch: heads=" << test.heads
                      << " width=" << test.width << " keys=" << test.visible
                      << " expected=" << test.expected << " actual=" << actual << '\n';
        }
    }
#endif
    std::cout << "split-capacity batch invariance: " << (failures ? "FAIL\n" : "PASS\n");
    return failures ? 1 : 0;
}

} // namespace

int main(int argc, char** argv) {
    const int split_policy = run_guarded("split capacity", run_split_capacity_batch_invariance_tests);
    if (split_policy != 0) return split_policy;
    if (argc == 2 && std::string_view(argv[1]) == "--split-capacity-only") return 0;
    if ((argc == 2 || argc == 3) && std::string_view(argv[1]) == "--width-consistency-only")
        return run_guarded("width consistency", [&] {
            return run_softmax_attention_width_consistency_tests(argc == 3 ? argv[2] : nullptr);
        });
    if (argc == 2 && std::string_view(argv[1]) == "--representative-range")
        return run_guarded("representative range", run_softmax_attention_representative_range_tests);
    if (argc == 3 && std::string_view(argv[1]) == "--represented-input")
        return run_guarded("represented input", [&] {
            return run_softmax_attention_represented_input_tests(argv[2]);
        });
    if (argc == 2 && std::string_view(argv[1]) == "--dflash2-only")
        return run_guarded("dflash2", run_softmax_attention_dflash2_tests);
    if (argc == 2 && std::string_view(argv[1]) == "--nvfp4-only") {
        return run_guarded("nvfp4", run_softmax_attention_nvfp4_tests);
    }
    if (argc == 2 && std::string_view(argv[1]) == "--k8v4-only") {
        return run_guarded("k8v4", run_softmax_attention_k8v4_tests);
    }
    if (argc != 1) {
        std::cerr
            << "usage: ninfer_softmax_attention_test [--split-capacity-only|--width-consistency-only [all|int8|rk8v4|packed|rk2v4-e8]|--represented-input DIR|--representative-range|--dflash2-only|--nvfp4-only|--k8v4-only]\n";
        return 2;
    }
    const int causal = run_guarded("causal cache", run_softmax_attention_causal_cache_tests);
    if (causal == 77) return 77;

    const int plain_and_packed = run_guarded("plain/packed", run_softmax_attention_plain_and_packed_tests);
    if (plain_and_packed == 77) return 77;

    const int context = run_guarded("context", run_softmax_attention_context_tests);
    if (context == 77) return 77;

    const int failures = causal + plain_and_packed + context;
    std::cout << (failures == 0 ? "softmax_attention: PASS\n" : "softmax_attention: FAIL\n");
    return failures == 0 ? 0 : 1;
}
