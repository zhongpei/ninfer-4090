#include "ninfer/ops/gated_delta_net.h"

#include "ops/gdn_ref.h"
#include "ops/quantized_weight.h"
#include "ops/op_tester.h"
#include "core/decode_graph.h"
#include "core/device.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <memory>
#include <array>
#include <type_traits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <optional>
#include <utility>
#include <cstdlib>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int kStateDim = 128;

constexpr ReductionCriterion gated_delta_net_output_bf16_criterion() {
    return {/*relative_l2=*/4.1e-3, /*gross_absolute=*/5.0e-6,
            /*gross_relative_to_max_reference=*/kBf16GrossRelativeFloor};
}

// The state output is FP32, so #20's BF16 floor was left off it on the grounds that dtype rounding
// of the *output* cannot be its floor. That is true and beside the point: measured, the error's
// source is BF16 anyway, and the floor does reach it.
//
// NINFER_OP_REPORT_STATS=1 over the whole matrix splits cleanly in two:
//
//     path                                  max_abs    max_reference   steps
//     decode / small-T / batch update      1.2e-8 .. 3.5e-8   ~0.09     0.00
//     exact chunk / chunk-tail / two-chunk 4.6e-4 .. 8.1e-4   ~0.23     0.53-0.88
//
// The non-chunked paths are exact to eight decimal places -- there is no accumulation to speak of.
// Everything above 1e-4 comes from the chunked recurrence, where the carried state crosses a BF16
// intermediate at each chunk boundary. Slightly under one BF16 rounding step of the state's own
// magnitude is exactly what one such round-trip costs, so this *is* the BF16 argument, arriving
// through the carried state rather than through the output dtype.
//
// That makes the previous 3.9e-3 the same mistake #20 was written to fix. It is about 1.0 rounding
// step, and the observed worst case is 0.88 of one -- 12% headroom, and the bound and the error
// are the same quantity. Use `kBf16GrossRelativeFloor` (two steps) like every other criterion the
// audit touched, which puts the worst observed case at 0.44 of its limit.
//
// `relative_l2` is untouched at 2.7e-3 against a measured 2.582e-3. It sits at 0.96, which is
// tight, but it is the criterion that constrains kernel accuracy and #20's convention leaves it
// alone deliberately -- loosening it would stop the chunked recurrence being checked at all.
constexpr ReductionCriterion gated_delta_net_state_fp32_criterion() {
    return {/*relative_l2=*/2.7e-3, /*gross_absolute=*/1.0e-5,
            /*gross_relative_to_max_reference=*/kBf16GrossRelativeFloor};
}

struct Case {
    const char* name;
    int qk_heads;
    int value_heads;
    int tokens;
    bool normalize_qk;
    bool near_zero_qk = false;
};

void fill_uniform(std::vector<float>& values, std::mt19937& generator, float low, float high) {
    std::uniform_real_distribution<float> distribution(low, high);
    for (float& value : values) { value = distribution(generator); }
}

void normalize_rows(std::vector<float>& values, int width) {
    const std::size_t rows = values.size() / static_cast<std::size_t>(width);
    for (std::size_t row = 0; row < rows; ++row) {
        float* base  = values.data() + row * static_cast<std::size_t>(width);
        double sumsq = 0.0;
        for (int d = 0; d < width; ++d) {
            const double value = static_cast<double>(base[d]);
            sumsq += value * value;
        }
        const double inv = 1.0 / std::sqrt(sumsq);
        for (int d = 0; d < width; ++d) {
            base[d] = static_cast<float>(static_cast<double>(base[d]) * inv);
        }
    }
}

gdn_ref::Inputs make_inputs(const Case& test_case, std::uint32_t seed) {
    gdn_ref::Inputs in;
    in.head_dim    = kStateDim;
    in.qk_heads    = test_case.qk_heads;
    in.value_heads = test_case.value_heads;
    in.tokens      = test_case.tokens;

    const std::size_t qk_size =
        static_cast<std::size_t>(kStateDim * test_case.qk_heads * test_case.tokens);
    const std::size_t value_size =
        static_cast<std::size_t>(kStateDim * test_case.value_heads * test_case.tokens);
    const std::size_t state_size =
        static_cast<std::size_t>(kStateDim * kStateDim * test_case.value_heads);
    in.q.resize(qk_size);
    in.k.resize(qk_size);
    in.v.resize(value_size);
    in.g.resize(static_cast<std::size_t>(test_case.value_heads * test_case.tokens));
    in.beta.resize(static_cast<std::size_t>(test_case.value_heads * test_case.tokens));
    in.state.resize(state_size);

    std::mt19937 generator(seed);
    fill_uniform(in.q, generator, -1.0f, 1.0f);
    fill_uniform(in.k, generator, -1.0f, 1.0f);
    fill_uniform(in.v, generator, -0.5f, 0.5f);
    fill_uniform(in.g, generator, -0.10f, -0.005f);
    fill_uniform(in.beta, generator, 0.05f, 0.95f);
    fill_uniform(in.state, generator, -0.02f, 0.02f);

    if (test_case.near_zero_qk) {
        for (float& value : in.q) { value *= 1.0e-4f; }
        for (float& value : in.k) { value *= 1.0e-4f; }
    } else if (!test_case.normalize_qk) {
        // Raw-Q/K mode still receives a stable, entirely valid public input. This host-side
        // generation choice is not part of the oracle.
        normalize_rows(in.q, kStateDim);
        normalize_rows(in.k, kStateDim);
    }

    round_to_bf16(in.q);
    round_to_bf16(in.k);
    round_to_bf16(in.v);
    return in;
}

std::vector<std::uint16_t> bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { bits[i] = f32_to_bf16(values[i]); }
    return bits;
}

std::vector<double> doubles(const std::vector<float>& values) {
    return std::vector<double>(values.begin(), values.end());
}

template <typename T>
int verify_exact(const std::string& label, const std::vector<T>& got,
                 const std::vector<T>& expected) {
    return ninfer::test::verify_exact(label.c_str(), got, expected);
}

int verify_recurrence(const std::string& label, const std::vector<double>& got,
                      const std::vector<double>& expected, const ReductionCriterion& criterion) {
    return verify_reduction(label.c_str(), got, expected, criterion);
}

std::vector<double> read_f32(const void* device, std::size_t count) {
    return doubles(from_device<float>(device, count));
}

int verify_common_inputs_unchanged(const std::string& label, const gdn_ref::Inputs& in,
                                   const DeviceBuffer& q, const DeviceBuffer& k,
                                   const DeviceBuffer& v, const DeviceBuffer& g,
                                   const DeviceBuffer& beta) {
    int failures = 0;
    failures += verify_exact(label + " q unchanged", from_device<std::uint16_t>(q, in.q.size()),
                             bf16_bits(in.q));
    failures += verify_exact(label + " k unchanged", from_device<std::uint16_t>(k, in.k.size()),
                             bf16_bits(in.k));
    failures += verify_exact(label + " v unchanged", from_device<std::uint16_t>(v, in.v.size()),
                             bf16_bits(in.v));
    failures += verify_exact(label + " g unchanged", from_device<float>(g, in.g.size()), in.g);
    failures +=
        verify_exact(label + " beta unchanged", from_device<float>(beta, in.beta.size()), in.beta);
    return failures;
}

struct DeviceInputs {
    explicit DeviceInputs(const gdn_ref::Inputs& in)
        : q(to_device_bf16(in.q)), k(to_device_bf16(in.k)), v(to_device_bf16(in.v)),
          g(to_device_f32(in.g)), beta(to_device_f32(in.beta)) {}

    DeviceBuffer q;
    DeviceBuffer k;
    DeviceBuffer v;
    DeviceBuffer g;
    DeviceBuffer beta;
};

struct CausalRun {
    std::vector<std::uint16_t> out;
    std::vector<float> state;
    int failures = 0;
};

gdn_ref::Inputs token_range(const gdn_ref::Inputs& in, int begin, int count) {
    gdn_ref::Inputs part = in;
    part.tokens = count;
    const auto range = [begin, count](auto& values, int stride) {
        values = std::vector<float>(values.begin() + std::size_t(begin) * stride,
                                    values.begin() + std::size_t(begin + count) * stride);
    };
    range(part.q, kStateDim * in.qk_heads);
    range(part.k, kStateDim * in.qk_heads);
    range(part.v, kStateDim * in.value_heads);
    range(part.g, in.value_heads);
    range(part.beta, in.value_heads);
    return part;
}

CausalRun causal_run(const gdn_ref::Inputs& in, bool normalize, bool graph_replay,
                    int future_begin = 0, int qk_offset = 0, int value_offset = 0) {
    DeviceInputs device(in);
    GuardedDeviceBuffer source(in.state.size() * sizeof(float));
    GuardedDeviceBuffer state(in.state.size() * sizeof(float));
    GuardedDeviceBuffer output(in.v.size() * sizeof(std::uint16_t));
    source.copy_from_host(in.state.data(), source.bytes());
    std::array<std::unique_ptr<GuardedDeviceBuffer>, 3> shifted;
    std::array<std::vector<std::uint8_t>, 3> stored;
    std::array<void*, 3> pointers{device.q.p, device.k.p, device.v.p};
    const std::array<const std::vector<float>*, 3> values{&in.q, &in.k, &in.v};
    const std::array<int, 3> offsets{qk_offset, qk_offset, value_offset};
    for (int i = 0; i < 3; ++i) {
        if (!offsets[i]) { continue; }
        const auto bits = bf16_bits(*values[i]);
        stored[i].resize(bits.size() * 2 + offsets[i], 0x37);
        std::memcpy(stored[i].data() + offsets[i], bits.data(), bits.size() * 2);
        shifted[i] = std::make_unique<GuardedDeviceBuffer>(stored[i].size());
        shifted[i]->copy_from_host(stored[i].data(), stored[i].size());
        pointers[i] = static_cast<std::uint8_t*>(shifted[i]->data()) + offsets[i];
    }
    const int heads = in.value_heads;
    const int qheads = in.qk_heads;
    Tensor q(pointers[0], DType::BF16, {kStateDim, qheads, int(in.tokens)});
    Tensor k(pointers[1], DType::BF16, {kStateDim, qheads, int(in.tokens)});
    Tensor v(pointers[2], DType::BF16, {kStateDim, heads, int(in.tokens)});
    Tensor g(device.g.p, DType::FP32, {heads, int(in.tokens)});
    Tensor beta(device.beta.p, DType::FP32, {heads, int(in.tokens)});
    Tensor s0(source.data(), DType::FP32, {kStateDim, kStateDim, heads});
    Tensor s1(state.data(), DType::FP32, {kStateDim, kStateDim, heads});
    Tensor out(output.data(), DType::BF16, {kStateDim, heads, int(in.tokens)});
    const float scale = 1.0f / std::sqrt(float(kStateDim));
    const auto bytes = ops::gated_delta_net_workspace_capacity_bytes(
        qheads, heads, normalize, in.tokens, in.tokens);
    WorkspaceArena workspace(std::max<std::size_t>(bytes, 256));
    DeviceContext context;
    const auto launch = [&] {
        ops::gated_delta_net(q, k, v, g, beta, scale, normalize, workspace, s0, s1, out,
                             context.stream);
    };
    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    if (graph_replay) {
        definition.capture(context.stream, launch);
        graph.instantiate(definition);
        // Fill the live inputs after capture, rather than accepting stale captured operands.
        const auto bits = bf16_bits(in.v);
        cuda_check(cudaMemcpy(pointers[2], bits.data(), bits.size() * sizeof(std::uint16_t),
                              cudaMemcpyHostToDevice), "GDN live value upload");
        graph.launch(context.stream);
    } else {
        launch();
    }
    cuda_check(cudaStreamSynchronize(context.stream), "GDN causal run");
    CausalRun result{from_device<std::uint16_t>(output.data(), in.v.size()),
                     from_device<float>(state.data(), in.state.size()), 0};
    if (graph_replay && future_begin > 0 && future_begin < in.tokens) {
        auto future = in.v;
        for (std::size_t i = std::size_t(future_begin) * heads * kStateDim; i < future.size(); ++i) {
            future[i] = -future[i];
        }
        const auto bits = bf16_bits(future);
        cuda_check(cudaMemcpy(pointers[2], bits.data(), bits.size() * sizeof(std::uint16_t),
                              cudaMemcpyHostToDevice), "GDN live value upload");
        graph.launch(context.stream);
        cuda_check(cudaStreamSynchronize(context.stream), "GDN live future input");
        const auto replay = from_device<std::uint16_t>(output.data(), in.v.size());
        if (!std::equal(result.out.begin(), result.out.begin() +
                        std::size_t(future_begin) * heads * kStateDim, replay.begin())) {
            std::cerr << "GDN future values changed a committed output prefix\n";
            ++result.failures;
        }
        const auto original = bf16_bits(in.v);
        cuda_check(cudaMemcpy(pointers[2], original.data(), original.size() * sizeof(std::uint16_t),
                              cudaMemcpyHostToDevice), "GDN restore live value");
    }
    result.failures += verify_exact("GDN causal source unchanged",
                                    from_device<float>(source.data(), in.state.size()), in.state);
    result.failures += verify_common_inputs_unchanged("GDN causal", in, device.q, device.k,
                                                     device.v, device.g, device.beta);
    result.failures += source.verify_guards("GDN causal source guards");
    result.failures += state.verify_guards("GDN causal state guards");
    result.failures += output.verify_guards("GDN causal output guards");
    for (int i = 0; i < 3; ++i) {
        if (!shifted[i]) { continue; }
        result.failures += verify_exact("GDN shifted operand immutable",
            from_device<std::uint8_t>(shifted[i]->data(), stored[i].size()), stored[i]);
        result.failures += shifted[i]->verify_guards("GDN shifted operand guards");
    }
    return result;
}

int causal_pair(const gdn_ref::Inputs& in, int prefix, bool normalize, bool sampled_oracle = false) {
    const auto short_input = token_range(in, 0, prefix);
    const auto short_run = causal_run(short_input, normalize, false);
    const auto full_run = causal_run(in, normalize, true, prefix);
    auto tail_input = token_range(in, prefix, in.tokens - prefix);
    tail_input.state = short_run.state; // Exact public FP32 state is the represented continuation input.
    const auto tail_run = causal_run(tail_input, normalize, true);
    int failures = short_run.failures + full_run.failures + tail_run.failures;
    const std::size_t cut = short_run.out.size();
    if (!std::equal(short_run.out.begin(), short_run.out.end(), full_run.out.begin())) {
        std::cerr << "GDN T=" << prefix << " vs " << in.tokens << ": common prefix differs\n";
        ++failures;
    }
    if (!std::equal(tail_run.out.begin(), tail_run.out.end(), full_run.out.begin() + cut)) {
        std::cerr << "GDN T=" << prefix << "+" << in.tokens - prefix << ": continuation differs\n";
        ++failures;
    }
    failures += verify_exact("GDN split final FP32 state", tail_run.state, full_run.state);
    auto continuation = token_range(in, in.tokens - 3, 3);
    continuation.state = tail_run.state;
    const auto after_split = causal_run(continuation, normalize, false);
    continuation.state = full_run.state;
    const auto after_full = causal_run(continuation, normalize, true);
    failures += after_split.failures + after_full.failures;
    failures += verify_exact("GDN subsequent output", after_split.out, after_full.out);
    failures += verify_exact("GDN subsequent FP32 state", after_split.state, after_full.state);
    const auto qualify = [&](const gdn_ref::Inputs& source, const CausalRun& run) {
        auto oracle_input = source;
        std::vector<double> actual;
        std::vector<double> actual_state;
        if (sampled_oracle) {
            // Predeclared real value heads 0,1,2 share real key head 0. No synthetic zero heads
            // enter either metric, and every token and state element of these heads is qualified.
            oracle_input.qk_heads = 1;
            oracle_input.value_heads = 3;
            oracle_input.q.clear(); oracle_input.k.clear(); oracle_input.v.clear();
            oracle_input.g.clear(); oracle_input.beta.clear();
            oracle_input.state.resize(3 * kStateDim * kStateDim);
            for (int t = 0; t < source.tokens; ++t) {
                const auto copy = [t](const auto& from, auto& to, int stride, int count) {
                    to.insert(to.end(), from.begin() + std::size_t(t) * stride,
                              from.begin() + std::size_t(t) * stride + count);
                };
                copy(source.q, oracle_input.q, source.qk_heads * kStateDim, kStateDim);
                copy(source.k, oracle_input.k, source.qk_heads * kStateDim, kStateDim);
                copy(source.v, oracle_input.v, source.value_heads * kStateDim, 3 * kStateDim);
                copy(source.g, oracle_input.g, source.value_heads, 3);
                copy(source.beta, oracle_input.beta, source.value_heads, 3);
                for (int r = 0; r < 3 * kStateDim; ++r) {
                    actual.push_back(bf16_to_f32(run.out[std::size_t(t) * source.value_heads *
                                                           kStateDim + r]));
                }
            }
            actual_state.assign(run.state.begin(), run.state.begin() + oracle_input.state.size());
        } else {
            for (auto bits : run.out) { actual.push_back(bf16_to_f32(bits)); }
            actual_state.assign(run.state.begin(), run.state.end());
        }
        const auto ref = gdn_ref::evaluate(oracle_input, double(1.0f / std::sqrt(float(kStateDim))),
                                          normalize);
        failures += verify_recurrence("GDN causal independent output", actual, ref.out,
                                      gated_delta_net_output_bf16_criterion());
        failures += verify_recurrence("GDN causal independent state", actual_state, ref.final_state,
                                      gated_delta_net_state_fp32_criterion());
    };
    qualify(short_input, short_run);
    qualify(in, full_run);
    qualify(tail_input, tail_run);
    return failures;
}

int causal_prefix_cases() {
    int failures = 0;
    for (bool normalize : {false, true}) {
        // 16-token Two-stage packet boundaries and arbitrary context-cache frontiers.
        // Keep the previous 59+5 and 123+5 regressions; add 15+1/16+1/
        // 64+1/128+1 to distinguish a packet-edge bug from generic FP32 drift.
        for (const auto [tokens, prefix] :
             {std::pair{16, 15}, std::pair{17, 16},
              std::pair{64, 59}, std::pair{65, 64},
              std::pair{128, 123}, std::pair{129, 128}}) {
            const auto in = make_inputs({"causal prefix", 16, 48, tokens, normalize},
                                        19000 + tokens);
            failures += causal_pair(in, prefix, normalize);
        }
    }
    return failures;
}

int represented_prefix_case(const std::string& directory) {
    const auto read = [&](const char* name, auto& values, std::size_t count) {
        using Value = typename std::decay_t<decltype(values)>::value_type;
        values.resize(count);
        std::ifstream file(directory + "/call3-" + name + ".bin", std::ios::binary);
        if (!file.read(reinterpret_cast<char*>(values.data()), count * sizeof(Value)) ||
            file.peek() != std::char_traits<char>::eof()) {
            throw std::runtime_error("invalid represented GDN input " + std::string(name));
        }
    };
    gdn_ref::Inputs in;
    in.head_dim = 128; in.qk_heads = 16; in.value_heads = 48; in.tokens = 1024;
    std::vector<std::uint16_t> bits;
    const auto read_bf16 = [&](const char* name, auto& values, int heads) {
        read(name, bits, std::size_t(1024) * heads * 128);
        values.resize(bits.size());
        for (std::size_t i = 0; i < bits.size(); ++i) { values[i] = bf16_to_f32(bits[i]); }
    };
    read_bf16("conv-q", in.q, 16); read_bf16("conv-k", in.k, 16);
    read_bf16("conv-v", in.v, 48);
    read("g", in.g, 48 * 1024); read("beta", in.beta, 48 * 1024);
    read("recurrent-before", in.state, 48 * 128 * 128);
    return causal_pair(in, 1019, true, true);
}

int unaligned_input_cases() {
    const auto in = make_inputs({"shifted contiguous inputs", 16, 48, 64, true}, 23064U);
    const auto aligned = causal_run(in, true, true);
    const auto shifted = causal_run(in, true, false, 0, 8, 8);
    // Q/K's existing eight-byte packed load requires eight-byte alignment. The public value
    // path loads BF16 scalars, so its two-byte alignment remains sufficient.
    const auto scalar_value = causal_run(in, true, true, 0, 8, 2);
    int failures = aligned.failures + shifted.failures + scalar_value.failures;
    failures += verify_exact("GDN offset8 output", shifted.out, aligned.out);
    failures += verify_exact("GDN offset8 final FP32 state", shifted.state, aligned.state);
    failures += verify_exact("GDN value offset2 output", scalar_value.out, aligned.out);
    failures += verify_exact("GDN value offset2 final FP32 state", scalar_value.state, aligned.state);
    const auto ref = gdn_ref::evaluate(in, double(1.0f / std::sqrt(float(kStateDim))), true);
    for (const auto* run : {&shifted, &scalar_value}) {
        std::vector<double> output;
        for (auto bits : run->out) { output.push_back(bf16_to_f32(bits)); }
        failures += verify_recurrence("GDN shifted independent output", output, ref.out,
                                      gated_delta_net_output_bf16_criterion());
        failures += verify_recurrence("GDN shifted independent state",
            std::vector<double>(run->state.begin(), run->state.end()), ref.final_state,
            gated_delta_net_state_fp32_criterion());
    }
    return failures;
}

// Qualifies the actual two-stage *dispatch alignment guard*, not merely the numerical
// "exact" mode where the fast kernel is disabled before checking alignment. For shifted
// BF16 views, enabling approximate fast-math must still produce byte-exact recurrent
// results (not just avoid a CUDA misaligned-address crash).
int unaligned_fast_guard_cases() {
    const char* precision = std::getenv("NINFER_GDN_TWO_STAGE_NUMERICS");
    if (precision == nullptr || std::string(precision) != "approx") {
        throw std::invalid_argument(
            "--unaligned-fast-only requires NINFER_GDN_TWO_STAGE_NUMERICS=approx");
    }
    const char* old = std::getenv("NINFER_GDN_TWO_STAGE");
    const std::optional<std::string> previous =
        old ? std::optional<std::string>(old) : std::nullopt;
    const auto set_force = [](const char* value) {
#if defined(_WIN32)
        if (_putenv_s("NINFER_GDN_TWO_STAGE", value ? value : "") != 0)
            throw std::runtime_error("cannot set GDN test route override");
#else
        if ((value ? setenv("NINFER_GDN_TWO_STAGE", value, 1)
                   : unsetenv("NINFER_GDN_TWO_STAGE")) != 0)
            throw std::runtime_error("cannot set GDN test route override");
#endif
    };
    const auto in = make_inputs({"fast GDN unaligned dispatch", 16, 48, 64, true}, 33064U);
    set_force("0");
    const auto recurrent8 = causal_run(in, true, false, 0, 8, 8);
    const auto recurrent2 = causal_run(in, true, false, 0, 8, 2);
    set_force("1");
    const auto guarded8 = causal_run(in, true, false, 0, 8, 8);
    const auto guarded2 = causal_run(in, true, false, 0, 8, 2);
    set_force(previous ? previous->c_str() : nullptr);
    int failures = recurrent8.failures + recurrent2.failures + guarded8.failures + guarded2.failures;
    failures += verify_exact("GDN fast-route Q/K/V offset8 output", guarded8.out, recurrent8.out);
    failures += verify_exact("GDN fast-route Q/K/V offset8 FP32 state", guarded8.state,
                             recurrent8.state);
    failures += verify_exact("GDN fast-route value offset2 output", guarded2.out, recurrent2.out);
    failures += verify_exact("GDN fast-route value offset2 FP32 state", guarded2.state,
                             recurrent2.state);
    return failures;
}

int state_cast_cases(); // also exercised by the exact-prefetch qualification below

// The exact double-buffer candidate changes only global-to-shared prefetch order.
// Exercise both implementations in this process with identical source tensors,
// including non-multiple-of-16 final tiles, graph replay, and continuation cuts.
int exact_prefetch_cases() {
    const char* previous_raw = std::getenv("NINFER_GDN_EXACT_PREFETCH");
    const std::optional<std::string> previous =
        previous_raw ? std::optional<std::string>(previous_raw) : std::nullopt;
    const auto set_route = [](const char* value) {
#if defined(_WIN32)
        if (_putenv_s("NINFER_GDN_EXACT_PREFETCH", value ? value : "") != 0)
            throw std::runtime_error("failed to select exact GDN prefetch route");
#else
        if ((value ? setenv("NINFER_GDN_EXACT_PREFETCH", value, 1)
                   : unsetenv("NINFER_GDN_EXACT_PREFETCH")) != 0)
            throw std::runtime_error("failed to select exact GDN prefetch route");
#endif
    };
    struct Restore {
        decltype(set_route)& setter;
        const std::optional<std::string>& original;
        ~Restore() {
            try { setter(original ? original->c_str() : nullptr); } catch (...) {}
        }
    } restore{set_route, previous};

    int failures = 0;
    // Qualify the two real GDN value-head geometries, not just 27B's h48.
    // The raw-FP32 Q/K path is also checked alongside normalized Q/K.
    for (int value_heads : {32, 48}) {
        for (bool normalize : {false, true}) {
            for (const auto [tokens, cut] :
                 {std::pair{32, 31}, std::pair{33, 32}, std::pair{59, 54},
                  std::pair{64, 59}, std::pair{65, 64},
                  std::pair{128, 123}, std::pair{129, 128}}) {
                const auto in = make_inputs({"exact GDN double-buffer", 16, value_heads,
                                             tokens, normalize},
                                            41000U + tokens + value_heads * 1000U);
                set_route("0");
                const CausalRun ordinary = causal_run(in, normalize, true);
                set_route("1");
                const CausalRun pipelined = causal_run(in, normalize, true);
                failures += ordinary.failures + pipelined.failures;
                failures += verify_exact("GDN pipelined exact output", pipelined.out, ordinary.out);
                failures += verify_exact("GDN pipelined exact FP32 state",
                                         pipelined.state, ordinary.state);
                failures += causal_pair(in, cut, normalize); // now uses pipelined prefills
            }
        }
    }
    set_route("1");
    failures += unaligned_input_cases(); // must take scalar-safe fallback
    failures += state_cast_cases();       // FP16/FP32 state conversion and alias variants
    return failures;
}

int state_cast_cases() {
    int failures = 0;
    for (bool source_half : {false, true}) {
        for (bool destination_half : {false, true}) {
            for (int tokens : {7, 64}) {
                auto in = make_inputs({"state casts", 16, 48, tokens, true}, 21000 + tokens);
                const auto encode = [](float x) { return quantized_weight::detail::f32_to_f16(x); };
                const auto decode = [](std::uint16_t x) {
                    return quantized_weight::detail::f16_to_f32(x);
                };
                std::vector<std::uint8_t> source_bits(in.state.size() * (source_half ? 2 : 4));
                for (std::size_t i = 0; i < in.state.size(); ++i) {
                    if (source_half) {
                        const auto bits = encode(in.state[i]);
                        std::memcpy(source_bits.data() + i * 2, &bits, 2);
                        in.state[i] = decode(bits);
                    } else {
                        std::memcpy(source_bits.data() + i * 4, &in.state[i], 4);
                    }
                }
                const float scale = 1.0f / std::sqrt(float(kStateDim));
                const auto ref = gdn_ref::evaluate(in, double(scale), true);
                DeviceInputs device(in);
                GuardedDeviceBuffer source(source_bits.size());
                GuardedDeviceBuffer state(in.state.size() * (destination_half ? 2 : 4));
                GuardedDeviceBuffer output(in.v.size() * 2);
                source.copy_from_host(source_bits.data(), source_bits.size());
                Tensor q(device.q.p, DType::BF16, {128, 16, tokens});
                Tensor k(device.k.p, DType::BF16, {128, 16, tokens});
                Tensor v(device.v.p, DType::BF16, {128, 48, tokens});
                Tensor g(device.g.p, DType::FP32, {48, tokens});
                Tensor beta(device.beta.p, DType::FP32, {48, tokens});
                Tensor s0(source.data(), source_half ? DType::FP16 : DType::FP32, {128, 128, 48});
                Tensor s1(state.data(), destination_half ? DType::FP16 : DType::FP32, {128, 128, 48});
                Tensor out(output.data(), DType::BF16, {128, 48, tokens});
                WorkspaceArena workspace(std::max<std::size_t>(256,
                    ops::gated_delta_net_workspace_capacity_bytes(16, 48, true, tokens, tokens)));
                ops::gated_delta_net(q, k, v, g, beta, scale, true, workspace, s0, s1, out, nullptr);
                cuda_synchronize();
                std::vector<double> actual_state;
                if (destination_half) {
                    const auto bits = from_device<std::uint16_t>(state.data(), in.state.size());
                    for (auto x : bits) { actual_state.push_back(decode(x)); }
                } else {
                    actual_state = read_f32(state.data(), in.state.size());
                }
                const auto label = std::string("GDN casts ") + (source_half ? "f16" : "f32") +
                                   "->" + (destination_half ? "f16" : "f32") +
                                   " T=" + std::to_string(tokens);
                failures += verify_recurrence(label + " out", from_device_bf16(output.data(),
                    in.v.size()), ref.out, gated_delta_net_output_bf16_criterion());
                failures += verify_recurrence(label + " state", actual_state, ref.final_state,
                                               gated_delta_net_state_fp32_criterion());
                failures += verify_exact(label + " source immutable",
                    from_device<std::uint8_t>(source.data(), source_bits.size()), source_bits);
                failures += source.verify_guards(label + " source guards");
                failures += state.verify_guards(label + " state guards");
                failures += output.verify_guards(label + " output guards");
            }
        }
    }
    return failures;
}

int inplace_case(const Case& test_case, std::uint32_t seed) {
    const gdn_ref::Inputs in = make_inputs(test_case, seed);
    const float scale        = 1.0f / std::sqrt(static_cast<float>(kStateDim));
    const gdn_ref::Result ref =
        gdn_ref::evaluate(in, static_cast<double>(scale), test_case.normalize_qk);
    DeviceInputs device(in);
    GuardedDeviceBuffer state(in.state.size() * sizeof(float));
    GuardedDeviceBuffer out(in.v.size() * sizeof(std::uint16_t));
    state.copy_from_host(in.state.data(), state.bytes());
    out.fill(0xff);

    Tensor q(device.q.p, DType::BF16, {kStateDim, test_case.qk_heads, test_case.tokens});
    Tensor k(device.k.p, DType::BF16, {kStateDim, test_case.qk_heads, test_case.tokens});
    Tensor v(device.v.p, DType::BF16, {kStateDim, test_case.value_heads, test_case.tokens});
    Tensor g(device.g.p, DType::FP32, {test_case.value_heads, test_case.tokens});
    Tensor beta(device.beta.p, DType::FP32, {test_case.value_heads, test_case.tokens});
    Tensor state_tensor(state.data(), DType::FP32, {kStateDim, kStateDim, test_case.value_heads});
    Tensor out_tensor(out.data(), DType::BF16,
                      {kStateDim, test_case.value_heads, test_case.tokens});
    const std::size_t workspace_bytes = ops::gated_delta_net_workspace_capacity_bytes(
        test_case.qk_heads, test_case.value_heads, test_case.normalize_qk, test_case.tokens,
        test_case.tokens);
    WorkspaceArena workspace(std::max<std::size_t>(workspace_bytes, 256));

    ops::gated_delta_net(q, k, v, g, beta, scale, test_case.normalize_qk, workspace, state_tensor,
                         out_tensor, nullptr);
    cuda_synchronize();

    const std::string label = std::string(test_case.name) + " inplace";
    int failures            = 0;
    failures += verify_recurrence(label + " out", from_device_bf16(out.data(), in.v.size()),
                                  ref.out, gated_delta_net_output_bf16_criterion());
    failures += verify_recurrence(label + " state", read_f32(state.data(), in.state.size()),
                                  ref.final_state, gated_delta_net_state_fp32_criterion());
    failures += state.verify_guards((label + " state").c_str());
    failures += out.verify_guards((label + " out").c_str());
    failures += verify_common_inputs_unchanged(label, in, device.q, device.k, device.v, device.g,
                                               device.beta);
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << label << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    return failures;
}

int distinct_state_case(const Case& test_case, std::uint32_t seed) {
    const gdn_ref::Inputs in = make_inputs(test_case, seed);
    const float scale        = 1.0f / std::sqrt(static_cast<float>(kStateDim));
    const gdn_ref::Result ref =
        gdn_ref::evaluate(in, static_cast<double>(scale), test_case.normalize_qk);
    DeviceInputs device(in);
    GuardedDeviceBuffer state_in(in.state.size() * sizeof(float));
    GuardedDeviceBuffer state_out(in.state.size() * sizeof(float));
    GuardedDeviceBuffer out(in.v.size() * sizeof(std::uint16_t));
    state_in.copy_from_host(in.state.data(), state_in.bytes());
    state_out.fill(0xff);
    out.fill(0xff);

    Tensor q(device.q.p, DType::BF16, {kStateDim, test_case.qk_heads, test_case.tokens});
    Tensor k(device.k.p, DType::BF16, {kStateDim, test_case.qk_heads, test_case.tokens});
    Tensor v(device.v.p, DType::BF16, {kStateDim, test_case.value_heads, test_case.tokens});
    Tensor g(device.g.p, DType::FP32, {test_case.value_heads, test_case.tokens});
    Tensor beta(device.beta.p, DType::FP32, {test_case.value_heads, test_case.tokens});
    Tensor state_in_tensor(state_in.data(), DType::FP32,
                           {kStateDim, kStateDim, test_case.value_heads});
    Tensor state_out_tensor(state_out.data(), DType::FP32,
                            {kStateDim, kStateDim, test_case.value_heads});
    Tensor out_tensor(out.data(), DType::BF16,
                      {kStateDim, test_case.value_heads, test_case.tokens});
    const std::size_t workspace_bytes = ops::gated_delta_net_workspace_capacity_bytes(
        test_case.qk_heads, test_case.value_heads, test_case.normalize_qk, test_case.tokens,
        test_case.tokens);
    WorkspaceArena workspace(std::max<std::size_t>(workspace_bytes, 256));

    ops::gated_delta_net(q, k, v, g, beta, scale, test_case.normalize_qk, workspace,
                         state_in_tensor, state_out_tensor, out_tensor, nullptr);
    cuda_synchronize();

    const std::string label = std::string(test_case.name) + " distinct-state";
    int failures            = 0;
    failures += verify_recurrence(label + " out", from_device_bf16(out.data(), in.v.size()),
                                  ref.out, gated_delta_net_output_bf16_criterion());
    failures += verify_recurrence(label + " state", read_f32(state_out.data(), in.state.size()),
                                  ref.final_state, gated_delta_net_state_fp32_criterion());
    failures += verify_exact(label + " state-in unchanged",
                             from_device<float>(state_in.data(), in.state.size()), in.state);
    failures += state_in.verify_guards((label + " state-in").c_str());
    failures += state_out.verify_guards((label + " state-out").c_str());
    failures += out.verify_guards((label + " out").c_str());
    failures += verify_common_inputs_unchanged(label, in, device.q, device.k, device.v, device.g,
                                               device.beta);
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << label << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    return failures;
}

int batch_update_case(const Case& test_case, const std::vector<int>& source_slots,
                      const std::vector<int>& destination_slots, int slots, std::uint32_t seed) {
    if (test_case.tokens != 1) { throw std::logic_error("batch_update_case requires W=1"); }
    if (source_slots.size() != destination_slots.size()) {
        throw std::logic_error("batch_update_case selector sizes differ");
    }
    const int batch   = static_cast<int>(source_slots.size());
    const int width   = 1;
    const float scale = 1.0f / std::sqrt(static_cast<float>(kStateDim));
    const std::size_t qk_row_size =
        static_cast<std::size_t>(kStateDim * test_case.qk_heads * width);
    const std::size_t value_row_size =
        static_cast<std::size_t>(kStateDim * test_case.value_heads * width);
    const std::size_t gate_row_size = static_cast<std::size_t>(test_case.value_heads * width);
    const std::size_t state_size =
        static_cast<std::size_t>(kStateDim * kStateDim * test_case.value_heads);

    gdn_ref::Inputs aggregate;
    aggregate.head_dim    = kStateDim;
    aggregate.qk_heads    = test_case.qk_heads;
    aggregate.value_heads = test_case.value_heads;
    aggregate.tokens      = static_cast<std::int64_t>(width) * batch;
    aggregate.q.reserve(qk_row_size * static_cast<std::size_t>(batch));
    aggregate.k.reserve(qk_row_size * static_cast<std::size_t>(batch));
    aggregate.v.reserve(value_row_size * static_cast<std::size_t>(batch));
    aggregate.g.reserve(gate_row_size * static_cast<std::size_t>(batch));
    aggregate.beta.reserve(gate_row_size * static_cast<std::size_t>(batch));

    std::vector<gdn_ref::Inputs> rows;
    rows.reserve(static_cast<std::size_t>(batch));
    std::vector<float> initial_states(state_size * static_cast<std::size_t>(slots), 0.125f);
    for (int row = 0; row < batch; ++row) {
        gdn_ref::Inputs input =
            make_inputs(test_case, seed + static_cast<std::uint32_t>(row) * 97U);
        aggregate.q.insert(aggregate.q.end(), input.q.begin(), input.q.end());
        aggregate.k.insert(aggregate.k.end(), input.k.begin(), input.k.end());
        aggregate.v.insert(aggregate.v.end(), input.v.begin(), input.v.end());
        aggregate.g.insert(aggregate.g.end(), input.g.begin(), input.g.end());
        aggregate.beta.insert(aggregate.beta.end(), input.beta.begin(), input.beta.end());
        std::copy(input.state.begin(), input.state.end(),
                  initial_states.begin() +
                      static_cast<std::size_t>(source_slots[static_cast<std::size_t>(row)]) *
                          state_size);
        rows.push_back(std::move(input));
    }

    std::vector<gdn_ref::Result> references;
    references.reserve(static_cast<std::size_t>(batch));
    std::vector<double> expected_output(value_row_size * static_cast<std::size_t>(batch), 0.0);
    std::vector<bool> written_slots(static_cast<std::size_t>(slots), false);
    for (int row = 0; row < batch; ++row) {
        gdn_ref::Result reference =
            gdn_ref::evaluate(rows[static_cast<std::size_t>(row)], static_cast<double>(scale),
                              test_case.normalize_qk);
        std::copy(reference.out.begin(), reference.out.end(),
                  expected_output.begin() + static_cast<std::size_t>(row) * value_row_size);
        written_slots[static_cast<std::size_t>(destination_slots[static_cast<std::size_t>(row)])] =
            true;
        references.push_back(std::move(reference));
    }

    DeviceInputs device(aggregate);
    GuardedDeviceBuffer states(initial_states.size() * sizeof(float));
    GuardedDeviceBuffer out(aggregate.v.size() * sizeof(std::uint16_t));
    states.copy_from_host(initial_states.data(), states.bytes());
    out.fill(0xff);
    DeviceBuffer device_source_slots      = to_device(source_slots);
    DeviceBuffer device_destination_slots = to_device(destination_slots);

    Tensor q(device.q.p, DType::BF16, {kStateDim, test_case.qk_heads, width, batch});
    Tensor k(device.k.p, DType::BF16, {kStateDim, test_case.qk_heads, width, batch});
    Tensor v(device.v.p, DType::BF16, {kStateDim, test_case.value_heads, width, batch});
    Tensor g(device.g.p, DType::FP32, {test_case.value_heads, width, batch});
    Tensor beta(device.beta.p, DType::FP32, {test_case.value_heads, width, batch});
    Tensor states_tensor(states.data(), DType::FP32,
                         {kStateDim, kStateDim, test_case.value_heads, slots});
    Tensor source_slots_tensor(device_source_slots.p, DType::I32, {batch});
    Tensor destination_slots_tensor(device_destination_slots.p, DType::I32, {batch});
    Tensor out_tensor(out.data(), DType::BF16, {kStateDim, test_case.value_heads, width, batch});
    ops::gated_delta_net_batch_update(q, k, v, g, beta, scale, test_case.normalize_qk,
                                      states_tensor, source_slots_tensor, destination_slots_tensor,
                                      out_tensor, nullptr);
    cuda_synchronize();

    const std::string label =
        std::string(test_case.name) + " batch update B=" + std::to_string(batch);
    int failures                         = 0;
    const std::vector<double> got_output = from_device_bf16(out.data(), aggregate.v.size());
    failures += verify_recurrence(label + " out", got_output, expected_output,
                                  gated_delta_net_output_bf16_criterion());

    const std::vector<float> got_states = from_device<float>(states.data(), initial_states.size());
    for (int row = 0; row < batch; ++row) {
        const std::size_t begin =
            static_cast<std::size_t>(destination_slots[static_cast<std::size_t>(row)]) * state_size;
        failures +=
            verify_recurrence(label + " row " + std::to_string(row) + " final state",
                              doubles(std::vector<float>(got_states.begin() + begin,
                                                         got_states.begin() + begin + state_size)),
                              references[static_cast<std::size_t>(row)].final_state,
                              gated_delta_net_state_fp32_criterion());
    }
    for (int slot = 0; slot < slots; ++slot) {
        if (written_slots[static_cast<std::size_t>(slot)]) continue;
        const std::size_t begin = static_cast<std::size_t>(slot) * state_size;
        failures += verify_exact(
            label + " untouched slot " + std::to_string(slot),
            std::vector<float>(got_states.begin() + begin, got_states.begin() + begin + state_size),
            std::vector<float>(initial_states.begin() + begin,
                               initial_states.begin() + begin + state_size));
    }
    failures +=
        verify_exact(label + " source selectors unchanged",
                     from_device_i32(device_source_slots, source_slots.size()), source_slots);
    failures += verify_exact(label + " destination selectors unchanged",
                             from_device_i32(device_destination_slots, destination_slots.size()),
                             destination_slots);
    failures += states.verify_guards((label + " states").c_str());
    failures += out.verify_guards((label + " out").c_str());
    failures += verify_common_inputs_unchanged(label, aggregate, device.q, device.k, device.v,
                                               device.g, device.beta);
    return failures;
}

int contract_rejection_cases() {
    DeviceBuffer q_buffer(kStateDim * 8 * sizeof(std::uint16_t));
    DeviceBuffer k_buffer(kStateDim * 8 * sizeof(std::uint16_t));
    DeviceBuffer v_buffer(kStateDim * 8 * sizeof(std::uint16_t));
    DeviceBuffer g_buffer(8 * sizeof(float));
    DeviceBuffer beta_buffer(8 * sizeof(float));
    DeviceBuffer state_buffer(kStateDim * kStateDim * 8 * sizeof(float));
    DeviceBuffer out_buffer(kStateDim * 8 * sizeof(std::uint16_t));
    WorkspaceArena workspace(256);
    const float scale = 1.0f / std::sqrt(static_cast<float>(kStateDim));

    auto is_rejected = [&](int activation_dim, int state_dim, int qk_heads, int value_heads) {
        Tensor q(q_buffer.p, DType::BF16, {activation_dim, qk_heads, 1});
        Tensor k(k_buffer.p, DType::BF16, {activation_dim, qk_heads, 1});
        Tensor v(v_buffer.p, DType::BF16, {activation_dim, value_heads, 1});
        Tensor g(g_buffer.p, DType::FP32, {value_heads, 1});
        Tensor beta(beta_buffer.p, DType::FP32, {value_heads, 1});
        Tensor state(state_buffer.p, DType::FP32, {state_dim, state_dim, value_heads});
        Tensor out(out_buffer.p, DType::BF16, {activation_dim, value_heads, 1});
        try {
            ops::gated_delta_net(q, k, v, g, beta, scale, true, workspace, state, out, nullptr);
        } catch (const std::invalid_argument&) { return true; }
        cuda_synchronize();
        return false;
    };

    int failures = 0;
    if (!is_rejected(64, kStateDim, 4, 8)) {
        std::cerr << "gated_delta_net accepted Q/K/V head dimension 64\n";
        ++failures;
    }
    if (!is_rejected(kStateDim, 64, 4, 8)) {
        std::cerr << "gated_delta_net accepted state dimension 64\n";
        ++failures;
    }
    if (!is_rejected(kStateDim, kStateDim, 4, 6)) {
        std::cerr << "gated_delta_net accepted a non-divisible head map\n";
        ++failures;
    }
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;

    for (const bool normalize_qk : {false, true}) {
        const std::size_t interval =
            ops::gated_delta_net_workspace_capacity_bytes(16, 48, normalize_qk, 63, 65);
        const std::size_t witness =
            ops::gated_delta_net_workspace_capacity_bytes(16, 48, normalize_qk, 65, 65);
        if (interval != witness) {
            std::cerr << "gated_delta_net interval capacity missed the chunk boundary\n";
            ++failures;
        }
    }
    try {
        (void)ops::gated_delta_net_workspace_capacity_bytes(16, 48, true, 0, 65);
        std::cerr << "gated_delta_net accepted an invalid token interval\n";
        ++failures;
    } catch (const std::invalid_argument&) {}
    try {
        (void)ops::gated_delta_net_workspace_capacity_bytes(4, 6, true, 1, 65);
        std::cerr << "gated_delta_net workspace accepted a non-divisible head map\n";
        ++failures;
    } catch (const std::invalid_argument&) {}
    failures += contract_rejection_cases();

    if (argc == 2 && std::string(argv[1]) == "--causal-prefix-only") {
        failures += causal_prefix_cases();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " GDN causal prefix\n";
        return failures == 0 ? 0 : 1;
    }
    if (argc == 3 && std::string(argv[1]) == "--represented-prefix-only") {
        failures += represented_prefix_case(argv[2]);
        std::cout << (failures == 0 ? "OK" : "FAIL") << " GDN represented prefix\n";
        return failures == 0 ? 0 : 1;
    }
    if (argc == 2 && std::string(argv[1]) == "--state-casts-only") {
        failures += state_cast_cases();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " GDN state casts\n";
        return failures == 0 ? 0 : 1;
    }
    if (argc == 2 && std::string(argv[1]) == "--exact-prefetch-only") {
        failures += exact_prefetch_cases();
        std::cout << (failures == 0 ? "OK" : "FAIL")
                  << " GDN exact double-buffer prefetch\n";
        return failures == 0 ? 0 : 1;
    }
    if (argc == 2 && std::string(argv[1]) == "--unaligned-fast-only") {
        failures += unaligned_fast_guard_cases();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " GDN approximate fast alignment guard\n";
        return failures == 0 ? 0 : 1;
    }
    if (argc == 2 && std::string(argv[1]) == "--unaligned-inputs-only") {
        failures += unaligned_input_cases();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " GDN shifted inputs\n";
        return failures == 0 ? 0 : 1;
    }
    failures += causal_prefix_cases() + state_cast_cases() + unaligned_input_cases();

    // Registered 27B/35B-A3B geometries, public state forms, and the recurrent/chunk/tail route
    // boundary are all qualified directly against the same complete FP64 recurrence.
    failures += inplace_case({"27b decode fused-qk-norm", 16, 48, 1, true}, 12001u);
    failures += distinct_state_case({"27b raw-qk small-T", 16, 48, 7, false}, 12007u);
    failures += distinct_state_case({"35b pre-chunk fused-qk-norm", 16, 32, 63, true}, 12063u);
    failures += distinct_state_case({"27b exact chunk fused-qk-norm", 16, 48, 64, true}, 12064u);
    failures += distinct_state_case({"27b exact chunk raw-qk", 16, 48, 64, false}, 12164u);
    failures += inplace_case({"35b chunk-tail fused-qk-norm", 16, 32, 65, true}, 12065u);
    failures += distinct_state_case({"generic grouped-map chunk-tail", 3, 12, 65, true}, 12365u);
    failures += distinct_state_case({"27b two-chunk fused-qk-norm", 16, 48, 128, true}, 12128u);
    failures += inplace_case({"35b two-chunk raw-qk", 16, 32, 128, false}, 12228u);

    // The production decode path updates selected state-pool slots in place at width one.
    failures += batch_update_case({"27b selected-slot fused-qk-norm", 16, 48, 1, true}, {7}, {7}, 8,
                                  12101u);
    failures += batch_update_case({"35b selected-slot near-zero", 16, 32, 1, true, true}, {6}, {6},
                                  8, 12201u);
    failures += batch_update_case({"35b ordinary", 16, 32, 1, true}, {8, 9, 10, 11, 12, 13, 14, 15},
                                  {8, 9, 10, 11, 12, 13, 14, 15}, 16, 13001u);
    failures += batch_update_case({"35b mixed fork destinations", 16, 32, 1, true}, {0, 2, 4, 6},
                                  {1, 3, 5, 7}, 8, 13101u);

    std::cout << (failures == 0 ? "OK" : "FAIL") << " gated_delta_net correctness\n";
    return failures == 0 ? 0 : 1;
}
