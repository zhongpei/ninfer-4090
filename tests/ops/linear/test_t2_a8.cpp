#include "ops/linear/linear_test_common.h"
#include "ops/linear/fixtures/t2_width_consistency.h"
#include "ops/op_tester.h"
#include "core/decode_graph.h"
#include "core/device.h"

#include <array>
#include <exception>
#include <algorithm>
#include <cstring>
#include <string_view>
#include <iostream>

namespace {

using namespace ninfer;
using namespace ninfer::test::linear;

constexpr Invocation a8(std::int32_t t) {
    return {t, CallForm::Policy, ops::LinearPolicy::AllowA8Int};
}

std::vector<std::uint8_t> fixture_bytes(std::string_view hex) {
    const auto nibble = [](char c) { return c <= '9' ? c - '0' : c - 'a' + 10; };
    std::vector<std::uint8_t> result(hex.size() / 2);
    for (std::size_t i = 0; i < result.size(); ++i)
        result[i] = static_cast<std::uint8_t>((nibble(hex[2 * i]) << 4) | nibble(hex[2 * i + 1]));
    return result;
}

int real_width_consistency() {
    constexpr int n = 1024, k = 5120, checked_rows = 68;
    auto host_weight = make_t2_g128_fp16_weight(n, k, 301U);
    std::fill(host_weight.payload.begin(), host_weight.payload.end(), 0);
    const auto codes = fixture_bytes(fixture::codes);
    const auto scales = fixture_bytes(fixture::scales);
    std::copy(codes.begin(), codes.end(), host_weight.payload.begin());
    std::copy(scales.begin(), scales.end(),
              host_weight.payload.begin() + host_weight.scale_plane_offset);
    const auto input_bytes = fixture_bytes(fixture::input);
    std::vector<std::uint16_t> input(k);
    std::memcpy(input.data(), input_bytes.data(), input_bytes.size());
    DeviceBuffer device_weight(host_weight.payload.size());
    device_weight.copy_from_host(host_weight.payload.data(), device_weight.bytes);
    const Weight weight = host_weight.device_weight(device_weight.p);
    // Independent decoding of the stored two-bit fields and binary16 scales; no production
    // decode or quantization kernel supplies the FP64 mathematical reference.
    std::array<double, checked_rows> reference{};
    for (int row = 0; row < checked_rows; ++row) {
        for (int j = 0; j < k; ++j) {
            const unsigned field = (codes[row * (k / 4) + j / 4] >> (2 * (j % 4))) & 3u;
            if (field == 2u) throw std::runtime_error("fixture contains illegal ternary code");
            const int code = field == 3u ? -1 : static_cast<int>(field);
            const int offset = 2 * (row * (k / 128) + j / 128);
            const std::uint16_t scale_bits = scales[offset] | (unsigned(scales[offset + 1]) << 8);
            reference[row] += double(code) *
                test::quantized_weight::detail::f16_to_f32(scale_bits) * test::bf16_to_f32(input[j]);
        }
    }
    std::array<std::uint16_t, checked_rows> canonical{};
    int failures = 0;
    for (int t : {1, 3, 7, 8, 9, 16, 17, 32, 50, 128, 192, 193, 1024}) {
        std::vector<std::uint16_t> columns(std::size_t(k) * t);
        for (int col = 0; col < t; ++col) std::copy(input.begin(), input.end(), columns.begin() + col * k);
        auto dx = test::to_device(columns);
        DeviceBuffer output(std::size_t(n) * t * sizeof(std::uint16_t));
        Tensor x(dx.p, DType::BF16, {k, t}), y(output.p, DType::BF16, {n, t});
        const auto capacity = ops::linear_workspace_capacity_bytes(
            weight.qtype, n, k, ops::LinearPolicy::AllowA8Int, t, t);
        DeviceArena scratch(std::max<std::size_t>(capacity, 256));
        DeviceContext context;
        DecodeGraphDefinition definition;
        DecodeGraphExecutable graph;
        definition.capture(context.stream, [&] {
            ops::linear(x, weight, y, ops::LinearPolicy::AllowA8Int, scratch, context.stream);
        });
        graph.instantiate(definition);
        graph.launch(context.stream);
        test::cuda_check(cudaStreamSynchronize(context.stream), "real T2 width graph");
        const auto bits = test::from_device<std::uint16_t>(output, std::size_t(n) * t);
        std::array<double, checked_rows> actual{};
        for (int row = 0; row < checked_rows; ++row) {
            if (t == 1) canonical[row] = bits[row];
            actual[row] = test::bf16_to_f32(bits[row]);
        }
        if (t == 1) {
            for (int row = 0; row < 4; ++row)
                std::cout << "T2 real counterexample " << row << " absolute error "
                          << std::abs(actual[row] - reference[row]) << '\n';
        }
        const std::string label = "T2 A8 real prefix T=" + std::to_string(t);
        failures += test::verify_reduction(label, actual, reference, {0.04, 1.0 / 256.0, 0.06});
        std::size_t differing = 0;
        for (int col = 0; col < t; ++col)
            for (int row = 0; row < checked_rows; ++row)
                differing += bits[col * n + row] != canonical[row];
        if (differing) {
            std::cerr << label << ": " << differing << " represented outputs differ from width1\n";
            ++failures;
        }
    }
    return failures;
}

// The ternary integer-activation routes through one policy: the small-T kernel (1..192 tokens, in
// launches of one, two and four column tiles) and above it the prefill GEMM, which pads T to its
// cheapest tile width inside.
int t2_a8_conformance() {
    int failures = 0;

    constexpr std::array kWidths{a8(1),   a8(2),   a8(3),    a8(4),   a8(5),   a8(7),   a8(8),
                                 a8(9),   a8(12),  a8(15),   a8(16),  a8(17),  a8(24),  a8(32),
                                 a8(33),  a8(40),  a8(63),   a8(64),  a8(65),  a8(95),  a8(100),
                                 a8(127), a8(128), a8(129),  a8(192), a8(193), a8(256), a8(300),
                                 a8(385), a8(600), a8(1007), a8(1024)};

    failures += run_shape("T2_A8", ActivationCompute::A8, make_t2_g128_fp16_weight,
                          {1024, 5120, 301U, Comparison::Full, true, kWidths});
    failures += run_shape("T2_A8", ActivationCompute::A8, make_t2_g128_fp16_weight,
                          {4096, 5120, 303U, Comparison::SampledRows, false, kWidths});
    failures += run_shape("T2_A8", ActivationCompute::A8, make_t2_g128_fp16_weight,
                          {7168, 5120, 305U, Comparison::SampledRows, false, kWidths});
    failures += run_shape("T2_A8", ActivationCompute::A8, make_t2_g128_fp16_weight,
                          {34816, 5120, 307U, Comparison::SampledRows, false, kWidths});
    failures += run_shape("T2_A8", ActivationCompute::A8, make_t2_g128_fp16_weight,
                          {5120, 6144, 309U, Comparison::Full, false, kWidths});
    failures += run_shape("T2_A8", ActivationCompute::A8, make_t2_g128_fp16_weight,
                          {5120, 17408, 311U, Comparison::Full, false, kWidths});

    return failures;
}

} // namespace

int main(int argc, char** argv) {
    if (!ninfer::test::linear::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    try {
        int failures = real_width_consistency();
        if (!(argc == 2 && std::string_view(argv[1]) == "--width-consistency-only"))
            failures += t2_a8_conformance();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " T2_A8 Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "T2_A8 Linear: " << error.what() << '\n';
        return 1;
    }
}
