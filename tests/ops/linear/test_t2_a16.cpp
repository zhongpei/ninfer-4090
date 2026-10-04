#include "ops/linear/linear_test_common.h"
#include "ops/linear/fixtures/t2_a16_width_consistency.h"
#include "ops/linear/fixtures/t2_a16_root_consistency.h"
#include "ops/op_tester.h"
#include "core/decode_graph.h"
#include "core/device.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <exception>
#include <iostream>
#include <string_view>

namespace {

using namespace ninfer;
using namespace ninfer::test::linear;

constexpr Invocation a16(std::int32_t t) { return {t}; }

constexpr Invocation convenience(std::int32_t t) { return {t, CallForm::A16Convenience}; }

std::vector<std::uint8_t> represented_bytes(std::string_view hex) {
    const auto nibble = [](char c) {
        return c >= 'a' ? c - 'a' + 10 : c - '0';
    };
    std::vector<std::uint8_t> bytes(hex.size() / 2);
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        bytes[i] = static_cast<std::uint8_t>((nibble(hex[2 * i]) << 4) | nibble(hex[2 * i + 1]));
    }
    return bytes;
}

int represented_width_consistency(
    int n = 1024, bool long_widths = true,
    std::string_view codes_hex = a16_fixture::codes,
    std::string_view scales_hex = a16_fixture::scales,
    std::string_view input_hex = a16_fixture::input,
    int parent_rows = 0, int row_begin = 0) {
    constexpr int k = 5120, bases = 8, checked = a16_fixture::rows;
    auto host_weight = make_t2_g128_fp16_weight(parent_rows ? parent_rows : n, k, 301U);
    std::fill(host_weight.payload.begin(), host_weight.payload.end(), 0);
    const auto codes = represented_bytes(codes_hex);
    const auto scales = represented_bytes(scales_hex);
    std::copy(codes.begin(), codes.end(), host_weight.payload.begin() + row_begin * (k / 4));
    std::copy(scales.begin(), scales.end(),
              host_weight.payload.begin() + host_weight.scale_plane_offset + row_begin * (k / 128) * 2);
    const auto input_bytes = represented_bytes(input_hex);
    std::vector<std::uint16_t> input(bases * k);
    std::memcpy(input.data(), input_bytes.data(), input_bytes.size());
    for (int b = 1; b < bases; ++b) {
        for (int j = 0; j < k; ++j) {
            input[b * k + j] = input[j] ^ (((j / 128 + b) % bases < b) ? 0x8000U : 0U);
        }
    }
    // No production decode or projection supplies the mathematical reference. Only represented
    // rows enter the criterion: the zero padding required by the registered shape cannot dilute it.
    std::array<std::array<double, checked>, bases> reference{};
    for (int b = 0; b < bases; ++b) {
        for (int row = 0; row < checked; ++row) {
            for (int j = 0; j < k; ++j) {
                const unsigned field = (codes[row * (k / 4) + j / 4] >> (2 * (j % 4))) & 3u;
                if (field == 2u) { throw std::runtime_error("illegal represented ternary field"); }
                const int code = field == 3u ? -1 : static_cast<int>(field);
                const int offset = 2 * (row * (k / 128) + j / 128);
                const std::uint16_t scale = scales[offset] | (unsigned(scales[offset + 1]) << 8);
                reference[b][row] += double(code) *
                    test::quantized_weight::detail::f16_to_f32(scale) *
                    test::bf16_to_f32(input[b * k + j]);
            }
        }
    }
    DeviceBuffer device_weight(host_weight.payload.size());
    device_weight.copy_from_host(host_weight.payload.data(), device_weight.bytes);
    Weight weight = host_weight.device_weight(device_weight.p);
    // Independently construct the two-plane row view over its resident parent. Codes and scales
    // have different offsets; retaining the parent scale-plane start protects the actual z slice.
    if (parent_rows) {
        weight.n = weight.shape[0] = weight.padded_shape[0] = n;
        weight.qdata = static_cast<const std::uint8_t*>(weight.qdata) + row_begin * (k / 4);
        weight.scales = static_cast<const std::uint8_t*>(weight.scales) + row_begin * (k / 128) * 2;
    }
    std::array<std::array<std::uint16_t, checked>, bases> canonical{};
    DeviceArena workspace(256);
    for (int b = 0; b < bases; ++b) {
        DeviceBuffer activation(k * sizeof(std::uint16_t));
        activation.copy_from_host(input.data() + b * k, activation.bytes);
        DeviceBuffer output(n * sizeof(std::uint16_t));
        Tensor x(activation.p, DType::BF16, {k, 1});
        Tensor y(output.p, DType::BF16, {n, 1});
        ops::linear(x, weight, y, ops::LinearPolicy::A16Only, workspace, nullptr);
        test::cuda_synchronize();
        const auto bits = test::from_device<std::uint16_t>(output, n);
        std::copy_n(bits.begin(), checked, canonical[b].begin());
    }
    int failures = 0;
    for (int width : {1, 3, 7, 8, 9, 12, 16, 17, 32, 50, 128, 129, 1024}) {
        if (!long_widths && width > 50) { continue; }
        const auto capacity = ops::linear_workspace_capacity_bytes(
            weight.qtype, n, k, ops::LinearPolicy::A16Only, width, width);
        if (capacity != 0) {
            std::cerr << "T2 A16 unexpectedly requires workspace\n";
            ++failures;
        }
        test::GuardedDeviceBuffer activation(std::size_t(k) * width * 2);
        test::GuardedDeviceBuffer output(std::size_t(n) * width * 2);
        Tensor x(activation.data(), DType::BF16, {k, width});
        Tensor y(output.data(), DType::BF16, {n, width});
        DeviceContext context;
        DecodeGraphDefinition definition;
        DecodeGraphExecutable graph;
        definition.capture(context.stream, [&] {
            ops::linear(x, weight, y, ops::LinearPolicy::A16Only, workspace, context.stream);
        });
        graph.instantiate(definition);
        for (int phase = 0; phase < 2; ++phase) {
            std::vector<std::uint16_t> columns(std::size_t(k) * width);
            for (int col = 0; col < width; ++col) {
                const int b = (col + 3 * phase) % bases;
                std::copy_n(input.begin() + b * k, k, columns.begin() + std::size_t(col) * k);
            }
            activation.copy_from_host(columns.data(), columns.size() * 2);
            graph.launch(context.stream);
            test::cuda_check(cudaStreamSynchronize(context.stream), "represented A16 graph");
            const auto bits = test::from_device<std::uint16_t>(output.data(), std::size_t(n) * width);
            std::vector<double> actual, expected;
            std::size_t differing = 0;
            for (int col = 0; col < width; ++col) {
                const int b = (col + 3 * phase) % bases;
                for (int row = 0; row < checked; ++row) {
                    actual.push_back(test::bf16_to_f32(bits[std::size_t(col) * n + row]));
                    expected.push_back(reference[b][row]);
                    differing += bits[std::size_t(col) * n + row] != canonical[b][row];
                }
            }
            const std::string label = "T2 A16 represented N=" + std::to_string(n) +
                                      " W=" + std::to_string(width) +
                                      " phase=" + std::to_string(phase);
            failures += test::verify_reduction(label, actual, expected,
                                               {1.0 / 256.0, 1.0 / 256.0, 2.0 / 256.0});
            if (differing != 0) {
                std::cerr << label << ": " << differing << " outputs differ from their W1 query\n";
                ++failures;
            }
            if (width == 1 && phase == 0) {
                for (int row = 0; row < 7; ++row) {
                    std::cout << "A16 represented counterexample " << row << " abs_error="
                              << std::abs(actual[row] - expected[row]) << '\n';
                }
            }
            if (phase == 1 && width > 1) {
                // Different compact-batch/ragged partitions retain the same represented columns.
                // Linear's public domain is [K,T], so batch packing is expressed as column slices.
                test::GuardedDeviceBuffer segmented(std::size_t(n) * width * 2);
                Tensor split_output(segmented.data(), DType::BF16, {n, width});
                constexpr std::array chunks{1, 3, 7, 11, 16, 17, 32};
                int start = 0, chunk = 0;
                while (start < width) {
                    const int count = std::min(chunks[chunk++ % chunks.size()], width - start);
                    Tensor x_part = x.slice(1, start, count);
                    Tensor y_part = split_output.slice(1, start, count);
                    ops::linear(x_part, weight, y_part, ops::LinearPolicy::A16Only, workspace,
                                context.stream);
                    start += count;
                }
                test::cuda_check(cudaStreamSynchronize(context.stream), "A16 split graph columns");
                const auto split = test::from_device<std::uint16_t>(segmented.data(), bits.size());
                if (split != bits) {
                    std::cerr << label << ": segmentation changes output bits\n";
                    ++failures;
                }
                failures += segmented.verify_guards(label + " split guards");
            }
            const auto preserved = test::from_device<std::uint16_t>(activation.data(), columns.size());
            if (preserved != columns) {
                std::cerr << label << ": input was modified\n";
                ++failures;
            }
        }
        failures += activation.verify_guards("A16 represented activation guards");
        failures += output.verify_guards("A16 represented output guards");
    }
    const auto stored_after = test::from_device<std::uint8_t>(device_weight, host_weight.payload.size());
    if (stored_after != host_weight.payload) {
        std::cerr << "A16 represented stored weights were modified\n";
        ++failures;
    }
    if (workspace.used() != 0 || workspace.peak_used() != 0) {
        std::cerr << "A16 represented workspace was consumed\n";
        ++failures;
    }
    return failures;
}

int root_width_consistency() {
    using namespace a16_root_fixture;
    int failures = represented_width_consistency(4096, true, qk_codes, qk_scales, input);
    failures += represented_width_consistency(6144, true, value_codes, value_scales,
                                             input, 12288, 0);
    failures += represented_width_consistency(6144, true, z_codes, z_scales,
                                             input, 12288, 6144);
    return failures;
}

// Qualify every registered N/K against the independent FP64 oracle. The short widths cover
// both column tiles, and the long invocation protects prefill without sampling query columns.
int canonical_a16_conformance() {
    int failures = represented_width_consistency() + root_width_consistency();
    failures += represented_width_consistency(34816, false);
    failures += represented_width_consistency(131072, false);
    constexpr std::array widths{a16(1), a16(7), a16(8), a16(16),
                                a16(17), a16(50), Invocation{1024, CallForm::Policy,
                                    ops::LinearPolicy::A16Only, true}};
    constexpr std::array shapes{
        std::array{1024, 5120}, std::array{4096, 5120}, std::array{6144, 5120},
        std::array{7168, 5120}, std::array{12288, 5120}, std::array{14336, 5120},
        std::array{16384, 5120}, std::array{34816, 5120}, std::array{131072, 5120},
        std::array{248320, 5120}, std::array{5120, 6144}, std::array{5120, 17408}};
    std::uint32_t seed = 401;
    for (const auto& shape : shapes) {
        failures += run_shape("T2_A16 canonical", ActivationCompute::A16,
                              make_t2_g128_fp16_weight,
                              {shape[0], shape[1], seed++, Comparison::SampledRows, true, widths});
    }
    return failures;
}

// The ternary profile's problems across the decode, small-T and
// prefill widths.
int t2_a16_conformance() {
    int failures = 0;

    constexpr std::array kDecode{convenience(1), a16(1),   a16(2),   a16(3),   a16(4),
                                 a16(5),         a16(8),   a16(9),   a16(12),  a16(16),
                                 a16(17),        a16(18),  a16(32),  a16(33),  a16(64),
                                 a16(65),        a16(128), a16(129), a16(1024)};

    failures += run_shape("T2_A16", ActivationCompute::A16, make_t2_g128_fp16_weight,
                          {1024, 5120, 201U, Comparison::Full, true, kDecode});
    failures += run_shape("T2_A16", ActivationCompute::A16, make_t2_g128_fp16_weight,
                          {7168, 5120, 203U, Comparison::SampledRows, false, kDecode});
    failures += run_shape("T2_A16", ActivationCompute::A16, make_t2_g128_fp16_weight,
                          {34816, 5120, 205U, Comparison::SampledRows, false, kDecode});
    failures += run_shape("T2_A16", ActivationCompute::A16, make_t2_g128_fp16_weight,
                          {5120, 6144, 207U, Comparison::Full, false, kDecode});
    failures += run_shape("T2_A16", ActivationCompute::A16, make_t2_g128_fp16_weight,
                          {5120, 17408, 209U, Comparison::Full, false, kDecode});
    failures += run_shape("T2_A16", ActivationCompute::A16, make_t2_g128_fp16_weight,
                          {131072, 5120, 211U, Comparison::SampledRows, false, kDecode});
    failures += run_shape("T2_A16", ActivationCompute::A16, make_t2_g128_fp16_weight,
                          {248320, 5120, 213U, Comparison::SampledRows, false, kDecode});

    return failures;
}

} // namespace

int main(int argc, char** argv) {
    if (!ninfer::test::linear::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    try {
        int failures = 0;
        if (argc == 2 && std::string_view(argv[1]) == "--real-width-consistency-only") {
            failures = represented_width_consistency();
        } else if (argc == 2 && std::string_view(argv[1]) == "--root-width-consistency-only") {
            failures = root_width_consistency();
        } else if (argc == 2 && std::string_view(argv[1]) == "--canonical-a16-only") {
            failures = canonical_a16_conformance();
        } else if (argc == 2 && std::string_view(argv[1]) == "--small-t-publisher-only") {
            constexpr std::array widths{a16(1), a16(7), a16(16)};
            failures += run_shape("T2_A16", ActivationCompute::A16, make_t2_g128_fp16_weight,
                                  {1024, 5120, 201U, Comparison::Full, true, widths});
            failures += run_shape("T2_A16", ActivationCompute::A16, make_t2_g128_fp16_weight,
                                  {7168, 5120, 203U, Comparison::SampledRows, false, widths});
        } else {
            failures = represented_width_consistency() + root_width_consistency() +
                       represented_width_consistency(34816, false) +
                       represented_width_consistency(131072, false) + t2_a16_conformance();
        }
        std::cout << (failures == 0 ? "OK" : "FAIL") << " T2_A16 Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "T2_A16 Linear: " << error.what() << '\n';
        return 1;
    }
}
