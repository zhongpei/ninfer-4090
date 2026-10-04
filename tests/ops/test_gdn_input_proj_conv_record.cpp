#include "core/weight.h"
#include "core/device.h"
#include "ninfer/ops/gdn_input_proj.h"

#include "ops/input_projection_test_common.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::input_projection;

namespace {

bool t1_only = false;
bool domain_only = false;
bool a8_oracle_negative_control = false;
bool t2_a16_b2_only = false;
bool policy_only = false;
bool t2_math_only = false;
constexpr std::int32_t kQueryRows = 2048;
constexpr std::int32_t kKeyRows   = 2048;
// Existing complete ConvSnapshot A16 / A4 and GDN input A8 qualification contracts.
constexpr ReductionCriterion kRecordA16{3.15e-3, 4.0e-3, kBf16GrossRelativeFloor};
constexpr ReductionCriterion kRecordA8{0.04, 1.0 / 256.0, 0.06};
constexpr ReductionCriterion kRecordA4{0.16, 4.0e-3, 0.16};

std::vector<std::uint16_t> make_bf16_bits(std::size_t elements, std::uint32_t seed, float low,
                                          float high) {
    std::vector<float> values(elements);
    fill_uniform(values, seed, low, high);
    round_to_bf16(values);
    return bf16_bits(values);
}

int verify_equal(std::string_view label, const std::vector<std::uint16_t>& lhs,
                 const std::vector<std::uint16_t>& rhs) {
    if (lhs.size() == rhs.size() && std::equal(lhs.begin(), lhs.end(), rhs.begin())) { return 0; }
    std::cerr << label << ": BF16 bits differ\n";
    return 1;
}

int verify_zero_tail(std::string_view label, const std::vector<std::uint16_t>& values,
                     std::int32_t rows, std::int32_t width, std::int32_t batch,
                     const std::vector<std::int32_t>& valid_columns) {
    for (std::int32_t batch_row = 0; batch_row < batch; ++batch_row) {
        for (std::int32_t token = valid_columns[static_cast<std::size_t>(batch_row)]; token < width;
             ++token) {
            const std::size_t base = static_cast<std::size_t>(batch_row * width + token) * rows;
            for (std::int32_t row = 0; row < rows; ++row) {
                if (values[base + row] != 0) {
                    std::cerr << label << ": invalid tail is not exact zero\n";
                    return 1;
                }
            }
        }
    }
    return 0;
}

int verify_conv_record(std::string_view label, const std::vector<std::uint16_t>& snapshot_state,
                       const std::vector<std::uint16_t>& record, std::int32_t channels,
                       std::int32_t width, std::int32_t batch,
                       const std::vector<std::int32_t>& valid_columns,
                       const std::vector<std::int32_t>& snapshot_bases) {
    const std::size_t slot_stride = static_cast<std::size_t>(channels) * 3;
    for (std::int32_t batch_row = 0; batch_row < batch; ++batch_row) {
        for (std::int32_t token = 0; token < valid_columns[static_cast<std::size_t>(batch_row)];
             ++token) {
            const std::size_t snapshot =
                static_cast<std::size_t>(snapshot_bases[static_cast<std::size_t>(batch_row)] +
                                         token) *
                    slot_stride +
                2ULL * channels;
            const std::size_t record_column =
                static_cast<std::size_t>(batch_row * width + token) * channels;
            if (!std::equal(snapshot_state.begin() + static_cast<std::ptrdiff_t>(snapshot),
                            snapshot_state.begin() +
                                static_cast<std::ptrdiff_t>(snapshot + channels),
                            record.begin() + static_cast<std::ptrdiff_t>(record_column))) {
                std::cerr << label << ": conv record differs from snapshot newest column\n";
                return 1;
            }
        }
    }
    return 0;
}

template <class SnapshotLaunch, class RecordLaunch>
int run_case(std::string_view label, std::int32_t hidden, std::int32_t value_rows,
             std::int32_t z_rows, std::int32_t width, std::int32_t batch,
             std::vector<std::int32_t> valid_columns, std::size_t snapshot_workspace_bytes,
             std::size_t record_workspace_bytes, SnapshotLaunch&& snapshot_launch,
             RecordLaunch&& record_launch, std::uint32_t seed,
             const quantized_weight::PackedWeight* first_weight = nullptr,
             const quantized_weight::PackedWeight* second_weight = nullptr,
             const ReductionCriterion& criterion = kRecordA16) {
    const std::int32_t channels          = kQueryRows + kKeyRows + value_rows;
    const std::int32_t aggregate_columns = width * batch;
    const std::int32_t source_slots      = 8;
    const std::int32_t slots             = aggregate_columns + source_slots;
    const bool snapshot_only             = width > 16;
    const bool dense                     = valid_columns.empty();
    if (valid_columns.empty()) { valid_columns.assign(static_cast<std::size_t>(batch), width); }

    const std::vector<float> activation = make_bf16_activation(hidden, aggregate_columns, seed);
    const std::vector<std::uint16_t> conv_weight_bits =
        make_bf16_bits(static_cast<std::size_t>(channels) * 4, seed + 1, -0.02F, 0.02F);
    const std::vector<std::uint16_t> state_before = make_bf16_bits(
        static_cast<std::size_t>(channels) * 3 * source_slots, seed + 2, -0.05F, 0.05F);

    std::vector<std::uint16_t> snapshot_before(static_cast<std::size_t>(channels) * 3 * slots);
    std::copy(state_before.begin(), state_before.end(),
              snapshot_before.begin() + static_cast<std::size_t>(channels) * 3 * aggregate_columns);
    std::vector<std::int32_t> snapshot_initial_slots(static_cast<std::size_t>(batch));
    std::vector<std::int32_t> initial_slots(static_cast<std::size_t>(batch));
    std::vector<std::int32_t> snapshot_bases(static_cast<std::size_t>(batch));
    for (std::int32_t batch_row = 0; batch_row < batch; ++batch_row) {
        snapshot_bases[static_cast<std::size_t>(batch_row)] = batch_row * width;
        // Read-only selectors may repeat; source capacity does not depend on W.
        initial_slots[static_cast<std::size_t>(batch_row)] =
            batch_row == 7 ? 7 : (batch_row * 3 + 7) % source_slots;
        snapshot_initial_slots[static_cast<std::size_t>(batch_row)] =
            aggregate_columns + initial_slots[static_cast<std::size_t>(batch_row)];
    }

    DeviceBuffer device_x           = to_device_bf16(activation);
    DeviceBuffer device_conv_weight = to_device(conv_weight_bits);
    DeviceBuffer snapshot_state     = to_device(snapshot_before);
    DeviceBuffer record_state       = to_device(state_before);
    DeviceBuffer device_valid;
    if (!dense) { device_valid = to_device(valid_columns); }
    DeviceBuffer device_initial          = to_device(initial_slots);
    DeviceBuffer device_snapshot_initial = to_device(snapshot_initial_slots);
    DeviceBuffer device_snapshot         = to_device(snapshot_bases);

    GuardedBf16Tensor snapshot_query(kQueryRows, aggregate_columns);
    GuardedBf16Tensor snapshot_key(kKeyRows, aggregate_columns);
    GuardedBf16Tensor snapshot_value(value_rows, aggregate_columns);
    GuardedBf16Tensor snapshot_z(z_rows, aggregate_columns);
    GuardedBf16Tensor record_query(kQueryRows, aggregate_columns);
    GuardedBf16Tensor record_key(kKeyRows, aggregate_columns);
    GuardedBf16Tensor record_value(value_rows, aggregate_columns);
    GuardedBf16Tensor record_z(z_rows, aggregate_columns);
    GuardedBf16Tensor conv_record(channels, aggregate_columns);

    Tensor x(device_x.p, DType::BF16, {hidden, width, batch});
    Tensor conv_weight(device_conv_weight.p, DType::BF16, {channels, 4});
    Tensor snapshot_state_view(snapshot_state.p, DType::BF16, {channels, 3, slots});
    Tensor record_state_view(record_state.p, DType::BF16, {channels, 3, source_slots});
    Tensor valid;
    if (!dense) { valid = Tensor(device_valid.p, DType::I32, {batch}); }
    Tensor initial(device_initial.p, DType::I32, {batch});
    Tensor snapshot_initial(device_snapshot_initial.p, DType::I32, {batch});
    Tensor snapshot_base(device_snapshot.p, DType::I32, {batch});
    Tensor snapshot_q(snapshot_query.data(), DType::BF16, {kQueryRows, width, batch});
    Tensor snapshot_k(snapshot_key.data(), DType::BF16, {kKeyRows, width, batch});
    Tensor snapshot_v(snapshot_value.data(), DType::BF16, {value_rows, width, batch});
    Tensor snapshot_z_view(snapshot_z.data(), DType::BF16, {z_rows, width, batch});
    Tensor record_q(record_query.data(), DType::BF16, {kQueryRows, width, batch});
    Tensor record_k(record_key.data(), DType::BF16, {kKeyRows, width, batch});
    Tensor record_v(record_value.data(), DType::BF16, {value_rows, width, batch});
    Tensor record_z_view(record_z.data(), DType::BF16, {z_rows, width, batch});
    Tensor conv_record_view(conv_record.data(), DType::BF16, {channels, width, batch});

    WorkspaceArena snapshot_workspace(std::max<std::size_t>(256, snapshot_workspace_bytes));
    WorkspaceArena record_workspace(std::max<std::size_t>(256, record_workspace_bytes));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cuda_synchronize();
    const auto launch = [&] {
        snapshot_launch(x, conv_weight, snapshot_state_view, valid, snapshot_initial, snapshot_base,
                        snapshot_q, snapshot_k, snapshot_v, snapshot_z_view, snapshot_workspace,
                        stream);
        if (!snapshot_only) {
        record_launch(x, conv_weight, record_state_view, valid, initial, conv_record_view, record_q,
                      record_k, record_v, record_z_view, record_workspace, stream);
        }
    };
    launch();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto activation_bits = bf16_bits(activation);
    if (width == 1 || width == 2 || width == 9 || width == 16 || snapshot_only) {
        cudaGraph_t graph;
        cudaGraphExec_t executable;
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        launch();
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphLaunch(executable, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        // The same graph must consume changed inputs at the captured addresses.
        activation_bits = bf16_bits(make_bf16_activation(hidden, aggregate_columns, seed + 19));
        CUDA_CHECK(cudaMemcpyAsync(device_x.p, activation_bits.data(),
                                   activation_bits.size() * sizeof(std::uint16_t),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaGraphLaunch(executable, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaGraphExecDestroy(executable));
        CUDA_CHECK(cudaGraphDestroy(graph));
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    if (a8_oracle_negative_control) {
        // A correlated defect is invisible to the snapshot/record comparison. Only an
        // independent mathematical oracle can reject these equally corrupted outputs.
        auto corrupted = record_z.bits();
        for (auto& bits : corrupted) { bits ^= 0x8000U; }
        CUDA_CHECK(cudaMemcpy(record_z_view.data, corrupted.data(), corrupted.size() * 2,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(snapshot_z_view.data, corrupted.data(), corrupted.size() * 2,
                              cudaMemcpyHostToDevice));
    }

    int failures = 0;
    if (!snapshot_only) {
    failures +=
        verify_equal(std::string(label) + " query", snapshot_query.bits(), record_query.bits());
    failures += verify_equal(std::string(label) + " key", snapshot_key.bits(), record_key.bits());
    failures +=
        verify_equal(std::string(label) + " value", snapshot_value.bits(), record_value.bits());
    failures += verify_equal(std::string(label) + " z", snapshot_z.bits(), record_z.bits());
    failures += verify_zero_tail(std::string(label) + " query", record_query.bits(), kQueryRows,
                                 width, batch, valid_columns);
    failures += verify_zero_tail(std::string(label) + " key", record_key.bits(), kKeyRows, width,
                                 batch, valid_columns);
    failures += verify_zero_tail(std::string(label) + " value", record_value.bits(), value_rows,
                                 width, batch, valid_columns);

    }

    const std::vector<std::uint16_t> snapshot_state_after =
        from_device<std::uint16_t>(snapshot_state, snapshot_before.size());
    const std::vector<std::uint16_t> record_state_after =
        from_device<std::uint16_t>(record_state, state_before.size());
    if (!snapshot_only) {
        failures += verify_conv_record(label, snapshot_state_after, conv_record.bits(), channels,
                                       width, batch, valid_columns, snapshot_bases);
    }
    if (first_weight != nullptr &&
        (width == 1 || first_weight->weight.qtype == QType::T2_G128_FP16)) {
        // Independently decode stored coefficients/scales and evaluate the complete dot in FP64.
        // BF16 record/output casts are assessed against this independent mathematical value.
        const auto record_bits = conv_record.bits();
        std::vector<float> represented_input(activation_bits.size());
        for (std::size_t i = 0; i < activation_bits.size(); ++i) {
            represented_input[i] = bf16_to_f32(activation_bits[i]);
        }
        const auto evaluate = [&](int offset, int rows, const GuardedBf16Tensor& output,
                                  bool convolve, std::string_view plane) {
            std::vector<double> actual, expected, actual_record, expected_record, cast_diagnostic;
            const auto output_bits = output.bits();
            for (int row : sampled_rows(rows, snapshot_only ? 8 : 32)) {
                const int global = offset + row;
                const bool second = second_weight != nullptr && global >= kQueryRows + kKeyRows;
                const auto& weight = second ? *second_weight : *first_weight;
                const int weight_row = second ? global - kQueryRows - kKeyRows : global;
                for (int b = 0; b < batch; ++b) {
                    const std::size_t initial = std::size_t(initial_slots[b]) * channels * 3;
                    double history[3]{};
                    if (convolve) {
                        for (int tap = 0; tap < 3; ++tap) {
                            history[tap] = bf16_to_f32(state_before[initial + tap * channels + global]);
                        }
                    }
                    std::vector<int> tokens;
                    if (snapshot_only) {
                        for (int t : {0, 1, 2, 3, 15, 16, 17, 49, 127, 128, width - 1}) {
                            if (t < valid_columns[b] &&
                                std::find(tokens.begin(), tokens.end(), t) == tokens.end()) {
                                tokens.push_back(t);
                            }
                        }
                    } else {
                        for (int t = 0; t < valid_columns[b]; ++t) { tokens.push_back(t); }
                    }
                    for (int token : tokens) {
                        if (snapshot_only && convolve) {
                            for (int tap = 0; tap < 3; ++tap) {
                                const int previous = token - 3 + tap;
                                if (previous < 0) {
                                    history[tap] = bf16_to_f32(state_before[
                                        initial + (previous + 3) * channels + global]);
                                } else {
                                    const double previous_p = quantized_weight::dot_fp64(
                                        weight, weight_row, represented_input.data() +
                                            (std::size_t(b) * width + previous) * hidden, hidden);
                                    history[tap] = bf16_to_f32(f32_to_bf16(float(previous_p)));
                                }
                            }
                        }
                        const auto column = std::size_t(b) * width + token;
                        const double projected = quantized_weight::dot_fp64(
                            weight, weight_row, represented_input.data() + column * hidden, hidden);
                        double value = projected;
                        double cast_value = value;
                        if (convolve) {
                            value = 0.0;
                            for (int tap = 0; tap < 3; ++tap) {
                                value += double(bf16_to_f32(conv_weight_bits[tap * channels + global])) *
                                         history[tap];
                            }
                            const double tap = bf16_to_f32(conv_weight_bits[3 * channels + global]);
                            cast_value = value + tap * bf16_to_f32(f32_to_bf16(float(projected)));
                            cast_value /= 1.0 + std::exp(-cast_value);
                            value += tap * projected;
                            value /= 1.0 + std::exp(-value);
                            const auto raw = snapshot_only
                                ? snapshot_state_after[(std::size_t(snapshot_bases[b]) + token) *
                                                           channels * 3 + 2 * channels + global]
                                : record_bits[column * channels + global];
                            actual_record.push_back(bf16_to_f32(raw));
                            expected_record.push_back(projected);
                            history[0] = history[1];
                            history[1] = history[2];
                            // Older columns are observable BF16 history, current p is unrounded.
                            history[2] = bf16_to_f32(f32_to_bf16(float(projected)));
                        }
                        actual.push_back(bf16_to_f32(output_bits[column * rows + row]));
                        expected.push_back(value);
                        cast_diagnostic.push_back(cast_value);
                    }
                }
            }
            const std::string prefix = std::string(label) + " " + std::string(plane);
            int count = compare(prefix + " FP64 output", actual, expected, criterion);
            if (convolve && error_stats_enabled()) {
                // Attribution only. The qualification above always uses the unrounded formula.
                const auto stats = compute_reduction_stats(actual.data(), cast_diagnostic.data(), actual.size());
                report_reduction_stats(prefix + " diagnostic projected BF16 cast", actual.size(), stats, criterion);
            }
            if (convolve) {
                count += compare(prefix + " FP64 record", actual_record,
                                 expected_record, criterion);
            }
            return count;
        };
        failures += evaluate(0, kQueryRows, snapshot_only ? snapshot_query : record_query, true, "query");
        failures += evaluate(kQueryRows, kKeyRows, snapshot_only ? snapshot_key : record_key, true, "key");
        failures += evaluate(kQueryRows + kKeyRows, value_rows, snapshot_only ? snapshot_value : record_value, true, "value");
        failures += evaluate(channels, z_rows, snapshot_only ? snapshot_z : record_z, false, "z");
    }

    if (first_weight != nullptr && first_weight->weight.qtype == QType::T2_G128_FP16 &&
        criterion.relative_l2 == kRecordA16.relative_l2 && width > 1) {
        DeviceBuffer split_state = to_device(snapshot_before);
        Tensor split_states(split_state.p, DType::BF16, {channels, 3, slots});
        GuardedBf16Tensor split_q(kQueryRows, aggregate_columns);
        GuardedBf16Tensor split_k(kKeyRows, aggregate_columns);
        GuardedBf16Tensor split_v(value_rows, aggregate_columns);
        GuardedBf16Tensor split_z(z_rows, aggregate_columns);
        DeviceBuffer split_initial(sizeof(std::int32_t));
        DeviceBuffer split_base(sizeof(std::int32_t));
        Tensor initial_one(split_initial.p, DType::I32, {1});
        Tensor base_one(split_base.p, DType::I32, {1});
        for (int b = 0; b < batch; ++b) {
            for (int token = 0; token < valid_columns[b]; ++token) {
                const std::int32_t source = token == 0 ? snapshot_initial_slots[b]
                                                        : snapshot_bases[b] + token - 1;
                const std::int32_t destination = snapshot_bases[b] + token;
                split_initial.copy_from_host(&source, sizeof(source));
                split_base.copy_from_host(&destination, sizeof(destination));
                const auto column = std::size_t(b) * width + token;
                Tensor x_one(static_cast<std::uint16_t*>(device_x.p) + column * hidden,
                             DType::BF16, {hidden, 1});
                Tensor q_one(static_cast<std::uint16_t*>(split_q.data()) + column * kQueryRows, DType::BF16, {kQueryRows, 1});
                Tensor k_one(static_cast<std::uint16_t*>(split_k.data()) + column * kKeyRows, DType::BF16, {kKeyRows, 1});
                Tensor v_one(static_cast<std::uint16_t*>(split_v.data()) + column * value_rows, DType::BF16, {value_rows, 1});
                Tensor z_one(static_cast<std::uint16_t*>(split_z.data()) + column * z_rows, DType::BF16, {z_rows, 1});
                snapshot_launch(x_one, conv_weight, split_states, Tensor{}, initial_one, base_one,
                                q_one, k_one, v_one, z_one, snapshot_workspace, nullptr);
            }
        }
        cuda_synchronize();
        const auto check_valid = [&](std::string_view plane, const auto& full, const auto& split,
                                     int rows) {
            auto a = full.bits();
            auto b = split.bits();
            for (int batch_row = 0; batch_row < batch; ++batch_row) {
                const auto start = std::size_t(batch_row) * width * rows;
                const auto count = std::size_t(valid_columns[batch_row]) * rows;
                if (!std::equal(a.begin() + start, a.begin() + start + count, b.begin() + start)) {
                    std::cerr << label << " split continuation " << plane << " bits differ\n";
                    return 1;
                }
            }
            return 0;
        };
        failures += check_valid("query", snapshot_query, split_q, kQueryRows);
        failures += check_valid("key", snapshot_key, split_k, kKeyRows);
        failures += check_valid("value", snapshot_value, split_v, value_rows);
        failures += check_valid("z", snapshot_z, split_z, z_rows);
        const auto split_state_bits = from_device<std::uint16_t>(split_state, snapshot_before.size());
        for (int b = 0; b < batch; ++b) {
            const auto start = std::size_t(snapshot_bases[b]) * channels * 3;
            const auto count = std::size_t(valid_columns[b]) * channels * 3;
            failures += verify_equal(std::string(label) + " split BF16 history",
                {snapshot_state_after.begin() + start, snapshot_state_after.begin() + start + count},
                {split_state_bits.begin() + start, split_state_bits.begin() + start + count});
        }
        failures += split_q.verify_guards(std::string(label) + " split query");
        failures += split_k.verify_guards(std::string(label) + " split key");
        failures += split_v.verify_guards(std::string(label) + " split value");
        failures += split_z.verify_guards(std::string(label) + " split z");
    }

    failures +=
        verify_equal(std::string(label) + " source state", state_before, record_state_after);

    failures += verify_preserved(std::string(label) + " x", device_x, activation_bits);
    failures +=
        verify_preserved(std::string(label) + " conv weight", device_conv_weight, conv_weight_bits);
    failures += verify_preserved(std::string(label) + " initial", device_initial, initial_slots);
    if (!dense) {
        failures += verify_preserved(std::string(label) + " valid", device_valid, valid_columns);
    }
    failures += snapshot_query.verify_guards(std::string(label) + " snapshot query");
    failures += snapshot_key.verify_guards(std::string(label) + " snapshot key");
    failures += snapshot_value.verify_guards(std::string(label) + " snapshot value");
    failures += snapshot_z.verify_guards(std::string(label) + " snapshot z");
    failures += record_query.verify_guards(std::string(label) + " record query");
    failures += record_key.verify_guards(std::string(label) + " record key");
    failures += record_value.verify_guards(std::string(label) + " record value");
    failures += record_z.verify_guards(std::string(label) + " record z");
    failures += conv_record.verify_guards(std::string(label) + " conv record");
    if (snapshot_workspace.used() != 0 ||
        snapshot_workspace.peak_used() != snapshot_workspace_bytes) {
        std::cerr << label << ": snapshot workspace query/execution mismatch\n";
        ++failures;
    }
    if (record_workspace.used() != 0 || record_workspace.peak_used() != record_workspace_bytes) {
        std::cerr << label << ": record workspace query/execution mismatch\n";
        ++failures;
    }
    return failures;
}

std::vector<std::int32_t> ragged(std::int32_t width, std::int32_t batch) {
    std::vector<std::int32_t> valid(batch);
    for (int b = 0; b < batch; ++b) valid[b] = b == 0 ? width : 1 + (3 * b) % width;
    return valid;
}

int run_q4_q5() {
    constexpr std::int32_t kHidden    = 5120;
    constexpr std::int32_t kValueRows = 6144;
    constexpr std::int32_t kZRows     = 6144;
    DevicePackedWeight qk(
        quantized_weight::make_patterned_weight(QType::Q4_G64_FP16, 4096, kHidden, 1401U));
    DevicePackedWeight value_z(
        quantized_weight::make_patterned_weight(QType::Q5_G64_FP16, 12288, kHidden, 1403U));

    int failures   = 0;
    const auto run = [&](std::int32_t width, std::int32_t batch, std::vector<std::int32_t> valid,
                         std::uint32_t seed) {
        const std::size_t snapshot_bytes =
            ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
                kQueryRows, kKeyRows, kValueRows, batch, width, width);
        const std::size_t record_bytes = ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
            kQueryRows, kKeyRows, kValueRows, batch, width, width);
        return run_case(
            "Q4/Q5 B=" + std::to_string(batch) + " T=" + std::to_string(width), kHidden, kValueRows,
            kZRows, width, batch, std::move(valid), snapshot_bytes, record_bytes,
            [&](const Tensor& x, const Tensor& conv, Tensor& state, const Tensor& valid_columns,
                const Tensor& initial, const Tensor& snapshot_base, Tensor& q, Tensor& k, Tensor& v,
                Tensor& z, WorkspaceArena& workspace, cudaStream_t stream) {
                ops::gdn_input_proj_conv_snapshot(x, qk.view(), value_z.view(), conv, state,
                                                  valid_columns, initial, snapshot_base, q, k, v, z,
                                                  workspace, stream);
            },
            [&](const Tensor& x, const Tensor& conv, const Tensor& state,
                const Tensor& valid_columns, const Tensor& initial, Tensor& record, Tensor& q,
                Tensor& k, Tensor& v, Tensor& z, WorkspaceArena& workspace, cudaStream_t stream) {
                ops::gdn_input_proj_conv_record(x, qk.view(), value_z.view(), conv, state,
                                                valid_columns, initial, record, q, k, v, z,
                                                workspace, stream);
            },
            seed, &qk.host, &value_z.host);
    };
    for (int batch = 1; batch <= 8; ++batch) { failures += run(1, batch, {}, 1900U + batch); }
    if (t1_only) { return failures; }
    for (int width = 2; width <= 16; ++width) {
        if (domain_only && width != 2 && width != 16) { continue; }
        failures += run(width, 1, {}, 1400U + width);
        failures += run(width, 8, ragged(width, 8), 1450U + width);
    }
    failures += run(5, 3, {5, 3, 1}, 1491U);
    failures += run(4, 4, {4, 3, 2, 1}, 1492U);
    failures += qk.verify_preserved("Q4 record qk weight");
    failures += value_z.verify_preserved("Q5 record value/z weight");
    return failures;
}

// The ternary pair under both policies: A16 and the integer-activation routes, whose records must
// stay bit-identical to a snapshot run under the same policy.
int run_t2() {
    constexpr std::int32_t kHidden    = 5120;
    constexpr std::int32_t kValueRows = 6144;
    constexpr std::int32_t kZRows     = 6144;
    DevicePackedWeight qk(
        quantized_weight::make_patterned_weight(QType::T2_G128_FP16, 4096, kHidden, 1411U));
    DevicePackedWeight value_z(
        quantized_weight::make_patterned_weight(QType::T2_G128_FP16, 12288, kHidden, 1413U));

    int failures   = 0;
    const auto run = [&](ops::LinearPolicy policy, std::int32_t width, std::int32_t batch,
                         std::vector<std::int32_t> valid, std::uint32_t seed) {
        const std::size_t snapshot_bytes =
            ops::gdn_input_proj_split_conv_snapshot_workspace_capacity_bytes(
                QType::T2_G128_FP16, QType::T2_G128_FP16, policy, batch, width, width);
        const std::size_t record_bytes = width > 16 ? 0 :
            ops::gdn_input_proj_split_conv_record_workspace_capacity_bytes(
                QType::T2_G128_FP16, QType::T2_G128_FP16, policy, batch, width, width);
        const std::string tag = policy == ops::LinearPolicy::A16Only ? "T2 A16" : "T2 A8I";
        return run_case(
            tag + " B=" + std::to_string(batch) + " T=" + std::to_string(width), kHidden,
            kValueRows, kZRows, width, batch, std::move(valid), snapshot_bytes, record_bytes,
            [&](const Tensor& x, const Tensor& conv, Tensor& state, const Tensor& valid_columns,
                const Tensor& initial, const Tensor& snapshot_base, Tensor& q, Tensor& k, Tensor& v,
                Tensor& z, WorkspaceArena& workspace, cudaStream_t stream) {
                ops::gdn_input_proj_conv_snapshot(x, qk.view(), value_z.view(), conv, state,
                                                  valid_columns, initial, snapshot_base, q, k, v, z,
                                                  policy, workspace, stream);
            },
            [&](const Tensor& x, const Tensor& conv, const Tensor& state,
                const Tensor& valid_columns, const Tensor& initial, Tensor& record, Tensor& q,
                Tensor& k, Tensor& v, Tensor& z, WorkspaceArena& workspace, cudaStream_t stream) {
                ops::gdn_input_proj_conv_record(x, qk.view(), value_z.view(), conv, state,
                                                valid_columns, initial, record, q, k, v, z, policy,
                                                workspace, stream);
            },
            seed, &qk.host, &value_z.host,
            policy == ops::LinearPolicy::AllowA8Int ? kRecordA8 : kRecordA16);
    };
    for (int batch = 1; batch <= 8; ++batch) {
        if (t2_a16_b2_only && batch != 2) { continue; }
        if (!a8_oracle_negative_control && !policy_only) {
            failures += run(ops::LinearPolicy::A16Only, 1, batch, {}, 1920U + batch);
        }
        if (!t2_a16_b2_only) {
            failures += run(ops::LinearPolicy::AllowA8Int, 1, batch, {}, 1930U + batch);
        }
    }
    if (t1_only) { return failures; }
    for (auto policy : {ops::LinearPolicy::A16Only, ops::LinearPolicy::AllowA8Int}) {
        for (int width : {2, 4, 7, 8, 16}) {
            failures += run(policy, width, 1, {}, 1420U + width);
            failures += run(policy, width, 8, ragged(width, 8), 1460U + width);
        }
        failures += run(policy, 5, 3, {5, 3, 1}, 1493U);
    }
    for (int width : {50, 1024}) {
        failures += run(ops::LinearPolicy::A16Only, width, 1, {}, 1940U + width);
    }
    failures += qk.verify_preserved("T2 record qk weight");
    failures += value_z.verify_preserved("T2 record value/z weight");
    return failures;
}

int run_q8() {
    constexpr std::int32_t kHidden    = 2048;
    constexpr std::int32_t kValueRows = 4096;
    constexpr std::int32_t kZRows     = 4096;
    DevicePackedWeight parent(
        quantized_weight::make_patterned_weight(QType::Q8_G32_FP16, 12288, kHidden, 1501U));

    int failures   = 0;
    const auto run = [&](std::int32_t width, std::int32_t batch, std::vector<std::int32_t> valid,
                         std::uint32_t seed) {
        const std::size_t snapshot_bytes =
            ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
                kQueryRows, kKeyRows, kValueRows, batch, width, width);
        const std::size_t record_bytes = ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
            kQueryRows, kKeyRows, kValueRows, batch, width, width);
        return run_case(
            "Q8 B=" + std::to_string(batch) + " T=" + std::to_string(width), kHidden, kValueRows,
            kZRows, width, batch, std::move(valid), snapshot_bytes, record_bytes,
            [&](const Tensor& x, const Tensor& conv, Tensor& state, const Tensor& valid_columns,
                const Tensor& initial, const Tensor& snapshot_base, Tensor& q, Tensor& k, Tensor& v,
                Tensor& z, WorkspaceArena& workspace, cudaStream_t stream) {
                ops::gdn_input_proj_conv_snapshot(x, parent.view(), conv, state, valid_columns,
                                                  initial, snapshot_base, q, k, v, z, workspace,
                                                  stream);
            },
            [&](const Tensor& x, const Tensor& conv, const Tensor& state,
                const Tensor& valid_columns, const Tensor& initial, Tensor& record, Tensor& q,
                Tensor& k, Tensor& v, Tensor& z, WorkspaceArena& workspace, cudaStream_t stream) {
                ops::gdn_input_proj_conv_record(x, parent.view(), conv, state, valid_columns,
                                                initial, record, q, k, v, z, workspace, stream);
            },
            seed, &parent.host);
    };
    failures += run(1, 1, {}, 1941U);
    failures += run(1, 8, {}, 1948U);
    if (t1_only) { return failures; }
    failures += run(2, 1, {1}, 1511U);
    failures += run(16, 1, {}, 1521U);
    failures += run(16, 8, {16, 13, 9, 7, 5, 3, 2, 1}, 1531U);
    failures += parent.verify_preserved("Q8 record parent weight");
    return failures;
}

int run_nvfp4() {
    constexpr std::int32_t kHidden    = 5120;
    constexpr std::int32_t kValueRows = 6144;
    constexpr std::int32_t kZRows     = 6144;
    constexpr std::int32_t kRows      = 16384;
    quantized_weight::PatternedWeightOptions options;
    options.weight_scale_divisor = 0.125F;
    options.input_scale_divisor  = 3.5F;
    DevicePackedWeight parent(
        quantized_weight::make_patterned_weight(QType::NVFP4, kRows, kHidden, 1601U, options));

    int failures   = 0;
    const auto run = [&](std::int32_t width, std::int32_t batch, std::vector<std::int32_t> valid,
                         ops::LinearPolicy policy, std::uint32_t seed) {
        const std::size_t snapshot_bytes =
            ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(QType::NVFP4, kRows, kHidden,
                                                                       policy, batch, width, width);
        const std::size_t record_bytes = ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
            QType::NVFP4, kRows, kHidden, policy, batch, width, width);
        return run_case(
            std::string("NVFP4 ") + (policy == ops::LinearPolicy::AllowA4 ? "A4" : "A16") +
                " B=" + std::to_string(batch) + " T=" + std::to_string(width),
            kHidden, kValueRows, kZRows, width, batch, std::move(valid), snapshot_bytes,
            record_bytes,
            [&](const Tensor& x, const Tensor& conv, Tensor& state, const Tensor& valid_columns,
                const Tensor& initial, const Tensor& snapshot_base, Tensor& q, Tensor& k, Tensor& v,
                Tensor& z, WorkspaceArena& workspace, cudaStream_t stream) {
                ops::gdn_input_proj_conv_snapshot(x, parent.view(), conv, state, valid_columns,
                                                  initial, snapshot_base, q, k, v, z, policy,
                                                  workspace, stream);
            },
            [&](const Tensor& x, const Tensor& conv, const Tensor& state,
                const Tensor& valid_columns, const Tensor& initial, Tensor& record, Tensor& q,
                Tensor& k, Tensor& v, Tensor& z, WorkspaceArena& workspace, cudaStream_t stream) {
                ops::gdn_input_proj_conv_record(x, parent.view(), conv, state, valid_columns,
                                                initial, record, q, k, v, z, policy, workspace,
                                                stream);
            },
            seed, &parent.host, nullptr,
            policy == ops::LinearPolicy::AllowA4 && batch > 1 ? kRecordA4 : kRecordA16);
    };
    if (!policy_only) {
        failures += run(1, 1, {}, ops::LinearPolicy::A16Only, 1951U);
        failures += run(1, 8, {}, ops::LinearPolicy::A16Only, 1958U);
    }
    for (int batch : {1, 8}) {
        failures += run_case_allowing_arch_skip("NVFP4 AllowA4 T1 B=" + std::to_string(batch), [&] {
            return run(1, batch, {}, ops::LinearPolicy::AllowA4, 1970U + batch);
        });
    }
    if (t1_only) { return failures; }
    // AllowA4 has no route on sm_86; the wrapper skips only genuine engine refusals, so the
    // A16 half of this sweep is still checked in full.
    for (auto policy : {ops::LinearPolicy::A16Only, ops::LinearPolicy::AllowA4}) {
        const std::string tag = policy == ops::LinearPolicy::A16Only
                                    ? "gdn_input_proj_conv_record NVFP4 A16"
                                    : "gdn_input_proj_conv_record NVFP4 A4";
        for (int width = 2; width <= 16; ++width) {
            if (domain_only && width != 2 && width != 16) { continue; }
            failures += run_case_allowing_arch_skip(tag + " W=" + std::to_string(width), [&] {
                return run(width, 1, {}, policy, 1600U + width);
            });
            failures += run_case_allowing_arch_skip(tag + " ragged W=" + std::to_string(width), [&] {
                return run(width, 8, ragged(width, 8), policy, 1650U + width);
            });
        }
    }
    failures += parent.verify_preserved("NVFP4 record parent weight");
    return failures;
}

int run_fp8_case(DevicePackedWeight& parent, std::int32_t width, std::int32_t batch,
                 std::vector<std::int32_t> valid, ops::LinearPolicy policy, std::uint32_t seed) {
    const std::size_t snapshot_bytes = ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
        QType::FP8_E4M3FN_ROW_BF16, 16384, 5120, policy, batch, width, width);
    const std::size_t record_bytes = ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
        QType::FP8_E4M3FN_ROW_BF16, 16384, 5120, policy, batch, width, width);
    return run_case(
        "FP8 policy=" + std::to_string(static_cast<int>(policy)) + " B=" + std::to_string(batch) +
            " W=" + std::to_string(width),
        5120, 6144, 6144, width, batch, std::move(valid), snapshot_bytes, record_bytes,
        [&](const Tensor& x, const Tensor& conv, Tensor& state, const Tensor& valid_columns,
            const Tensor& initial, const Tensor& snapshot_base, Tensor& q, Tensor& k, Tensor& v,
            Tensor& z, WorkspaceArena& workspace, cudaStream_t stream) {
            ops::gdn_input_proj_conv_snapshot(x, parent.view(), conv, state, valid_columns, initial,
                                              snapshot_base, q, k, v, z, policy, workspace, stream);
        },
        [&](const Tensor& x, const Tensor& conv, const Tensor& state, const Tensor& valid_columns,
            const Tensor& initial, Tensor& record, Tensor& q, Tensor& k, Tensor& v, Tensor& z,
            WorkspaceArena& workspace, cudaStream_t stream) {
            ops::gdn_input_proj_conv_record(x, parent.view(), conv, state, valid_columns, initial,
                                            record, q, k, v, z, policy, workspace, stream);
        },
        seed, &parent.host);
}

int run_fp8() {
    constexpr std::int32_t kHidden = 5120;
    constexpr std::int32_t kRows   = 16384;
    DevicePackedWeight parent(
        quantized_weight::make_patterned_weight(QType::FP8_E4M3FN_ROW_BF16, kRows, kHidden, 1701U));
    int failures = 0;
    if (!policy_only) {
        failures += run_fp8_case(parent, 1, 1, {}, ops::LinearPolicy::A16Only, 1961U);
        failures += run_fp8_case(parent, 1, 8, {}, ops::LinearPolicy::A16Only, 1968U);
    }
    // At T1 and B<=8 the existing FP8 AllowA8 planner selects A16 (A8 begins at B*T>=9).
    for (int batch : {1, 8}) {
        failures += run_case_allowing_arch_skip("FP8 AllowA8 T1 B=" + std::to_string(batch), [&] {
            return run_fp8_case(parent, 1, batch, {}, ops::LinearPolicy::AllowA8, 1980U + batch);
        });
    }
    if (t1_only) { return failures; }
    for (auto policy : {ops::LinearPolicy::A16Only, ops::LinearPolicy::AllowA8}) {
        const std::string tag = policy == ops::LinearPolicy::A16Only
                                    ? "gdn_input_proj_conv_record FP8 A16"
                                    : "gdn_input_proj_conv_record FP8 A8";
        for (int width = 2; width <= 16; ++width) {
            if (domain_only && width != 2 && width != 16) { continue; }
            failures += run_case_allowing_arch_skip(tag + " W=" + std::to_string(width), [&] {
                return run_fp8_case(parent, width, 1, {}, policy, 1700U + width);
            });
            failures += run_case_allowing_arch_skip(tag + " ragged W=" + std::to_string(width), [&] {
                return run_fp8_case(parent, width, 8, ragged(width, 8), policy, 1750U + width);
            });
        }
        for (int batch : {2, 3, 4}) {
            failures += run_case_allowing_arch_skip(tag + " B=" + std::to_string(batch), [&] {
                return run_fp8_case(parent, 4, batch, ragged(4, batch), policy, 1810U + batch);
            });
        }
    }
    failures += parent.verify_preserved("FP8 record parent weight");
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    a8_oracle_negative_control = argc == 2 &&
        std::string_view(argv[1]) == "--t1-a8-oracle-negative-control";
    t2_a16_b2_only = argc == 2 && std::string_view(argv[1]) == "--t1-t2-a16-b2-oracle-only";
    policy_only = argc == 2 && std::string_view(argv[1]) == "--t1-policy-oracle-only";
    t2_math_only = argc == 2 && std::string_view(argv[1]) == "--t2-math-regression-only";
    t1_only = a8_oracle_negative_control || t2_a16_b2_only || policy_only ||
        (argc == 2 && std::string_view(argv[1]) == "--t1-oracle-only");
    domain_only = argc == 2 && std::string_view(argv[1]) == "--record-domain-regression-only";
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    try {
        failures += run_t2();
        if (!a8_oracle_negative_control && !t2_a16_b2_only && !t2_math_only) {
            if (!policy_only) {
                failures += run_q4_q5();
                failures += run_q8();
            }
            failures += run_nvfp4();
            failures += run_fp8();
        }
    } catch (const std::exception& error) {
        std::cerr << "FAIL record oracle: " << error.what() << '\n';
        return 1;
    }
    std::cout << (failures == 0 ? "OK" : "FAIL") << " gdn_input_proj_conv_record\n";
    return failures == 0 ? 0 : 1;
}
