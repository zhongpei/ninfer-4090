#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "models/qwen3_5/program/planning/graph_profiles.h"
#include "ninfer/ops/softmax_attention.h"
#include "ops/op_check.h"

#include <cuda_fp16.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
namespace {
constexpr std::uint32_t capacity = 4096;
constexpr auto storage = KvCacheStorage::Fp8E4M3Row256;

// Independent decoding of the stored E4M3FN byte and FP16 scale, not a kernel comparison.
double fp8(std::uint8_t bits) {
    const int exponent = (bits >> 3) & 15;
    const int mantissa = bits & 7;
    const double magnitude = exponent == 0 ? std::ldexp(double(mantissa), -9) :
        std::ldexp(1.0 + double(mantissa) / 8.0, exponent - 7);
    return (bits & 128) ? -magnitude : magnitude;
}

void check_family(std::int32_t heads, std::int32_t batch, std::uint32_t drafts,
                  SpeculativeBackend backend) {
    const std::int32_t kv_heads = heads == 24 ? 4 : 2;
    const std::int32_t width = backend == SpeculativeBackend::None ? 1 : drafts + 1;
    const std::int32_t pages_per_row = capacity / kPagedKVPageSize;
    const std::int32_t pages = pages_per_row * batch;
    DeviceContext device(0);
    DeviceArena arena(64ULL << 20);
    auto q = arena.alloc(DType::BF16, {256, heads, width, batch});
    auto k = arena.alloc(DType::BF16, {256, kv_heads, width, batch});
    auto v = arena.alloc(DType::BF16, {256, kv_heads, width, batch});
    auto positions = arena.alloc(DType::I32, {width, batch});
    auto valid = arena.alloc(DType::I32, {batch});
    auto rows = arena.alloc(DType::I32, {batch});
    auto out = arena.alloc(DType::BF16, {256, heads, width, batch});
    auto ar_out = arena.alloc(DType::BF16, {256, heads, 1, batch});
    auto ar_positions = arena.alloc(DType::I32, {1, batch});
    auto ar_valid = arena.alloc(DType::I32, {batch});
    PagedKVBatchLayerView cache{
        arena.alloc(DType::FP8_E4M3FN, {256, 64, kv_heads, pages}),
        arena.alloc(DType::FP8_E4M3FN, {256, 64, kv_heads, pages}),
        arena.alloc(DType::FP16, {1, 64, kv_heads, pages}),
        arena.alloc(DType::FP16, {1, 64, kv_heads, pages}),
        arena.alloc(DType::I32, {pages_per_row, batch}), 256, kv_heads, storage};
    CUDA_CHECK(cudaMemset(q.data, 0, q.bytes()));
    CUDA_CHECK(cudaMemset(k.data, 0, k.bytes()));
    CUDA_CHECK(cudaMemset(cache.k_pages.data, 0, cache.k_pages.bytes()));
    CUDA_CHECK(cudaMemset(cache.v_pages.data, 0x38, cache.v_pages.bytes()));
    std::vector<std::uint16_t> values(v.numel(), 0x3e80); // represented BF16 0.25
    CUDA_CHECK(cudaMemcpy(v.data, values.data(), v.bytes(), cudaMemcpyHostToDevice));
    std::vector<std::uint16_t> scales(cache.k_scale_pages.numel(), 0x3400); // FP16 0.25
    CUDA_CHECK(cudaMemcpy(cache.k_scale_pages.data, scales.data(), cache.k_scale_pages.bytes(), cudaMemcpyHostToDevice));
    // Different physical rows make page IDs and row identity observable during replay.
    for (std::int32_t row = 0; row < batch; ++row) {
        std::fill(scales.begin() + row * pages_per_row * 64 * kv_heads,
                  scales.begin() + (row + 1) * pages_per_row * 64 * kv_heads,
                  row == 0 ? 0x3400 : 0x3800);
    }
    CUDA_CHECK(cudaMemcpy(cache.v_scale_pages.data, scales.data(), cache.v_scale_pages.bytes(), cudaMemcpyHostToDevice));
    std::vector<std::int32_t> tables(pages);
    for (std::int32_t page = 0; page < pages; ++page) { tables[page] = page; }
    CUDA_CHECK(cudaMemcpy(cache.block_tables.data, tables.data(), cache.block_tables.bytes(), cudaMemcpyHostToDevice));
    std::vector<std::int32_t> host_rows(batch), host_valid(batch, width), host_positions(width * batch);
    CUDA_CHECK(cudaMemcpy(valid.data, host_valid.data(), valid.bytes(), cudaMemcpyHostToDevice));
    WorkspaceArena workspace(ops::causal_softmax_attention_workspace_capacity_bytes(
        {256, heads, kv_heads}, storage, {1, capacity}, batch, 1, width));
    std::vector<std::int32_t> ar_counts(batch, 1), ar_host_positions(batch);
    CUDA_CHECK(cudaMemcpy(ar_valid.data, ar_counts.data(), ar_valid.bytes(), cudaMemcpyHostToDevice));
    using namespace models::qwen3_5::detail;
    const auto profiles = backend == SpeculativeBackend::None ?
        ordinary_graph_profiles(capacity, heads, storage, batch) :
        backend == SpeculativeBackend::Mtp ? mtp_graph_profiles(capacity, drafts, heads, storage, batch) :
        dflash_graph_profiles(backend, capacity, drafts, batch, heads, storage);
    std::map<std::uint32_t, DecodeGraphExecutable> executables;
    for (const auto& profile : profiles) {
        const auto visible = [&](std::uint32_t offset) {
            return static_cast<std::uint32_t>(std::min<std::uint64_t>(capacity,
                std::uint64_t(profile.max) + offset));
        };
        const auto launch = [&] {
            ops::causal_softmax_attention(q, k, v, positions, valid, rows, {256, heads, kv_heads},
                0.0625f, cache, {1, visible(width)}, workspace, out, device.stream);
            if (backend == SpeculativeBackend::Mtp) {
                for (std::uint32_t step = 0; step + 1 < drafts; ++step) {
                    auto qa = Tensor(q.data, DType::BF16, {256, heads, 1, batch});
                    auto ka = Tensor(k.data, DType::BF16, {256, kv_heads, 1, batch});
                    auto va = Tensor(v.data, DType::BF16, {256, kv_heads, 1, batch});
                    ops::causal_softmax_attention(qa, ka, va, ar_positions, ar_valid, rows,
                        {256, heads, kv_heads}, 0.0625f, cache,
                        {1, visible(drafts + step + 2)}, workspace, ar_out, device.stream);
                }
            }
        };
        for (int replay = 0; replay < 2; ++replay) {
            const std::int32_t base = replay == 0 ? 0 : visible(width) - width;
            for (std::int32_t row = 0; row < batch; ++row) {
                host_rows[row] = replay == 0 ? row : batch - row - 1;
                ar_host_positions[row] = base;
                for (std::int32_t col = 0; col < width; ++col) {
                    host_positions[row * width + col] = base + col;
                }
            }
            CUDA_CHECK(cudaMemcpy(rows.data, host_rows.data(), rows.bytes(), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(positions.data, host_positions.data(), positions.bytes(), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(ar_positions.data, ar_host_positions.data(), ar_positions.bytes(), cudaMemcpyHostToDevice));
            if (replay == 0) {
                launch();
                device.synchronize();
                DecodeGraphDefinition definition;
                definition.capture(device.stream, launch);
                auto [entry, inserted] = executables.try_emplace(profile.topology_class);
                if (inserted) { entry->second.instantiate(definition); }
                else {
                    try { entry->second.update(definition); }
                    catch (const std::exception& error) {
                        throw std::runtime_error("heads=" + std::to_string(heads) +
                            " B=" + std::to_string(batch) + " K=" + std::to_string(drafts) +
                            " frontier=" + std::to_string(profile.min) + ".." +
                            std::to_string(profile.max) + ": " + error.what());
                    }
                }
            }
            executables.at(profile.topology_class).launch(device.stream);
            device.synchronize();
            std::vector<std::uint8_t> codes(cache.v_pages.bytes());
            std::vector<__half> decoded_scales(cache.v_scale_pages.numel());
            std::vector<std::uint16_t> actual(out.numel()), ar_actual(ar_out.numel());
            CUDA_CHECK(cudaMemcpy(codes.data(), cache.v_pages.data, codes.size(), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(decoded_scales.data(), cache.v_scale_pages.data, cache.v_scale_pages.bytes(), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(actual.data(), out.data, out.bytes(), cudaMemcpyDeviceToHost));
            if (backend == SpeculativeBackend::Mtp) {
                CUDA_CHECK(cudaMemcpy(ar_actual.data(), ar_out.data, ar_out.bytes(), cudaMemcpyDeviceToHost));
            }
            const auto target_count = actual.size();
            const auto ar_count = backend == SpeculativeBackend::Mtp ? ar_actual.size() : 0;
            std::vector<double> got(target_count + ar_count), expected(got.size());
            for (int row = 0; row < batch; ++row) for (int col = 0; col < width; ++col)
            for (int head = 0; head < heads; ++head) {
                const int kv_head = head / (heads / kv_heads);
                double reference = 0;
                const int last = host_positions[row * width + col];
                for (int token = 0; token <= last; ++token) {
                    const int page = host_rows[row] * pages_per_row + token / 64;
                    const auto scale_index = (page * kv_heads + kv_head) * 64 + token % 64;
                    reference += fp8(codes[scale_index * 256]) *
                        double(__half2float(decoded_scales[scale_index]));
                }
                reference /= last + 1;
                // All represented V components are identical in this fixture.
                for (int dim = 0; dim < 256; ++dim) {
                    const auto index = ((row * width + col) * heads + head) * 256 + dim;
                    got[index] = std::bit_cast<float>(std::uint32_t(actual[index]) << 16);
                    expected[index] = reference;
                    if (backend == SpeculativeBackend::Mtp && col == 0) {
                        const auto ar_index = (row * heads + head) * 256 + dim;
                        got[target_count + ar_index] =
                            std::bit_cast<float>(std::uint32_t(ar_actual[ar_index]) << 16);
                        expected[target_count + ar_index] = reference;
                    }
                }
            }
            // The same FP8 criterion used by the attention Op qualification suite.
            const ninfer::test::ReductionCriterion criterion{1.2e-2, 4.0e-3, 9.0e-3};
            const auto stats = ninfer::test::compute_reduction_stats(got.data(), expected.data(), got.size());
            if (!ninfer::test::reduction_passes(stats, got.size(), criterion)) {
                throw std::runtime_error("FP8 graph replay disagrees with independent uniform-attention oracle");
            }
        }
    }
}
} // namespace

int main() {
    int count = 0;
    const auto status = cudaGetDeviceCount(&count);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver || (status == cudaSuccess && count == 0)) { return 77; }
    try {
        for (const int heads : {16, 24}) for (const int batch : {1, 2}) {
            check_family(heads, batch, 3, SpeculativeBackend::Mtp);
            check_family(heads, batch, 0, SpeculativeBackend::None);
            check_family(heads, batch, 7, SpeculativeBackend::DFlash);
        }
        std::cout << "FP8 attention graph profiles update and replay against independent oracle\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
