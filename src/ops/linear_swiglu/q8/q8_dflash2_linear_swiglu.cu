#include "core/weight.h"
#include "ops/linear_swiglu/q8/q8_linear_swiglu_kernels.h"

#include "core/device.h"
#include "ops/linear/q8/q8_ksplit_config.h"
#include "ops/linear/q8/q8_rowsplit_output.cuh"
#include "ops/linear/q8/q8_rowsplit_gemm_mma.cuh"
#include "ops/linear/q8/q8_ksplit_mma.cuh"
#include "ops/linear_swiglu/q8/q8_linear_swiglu_output.cuh"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

using Geometry                       = Q8LinearGeometry<34816, 5120>;
constexpr std::int32_t kIntermediate = Geometry::kOutputRows / 2;
constexpr std::int32_t kFirstSmallT  = 1;
constexpr std::int32_t kLastSmallT   = 40;
using Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

template <int Capacity>
void launch_tile(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Schedule =
        Q8KSplitSchedule<Capacity == 24 ? 8 : 4, Capacity, 2, Q8KSplitScaleAccess::Shared>;
    using RowPolicy = Q8SwiGluPairedRows<kIntermediate>;
    static_assert((Geometry::kInputRows % Schedule::kGroupK) == 0);
    static_assert((kIntermediate % RowPolicy::kOutputRowsPerCta) == 0);

    const Q8ContiguousOutput ignored_output{static_cast<__nv_bfloat16*>(out.data), kIntermediate};
    const Q8SwiGluDirectEpilogue epilogue{static_cast<__nv_bfloat16*>(out.data), kIntermediate};
    const RowPolicy row_policy{};
    constexpr int kBlocks = kIntermediate / RowPolicy::kOutputRowsPerCta;
    q8_ksplit_mma_kernel<Geometry, Capacity, Schedule, Q8ContiguousOutput, Q8SwiGluDirectEpilogue,
                         RowPolicy, true, true><<<kBlocks, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), ignored_output, epilogue, row_policy,
        x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

template <int Columns>
void launch_sm89_occ_tile(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    static_assert(Columns == 8 || Columns == 16 || Columns == 32);
    using Schedule = Q8KSplitSchedule<4, Columns, 4, Q8KSplitScaleAccess::Shared>;
    using RowPolicy = Q8SwiGluPairedRows<kIntermediate>;
    const Q8ContiguousOutput ignored_output{static_cast<__nv_bfloat16*>(out.data), kIntermediate};
    const Q8SwiGluDirectEpilogue epilogue{static_cast<__nv_bfloat16*>(out.data), kIntermediate};
    constexpr int kBlocks = kIntermediate / RowPolicy::kOutputRowsPerCta;
    q8_ksplit_mma_kernel<Geometry, Columns, Schedule, Q8ContiguousOutput, Q8SwiGluDirectEpilogue,
                         RowPolicy, true, false>
        <<<kBlocks, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), ignored_output, epilogue, RowPolicy{},
            Columns);
    CUDA_CHECK(cudaGetLastError());
}

template <std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<Launch, sizeof...(Offsets)>{
        &launch_tile<8 * (1 + static_cast<int>(Offsets))>...};
}

constexpr auto kLaunchers = make_launchers(std::make_index_sequence<kLastSmallT / 8>{});

} // namespace

void q8_dflash2_linear_swiglu_small_t_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                             cudaStream_t stream) {
    if (x.ne[1] < kFirstSmallT || x.ne[1] > kLastSmallT) {
        throw std::invalid_argument("Q8 DFlash2 LinearSwiGLU small-T: unsupported T");
    }
    const std::size_t index = static_cast<std::size_t>((x.ne[1] - 1) / 8);
    kLaunchers[index](x, weight, out, stream);
}

void q8_dflash2_linear_swiglu_sm89_occ_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                               cudaStream_t stream) {
    switch (x.ne[1]) {
    case 8:
        launch_sm89_occ_tile<8>(x, weight, out, stream);
        return;
    case 16:
        launch_sm89_occ_tile<16>(x, weight, out, stream);
        return;
    case 32:
        launch_sm89_occ_tile<32>(x, weight, out, stream);
        return;
    case 64: {
        // The inherited T64 path asks ptxas for two CTAs/SM. Ada's 128 SMs can benefit from a
        // three-CTA register budget without changing the tile or arithmetic.
        using Schedule = Q8RowSplitMmaGemmSchedule<64, 64, 32, 16, 3, 2, 128, 1>;
        const Q8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]};
        const dim3 grid(out.ne[0] / (Schedule::BM / 2), 1u, 1u);
        q8_rowsplit_gemm_mma_kernel<Schedule, true, Q8Epilogue::SwiGluSplitHalf>
            <<<grid, Schedule::THREADS, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const std::uint8_t*>(weight.qdata),
                static_cast<const std::uint8_t*>(weight.scales), output, weight.n, weight.k, 64,
                weight.padded_shape[1]);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    default:
        throw std::invalid_argument("Q8 DFlash2 sm89 occupancy candidate requires T=8,16,32,64");
    }
}

} // namespace ninfer::ops::detail
