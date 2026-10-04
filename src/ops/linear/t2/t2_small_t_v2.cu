#include "ops/linear/t2/t2_small_t_v2.cuh"

#include "core/device.h"
#include "ops/common/token_slices.h"
#include "ops/linear/t2/t2_launch.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

template <class Schedule>
void launch(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const std::int32_t rows = out.ne[0];
    const std::int32_t k    = w.padded_shape[1];
    const std::int32_t cols = x.ne[1];
    if ((rows % Schedule::kRows) != 0 || (k % Schedule::kSlabK) != 0 || k != x.ne[0] ||
        cols <= 0) {
        throw std::invalid_argument("t2 A16: unsupported shape");
    }
    for_each_token_slice(cols, Schedule::kColumns,
                         [&](std::int32_t offset, std::int32_t count) {
                             const Tensor input = x.slice(1, offset, count);
                             Tensor output      = out.slice(1, offset, count);
                             const dim3 grid(
                                 static_cast<unsigned>(rows / Schedule::kRows),
                                 static_cast<unsigned>((count + Schedule::kColumns - 1) /
                                                       Schedule::kColumns), 1u);
                             t2_small_t_v2_kernel<Schedule>
                                 <<<grid, Schedule::kThreads, 0, stream>>>(
                                     static_cast<const __nv_bfloat16*>(input.data),
                                     static_cast<const std::uint8_t*>(w.qdata),
                                     static_cast<const std::uint8_t*>(w.scales),
                                     static_cast<__nv_bfloat16*>(output.data), rows, k, count);
                             CUDA_CHECK(cudaGetLastError());
                         });
}

template <int ColumnTiles>
void launch_rows(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    // Both row tiles keep four K warps and the same per-group FMA and final reduction.
    if (out.ne[0] >= 131072) {
        launch<T2SmallTv2Schedule<4, 2, ColumnTiles, 2, 4>>(x, w, out, stream);
    } else {
        launch<T2SmallTv2Schedule<4, 1, ColumnTiles, 2, 5>>(x, w, out, stream);
    }
}

} // namespace

void launch_t2_small_t_v2(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    if (x.ne[1] <= 8) {
        launch_rows<1>(x, w, out, stream);
    } else {
        launch_rows<2>(x, w, out, stream);
    }
}

} // namespace ninfer::ops::detail
