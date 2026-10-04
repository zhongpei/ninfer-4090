#include "ops/linear/t2/t2_dispatch.h"

#include <stdexcept>

namespace ninfer::ops::detail {

// All registered A16 problems share the same K partition and scale accumulation, including
// decode and prefill. Column and row tiling may vary without changing a query's arithmetic.
T2Launch select_t2_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t) {
    if (t <= 0) { throw std::invalid_argument("t2 linear: unsupported shape or T"); }

    switch (k) {
    case 5120:
        switch (n) {
        case 1024:
        case 4096:
        case 6144:
        case 7168:
        case 12288:
        case 14336:
        case 16384:
        case 34816:
            return launch_t2_small_t_v2;
        case 131072:
        case 248320:
            return launch_t2_small_t_v2;
        default:
            break;
        }
        break;
    case 6144:
        if (n == 5120) {
            return launch_t2_small_t_v2;
        }
        break;
    case 17408:
        if (n == 5120) {
            return launch_t2_small_t_v2;
        }
        break;
    default:
        break;
    }

    throw std::invalid_argument("t2 linear: unsupported shape or T");
}

T2Launch select_t2_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
    case LinearPolicy::AllowA8Int:
    case LinearPolicy::AllowA8IntDecode:
    case LinearPolicy::AllowPrefillCublas:
        return select_t2_a16_launch(n, k, t);
    case LinearPolicy::AllowA4:
        break;
    }
    throw std::invalid_argument("t2 linear: unsupported policy");
}

void t2_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                 cudaStream_t stream) {
    // The kernels address row-major records; the panel permutation is a Q4-family device form.
    if (w.layout != QuantLayout::RowSplit || w.qdata == nullptr || w.scales == nullptr) {
        throw std::invalid_argument("t2 linear: requires a row-split weight");
    }
    const T2Launch launch = select_t2_launch(w.n, w.k, x.ne[1], policy);
    launch(x, w, out, stream);
}

} // namespace ninfer::ops::detail
