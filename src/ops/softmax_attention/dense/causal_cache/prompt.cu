// ninfer::ops - causal_softmax_attention prompt-scale launcher: fill k/v at device
// positions then launch causal attention over absolute cached history.
#include "ops/softmax_attention/dense/causal_cache/launch.h"

#include "ops/common/math.h"
#include "ops/common/device_route.h"
#include "ops/common/device_multiprocessors.h"
#include "ops/softmax_attention/dense/causal_cache/prompt_i8_fast.cuh"
#include <cstdlib>

#include "ops/kv_cache/append/launch.h"
#include "ops/softmax_attention/dense/causal_cache/prompt_bf16.cuh"
#include "ops/softmax_attention/dense/causal_cache/prompt_i8.cuh"
#include "core/device.h" // CUDA_CHECK

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

// Fast prompt is a separate opt-in candidate. The original INT8-family implementation remains
// the default until the user's SM89 numerical/perplexity and end-to-end A/B gates pass.
bool fast_prompt_route() {
    // Re-evaluate the explicit override for each Engine: resident route experiments
    // create fresh Programs in the same process with different environment profiles.
    if (const char* forced = std::getenv("NINFER_PROMPT_FAST")) {
        return forced[0] == '1';
    }
    return device_route_schedule("attn_prompt_fast", 1) == "on";
}

template <typename Geometry, bool PackedValues, typename CacheView, typename Metadata>
void launch_fast_prompt_candidate(const Tensor& q, const Tensor& positions, float scale,
                                  const CacheView& cache, Metadata metadata,
                                  Tensor& out, cudaStream_t stream) {
    using Wide = CausalPromptI8FastShape<8, PackedValues>;
    using Narrow = CausalPromptI8FastShape<4, PackedValues>;
    static const cudaError_t attr_wide =
        cudaFuncSetAttribute(
            causal_attention_prompt_i8_fast_kernel<Geometry, Metadata, 8, PackedValues,
                                                    KvKeyCoding::Int8>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, Wide::SmemBytes);
    static const cudaError_t attr_narrow =
        cudaFuncSetAttribute(
            causal_attention_prompt_i8_fast_kernel<Geometry, Metadata, 4, PackedValues,
                                                    KvKeyCoding::Int8>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, Narrow::SmemBytes);
    CUDA_CHECK(attr_wide);
    CUDA_CHECK(attr_narrow);
    const auto tokens = static_cast<std::int32_t>(q.ne[2]);
    const int sms = current_device_multiprocessors(128);
    const int wide_waves = div_up(div_up(tokens, Wide::Br) * Geometry::QHeads, sms);
    const int narrow_waves = div_up(div_up(tokens, Narrow::Br) * Geometry::QHeads, sms);
    const bool narrow = narrow_waves * 72 < wide_waves * 100;
    const auto launch = [&]<int Warps>() {
        using Shape = CausalPromptI8FastShape<Warps, PackedValues>;
        const dim3 grid(static_cast<unsigned>(div_up(tokens, Shape::Br)),
                        static_cast<unsigned>(Geometry::QHeads), 1U);
        causal_attention_prompt_i8_fast_kernel<Geometry, Metadata, Warps, PackedValues,
                                               KvKeyCoding::Int8>
            <<<grid, Shape::Threads, Shape::SmemBytes, stream>>>(
                static_cast<const __nv_bfloat16*>(q.data),
                static_cast<const std::int8_t*>(cache.k_pages.data),
                static_cast<const std::int8_t*>(cache.v_pages.data),
                static_cast<const __half*>(cache.k_scale_pages.data),
                static_cast<const __half*>(cache.v_scale_pages.data),
                metadata, static_cast<const std::int32_t*>(positions.data), scale,
                static_cast<__nv_bfloat16*>(out.data), tokens);
    };
    if (narrow) launch.template operator()<4>();
    else launch.template operator()<8>();
    CUDA_CHECK(cudaGetLastError());
    // The local rk8v4 storage rotates V. Upstream's raw kernel expects its caller to undo
    // that representation; preserve the local published BF16 output contract.
    if (kv_fork_mode_flags(cache.storage).rotate_v) {
        kv_cache_inverse_rotate_output_kernel<Geometry::QHeads>
            <<<tokens * Geometry::QHeads * kKVCacheInt8Groups, 32, 0, stream>>>(
                static_cast<__nv_bfloat16*>(out.data), tokens, tokens, 0, nullptr);
        CUDA_CHECK(cudaGetLastError());
    }
}

template <typename Geometry, typename CacheView, typename Metadata>
void causal_attention_prompt_attention_launch_for(const Tensor& q, const Tensor& positions,
                                                  float scale, const CacheView& cache,
                                                  Metadata metadata, Tensor& out,
                                                  cudaStream_t stream) {
    if (fast_prompt_route()) {
        if (cache.storage == KvCacheStorage::Int8Group64) {
            launch_fast_prompt_candidate<Geometry, false>(q, positions, scale, cache, metadata,
                                                          out, stream);
            return;
        }
        if (cache.storage == KvCacheStorage::RotatedInt8KeyInt4ValueGroup64) {
            launch_fast_prompt_candidate<Geometry, true>(q, positions, scale, cache, metadata,
                                                         out, stream);
            return;
        }
    }
    const Tensor& cache_k = cache.k_pages;
    const Tensor& cache_v = cache.v_pages;
    // Both dtype-specialized kernels exceed the default 48 KiB dynamic-smem ceiling.
    static const cudaError_t attr_bf16 =
        cudaFuncSetAttribute(causal_attention_prompt_bf16_kernel<Geometry, Metadata>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, kCausalPromptSmemBytes);
    CUDA_CHECK(attr_bf16);

    const auto tokens = static_cast<std::int32_t>(q.ne[2]);
    if (kv_storage_is_int8_family(cache.storage)) {
        const KvForkModeFlags mode = kv_fork_mode_flags(cache.storage);
        const dim3 attention_grid(static_cast<unsigned>(div_up(tokens, kCausalPromptI8Br)),
                                  static_cast<unsigned>(Geometry::QHeads), 1u);
        const Tensor& cache_k_scale = cache.k_scale_pages;
        const Tensor& cache_v_scale = cache.v_scale_pages;
        const auto launch_i8 = [&]<bool PackedV, bool RotateK, bool RotateV, bool PackedK,
                                   bool E8Root>() {
            static const cudaError_t attr_i8 = cudaFuncSetAttribute(
                causal_attention_prompt_i8_kernel<Geometry, PackedV, RotateK, RotateV, PackedK,
                                                  E8Root, Metadata>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, kCausalPromptI8SmemBytes);
            CUDA_CHECK(attr_i8);
            causal_attention_prompt_i8_kernel<Geometry, PackedV, RotateK, RotateV, PackedK, E8Root,
                                              Metadata>
                <<<attention_grid, kCausalPromptI8Threads, kCausalPromptI8SmemBytes, stream>>>(
                    static_cast<const __nv_bfloat16*>(q.data),
                    static_cast<const std::int8_t*>(cache_k.data),
                    static_cast<const std::uint8_t*>(cache_v.data),
                    static_cast<const __half*>(cache_k_scale.data),
                    static_cast<const __half*>(cache_v_scale.data), metadata,
                    static_cast<const std::int32_t*>(positions.data), scale,
                    static_cast<__nv_bfloat16*>(out.data), tokens);
        };
        if (mode.e8_root) {
            launch_i8.template operator()<true, true, true, false, true>();
        } else if (mode.packed_k) {
            launch_i8.template operator()<true, true, true, true, false>();
        } else if (mode.packed_v) {
            launch_i8.template operator()<true, true, true, false, false>();
        } else {
            launch_i8.template operator()<false, false, false, false, false>();
        }
    } else {
        const dim3 attention_grid(static_cast<unsigned>(div_up(tokens, kCausalPromptBr)),
                                  static_cast<unsigned>(Geometry::QHeads), 1u);
        causal_attention_prompt_bf16_kernel<Geometry, Metadata>
            <<<attention_grid, kCausalPromptThreads, kCausalPromptSmemBytes, stream>>>(
                static_cast<const __nv_bfloat16*>(q.data),
                static_cast<const __nv_bfloat16*>(cache_k.data),
                static_cast<const __nv_bfloat16*>(cache_v.data), metadata,
                static_cast<const std::int32_t*>(positions.data), scale,
                static_cast<__nv_bfloat16*>(out.data), tokens);
    }
    CUDA_CHECK(cudaGetLastError());
    if (kv_fork_mode_flags(cache.storage).rotate_v) {
        kv_cache_inverse_rotate_output_kernel<Geometry::QHeads>
            <<<tokens * Geometry::QHeads * kKVCacheInt8Groups, 32, 0, stream>>>(
                static_cast<__nv_bfloat16*>(out.data), tokens, tokens, 0, nullptr);
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace

void causal_attention_prompt_attention_launch(const Tensor& q, const Tensor& positions, float scale,
                                              const PagedKVLayerView& cache, Tensor& out,
                                              cudaStream_t stream) {
    if (cache.storage == KvCacheStorage::Fp8KeyNvfp4Value) {
        causal_attention_prompt_k8v4_attention_launch(q, positions, scale, cache, out, stream);
        return;
    }
    if (cache.storage == KvCacheStorage::Nvfp4Group16) {
        causal_attention_prompt_nvfp4_attention_launch(q, positions, scale, cache, out, stream);
        return;
    }
    if (cache.storage == KvCacheStorage::Fp8E4M3Row256) {
        causal_attention_prompt_fp8_attention_launch(q, positions, scale, cache, out, stream);
        return;
    }
    const PagedKVDirectMetadata metadata{static_cast<const std::int32_t*>(cache.block_table.data)};
    if (q.ne[1] == CausalD256H24Kv4::QHeads) {
        causal_attention_prompt_attention_launch_for<CausalD256H24Kv4>(q, positions, scale, cache,
                                                                       metadata, out, stream);
        return;
    }
    causal_attention_prompt_attention_launch_for<CausalD256H16Kv2>(q, positions, scale, cache,
                                                                   metadata, out, stream);
}

void causal_attention_prompt_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                    const Tensor& positions, const Tensor& valid_columns,
                                    const Tensor& table_rows, float scale,
                                    PagedKVBatchLayerView cache, Tensor& out, cudaStream_t stream) {
    if (cache.storage == KvCacheStorage::Fp8KeyNvfp4Value) {
        causal_attention_prompt_k8v4_launch(q, k, v, positions, valid_columns, table_rows, scale,
                                            cache, out, stream);
        return;
    }
    if (cache.storage == KvCacheStorage::Nvfp4Group16) {
        causal_attention_prompt_nvfp4_launch(q, k, v, positions, valid_columns, table_rows, scale,
                                             cache, out, stream);
        return;
    }
    if (cache.storage == KvCacheStorage::Fp8E4M3Row256) {
        causal_attention_prompt_fp8_launch(q, k, v, positions, valid_columns, table_rows, scale,
                                           cache, out, stream);
        return;
    }
    kv_cache_append_batch_launch(k, v, positions, valid_columns, table_rows, cache, stream);
    const auto launch = [&]<bool Masked>() {
        const PagedKVBatchMetadata<Masked> metadata{
            .tables = static_cast<const std::int32_t*>(cache.block_tables.data),
            .valid_columns =
                Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
            .table_rows   = static_cast<const std::int32_t*>(table_rows.data),
            .table_stride = cache.block_tables.ne[0],
        };
        if (q.ne[1] == CausalD256H24Kv4::QHeads) {
            causal_attention_prompt_attention_launch_for<CausalD256H24Kv4>(
                q, positions, scale, cache, metadata, out, stream);
            return;
        }
        causal_attention_prompt_attention_launch_for<CausalD256H16Kv2>(q, positions, scale, cache,
                                                                       metadata, out, stream);
    };
    if (valid_columns.data == nullptr) {
        launch.template operator()<false>();
    } else {
        launch.template operator()<true>();
    }
}

} // namespace ninfer::ops::detail
