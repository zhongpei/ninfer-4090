#pragma once

#include "core/dtype.h"
#include "ninfer/types.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ninfer {

inline constexpr std::int32_t kD256KVCacheHeadDim = 256;

/** Physical data/scale planes for one K or V vector. */
struct PagedKVVectorLayout {
    DType data_dtype                  = DType::BF16;
    std::int32_t data_leading_extent  = 0;
    DType scale_dtype                 = DType::U8;
    std::int32_t scale_leading_extent = 0;

    [[nodiscard]] constexpr bool has_scale() const noexcept { return scale_leading_extent != 0; }

    [[nodiscard]] std::size_t physical_bytes() const {
        return static_cast<std::size_t>(data_leading_extent) * dtype_size(data_dtype) +
               static_cast<std::size_t>(scale_leading_extent) * dtype_size(scale_dtype);
    }

    friend bool operator==(const PagedKVVectorLayout&, const PagedKVVectorLayout&) = default;
};

/** Resolved physical plane schema for one paged K/V layer. */
struct PagedKVStorageLayout {
    KvCacheStorage storage = KvCacheStorage::BFloat16;
    std::int32_t head_dim  = 0;
    PagedKVVectorLayout key;
    PagedKVVectorLayout value;

    [[nodiscard]] constexpr std::size_t planes_per_layer() const noexcept {
        return 2ULL + static_cast<std::size_t>(key.has_scale()) +
               static_cast<std::size_t>(value.has_scale());
    }

    [[nodiscard]] constexpr std::size_t logical_vector_bytes() const noexcept {
        return static_cast<std::size_t>(head_dim) * sizeof(std::uint16_t);
    }

    [[nodiscard]] constexpr std::size_t logical_bytes_per_token_head() const noexcept {
        return 2ULL * logical_vector_bytes();
    }

    [[nodiscard]] std::size_t physical_bytes_per_token_head() const {
        return key.physical_bytes() + value.physical_bytes();
    }

    friend bool operator==(const PagedKVStorageLayout&, const PagedKVStorageLayout&) = default;
};

[[nodiscard]] inline PagedKVStorageLayout paged_kv_storage_layout(KvCacheStorage storage,
                                                                  std::int32_t head_dim) {
    if (head_dim <= 0) { throw std::invalid_argument("KV-cache head dimension must be positive"); }

    const auto symmetric = [=](PagedKVVectorLayout vector) {
        return PagedKVStorageLayout{storage, head_dim, vector, vector};
    };
    switch (storage) {
    case KvCacheStorage::BFloat16:
        return symmetric({DType::BF16, head_dim, DType::U8, 0});
    case KvCacheStorage::Int8Group64:
        if (head_dim == kD256KVCacheHeadDim) { return symmetric({DType::I8, 256, DType::FP16, 4}); }
        break;
    case KvCacheStorage::RotatedInt8KeyInt4ValueGroup64:
        // Existing RotorQuant rk8v4: D256-rotated INT8 K plus unrotated packed-int4 V with
        // G32 value scales. Keep this format byte-for-byte compatible with franken/v0.11.
        if (head_dim == kD256KVCacheHeadDim) {
            return {storage,
                    head_dim,
                    {DType::I8, head_dim, DType::FP16, 4},
                    {DType::U8, head_dim / 2, DType::FP16, 8}};
        }
        break;
    case KvCacheStorage::RotatedInt4KeyInt4ValueGroup64:
    case KvCacheStorage::RK4V4E8:
        // E8-family modes use independently H64-rotated G64 planes. Codes are packed two
        // dimensions per byte and each 64-D group carries one FP16 scale.
        if (head_dim == kD256KVCacheHeadDim) {
            return symmetric({DType::U8, head_dim / 2, DType::FP16, 4});
        }
        break;
    case KvCacheStorage::RK2V4E8:
        if (head_dim == kD256KVCacheHeadDim) {
            return {storage,
                    head_dim,
                    {DType::U8, head_dim / 4, DType::FP16, 4},
                    {DType::U8, head_dim / 2, DType::FP16, 4}};
        }
        break;
    case KvCacheStorage::Fp8E4M3Row256:
        if (head_dim == kD256KVCacheHeadDim) {
            return symmetric({DType::FP8_E4M3FN, 256, DType::FP16, 1});
        }
        break;
    case KvCacheStorage::Nvfp4Group16:
        if (head_dim == kD256KVCacheHeadDim) { return symmetric({DType::U8, 128, DType::U8, 16}); }
        break;
    case KvCacheStorage::Fp8KeyNvfp4Value:
        if (head_dim == kD256KVCacheHeadDim) {
            return {storage,
                    head_dim,
                    {DType::FP8_E4M3FN, 256, DType::FP16, 1},
                    {DType::U8, 128, DType::U8, 16}};
        }
        break;
    }
    throw std::invalid_argument("unsupported paged KV-cache storage geometry");
}

[[nodiscard]] constexpr bool kv_storage_is_e8_family(KvCacheStorage storage) noexcept {
    return storage == KvCacheStorage::RotatedInt4KeyInt4ValueGroup64 ||
           storage == KvCacheStorage::RK4V4E8 || storage == KvCacheStorage::RK2V4E8;
}

[[nodiscard]] constexpr bool kv_storage_is_int8_family(KvCacheStorage storage) noexcept {
    return storage == KvCacheStorage::Int8Group64 ||
           storage == KvCacheStorage::RotatedInt8KeyInt4ValueGroup64 ||
           kv_storage_is_e8_family(storage);
}

struct KvE8ModeFlags {
    bool packed_k   = false;
    bool e8_lattice = false;
    bool e8_root    = false;
};

[[nodiscard]] constexpr KvE8ModeFlags kv_e8_mode_flags(KvCacheStorage storage) noexcept {
    switch (storage) {
    case KvCacheStorage::RotatedInt4KeyInt4ValueGroup64:
        return {.packed_k = true};
    case KvCacheStorage::RK4V4E8:
        return {.packed_k = true, .e8_lattice = true};
    case KvCacheStorage::RK2V4E8:
        return {.e8_root = true};
    default:
        return {};
    }
}

} // namespace ninfer
