#pragma once

#include "core/tensor.h"
#include "ninfer/types.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <stdexcept>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

// Offline-only trace writer for training a DFlash2 companion against the exact deployed .ninfer
// target. The binary stores committed linear target rows, fused target taps, and full BF16 logits;
// Python tooling performs top-K/log-softmax offline so runtime tracing adds no extra GPU kernels.
class DFlashTeacherTraceWriter {
public:
    DFlashTeacherTraceWriter() = default;
    DFlashTeacherTraceWriter(std::filesystem::path path, std::uint32_t max_records,
                             std::uint32_t token_domain);

    [[nodiscard]] bool enabled() const noexcept { return output_.is_open(); }
    [[nodiscard]] std::uint64_t record_count() const noexcept { return record_count_; }

    void append(std::uint64_t request_id, std::uint32_t frontier, std::uint32_t rows,
                const Tensor& target_input_ids, const Tensor& fused_target_features,
                const Tensor& target_logits, std::span<const TokenId> labels);

private:
    void write_header(std::uint32_t feature_rows, std::uint32_t physical_rows,
                      std::uint32_t physical_width);
    template <class T>
    void write_scalar(const T& value) {
        output_.write(reinterpret_cast<const char*>(&value), sizeof(T));
        if (!output_) { throw std::runtime_error("failed writing DFlash teacher trace"); }
    }
    void write_bytes(const void* data, std::size_t bytes);

    std::filesystem::path path_;
    std::ofstream output_;
    std::uint32_t max_records_ = 0;
    std::uint32_t token_domain_ = 0;
    std::uint32_t feature_rows_ = 0;
    std::uint32_t physical_rows_ = 0;
    std::uint32_t physical_width_ = 0;
    std::uint64_t record_count_ = 0;
    bool header_written_ = false;

    std::vector<TokenId> host_ids_;
    std::vector<std::uint16_t> host_features_;
    std::vector<std::uint16_t> host_logits_;
};

} // namespace ninfer::models::qwen3_5::detail
