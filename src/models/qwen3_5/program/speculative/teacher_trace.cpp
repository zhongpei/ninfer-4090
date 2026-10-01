#include "models/qwen3_5/program/speculative/teacher_trace.h"

#include "core/device.h"

#include <cuda_runtime.h>

#include <array>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <system_error>

namespace ninfer::models::qwen3_5::detail {
namespace {

constexpr std::array<char, 8> kTraceMagic{'N','I','F','T','R','C','1','\0'};
constexpr std::uint32_t kTraceVersion = 1;
constexpr std::uint32_t kRecordMagic = 0x31434552U; // "REC1" little-endian

void require_vector_prefix(const Tensor& tensor, DType dtype, std::uint32_t rows,
                           const char* name) {
    if (tensor.data == nullptr || tensor.dtype != dtype || !tensor.is_contiguous() ||
        tensor.ne[0] < static_cast<std::int32_t>(rows) || tensor.ne[1] != 1 ||
        tensor.ne[2] != 1 || tensor.ne[3] != 1) {
        throw std::invalid_argument(std::string("DFlash teacher trace: invalid ") + name);
    }
}

void require_matrix_prefix(const Tensor& tensor, DType dtype, std::uint32_t rows,
                           const char* name) {
    if (tensor.data == nullptr || tensor.dtype != dtype || !tensor.is_contiguous() ||
        tensor.ne[0] <= 0 || tensor.ne[1] < static_cast<std::int32_t>(rows) ||
        tensor.ne[2] != 1 || tensor.ne[3] != 1) {
        throw std::invalid_argument(std::string("DFlash teacher trace: invalid ") + name);
    }
}

} // namespace

DFlashTeacherTraceWriter::DFlashTeacherTraceWriter(std::filesystem::path path,
                                                   std::uint32_t max_records,
                                                   std::uint32_t token_domain)
    : path_(std::move(path)), max_records_(max_records), token_domain_(token_domain) {
    if (path_.empty()) return;
    if (token_domain_ == 0) {
        throw std::invalid_argument("DFlash teacher trace token domain must be positive");
    }
    std::error_code ec;
    const auto parent = path_.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent, ec);
        if (ec) {
            throw std::runtime_error("cannot create DFlash teacher trace directory: " +
                                     parent.string());
        }
    }
    output_.open(path_, std::ios::binary | std::ios::trunc);
    if (!output_) {
        throw std::runtime_error("cannot open DFlash teacher trace: " + path_.string());
    }
}

void DFlashTeacherTraceWriter::write_bytes(const void* data, std::size_t bytes) {
    if (bytes == 0) return;
    output_.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    if (!output_) { throw std::runtime_error("failed writing DFlash teacher trace"); }
}

void DFlashTeacherTraceWriter::write_header(std::uint32_t feature_rows,
                                            std::uint32_t physical_rows,
                                            std::uint32_t physical_width) {
    feature_rows_ = feature_rows;
    physical_rows_ = physical_rows;
    physical_width_ = physical_width;
    write_bytes(kTraceMagic.data(), kTraceMagic.size());
    write_scalar(kTraceVersion);
    write_scalar(token_domain_);
    write_scalar(feature_rows_);
    write_scalar(physical_rows_);
    write_scalar(physical_width_);
    const std::uint32_t reserved = 0;
    write_scalar(reserved);
    header_written_ = true;
}

void DFlashTeacherTraceWriter::append(std::uint64_t request_id, std::uint32_t frontier,
                                      std::uint32_t rows, const Tensor& target_input_ids,
                                      const Tensor& fused_target_features,
                                      const Tensor& target_logits,
                                      std::span<const TokenId> labels) {
    if (!enabled() || rows == 0 || (max_records_ != 0 && record_count_ >= max_records_)) return;
    require_vector_prefix(target_input_ids, DType::I32, rows, "target input ids");
    require_matrix_prefix(fused_target_features, DType::BF16, rows, "target features");
    require_matrix_prefix(target_logits, DType::BF16, rows, "target logits");
    if (labels.size() < rows) {
        throw std::invalid_argument("DFlash teacher trace: label prefix is shorter than rows");
    }

    const auto feature_rows = static_cast<std::uint32_t>(fused_target_features.ne[0]);
    const auto physical_rows = static_cast<std::uint32_t>(target_logits.ne[0]);
    const auto physical_width = static_cast<std::uint32_t>(target_logits.ne[1]);
    if (!header_written_) {
        write_header(feature_rows, physical_rows, physical_width);
    } else if (feature_rows != feature_rows_ || physical_rows != physical_rows_ ||
               physical_width != physical_width_) {
        throw std::logic_error("DFlash teacher trace tensor geometry changed after header");
    }

    host_ids_.resize(rows);
    host_features_.resize(static_cast<std::size_t>(feature_rows) * rows);
    host_logits_.resize(static_cast<std::size_t>(physical_rows) * rows);

    CUDA_CHECK(cudaMemcpy(host_ids_.data(), target_input_ids.data,
                          static_cast<std::size_t>(rows) * sizeof(TokenId),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_features_.data(), fused_target_features.data,
                          host_features_.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_logits_.data(), target_logits.data,
                          host_logits_.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost));

    write_scalar(kRecordMagic);
    write_scalar(request_id);
    write_scalar(frontier);
    write_scalar(rows);
    write_bytes(host_ids_.data(), host_ids_.size() * sizeof(TokenId));
    write_bytes(labels.data(), static_cast<std::size_t>(rows) * sizeof(TokenId));
    write_bytes(host_features_.data(), host_features_.size() * sizeof(std::uint16_t));
    write_bytes(host_logits_.data(), host_logits_.size() * sizeof(std::uint16_t));
    output_.flush();
    if (!output_) { throw std::runtime_error("failed flushing DFlash teacher trace"); }
    ++record_count_;
}

} // namespace ninfer::models::qwen3_5::detail
