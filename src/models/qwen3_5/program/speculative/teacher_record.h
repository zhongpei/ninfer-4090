#pragma once

#include "ninfer/types.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

// Host-only writer for exact .ninfer teacher observations. One file is emitted per logical
// sequence so prefill chunks can be concatenated without making prefill_chunk part of the dataset.
class DFlashTeacherWriter {
public:
    static constexpr std::uint32_t kVersion = 1;
    static constexpr std::uint32_t kTopK = 16;

    DFlashTeacherWriter() = default;

    DFlashTeacherWriter(std::filesystem::path directory, std::uint32_t feature_rows,
                        std::span<const std::uint32_t> target_layers)
        : directory_(std::move(directory)), feature_rows_(feature_rows),
          target_layers_(target_layers.begin(), target_layers.end()) {
        if (directory_.empty() || feature_rows_ == 0 || target_layers_.empty() ||
            target_layers_.size() > 32) {
            throw std::invalid_argument("invalid DFlash teacher writer configuration");
        }
        std::filesystem::create_directories(directory_);
        const auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        session_ = static_cast<std::uint64_t>(now);
    }

    [[nodiscard]] bool enabled() const noexcept { return !directory_.empty(); }

    void write_chunk(std::uint64_t sequence_id, std::uint32_t chunk_begin,
                     std::span<const TokenId> ids, std::span<const std::int32_t> positions,
                     std::span<const std::uint16_t> fused_bf16,
                     std::span<const TokenId> top_ids, std::span<const float> top_scores) {
        if (!enabled() || ids.empty() || positions.size() != ids.size() ||
            fused_bf16.size() != static_cast<std::size_t>(feature_rows_) * ids.size() ||
            top_ids.size() != static_cast<std::size_t>(kTopK) * ids.size() ||
            top_scores.size() != static_cast<std::size_t>(kTopK) * ids.size()) {
            throw std::invalid_argument("invalid DFlash teacher chunk");
        }

        StreamState& state = stream(sequence_id);
        if (state.next_begin != chunk_begin) {
            throw std::logic_error("DFlash teacher chunks are not contiguous");
        }

        const std::uint32_t token_count = static_cast<std::uint32_t>(ids.size());
        write_scalar(state.out, token_count);
        write_scalar(state.out, chunk_begin);
        write_bytes(state.out, ids.data(), ids.size_bytes());
        write_bytes(state.out, positions.data(), positions.size_bytes());
        write_bytes(state.out, fused_bf16.data(), fused_bf16.size_bytes());
        write_bytes(state.out, top_ids.data(), top_ids.size_bytes());
        write_bytes(state.out, top_scores.data(), top_scores.size_bytes());
        state.out.flush();
        if (!state.out) {
            throw std::runtime_error("failed writing DFlash teacher chunk");
        }
        state.next_begin += token_count;
    }

private:
    struct StreamState {
        std::ofstream out;
        std::uint32_t next_begin = 0;
    };

    template <class T>
    static void write_scalar(std::ofstream& out, const T& value) {
        write_bytes(out, &value, sizeof(value));
    }

    static void write_bytes(std::ofstream& out, const void* data, std::size_t bytes) {
        out.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
        if (!out) { throw std::runtime_error("failed writing DFlash teacher file"); }
    }

    StreamState& stream(std::uint64_t sequence_id) {
        auto found = streams_.find(sequence_id);
        if (found != streams_.end()) { return *found->second; }

        auto state = std::make_unique<StreamState>();
        const std::filesystem::path path =
            directory_ / ("teacher-" + hex(session_) + "-" + hex(sequence_id) + ".ndft");
        state->out.open(path, std::ios::binary | std::ios::trunc);
        if (!state->out) {
            throw std::runtime_error("cannot create DFlash teacher file: " + path.string());
        }
        static constexpr std::array<char, 8> magic{'N','I','N','F','D','F','T','1'};
        write_bytes(state->out, magic.data(), magic.size());
        write_scalar(state->out, kVersion);
        write_scalar(state->out, feature_rows_);
        const std::uint32_t layers = static_cast<std::uint32_t>(target_layers_.size());
        write_scalar(state->out, layers);
        write_scalar(state->out, kTopK);
        write_bytes(state->out, target_layers_.data(),
                    target_layers_.size() * sizeof(std::uint32_t));
        auto [it, inserted] = streams_.emplace(sequence_id, std::move(state));
        (void)inserted;
        return *it->second;
    }

    static std::string hex(std::uint64_t value) {
        static constexpr char digits[] = "0123456789abcdef";
        std::array<char, 16> out{};
        for (int i = 15; i >= 0; --i) {
            out[static_cast<std::size_t>(i)] = digits[value & 0xfU];
            value >>= 4U;
        }
        return std::string(out.data(), out.size());
    }

    std::filesystem::path directory_;
    std::uint32_t feature_rows_ = 0;
    std::vector<std::uint32_t> target_layers_;
    std::uint64_t session_ = 0;
    std::unordered_map<std::uint64_t, std::unique_ptr<StreamState>> streams_;
};

} // namespace ninfer::models::qwen3_5::detail
