#include "models/qwen3_5/program/proposal_view.h"
#include "models/qwen3_5/program/round_buffers.h"

#include <algorithm>
#include <iterator>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <typeinfo>

namespace {
constexpr const char* source = "test_dflash_proposal_view";
void require(bool ok, const char* message) {
    if (!ok) { throw std::runtime_error(message); }
}
template <typename Error>
void rejected(const ninfer::models::qwen3_5::DFlashDecodeState& frame, std::uint32_t drafts) {
    try { (void)ninfer::models::qwen3_5::selected_dflash_proposal_view(frame, drafts); }
    catch (const Error& error) {
        require(typeid(error) == typeid(Error), "proposal rejection used the wrong exception type");
        return;
    }
    throw std::runtime_error("unsupported compact proposal was accepted");
}
}

int main() try {
    for (const std::uint32_t batch : {1U, 2U, 4U, 8U}) {
        ninfer::LayoutBuilder builder;
        auto layout = ninfer::models::qwen3_5::begin_round_state_layout(builder,
            ninfer::models::qwen3_5::RoundStateSpec{
                .hidden = 16, .output_rows = 32, .batch_capacity = batch,
                .draft_window = 15, .backend = ninfer::SpeculativeBackend::DFlash2,
                .calibrated_routing = true,
            });
        ninfer::models::qwen3_5::complete_round_state_layout(builder, layout);
        const auto bytes = builder.finish(256, source);
        std::unique_ptr<void, decltype(&std::free)> storage(std::aligned_alloc(256, bytes), &std::free);
        require(storage != nullptr, "host layout allocation failed");
        ninfer::models::qwen3_5::RoundState round({storage.get(), bytes}, layout);
        require(round.dflash_decode.has_value(), "DFlash2 frame absent");
        const auto& frame = *round.dflash_decode;
        for (const std::uint32_t action : {7U, 11U, 15U}) {
            const auto selected = ninfer::models::qwen3_5::selected_dflash_proposal_view(frame, action);
            const auto& drafts_buffer = action == 15 ? frame.draft_tokens : frame.target_draft_tokens;
            const auto& candidates_buffer = action == 15 ? frame.candidate_ids : frame.target_candidate_ids;
            const auto& probabilities_buffer = action == 15 ? frame.proposal_q : frame.target_proposal_q;
            require(selected.proposal_ids.ne[0] == static_cast<int>(action + 1) &&
                    selected.proposal_ids.ne[1] == static_cast<int>(batch), "wrong physical proposal shape");
            require(selected.proposal_ids.data == frame.proposal_ids.data &&
                    selected.proposal_positions.data == frame.proposal_positions.data,
                    "compact ingress must reuse stable storage");
            require(selected.draft_tokens.data == drafts_buffer.data &&
                    selected.candidate_ids.data == candidates_buffer.data &&
                    selected.proposal_q.data == probabilities_buffer.data,
                    "proposal outputs must directly feed selected target storage");
            require(std::equal(std::begin(selected.append_positions.ne), std::end(selected.append_positions.ne), frame.append_positions.ne) &&
                    std::equal(std::begin(selected.append_positions.nb), std::end(selected.append_positions.nb), frame.append_positions.nb) &&
                    selected.append_positions.data == frame.append_positions.data &&
                    std::equal(std::begin(selected.append_counts.ne), std::end(selected.append_counts.ne), frame.append_counts.ne) &&
                    std::equal(std::begin(selected.append_counts.nb), std::end(selected.append_counts.nb), frame.append_counts.nb) &&
                    selected.append_counts.data == frame.append_counts.data,
                    "previous-round append capacity must not be narrowed");
            for (std::uint32_t lane = 0; lane < batch; ++lane) {
                const auto address = selected.proposal_ids.slice(1, static_cast<int>(lane), 1).data;
                require(static_cast<std::byte*>(address) == static_cast<std::byte*>(frame.proposal_ids.data) +
                            lane * (action + 1) * sizeof(std::int32_t), "proposal lane retained resident-width stride");
                const auto drafts = selected.draft_tokens.slice(1, static_cast<int>(lane), 1).data;
                require(static_cast<std::byte*>(drafts) == static_cast<std::byte*>(drafts_buffer.data) +
                            lane * action * sizeof(std::int32_t), "draft output lane retained resident-width stride");
                const auto candidates = selected.candidate_ids.slice(2, static_cast<int>(lane), 1).data;
                require(static_cast<std::byte*>(candidates) == static_cast<std::byte*>(candidates_buffer.data) +
                            lane * action * 16 * sizeof(std::int32_t), "candidate lane retained resident-width stride");
                const auto q = selected.proposal_q.slice(2, static_cast<int>(lane), 1).data;
                require(static_cast<std::byte*>(q) == static_cast<std::byte*>(probabilities_buffer.data) +
                            lane * action * 16 * sizeof(float), "proposal probability lane retained resident-width stride");
                const auto positions = selected.proposal_positions.slice(1, static_cast<int>(lane), 1).data;
                require(static_cast<std::byte*>(positions) == static_cast<std::byte*>(frame.proposal_positions.data) +
                            lane * (action + 1) * sizeof(std::int32_t), "proposal position lane retained resident-width stride");
            }
        }
        require(frame.proposal_ids.ne[0] == 16 && frame.draft_tokens.ne[0] == 15,
                "selected view mutated resident frame");
        rejected<std::invalid_argument>(frame, 0);
        rejected<std::invalid_argument>(frame, 16);
        auto invalid = frame;
        invalid.proposal_ids.data = nullptr;
        rejected<std::logic_error>(invalid, 7);
        invalid = frame;
        invalid.proposal_positions.dtype = ninfer::DType::FP32;
        rejected<std::logic_error>(invalid, 7);
        invalid = frame;
        invalid.proposal_ids.ne[0] = 7;
        rejected<std::logic_error>(invalid, 7);
        invalid = frame;
        invalid.target_proposal_q.data = nullptr;
        rejected<std::logic_error>(invalid, 7);
    }
    std::cout << "ok compact proposal views: K7/K11/K15, B1/B2/B4/B8, direct output and full append\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << source << ": " << error.what() << '\n';
    return 1;
}
