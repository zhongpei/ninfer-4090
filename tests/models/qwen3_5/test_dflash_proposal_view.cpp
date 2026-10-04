#include "models/qwen3_5/program/proposal_view.h"
#include "models/qwen3_5/state/round_buffers.h"
#include "test_support.h"

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace {
constexpr const char* source = "test_dflash_proposal_view";
void require(bool ok, const char* message) {
    ninfer::test::require(ok, source, message);
}
void rejected(const ninfer::qwen3_5::DFlashDecodeState& frame, std::uint32_t drafts) {
    try { (void)ninfer::qwen3_5::selected_dflash_proposal_view(frame, drafts); }
    catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("unsupported compact proposal was accepted");
}
}

int main() try {
    for (const std::uint32_t batch : {1U, 2U, 4U, 8U}) {
        ninfer::LayoutBuilder builder;
        const auto layout = ninfer::qwen3_5::plan_round_state(builder,
            ninfer::qwen3_5::RoundStateSpec{
                .hidden = 16, .output_rows = 32, .batch_capacity = batch,
                .draft_window = 15, .backend = ninfer::SpeculativeBackend::DFlash2,
                .calibrated_routing = true,
            });
        const auto bytes = builder.finish(256, source);
        std::unique_ptr<void, decltype(&std::free)> storage(std::aligned_alloc(256, bytes), &std::free);
        require(storage != nullptr, "host layout allocation failed");
        auto round = ninfer::qwen3_5::bind_round_state(layout, static_cast<std::byte*>(storage.get()), bytes);
        require(round.dflash_decode.has_value(), "DFlash2 frame absent");
        const auto& frame = *round.dflash_decode;
        for (const std::uint32_t action : {7U, 11U, 15U}) {
            const auto selected = ninfer::qwen3_5::selected_dflash_proposal_view(frame, action);
            const auto target = frame.target_view(action);
            require(selected.proposal_ids.ne[0] == static_cast<int>(action + 1) &&
                    selected.proposal_ids.ne[1] == static_cast<int>(batch), "wrong physical proposal shape");
            require(selected.proposal_ids.data == frame.proposal_ids.data &&
                    selected.proposal_positions.data == frame.proposal_positions.data,
                    "compact ingress must reuse stable storage");
            require(selected.draft_tokens.data == target.draft_tokens.data &&
                    selected.candidate_ids.data == target.candidate_ids.data &&
                    selected.proposal_q.data == target.proposal_q.data,
                    "proposal outputs must directly feed selected target storage");
            require(selected.append_positions.ne == frame.append_positions.ne &&
                    selected.append_positions.data == frame.append_positions.data,
                    "previous-round append capacity must not be narrowed");
            for (std::uint32_t lane = 0; lane < batch; ++lane) {
                const auto address = selected.proposal_ids.slice(1, static_cast<int>(lane), 1).data;
                require(static_cast<std::byte*>(address) == static_cast<std::byte*>(frame.proposal_ids.data) +
                            lane * (action + 1) * sizeof(std::int32_t), "proposal lane retained resident-width stride");
                const auto drafts = selected.draft_tokens.slice(1, static_cast<int>(lane), 1).data;
                require(static_cast<std::byte*>(drafts) == static_cast<std::byte*>(target.draft_tokens.data) +
                            lane * action * sizeof(std::int32_t), "draft output lane retained resident-width stride");
                const auto candidates = selected.candidate_ids.slice(2, static_cast<int>(lane), 1).data;
                require(static_cast<std::byte*>(candidates) == static_cast<std::byte*>(target.candidate_ids.data) +
                            lane * action * 16 * sizeof(std::int32_t), "candidate lane retained resident-width stride");
            }
        }
        require(frame.proposal_ids.ne[0] == 16 && frame.draft_tokens.ne[0] == 15,
                "selected view mutated resident frame");
        rejected(frame, 0);
        rejected(frame, 16);
        auto invalid = frame;
        invalid.proposal_ids.data = nullptr;
        rejected(invalid, 7);
    }
    std::cout << "ok compact proposal views: K7/K11/K15, B1/B2/B4/B8, direct output and full append\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << source << ": " << error.what() << '\n';
    return 1;
}
