#include "runtime/engine/speculative_routing_profile.h"
#include <nlohmann/json.hpp>
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
using Json = nlohmann::json;
void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}
struct Fixture {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("ninfer-router-profile-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".json");
    ninfer::SpeculativeRoutingProfileIdentity identity;
    Json document;
    Fixture() {
        identity.artifact_id = "0123456789abcdef0123456789abcdef";
        identity.prefill_signature = "represented-model-signature";
        identity.hardware_class = "RTX4090-sm89";
        identity.kv_storage = ninfer::KvCacheStorage::Int8Group64;
        identity.max_concurrency = 8;
        identity.max_context = 32768;
        identity.prefill_chunk = 1024;
        identity.resolved_kv_capacity = 65536;
        identity.context_cache = {true, 8, 8, 8ULL << 30, 16, 8, 2, 4};
        document = {{"schema_version", 1}, {"artifact_type", "ninfer_spec_router_profile"},
            {"identity", ninfer::runtime::routing_profile_identity_json(identity)},
            {"cells", Json::array({{{"active_batch", 1}, {"frontier_upper", 1024}, {"draft_tokens", 7}},
                {{"active_batch", 1}, {"frontier_upper", 8192}, {"draft_tokens", 11}},
                {{"active_batch", 8}, {"frontier_upper", 32768}, {"draft_tokens", 15}}})}};
    }
    ~Fixture() { std::error_code error; std::filesystem::remove(path, error); }
    auto load(const Json& value) {
        { std::ofstream stream(path); stream << value; }
        return ninfer::runtime::load_calibrated_routing_profile(path, identity);
    }
    void rejects(const Json& value) {
        try { (void)load(value); }
        catch (const std::invalid_argument&) { return; }
        throw std::runtime_error("invalid profile was accepted");
    }
};
}
int main() {
    try {
        Fixture fixture;
        const auto table = fixture.load(fixture.document);
        require(table.select(1, 0) == 7 && table.select(1, 1024) == 7, "first interval");
        require(table.select(1, 1025) == 11 && table.select(1, 8192) == 11, "second interval");
        require(table.select(8, 8193) == 15 && table.select(8, 32768) == 15, "third interval");
        require(table.select(1, 8193) == 0 && table.select(2, 1024) == 0 &&
            table.select(8, 32769) == 0 && table.select(0, 12) == 0 && table.select(9, 12) == 0,
            "uncovered or out-of-domain must be target-only");
        require(table.proposal_compute == ninfer::runtime::DFlashProposalCompute::Full &&
                table.proposal_width(0) == 0 && table.proposal_width(7) == 16 &&
                table.proposal_width(11) == 16 && table.proposal_width(15) == 16,
                "legacy profile must retain physical full proposals");
        for (const char* mode : {"full", "selected"}) {
            auto versioned = fixture.document;
            versioned["schema_version"] = 2;
            versioned["proposal_compute"] = mode;
            const auto loaded = fixture.load(versioned);
            const bool selected = std::string(mode) == "selected";
            require((loaded.proposal_compute == ninfer::runtime::DFlashProposalCompute::Selected) == selected,
                    "v2 compute mode was not loaded");
            for (const auto action : ninfer::runtime::kCalibratedDraftActions) {
                require(loaded.proposal_width(action) == (action == 0 ? 0U : selected ? action + 1U : 16U),
                        "physical proposal width differs from mode/action");
            }
            require(loaded.select(1, 1024) == 7 && loaded.select(1, 1025) == 11 &&
                    loaded.select(8, 8193) == 15 && loaded.select(2, 1024) == 0,
                    "compute mode must not mutate route selection or uncovered skips");
        }
        {
            auto baseline = fixture.document;
            baseline["schema_version"] = 3;
            baseline["proposal_compute"] = "selected";
            baseline["default_action"] = 7;
            baseline["cells"] = Json::array({
                {{"active_batch", 1}, {"frontier_upper", 8192}, {"draft_tokens", 11}},
                {{"active_batch", 8}, {"frontier_upper", 32768}, {"draft_tokens", 0}},
            });
            const auto loaded = fixture.load(baseline);
            require(loaded.default_action == 7, "schema3 must publish K7 baseline");
            require(loaded.select(1, 0) == 7 && loaded.select(2, 1024) == 7 &&
                    loaded.select(8, 8192) == 7, "schema3 uncovered cells must inherit K7");
            require(loaded.select(1, 1025) == 11, "schema3 K11 override");
            require(loaded.select(8, 8193) == 0, "schema3 explicit K0 override");
            require(loaded.proposal_compute == ninfer::runtime::DFlashProposalCompute::Selected &&
                    loaded.proposal_width(7) == 8 && loaded.proposal_width(11) == 12,
                    "schema3 selected proposal widths");

            auto invalid = baseline;
            invalid["default_action"] = 0; fixture.rejects(invalid);
            invalid = baseline; invalid["default_action"] = 11; fixture.rejects(invalid);
            invalid = baseline; invalid["cells"][0]["draft_tokens"] = 7; fixture.rejects(invalid);
            invalid = baseline; invalid["cells"][0]["draft_tokens"] = 15; fixture.rejects(invalid);
            invalid = baseline; invalid.erase("default_action"); fixture.rejects(invalid);
            invalid = baseline; invalid.erase("proposal_compute"); fixture.rejects(invalid);
        }
        for (const Json invalid_mode : {Json("auto"), Json(""), Json(true), Json(8), Json(nullptr)}) {
            auto versioned = fixture.document;
            versioned["schema_version"] = 2; versioned["proposal_compute"] = invalid_mode;
            fixture.rejects(versioned);
        }
        {
            auto versioned = fixture.document;
            versioned["proposal_compute"] = "selected"; fixture.rejects(versioned);
            versioned["schema_version"] = 3; fixture.rejects(versioned);
            versioned["schema_version"] = 2.0; fixture.rejects(versioned);
            versioned["schema_version"] = 2; versioned["extra"] = true; fixture.rejects(versioned);
        }
        for (const auto* suffix : {" trailing-garbage", " {}"}) {
            { std::ofstream stream(fixture.path); stream << fixture.document << suffix; }
            bool rejected = false;
            try { (void)ninfer::runtime::load_calibrated_routing_profile(fixture.path, fixture.identity); }
            catch (const std::invalid_argument&) { rejected = true; }
            require(rejected, "valid JSON followed by trailing garbage was accepted");
        }
        auto value = fixture.document;
        value["cells"].push_back(value["cells"][0]); fixture.rejects(value);
        for (const auto& [key, invalid_value] : std::vector<std::pair<std::string, Json>>{
                 {"active_batch", 0}, {"active_batch", 9}, {"active_batch", 1.0},
                 {"frontier_upper", 1025}, {"draft_tokens", 3}, {"draft_tokens", -1}}) {
            value = fixture.document; value["cells"][0][key] = invalid_value; fixture.rejects(value);
        }
        for (const auto& key : {"schema_version", "artifact_type", "identity", "cells"}) {
            value = fixture.document; value.erase(key); fixture.rejects(value);
        }
        value = fixture.document; value["extra"] = true; fixture.rejects(value);
        value = fixture.document; value["schema_version"] = 2; fixture.rejects(value);
        value = fixture.document; value["provenance"] = false; fixture.rejects(value);
        value = fixture.document; value["cells"][0]["extra"] = 1; fixture.rejects(value);
        value = fixture.document; value["identity"]["context_cache"]["host_kv_capacity_bytes"] = 0; fixture.rejects(value);
        value = fixture.document; value["identity"]["execution_options"]["prefill_a8"] = false; fixture.rejects(value);
        value = fixture.document; value["identity"]["execution_options"]["rope_scaling_factor"] = 1.00001; fixture.rejects(value);
        value = fixture.document; value["identity"]["max_context"] = 32768.0; fixture.rejects(value);
        value = fixture.document; value["identity"]["hardware_class"] = "different"; fixture.rejects(value);
        value = fixture.document; value["identity"]["execution_options"]["extra"] = false; fixture.rejects(value);
        value = fixture.document; value["provenance"] = {{"purpose", "measurement_control"}, {"qualified", false}};
        require(fixture.load(value).select(1, 1024) == 7, "control provenance is accepted");
        value = fixture.document; value["cells"][0]["draft_tokens"] = 0;
        require(fixture.load(value).select(1, 1024) == 0, "explicit target-only action");
        // Keep the measured document fixed while changing the actual expected startup identity.
        // This catches omitted serializer fields as well as incorrect value comparisons.
        const auto reject_expected = [&](const ninfer::SpeculativeRoutingProfileIdentity& expected) {
            { std::ofstream stream(fixture.path); stream << fixture.document; }
            try { (void)ninfer::runtime::load_calibrated_routing_profile(fixture.path, expected); }
            catch (const std::invalid_argument&) { return; }
            throw std::runtime_error("mismatched expected startup identity was accepted");
        };
#define EXPECT_CHANGE(expression) { auto expected = fixture.identity; expression; reject_expected(expected); }
        EXPECT_CHANGE(expected.artifact_id[0] = 'f');
        EXPECT_CHANGE(expected.prefill_signature += "different");
        EXPECT_CHANGE(expected.hardware_class += "different");
        EXPECT_CHANGE(expected.backend = ninfer::SpeculativeBackend::None);
        EXPECT_CHANGE(expected.kv_storage = ninfer::KvCacheStorage::BFloat16);
        EXPECT_CHANGE(expected.startup_draft_tokens = 11);
        EXPECT_CHANGE(expected.proposal_head = ninfer::ProposalHead::Optimized);
        EXPECT_CHANGE(expected.use_cuda_graph = false);
        EXPECT_CHANGE(++expected.max_concurrency);
        EXPECT_CHANGE(++expected.max_context);
        EXPECT_CHANGE(++expected.prefill_chunk);
        EXPECT_CHANGE(++expected.resolved_kv_capacity);
        EXPECT_CHANGE(expected.context_cache.enabled = false);
        EXPECT_CHANGE(++expected.context_cache.extra_device_state_slots);
        EXPECT_CHANGE(++expected.context_cache.host_state_slots);
        EXPECT_CHANGE(++expected.context_cache.host_kv_capacity_bytes);
        EXPECT_CHANGE(++expected.context_cache.max_private_continuations);
        EXPECT_CHANGE(++expected.context_cache.max_shared_prefixes);
        EXPECT_CHANGE(++expected.context_cache.max_long_anchors_per_continuation);
        EXPECT_CHANGE(++expected.context_cache.max_cache_markers_per_request);
#define FLAG(name) EXPECT_CHANGE(expected.execution_options.name = !expected.execution_options.name)
        FLAG(lm_head_q4); FLAG(lm_head_q6); FLAG(embedding_q4); FLAG(embedding_q6);
        FLAG(gdn_state_fp16); FLAG(mlp_a8_decode); FLAG(prefill_a8); FLAG(prefill_cublas);
        FLAG(prefill_cublas_projections); FLAG(mtp_experts_q4); FLAG(enable_vision);
#undef FLAG
        EXPECT_CHANGE(expected.execution_options.vision_residency = ninfer::VisionResidency::Overlay);
        EXPECT_CHANGE(++expected.execution_options.vision_max_merged_tokens);
        EXPECT_CHANGE(expected.execution_options.rope_scaling_factor = 2.25F);
        EXPECT_CHANGE(++expected.execution_options.rope_scaling_original_context);
#undef EXPECT_CHANGE
        ninfer::SpeculativeOptions options;
        options.backend = ninfer::SpeculativeBackend::DFlash2;
        options.draft_tokens = 15;
        options.routing.mode = ninfer::SpeculativeRoutingMode::Calibrated;
        options.routing.profile_path = fixture.path;
        ninfer::runtime::validate_speculative_routing_options(options);
        const auto reject_options = [](const ninfer::SpeculativeOptions& input) {
            try { ninfer::runtime::validate_speculative_routing_options(input); }
            catch (const std::invalid_argument&) { return; }
            throw std::runtime_error("invalid calibrated options were accepted");
        };
        auto changed = options; changed.routing.profile_path.clear(); reject_options(changed);
        changed = options; changed.routing.mode = ninfer::SpeculativeRoutingMode::Fixed; reject_options(changed);
        changed = options; changed.backend = ninfer::SpeculativeBackend::None; reject_options(changed);
        changed = options; changed.draft_tokens = 11; reject_options(changed);
        changed = options; changed.lookup_ngram = 3; reject_options(changed);
        changed = options; changed.tree.mode = ninfer::SpeculativeTreeMode::Lattice; reject_options(changed);
        changed = options; changed.routing.scope = ninfer::SpeculativeRouterScope::Engine; reject_options(changed);
        changed = options; changed.routing.widths[0] = 7; reject_options(changed);
        changed = options; changed.routing.state_path = "state"; reject_options(changed);
        std::cout << "ok calibrated profile: legacy K0 fallback, schema3 K7 baseline/overrides, strict identity and proposal compute\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
