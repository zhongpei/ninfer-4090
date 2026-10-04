#include "runtime/engine/speculative_routing_profile.h"

#include <nlohmann/json.hpp>
#include <fstream>
#include <cmath>
#include <stdexcept>

namespace ninfer::runtime {
namespace {
using Json = nlohmann::json;
[[noreturn]] void invalid(const std::string& message) {
    throw std::invalid_argument("speculative routing profile: " + message);
}
const char* kv_name(KvCacheStorage value) {
    switch (value) {
    case KvCacheStorage::BFloat16: return "bf16";
    case KvCacheStorage::Int8Group64: return "int8";
    case KvCacheStorage::Fp8E4M3Row256: return "fp8";
    case KvCacheStorage::RotatedInt8KeyInt4ValueGroup64: return "rk8v4";
    case KvCacheStorage::RotatedInt4KeyInt4ValueGroup64: return "rk4v4";
    case KvCacheStorage::RK4V4E8: return "rk4v4-e8";
    case KvCacheStorage::RK2V4E8: return "rk2v4-e8";
    case KvCacheStorage::Nvfp4Group16: return "nvfp4";
    case KvCacheStorage::Fp8KeyNvfp4Value: return "k8v4";
    }
    invalid("invalid KV storage identity");
}
void same_identity(const Json& actual, const Json& expected, const std::string& key) {
    if (expected.is_object()) {
        if (!actual.is_object() || actual.size() != expected.size()) {
            invalid(key + " fields do not match");
        }
        for (const auto& [name, value] : expected.items()) {
            if (!actual.contains(name)) { invalid(key + " missing " + name); }
            same_identity(actual.at(name), value, key + "." + name);
        }
        return;
    }
    if (expected.is_number_float()) {
        if (!actual.is_number() || !std::isfinite(actual.get<double>()) ||
            actual.get<float>() != expected.get<float>()) { invalid(key + " mismatch"); }
    } else if (expected.is_number_integer()) {
        if (!actual.is_number_integer() || actual != expected) { invalid(key + " mismatch"); }
    } else if (actual.type() != expected.type() || actual != expected) {
        invalid(key + " mismatch");
    }
}
std::uint32_t integer(const Json& value, const char* field) {
    if (!value.is_number_integer() || value < 0 || value > UINT32_MAX) {
        invalid(std::string(field) + " must be a uint32 integer");
    }
    return value.get<std::uint32_t>();
}
} // namespace

nlohmann::json routing_profile_identity_json(const SpeculativeRoutingProfileIdentity& identity) {
    if (identity.backend != SpeculativeBackend::DFlash2) { invalid("identity requires DFlash2 backend"); }
    const auto& cache = identity.context_cache;
    const auto& execution = identity.execution_options;
#define FIELD(owner, name) {#name, owner.name}
    Json cache_json = {FIELD(cache, enabled), FIELD(cache, extra_device_state_slots),
        FIELD(cache, host_state_slots), FIELD(cache, host_kv_capacity_bytes),
        FIELD(cache, max_private_continuations), FIELD(cache, max_shared_prefixes),
        FIELD(cache, max_long_anchors_per_continuation), FIELD(cache, max_cache_markers_per_request)};
    Json execution_json = {FIELD(execution, lm_head_q4), FIELD(execution, lm_head_q6),
        FIELD(execution, embedding_q4), FIELD(execution, embedding_q6),
        FIELD(execution, gdn_state_fp16), FIELD(execution, mlp_a8_decode),
        FIELD(execution, prefill_a8), FIELD(execution, prefill_cublas),
        FIELD(execution, prefill_cublas_projections), FIELD(execution, mtp_experts_q4),
        FIELD(execution, enable_vision),
        {"vision_residency", execution.vision_residency == VisionResidency::Resident ? "resident" : "overlay"},
        FIELD(execution, vision_max_merged_tokens), FIELD(execution, rope_scaling_factor),
        FIELD(execution, rope_scaling_original_context)};
    Json result = {FIELD(identity, artifact_id), FIELD(identity, prefill_signature),
        FIELD(identity, hardware_class), {"backend", "dflash2"},
        {"kv_storage", kv_name(identity.kv_storage)}, FIELD(identity, startup_draft_tokens),
        {"proposal_head", identity.proposal_head == ProposalHead::Full ? "full" : "optimized"},
        FIELD(identity, use_cuda_graph), FIELD(identity, max_concurrency), FIELD(identity, max_context),
        FIELD(identity, prefill_chunk), FIELD(identity, resolved_kv_capacity),
        {"context_cache", std::move(cache_json)}, {"execution_options", std::move(execution_json)}};
#undef FIELD
    return result;
}

void validate_speculative_routing_options(const SpeculativeOptions& options) {
    const auto& routing = options.routing;
    if (routing.mode != SpeculativeRoutingMode::Calibrated) {
        if (!routing.profile_path.empty()) { invalid("profile_path requires calibrated mode"); }
        return;
    }
    if (routing.profile_path.empty()) { invalid("calibrated mode requires profile_path"); }
    if (options.backend != SpeculativeBackend::DFlash2 || options.draft_tokens != 15 ||
        options.tree.mode != SpeculativeTreeMode::Off || options.lookup_ngram != 0 ||
        options.lookup.dflash_mode != LookupDFlashMode::Off ||
        options.lookup.persistent_tokens != 0 || !options.lookup.persistent_path.empty() ||
        !options.lookup.corpus_prefix.empty()) {
        invalid("calibrated mode requires DFlash2 chain K15 without tree or lookup");
    }
    const SpeculativeRoutingOptions defaults;
    if (routing.scope != defaults.scope || !routing.state_path.empty() ||
        routing.widths != defaults.widths || routing.verify_costs != defaults.verify_costs ||
        routing.draft_cost != defaults.draft_cost || routing.prior_acceptance != defaults.prior_acceptance ||
        routing.prior_weight != defaults.prior_weight || routing.warmup_rounds != defaults.warmup_rounds ||
        routing.probe_period != defaults.probe_period || routing.switch_margin != defaults.switch_margin) {
        invalid("calibrated mode does not accept Stair policy options");
    }
}

CalibratedRoutingTable load_calibrated_routing_profile(
    const std::filesystem::path& path, const SpeculativeRoutingProfileIdentity& identity) {
    std::ifstream stream(path);
    if (!stream) { invalid("cannot open " + path.string()); }
    Json root;
    try { root = Json::parse(stream); }
    catch (const Json::exception& error) { invalid(error.what()); }
    if (!root.is_object() || (root.size() != 4 && root.size() != 5) ||
        !root.contains("schema_version") || !root.contains("artifact_type") ||
        !root.contains("identity") || !root.contains("cells") ||
        (root.size() == 5 && !root.contains("provenance"))) { invalid("invalid top-level fields"); }
    if (integer(root.at("schema_version"), "schema_version") != 1 ||
        root.at("artifact_type") != "ninfer_spec_router_profile") { invalid("unsupported schema"); }
    if (root.contains("provenance") && !root.at("provenance").is_object()) {
        invalid("provenance must be an object");
    }
    same_identity(root.at("identity"), routing_profile_identity_json(identity), "identity");
    if (!root.at("cells").is_array()) { invalid("cells must be an array"); }
    CalibratedRoutingTable table;
    std::array<std::array<bool, 3>, kMaximumConcurrency> seen{};
    for (const auto& cell : root.at("cells")) {
        if (!cell.is_object() || cell.size() != 3 || !cell.contains("active_batch") ||
            !cell.contains("frontier_upper") || !cell.contains("draft_tokens")) {
            invalid("invalid cell fields");
        }
        const auto batch = integer(cell.at("active_batch"), "active_batch");
        const auto frontier = integer(cell.at("frontier_upper"), "frontier_upper");
        const auto draft = integer(cell.at("draft_tokens"), "draft_tokens");
        if (batch < 1 || batch > kMaximumConcurrency ||
            (frontier != 1024 && frontier != 8192 && frontier != 32768) ||
            (draft != 0 && draft != 7 && draft != 11 && draft != 15)) { invalid("invalid cell value"); }
        const std::size_t column = frontier == 1024 ? 0 : frontier == 8192 ? 1 : 2;
        if (seen[batch - 1][column]) { invalid("duplicate cell"); }
        seen[batch - 1][column] = true;
        table.draft_tokens[batch - 1][column] = draft;
    }
    return table;
}
} // namespace ninfer::runtime
