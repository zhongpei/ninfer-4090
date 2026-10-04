#include "runtime/engine/model_instance.h"
#include "runtime/engine/resident_model.h"
#include "runtime/engine/speculative_routing_profile.h"
#include "artifact/reader.h"
#include "artifact/formats.h"
#include "core/startup.h"
#include "models/qwen3_5/load.h"
#include "models/qwen3_5/measurement.h"

#include <algorithm>
#include <chrono>
#include <set>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace ninfer::runtime {
namespace {
using Clock = std::chrono::steady_clock;

void validate_options(const EngineOptions& options) {
    validate_speculative_routing_options(options.speculative);
    if (options.artifact_path.empty()) {
        throw std::invalid_argument("Engine artifact_path must not be empty");
    }
    if (options.artifact_path.extension() != ".ninfer") {
        throw std::invalid_argument("NInfer accepts only .ninfer artifacts");
    }
    if (options.max_context == 0) {
        throw std::invalid_argument("Engine max_context must be nonzero");
    }
    switch (options.kv_capacity.mode) {
    case KvCapacityMode::Explicit:
        if (options.kv_capacity.explicit_tokens == 0) {
            throw std::invalid_argument("Engine explicit kv_capacity must be nonzero");
        }
        if (options.kv_capacity.automatic_headroom_bytes != 0) {
            throw std::invalid_argument(
                "Engine explicit kv_capacity must not carry automatic headroom");
        }
        break;
    case KvCapacityMode::Automatic:
        if (options.kv_capacity.explicit_tokens != 0) {
            throw std::invalid_argument(
                "Engine automatic kv_capacity must not carry explicit tokens");
        }
        break;
    default:
        throw std::invalid_argument("Engine kv_capacity mode is invalid");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("Engine max_concurrency must be in [1,8]");
    }
    if (options.max_pending_requests == 0 || options.pending_timeout_ms == 0) {
        throw std::invalid_argument("Engine pending request capacity and timeout must be nonzero");
    }
    if (options.enable_vision && options.media_live_bytes == 0) {
        throw std::invalid_argument(
            "Engine media_live_bytes must be nonzero when Vision is enabled");
    }
    if (options.media_preprocess_threads > 64) {
        throw std::invalid_argument("Engine media_preprocess_threads must be in [0,64]");
    }
}

SpeculativeRoutingProfileIdentity routing_identity(const EngineOptions& options,
    const artifact::ArtifactId& artifact_id, const std::string& signature,
    const std::string& hardware_class, std::uint32_t resolved_kv_capacity) {
    SpeculativeRoutingProfileIdentity result;
    constexpr char hex[] = "0123456789abcdef";
    for (const auto byte : artifact_id) {
        const auto value = std::to_integer<unsigned>(byte);
        result.artifact_id.push_back(hex[value >> 4]);
        result.artifact_id.push_back(hex[value & 15]);
    }
    result.prefill_signature = signature;
    result.hardware_class = hardware_class;
    result.backend = options.speculative.backend;
    result.kv_storage = options.kv_cache;
    result.startup_draft_tokens = options.speculative.draft_tokens;
    result.proposal_head = options.speculative.proposal_head;
    result.use_cuda_graph = options.use_cuda_graph;
    result.max_concurrency = options.max_concurrency;
    result.max_context = options.max_context;
    result.prefill_chunk = options.prefill_chunk;
    result.resolved_kv_capacity = resolved_kv_capacity;
    const auto& cache = options.context_cache;
    result.context_cache = {cache.enabled, cache.device_state_slots.value(), cache.host_state_slots,
        cache.host_kv_capacity_bytes, cache.max_private_continuations.value(),
        cache.max_shared_prefixes.value(), cache.max_long_anchors_per_continuation.value(),
        cache.max_cache_markers_per_request.value()};
#define COPY(name) result.execution_options.name = options.name
    COPY(lm_head_q4); COPY(lm_head_q6); COPY(embedding_q4); COPY(embedding_q6);
    COPY(gdn_state_fp16); COPY(mlp_a8_decode); COPY(prefill_a8); COPY(prefill_cublas);
    COPY(prefill_cublas_projections); COPY(mtp_experts_q4); COPY(enable_vision);
    COPY(vision_residency); COPY(vision_max_merged_tokens); COPY(rope_scaling_factor);
    COPY(rope_scaling_original_context);
#undef COPY
    return result;
}

std::size_t current_free_device_bytes() {
    std::size_t free_bytes  = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    return free_bytes;
}

} // namespace

EngineOptions normalize_engine_options(EngineOptions options) {
    switch (options.purpose) {
    case EnginePurpose::Generation:
        break;
    case EnginePurpose::CausalScoring:
        options.max_concurrency      = 1;
        options.max_pending_requests = 1;
        options.prefill_chunk        = 1024;
        options.kv_capacity          = KvCapacityPolicy::explicit_capacity(options.max_context);
        options.speculative          = {};
        options.enable_vision        = false;
        options.use_cuda_graph       = false;
        options.context_cache        = ContextCacheOptions{.enabled = false};
        break;
    default:
        throw std::invalid_argument("Engine purpose is invalid");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("Engine max_concurrency must be in [1,8]");
    }

    ContextCacheOptions& cache      = options.context_cache;
    const std::uint32_t concurrency = options.max_concurrency;
    if (!cache.enabled) {
        if ((cache.device_state_slots && *cache.device_state_slots != 0) ||
            (cache.max_private_continuations && *cache.max_private_continuations != concurrency) ||
            (cache.max_shared_prefixes && *cache.max_shared_prefixes != 0) ||
            (cache.max_long_anchors_per_continuation &&
             *cache.max_long_anchors_per_continuation != 0)) {
            throw std::invalid_argument("disabled context cache accepts only root-only capacities");
        }
        cache.device_state_slots                = 0;
        cache.host_state_slots                  = 0;
        cache.host_kv_capacity_bytes            = 0;
        cache.max_private_continuations         = concurrency;
        cache.max_shared_prefixes               = 0;
        cache.max_long_anchors_per_continuation = 0;
        cache.max_cache_markers_per_request     = cache.max_cache_markers_per_request.value_or(4U);
        return options;
    }

    cache.device_state_slots            = cache.device_state_slots.value_or(concurrency);
    const std::uint64_t default_private = 2ULL * concurrency;
    cache.max_private_continuations =
        cache.max_private_continuations.value_or(static_cast<std::uint32_t>(default_private));
    cache.max_shared_prefixes = cache.max_shared_prefixes.value_or(
        std::max(concurrency, static_cast<std::uint32_t>(kMaximumExplicitPromptCacheMarkers)));
    cache.max_long_anchors_per_continuation = cache.max_long_anchors_per_continuation.value_or(2U);
    cache.max_cache_markers_per_request     = cache.max_cache_markers_per_request.value_or(4U);

    if (*cache.max_private_continuations < concurrency) {
        throw std::invalid_argument(
            "context cache max_private_continuations must cover every active request");
    }
    const std::uint64_t total_device_state_slots =
        static_cast<std::uint64_t>(concurrency) + *cache.device_state_slots;
    if (total_device_state_slots > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("context cache Device state capacity exceeds uint32");
    }
    const std::uint64_t address_spaces =
        static_cast<std::uint64_t>(*cache.max_private_continuations) + *cache.max_shared_prefixes;
    if (address_spaces > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("context cache address-space capacity exceeds uint32");
    }
    if (*cache.max_long_anchors_per_continuation != 0 &&
        *cache.max_private_continuations >
            std::numeric_limits<std::size_t>::max() / *cache.max_long_anchors_per_continuation) {
        throw std::overflow_error("context cache long-anchor capacity exceeds size_t");
    }
    return options;
}

ModelInstance::ModelInstance(std::shared_ptr<models::qwen3_5::Model> source,
                             const EngineOptions& options, DeviceContext* execution_device)
    : model(std::move(source)), parameters(*model, options.prefill_a8),
      frontend(models::qwen3_5::make_frontend(
          model->resources(), {.chat_template_path       = options.chat_template_path,
                               .architecture             = model->config().text.architecture,
                               .vision_enabled           = options.enable_vision,
                               .max_context              = options.max_context,
                               .media_cache_bytes        = options.media_cache_bytes,
                               .media_live_bytes         = options.media_live_bytes,
                               .media_preprocess_threads = options.media_preprocess_threads,
                               .vision_max_merged_tokens = options.vision_max_merged_tokens})),
      capacity(options.max_context), execution_device_(execution_device) {}

ModelInstance::~ModelInstance() {
    if (execution_device_ != nullptr) {
        execution_device_->bind_to_current_thread_noexcept();
        try { execution_device_->synchronize(); } catch (...) {}
        program.reset();
        try { execution_device_->synchronize(); } catch (...) {}
    }
}

namespace {
ConstructedModel prepare_model(const EngineOptions& options, DeviceContext& device,
                               std::shared_ptr<models::qwen3_5::Model> model,
                               Clock::time_point start, DeviceContext* execution_device = nullptr) {
    StartupPhaseScope frontend(options.startup_observer, StartupPhase::FrontendInitialize);
    auto instance = std::make_unique<ModelInstance>(std::move(model), options, execution_device);
    frontend.complete();
    StartupPhaseScope planning(options.startup_observer, StartupPhase::TargetFinalize);
    const std::size_t overlay_window_bytes =
        models::qwen3_5::prepare_vision_overlay(instance->parameters, device, options);
    const auto signature = models::qwen3_5::prefill_signature(*instance->model);
    auto context_cost    = resolve_context_machine_cost(
        {.hardware_class =
                context_cost_hardware_class(device.props.name, device.props.major, device.props.minor),
            .prefill_signature = signature},
        options.context_cost.preset_path);
    auto planner    = models::qwen3_5::make_sequence_planner(instance->parameters, device, options);
    auto resolution = resolve_kv_capacity(options.kv_capacity, planner.capacity_curve(),
                                          current_free_device_bytes());
    auto sequence   = std::move(planner).finalize(resolution.main_page_groups);
    if (sequence.device_reservation_bytes() != resolution.runtime_reservation_bytes ||
        sequence.kv_capacity() != resolution.resolved_tokens) {
        throw std::logic_error("resolved KV capacity does not match the finalized Program plan");
    }
    std::optional<SpeculativeRoutingProfileIdentity> speculative_identity;
    if (options.speculative.backend == SpeculativeBackend::DFlash2 &&
        options.speculative.draft_tokens == 15) {
        speculative_identity = routing_identity(options, instance->model->info().artifact_id,
            signature, context_cost.summary.hardware_class, resolution.resolved_tokens);
    }
    if (options.speculative.routing.mode == SpeculativeRoutingMode::Calibrated) {
        sequence.set_calibrated_routing(load_calibrated_routing_profile(
            options.speculative.routing.profile_path, *speculative_identity));
    }
    instance->kv_capacity_resolution = resolution;
    planning.complete();
    StartupPhaseScope program(options.startup_observer, StartupPhase::ProgramInitialize);
    instance->program = models::qwen3_5::create_program(instance->parameters, std::move(sequence),
                                                        device, options.startup_observer);
    device.synchronize();
    program.complete();
    instance->kv_capacity_resolution.available_after_startup_bytes = current_free_device_bytes();
    const auto& stats = instance->model->storage_stats();
    LoadSummary summary;
    summary.speculative_routing_identity = std::move(speculative_identity);
    summary.architecture = models::architecture_name(instance->model->config().text.architecture);
    summary.model_name   = instance->model->info().name;
    summary.prefill_signature = signature;
    std::set<std::string> formats;
    for (const auto& weight : instance->model->weight_data()) {
        for (const auto& part : weight.view.parts) {
            formats.emplace(artifact::format_name(part.parent->geometry.format));
        }
    }
    summary.weight_formats.assign(formats.begin(), formats.end());
    summary.load_seconds         = std::chrono::duration<double>(Clock::now() - start).count();
    summary.upload_seconds       = stats.upload_seconds;
    summary.artifact_bytes_read  = stats.read_bytes;
    summary.host_to_device_bytes = stats.h2d_bytes;
    summary.peak_staging_bytes   = stats.peak_staging_bytes;
    summary.pinned_weight_bytes  = stats.pinned_bytes;
    summary.overlay_window_bytes = overlay_window_bytes;
    summary.device_object_count  = stats.device_object_count;
    summary.host_object_count    = stats.host_object_count;
    summary.context_cost         = std::move(context_cost.summary);
    return {std::move(instance), std::move(summary), std::move(context_cost.model)};
}

} // namespace

ConstructedModel construct_model(const EngineOptions& options, DeviceContext& device) {
    validate_options(options);
    const auto start = Clock::now();
    StartupPhaseScope inspect(options.startup_observer, StartupPhase::ArtifactInspect);
    artifact::Reader reader(options.artifact_path);
    inspect.complete();
    StartupPhaseScope binding(options.startup_observer, StartupPhase::TargetPlan);
    auto plan = models::qwen3_5::plan_load(reader, models::load_options(options));
    binding.complete();
    auto model =
        models::qwen3_5::materialize_model(std::move(plan), device, &options.startup_observer);
    device.synchronize();
    return prepare_model(options, device, std::move(model), start);
}

struct ResidentModelSession::State {
    explicit State(EngineOptions source)
        : options(std::move(source)), device(options.devices.empty() ? options.device
                                                                  : options.devices.front()) {
        StartupPhaseScope inspect(options.startup_observer, StartupPhase::ArtifactInspect);
        artifact::Reader reader(options.artifact_path);
        inspect.complete();
        StartupPhaseScope binding(options.startup_observer, StartupPhase::TargetPlan);
        auto plan = models::qwen3_5::plan_load(reader, models::load_options(options));
        binding.complete();
        model = models::qwen3_5::materialize_model(std::move(plan), device,
                                                   &options.startup_observer);
        ++materialize_count;
        device.synchronize();
    }
    ~State() {
        device.bind_to_current_thread_noexcept();
        try { device.synchronize(); } catch (...) {}
        model.reset();
    }

    EngineOptions options;
    DeviceContext device;
    std::unique_ptr<models::qwen3_5::Model> model;
    std::size_t materialize_count = 0;
    std::mutex mutex;
    bool borrowed = false;
};

namespace {
void validate_resident_load(const EngineOptions& options) {
    validate_options(options);
    if (options.purpose != EnginePurpose::Generation || options.enable_vision ||
        options.devices.size() > 1 || options.speculative.backend != SpeculativeBackend::DFlash2 ||
        options.speculative.proposal_head != ProposalHead::Full) {
        throw std::invalid_argument(
            "resident model requires single-GPU Generation with DFlash2 Full and no Vision");
    }
}
}

ResidentModelSession::ResidentModelSession(const EngineOptions& load_options) {
    validate_resident_load(load_options);
    state_ = std::make_shared<State>(normalize_engine_options(load_options));
}

void ResidentModelSession::validate_instance_options(const EngineOptions& options) const {
    validate_options(options);
    const auto& loaded = state_->options;
    const int requested_device = options.devices.empty() ? options.device : options.devices.front();
    if (options.purpose != EnginePurpose::Generation || options.enable_vision ||
        options.devices.size() > 1 || requested_device != state_->device.device ||
        std::filesystem::absolute(options.artifact_path).lexically_normal() !=
            std::filesystem::absolute(loaded.artifact_path).lexically_normal() ||
        !models::resident_load_options_compatible(state_->model->options(),
                                                    models::load_options(options))) {
        throw std::invalid_argument("resident model does not match requested artifact, GPU or load options");
    }
}

ConstructedModel ResidentModelSession::make_instance(const EngineOptions& source,
                                                    DeviceContext& device) {
    validate_instance_options(source);
    if (device.size() != 1 || device.device != state_->device.device) {
        throw std::invalid_argument("resident Program requires the resident model GPU");
    }
    const auto options = normalize_engine_options(source);
    struct Lease {
        explicit Lease(std::shared_ptr<State> owner) : state(std::move(owner)) {
            std::lock_guard lock(state->mutex);
            if (state->borrowed) throw std::logic_error("resident model already has an active Program");
            state->borrowed = true;
        }
        ~Lease() {
            std::lock_guard lock(state->mutex);
            state->borrowed = false;
        }
        std::shared_ptr<State> state;
    };
    auto lease = std::make_shared<Lease>(state_);
    std::shared_ptr<models::qwen3_5::Model> borrowed(lease, state_->model.get());
    auto constructed = prepare_model(options, device, std::move(borrowed), Clock::now(), &device);
    // The session performed these transfers once. Creating a Program does not repeat them.
    constructed.load.upload_seconds = 0;
    constructed.load.artifact_bytes_read = 0;
    constructed.load.host_to_device_bytes = 0;
    constructed.load.peak_staging_bytes = 0;
    return constructed;
}

std::size_t ResidentModelSession::model_load_count() const noexcept {
    return state_->materialize_count;
}

std::size_t ResidentModelSession::resident_weight_bytes() const noexcept {
    return state_->model->storage_stats().device_capacity_bytes;
}

} // namespace ninfer::runtime
