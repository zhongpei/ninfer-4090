#pragma once

#include "models/qwen3_5/model.h"
#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/program/runtime_types.h"
#include "runtime/engine/context_cache/context_cost.h"
#include "runtime/engine/kv_capacity.h"

#include <memory>

namespace ninfer::runtime {

[[nodiscard]] EngineOptions normalize_engine_options(EngineOptions options);

struct ModelInstance {
    using ModelContract = models::qwen3_5::RuntimeTypes;

    std::shared_ptr<models::qwen3_5::Model> model;
    const models::qwen3_5::execution::Parameters parameters;
    models::qwen3_5::Frontend frontend;
    KvCapacityResolution kv_capacity_resolution;
    const std::uint32_t capacity;
    std::unique_ptr<models::qwen3_5::Program> program;

    ModelInstance(std::shared_ptr<models::qwen3_5::Model> model, const EngineOptions& options,
                  DeviceContext* execution_device = nullptr);
    ~ModelInstance();
    ModelInstance(const ModelInstance&)            = delete;
    ModelInstance& operator=(const ModelInstance&) = delete;

private:
    // Borrowed resident Programs must complete before their exclusive model lease is released.
    DeviceContext* execution_device_ = nullptr;
};

struct ConstructedModel {
    std::unique_ptr<ModelInstance> instance;
    LoadSummary load;
    ContextMachineCostModel context_cost;
    // Normalized startup options after hardware-dependent policies (currently auto prefill chunk)
    // have resolved. Engine publishes these rather than the unresolved user sentinel/configuration.
    EngineOptions resolved_options;
};

[[nodiscard]] ConstructedModel construct_model(const EngineOptions& options, DeviceContext& device);

} // namespace ninfer::runtime
