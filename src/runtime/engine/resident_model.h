#pragma once

#include "ninfer/engine.h"
#include "runtime/engine/model_instance.h"

#include <cstddef>
#include <memory>

namespace ninfer::runtime {

// Internal sequential test/benchmark owner. Immutable weights stay resident; each borrowed
// Engine or ModelInstance owns an independent Program and must close before the next borrow.
class ResidentModelSession {
public:
    explicit ResidentModelSession(const EngineOptions& load_options);
    ResidentModelSession(const ResidentModelSession&) = delete;
    ResidentModelSession& operator=(const ResidentModelSession&) = delete;
    [[nodiscard]] Engine make_engine(const EngineOptions& options);
    [[nodiscard]] ConstructedModel make_instance(const EngineOptions& options,
                                                DeviceContext& device);
    [[nodiscard]] std::size_t model_load_count() const noexcept;
    [[nodiscard]] std::size_t resident_weight_bytes() const noexcept;

private:
    struct State;
    std::shared_ptr<State> state_;
    void validate_instance_options(const EngineOptions& options) const;
};

} // namespace ninfer::runtime
