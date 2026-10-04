#pragma once

#include "runtime/engine/resident_model.h"
#include <nlohmann/json.hpp>

namespace ninfer::test {
// Qualify both proposal modes against one immutable weight residency. Each case owns fresh state.
nlohmann::json run_calibrated_resident_suite(runtime::ResidentModelSession& session,
                                           const char* artifact);
} // namespace ninfer::test
