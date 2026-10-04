#pragma once

#include "runtime/contract/speculative_routing.h"
#include <nlohmann/json.hpp>

namespace ninfer::runtime {

[[nodiscard]] nlohmann::json routing_profile_identity_json(const SpeculativeRoutingProfileIdentity& identity);

void validate_speculative_routing_options(const SpeculativeOptions& options);
[[nodiscard]] CalibratedRoutingTable load_calibrated_routing_profile(
    const std::filesystem::path& path, const SpeculativeRoutingProfileIdentity& identity);

} // namespace ninfer::runtime
