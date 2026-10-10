#include "ops/common/device_route_profile.h"
#include "ops/common/device_route.h"
#include "ops/common/device_routes_sm89_builtin.h"

#include <cuda_runtime_api.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::ops {
namespace {
using Json = nlohmann::json;

std::string trim(std::string text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const auto last = text.find_last_not_of(" \t\r\n");
    return text.substr(first, last - first + 1);
}

std::string env(std::string_view key) {
    const std::string name(key);
    const char* raw = std::getenv(name.c_str());
    return raw == nullptr ? std::string() : trim(raw);
}

std::string class_name(const cudaDeviceProp& prop) {
    std::string result;
    bool previous_dash = false;
    for (const char* c = prop.name; *c != '\0'; ++c) {
        const auto value = static_cast<unsigned char>(*c);
        if (std::isalnum(value)) {
            result += static_cast<char>(std::tolower(value));
            previous_dash = false;
        } else if (!previous_dash) {
            result += '-';
            previous_dash = true;
        }
    }
    if (!result.empty() && result.back() == '-') result.pop_back();
    // CUDA usually returns "NVIDIA GeForce RTX 4090"; avoid nvidia-nvidia-.
    if (result.rfind("nvidia-", 0) != 0) result = "nvidia-" + result;
    result += "-sm" + std::to_string(prop.major) + std::to_string(prop.minor);
    return result;
}

std::shared_ptr<DeviceRouteProfile> parse_profile(const Json& source,
                                                   const std::string& hardware_class,
                                                   int multiprocessors) {
    if (!source.is_object()) throw std::invalid_argument("device routes: root must be an object");
    if (source.contains("schema") &&
        (source.value("schema", std::string()) != "ninfer.device-route-profiles" ||
         source.value("schema_version", 0) != 2)) {
        throw std::invalid_argument("device routes: schema/version mismatch");
    }
    const Json devices = source.contains("devices") ? source.at("devices") : Json::array({source});
    if (!devices.is_array()) throw std::invalid_argument("device routes: devices must be an array");
    for (const Json& d : devices) {
        if (!d.is_object() || !d.contains("hardware_class") || !d.contains("routes"))
            throw std::invalid_argument("device routes: device requires hardware_class and routes");
        if (!d.at("hardware_class").is_string() ||
            d.at("hardware_class").get<std::string>() != hardware_class ||
            d.value("multiprocessors", 0) != multiprocessors) continue;
        if (!d.at("routes").is_object())
            throw std::invalid_argument("device routes: routes must be an object");
        auto out = std::make_shared<DeviceRouteProfile>();
        out->hardware_class = hardware_class;
        out->multiprocessors = multiprocessors;
        out->origin = d.value("origin", std::string("profile file"));
        for (auto it = d.at("routes").begin(); it != d.at("routes").end(); ++it) {
            if (!it.value().is_array() || it.value().empty())
                throw std::invalid_argument("device routes: empty/invalid route band");
            int previous = 0;
            std::vector<DeviceRouteBand> bands;
            for (const auto& row : it.value()) {
                if (!row.is_array() || row.size() != 2 || !row[0].is_number_integer() ||
                    !row[1].is_string())
                    throw std::invalid_argument("device routes: expected [last_width,schedule]");
                const int last = row[0].get<int>();
                if (last <= previous)
                    throw std::invalid_argument("device routes: width bands must be increasing");
                bands.push_back(DeviceRouteBand{last, row[1].get<std::string>()});
                previous = last;
            }
            out->routes.emplace(it.key(), std::move(bands));
        }
        return out;
    }
    return nullptr;
}

std::shared_ptr<DeviceRouteProfile> read_profile_file(const std::string& path,
                                                       const std::string& hardware_class,
                                                       int sm) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("device routes: cannot open " + path);
    return parse_profile(Json::parse(in), hardware_class, sm);
}

void restrict_to_keys(DeviceRouteProfile& profile, const std::string& only) {
    if (only.empty()) return;
    std::vector<std::string> selected;
    for (std::size_t first = 0; first < only.size();) {
        const std::size_t last = only.find(',', first);
        const std::string key = trim(only.substr(first, last - first));
        if (!key.empty()) selected.push_back(key);
        if (last == std::string::npos) break;
        first = last + 1;
    }
    for (auto it = profile.routes.begin(); it != profile.routes.end();) {
        if (std::find(selected.begin(), selected.end(), it->first) == selected.end())
            it = profile.routes.erase(it);
        else ++it;
    }
}

// Full-band overrides are for bisecting candidate routes. A JSON profile supports width bands.
void apply_overrides(DeviceRouteProfile& profile, const std::string& overrides) {
    for (std::size_t first = 0; first < overrides.size();) {
        const std::size_t last = overrides.find(';', first);
        const std::string item = trim(overrides.substr(first, last - first));
        if (!item.empty()) {
            const auto equal = item.find('=');
            if (equal == std::string::npos || equal == 0)
                throw std::invalid_argument("device routes: override must be key=schedule");
            const std::string key = trim(item.substr(0, equal));
            const std::string schedule = trim(item.substr(equal + 1));
            if (key.empty()) throw std::invalid_argument("device routes: empty override key");
            profile.routes[key] = {{INT_MAX, schedule}};
        }
        if (last == std::string::npos) break;
        first = last + 1;
    }
}

} // namespace

void configure_device_routes_from_environment() {
    const std::string mode = env("NINFER_DEVICE_ROUTE_MODE");
    if (mode.empty() || mode == "off") {
        install_device_route_profile(nullptr);
        return;
    }
    if (mode != "auto" && mode != "builtin" && mode != "file")
        throw std::invalid_argument("NINFER_DEVICE_ROUTE_MODE: expected off|auto|builtin|file");
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess)
        throw std::runtime_error("device routes: cannot determine CUDA device");
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess)
        throw std::runtime_error("device routes: cannot query CUDA device");
    const std::string hardware_class = class_name(prop);
    const std::string file_path = !env("NINFER_DEVICE_PROFILE_PATH").empty()
                                      ? env("NINFER_DEVICE_PROFILE_PATH")
                                      : env("NINFER_DEVICE_PROFILES");
    std::shared_ptr<DeviceRouteProfile> profile;
    if (mode == "file" || (mode == "auto" && !file_path.empty())) {
        if (file_path.empty())
            throw std::invalid_argument("device routes: file mode requires PROFILE_PATH");
        profile = read_profile_file(file_path, hardware_class, prop.multiProcessorCount);
        if (!profile && mode == "file")
            throw std::invalid_argument("device routes: profile GPU identity does not match");
    }
    if (!profile && (mode == "builtin" || mode == "auto") &&
        hardware_class == "nvidia-geforce-rtx-4090-sm89" &&
        prop.multiProcessorCount == 128) {
        profile = parse_profile(Json::parse(kSm89BuiltinRouteJson.begin(),
                                           kSm89BuiltinRouteJson.end()),
                                hardware_class, prop.multiProcessorCount);
    }
    if (!profile) {
        if (mode != "auto")
            throw std::invalid_argument("device routes: no compatible GPU profile");
        install_device_route_profile(device, nullptr);
        return;
    }
    restrict_to_keys(*profile, env("NINFER_DEVICE_ROUTE_ONLY"));
    apply_overrides(*profile, env("NINFER_DEVICE_ROUTE_OVERRIDES"));
    std::fprintf(stderr, "ninfer device routes: mode=%s device=%d gpu=%s source=%s keys=%zu\n",
                 mode.c_str(), device, hardware_class.c_str(), profile->origin.c_str(),
                 profile->routes.size());
    install_device_route_profile(device, std::move(profile));
}
} // namespace ninfer::ops
