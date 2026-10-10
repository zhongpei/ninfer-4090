#pragma once

// Experimental opt-in device-wide route profile for the current CUDA device.
// Installs before model planning: consumers must not change profiles during active kernels.
namespace ninfer::ops {
void configure_device_routes_from_environment();
}
