// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <openframegen/core/runtime.hpp>

#ifndef OFG_VERSION_STRING
#define OFG_VERSION_STRING "0.0.0"
#endif

namespace ofg {

Platform current_platform() noexcept {
#if defined(_WIN32)
    return Platform::Windows;
#elif defined(__APPLE__)
    return Platform::MacOS;
#elif defined(__linux__)
    return Platform::Linux;
#else
    return Platform::Unknown;
#endif
}

const char* platform_name() noexcept {
    switch (current_platform()) {
        case Platform::Linux:
            return "Linux";
        case Platform::Windows:
            return "Windows";
        case Platform::MacOS:
            return "macOS";
        case Platform::Unknown:
        default:
            return "Unknown";
    }
}

const char* version() noexcept {
    return OFG_VERSION_STRING;
}

RuntimeInfo runtime_info() noexcept {
    return RuntimeInfo{
        .platform = current_platform(),
        .platform_name = platform_name(),
        .version = version(),
    };
}

} // namespace ofg
