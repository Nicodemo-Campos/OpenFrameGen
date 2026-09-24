// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

namespace ofg {

enum class Platform {
    Linux,
    Windows,
    MacOS,
    Unknown
};

struct RuntimeInfo {
    Platform platform;
    const char* platform_name;
    const char* version;
};

[[nodiscard]] Platform current_platform() noexcept;
[[nodiscard]] const char* platform_name() noexcept;
[[nodiscard]] const char* version() noexcept;
[[nodiscard]] RuntimeInfo runtime_info() noexcept;

} // namespace ofg
