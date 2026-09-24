// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <openframegen/core/runtime.hpp>

#include <cassert>
#include <cstring>

int main() {
    const auto info = ofg::runtime_info();

    assert(info.version != nullptr);
    assert(std::strlen(info.version) > 0);

    assert(info.platform_name != nullptr);
    assert(std::strlen(info.platform_name) > 0);

#if defined(_WIN32) || defined(__APPLE__) || defined(__linux__)
    assert(info.platform != ofg::Platform::Unknown);
#endif

    return 0;
}
