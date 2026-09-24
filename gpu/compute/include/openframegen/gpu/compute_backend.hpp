// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include <cstdint>

namespace ofg::compute {

enum class BackendApi : std::uint8_t {
    Unknown,
    Vulkan,
    Direct3D12,
    Metal,
};

enum class PixelFormat : std::uint8_t {
    Unknown,
    Bgra8Unorm,
    Bgra8Srgb,
    Rgba8Unorm,
    Rgba8Srgb,
    Rgb10A2Unorm,
    Rgba16Float,
};

struct Extent2D {
    std::uint32_t width = 0;
    std::uint32_t height = 0;

    [[nodiscard]] constexpr bool valid() const noexcept {
        return width != 0 && height != 0;
    }
};

struct ImageDescription {
    Extent2D extent{};
    PixelFormat format = PixelFormat::Unknown;

    [[nodiscard]] constexpr bool valid() const noexcept {
        return extent.valid() && format != PixelFormat::Unknown;
    }
};

struct BackendCapabilities {
    bool pass_through = false;
    bool bilinear_scaling = false;
    bool bicubic_scaling = false;
    bool sharpening = false;
    bool timestamp_queries = false;
};

class ComputeBackend {
public:
    virtual ~ComputeBackend() = default;

    [[nodiscard]] virtual BackendApi api() const noexcept = 0;
    [[nodiscard]] virtual const char* name() const noexcept = 0;
    [[nodiscard]] virtual BackendCapabilities capabilities()
        const noexcept = 0;
    [[nodiscard]] virtual bool supports_format(
        PixelFormat format) const noexcept = 0;
};

} // namespace ofg::compute
