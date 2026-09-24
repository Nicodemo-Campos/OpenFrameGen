// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <openframegen/gpu/compute_backend.hpp>

#include <cassert>
#include <cstring>

namespace {

class FakeComputeBackend final : public ofg::compute::ComputeBackend {
public:
    [[nodiscard]] ofg::compute::BackendApi api() const noexcept override {
        return ofg::compute::BackendApi::Vulkan;
    }

    [[nodiscard]] const char* name() const noexcept override {
        return "fake-vulkan";
    }

    [[nodiscard]] ofg::compute::BackendCapabilities capabilities()
        const noexcept override {
        return ofg::compute::BackendCapabilities{
            .pass_through = true,
            .timestamp_queries = true,
        };
    }

    [[nodiscard]] bool supports_format(
        ofg::compute::PixelFormat format) const noexcept override {
        return format == ofg::compute::PixelFormat::Bgra8Unorm ||
               format == ofg::compute::PixelFormat::Rgba8Unorm;
    }
};

} // namespace

int main() {
    using namespace ofg::compute;

    constexpr ImageDescription valid_image{
        .extent = Extent2D{1920, 1080},
        .format = PixelFormat::Bgra8Unorm,
    };

    constexpr ImageDescription invalid_image{};

    static_assert(valid_image.valid());
    static_assert(!invalid_image.valid());

    FakeComputeBackend backend;

    assert(backend.api() == BackendApi::Vulkan);
    assert(backend.name() != nullptr);
    assert(std::strcmp(backend.name(), "fake-vulkan") == 0);

    const auto capabilities = backend.capabilities();
    assert(capabilities.pass_through);
    assert(capabilities.timestamp_queries);
    assert(!capabilities.bilinear_scaling);
    assert(!capabilities.bicubic_scaling);
    assert(!capabilities.sharpening);

    assert(backend.supports_format(PixelFormat::Bgra8Unorm));
    assert(backend.supports_format(PixelFormat::Rgba8Unorm));
    assert(!backend.supports_format(PixelFormat::Rgba16Float));

    return 0;
}
