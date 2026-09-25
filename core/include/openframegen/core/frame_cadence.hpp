// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include <cstdint>

namespace ofg {

struct FrameCadence2xPlan {
    std::uint64_t previous_source_time_ns = 0;
    std::uint64_t interpolated_time_ns = 0;
    std::uint64_t current_source_time_ns = 0;
    std::uint64_t source_interval_ns = 0;
    std::uint64_t previous_to_interpolated_ns = 0;
    std::uint64_t interpolated_to_current_ns = 0;
};

class FrameCadence2xPlanner {
public:
    [[nodiscard]] bool observe_source_frame(
        std::uint64_t source_time_ns,
        FrameCadence2xPlan& plan) noexcept;

    void reset() noexcept;

    [[nodiscard]] bool primed() const noexcept;
    [[nodiscard]] std::uint64_t last_source_time_ns() const noexcept;

private:
    std::uint64_t previous_source_time_ns_ = 0;
    bool has_previous_source_ = false;
};

} // namespace ofg
