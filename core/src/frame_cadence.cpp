// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <openframegen/core/frame_cadence.hpp>

namespace ofg {

bool FrameCadence2xPlanner::observe_source_frame(
    std::uint64_t source_time_ns,
    FrameCadence2xPlan& plan) noexcept {
    plan = {};

    if (!has_previous_source_) {
        previous_source_time_ns_ = source_time_ns;
        has_previous_source_ = true;
        return false;
    }

    if (source_time_ns <= previous_source_time_ns_) {
        previous_source_time_ns_ = source_time_ns;
        return false;
    }

    const std::uint64_t interval =
        source_time_ns - previous_source_time_ns_;
    const std::uint64_t first_half = interval / 2u;
    const std::uint64_t second_half = interval - first_half;

    plan.previous_source_time_ns = previous_source_time_ns_;
    plan.interpolated_time_ns =
        previous_source_time_ns_ + first_half;
    plan.current_source_time_ns = source_time_ns;
    plan.source_interval_ns = interval;
    plan.previous_to_interpolated_ns = first_half;
    plan.interpolated_to_current_ns = second_half;

    previous_source_time_ns_ = source_time_ns;
    return true;
}

void FrameCadence2xPlanner::reset() noexcept {
    previous_source_time_ns_ = 0;
    has_previous_source_ = false;
}

bool FrameCadence2xPlanner::primed() const noexcept {
    return has_previous_source_;
}

std::uint64_t FrameCadence2xPlanner::last_source_time_ns() const noexcept {
    return previous_source_time_ns_;
}

} // namespace ofg
