// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <openframegen/core/frame_cadence.hpp>

#include <cassert>

int main() {
    ofg::FrameCadence2xPlanner planner;
    ofg::FrameCadence2xPlan plan{};

    assert(!planner.primed());
    assert(!planner.observe_source_frame(1'000'000u, plan));
    assert(planner.primed());
    assert(planner.last_source_time_ns() == 1'000'000u);

    assert(planner.observe_source_frame(17'666'666u, plan));
    assert(plan.previous_source_time_ns == 1'000'000u);
    assert(plan.current_source_time_ns == 17'666'666u);
    assert(plan.source_interval_ns == 16'666'666u);
    assert(plan.previous_to_interpolated_ns == 8'333'333u);
    assert(plan.interpolated_to_current_ns == 8'333'333u);
    assert(plan.interpolated_time_ns == 9'333'333u);

    assert(planner.observe_source_frame(17'666'671u, plan));
    assert(plan.source_interval_ns == 5u);
    assert(plan.previous_to_interpolated_ns == 2u);
    assert(plan.interpolated_to_current_ns == 3u);
    assert(plan.interpolated_time_ns == 17'666'668u);

    assert(!planner.observe_source_frame(17'666'671u, plan));
    assert(!planner.observe_source_frame(12u, plan));
    assert(planner.last_source_time_ns() == 12u);

    assert(planner.observe_source_frame(20u, plan));
    assert(plan.source_interval_ns == 8u);
    assert(plan.interpolated_time_ns == 16u);

    planner.reset();
    assert(!planner.primed());
    assert(planner.last_source_time_ns() == 0u);

    return 0;
}
