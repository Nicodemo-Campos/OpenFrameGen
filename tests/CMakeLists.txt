add_executable(ofg_core_tests
    core_runtime_tests.cpp
)

target_link_libraries(ofg_core_tests
    PRIVATE
        OpenFrameGen::Core
)

add_test(
    NAME ofg_core_runtime
    COMMAND ofg_core_tests
)

add_executable(ofg_frame_cadence_tests
    frame_cadence_tests.cpp
)

target_link_libraries(ofg_frame_cadence_tests
    PRIVATE
        OpenFrameGen::Core
)

add_test(
    NAME ofg_frame_cadence
    COMMAND ofg_frame_cadence_tests
)

add_executable(ofg_compute_tests
    compute_backend_tests.cpp
)

target_link_libraries(ofg_compute_tests
    PRIVATE
        OpenFrameGen::Compute
)

add_test(
    NAME ofg_compute_backend
    COMMAND ofg_compute_tests
)
