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
