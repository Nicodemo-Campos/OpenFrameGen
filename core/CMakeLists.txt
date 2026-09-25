add_library(ofg_core
    src/runtime.cpp
    src/frame_cadence.cpp
)

add_library(OpenFrameGen::Core ALIAS ofg_core)

target_include_directories(ofg_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(ofg_core PUBLIC cxx_std_20)

target_compile_definitions(ofg_core
    PRIVATE
        OFG_VERSION_STRING="${PROJECT_VERSION}"
)

if(MSVC)
    target_compile_options(ofg_core PRIVATE /W4 /permissive-)
else()
    target_compile_options(ofg_core PRIVATE -Wall -Wextra -Wpedantic)
endif()
