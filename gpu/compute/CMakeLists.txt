add_library(ofg_compute INTERFACE)

add_library(OpenFrameGen::Compute ALIAS ofg_compute)

target_include_directories(ofg_compute
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(ofg_compute INTERFACE cxx_std_20)
