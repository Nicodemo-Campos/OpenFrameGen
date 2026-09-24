add_library(ofg_vulkan_layer SHARED
    src/layer.cpp
)

target_include_directories(ofg_vulkan_layer
    PRIVATE
        ${OFG_VULKAN_INCLUDE_DIR}
)

target_compile_features(ofg_vulkan_layer PRIVATE cxx_std_20)

if(MSVC)
    target_compile_options(ofg_vulkan_layer PRIVATE /W4 /permissive-)
else()
    target_compile_options(ofg_vulkan_layer PRIVATE -Wall -Wextra -Wpedantic)
endif()

set(OFG_LAYER_OUTPUT_DIR "${CMAKE_BINARY_DIR}/ofg-layer")

set_target_properties(ofg_vulkan_layer PROPERTIES
    OUTPUT_NAME "openframegen_vulkan_layer"
    PREFIX ""
    RUNTIME_OUTPUT_DIRECTORY "${OFG_LAYER_OUTPUT_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${OFG_LAYER_OUTPUT_DIR}"
    ARCHIVE_OUTPUT_DIRECTORY "${OFG_LAYER_OUTPUT_DIR}"
)

foreach(config DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set_target_properties(ofg_vulkan_layer PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY_${config} "${OFG_LAYER_OUTPUT_DIR}"
        LIBRARY_OUTPUT_DIRECTORY_${config} "${OFG_LAYER_OUTPUT_DIR}"
        ARCHIVE_OUTPUT_DIRECTORY_${config} "${OFG_LAYER_OUTPUT_DIR}"
    )
endforeach()

set(
    OFG_LAYER_LIBRARY_RELATIVE_PATH
    "./openframegen_vulkan_layer${CMAKE_SHARED_LIBRARY_SUFFIX}"
)

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/manifest/VkLayer_OpenFrameGen.json.in"
    "${OFG_LAYER_OUTPUT_DIR}/VkLayer_OpenFrameGen.json"
    @ONLY
)
