add_library(ofg_vulkan_layer SHARED
    src/layer.cpp
    src/passthrough_pipeline.cpp
)

target_include_directories(ofg_vulkan_layer
    PRIVATE
        ${OFG_VULKAN_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_features(ofg_vulkan_layer PRIVATE cxx_std_20)

find_program(
    OFG_GLSLC_EXECUTABLE
    NAMES glslc glslc.exe
    HINTS
        "$ENV{VULKAN_SDK}/Bin"
        "$ENV{VULKAN_SDK}/bin"
)

if(OFG_GLSLC_EXECUTABLE)
    set(OFG_GENERATED_SHADER_DIR
        "${CMAKE_CURRENT_BINARY_DIR}/generated"
    )
    file(MAKE_DIRECTORY "${OFG_GENERATED_SHADER_DIR}")

    set(OFG_PASSTHROUGH_SPV
        "${OFG_GENERATED_SHADER_DIR}/passthrough.comp.spv"
    )
    set(OFG_PASSTHROUGH_HEADER
        "${OFG_GENERATED_SHADER_DIR}/passthrough_spv.hpp"
    )
    set(OFG_SHARPEN_SPV
        "${OFG_GENERATED_SHADER_DIR}/sharpen.comp.spv"
    )
    set(OFG_SHARPEN_HEADER
        "${OFG_GENERATED_SHADER_DIR}/sharpen_spv.hpp"
    )

    add_custom_command(
        OUTPUT "${OFG_PASSTHROUGH_SPV}"
        COMMAND
            "${OFG_GLSLC_EXECUTABLE}"
            -fshader-stage=compute
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/passthrough.comp"
            -o "${OFG_PASSTHROUGH_SPV}"
        DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/passthrough.comp"
        VERBATIM
    )

    add_custom_command(
        OUTPUT "${OFG_PASSTHROUGH_HEADER}"
        COMMAND
            "${CMAKE_COMMAND}"
            "-DINPUT=${OFG_PASSTHROUGH_SPV}"
            "-DOUTPUT=${OFG_PASSTHROUGH_HEADER}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_spirv.cmake"
        DEPENDS
            "${OFG_PASSTHROUGH_SPV}"
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_spirv.cmake"
        VERBATIM
    )

    add_custom_command(
        OUTPUT "${OFG_SHARPEN_SPV}"
        COMMAND
            "${OFG_GLSLC_EXECUTABLE}"
            -fshader-stage=compute
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/sharpen.comp"
            -o "${OFG_SHARPEN_SPV}"
        DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/sharpen.comp"
        VERBATIM
    )

    add_custom_command(
        OUTPUT "${OFG_SHARPEN_HEADER}"
        COMMAND
            "${CMAKE_COMMAND}"
            "-DINPUT=${OFG_SHARPEN_SPV}"
            "-DOUTPUT=${OFG_SHARPEN_HEADER}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_spirv.cmake"
        DEPENDS
            "${OFG_SHARPEN_SPV}"
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_spirv.cmake"
        VERBATIM
    )

    target_sources(
        ofg_vulkan_layer
        PRIVATE
            "${OFG_PASSTHROUGH_HEADER}"
            "${OFG_SHARPEN_HEADER}"
    )

    target_include_directories(
        ofg_vulkan_layer
        PRIVATE
            "${OFG_GENERATED_SHADER_DIR}"
    )

    target_compile_definitions(
        ofg_vulkan_layer
        PRIVATE
            OFG_VULKAN_PASSTHROUGH_ENABLED=1
    )

    message(STATUS
        "OpenFrameGen Vulkan compute pass-through: enabled with ${OFG_GLSLC_EXECUTABLE}"
    )
else()
    target_compile_definitions(
        ofg_vulkan_layer
        PRIVATE
            OFG_VULKAN_PASSTHROUGH_ENABLED=0
    )

    message(STATUS
        "OpenFrameGen Vulkan compute pass-through: disabled (glslc not found)"
    )
endif()

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

if(WIN32)
    # The Windows Vulkan loader identifies paths using backslashes.
    # JSON requires each backslash to be escaped, so the configured
    # manifest must contain ".\\openframegen_vulkan_layer.dll".
    set(
        OFG_LAYER_LIBRARY_RELATIVE_PATH
        ".\\\\openframegen_vulkan_layer${CMAKE_SHARED_LIBRARY_SUFFIX}"
    )
else()
    set(
        OFG_LAYER_LIBRARY_RELATIVE_PATH
        "./openframegen_vulkan_layer${CMAKE_SHARED_LIBRARY_SUFFIX}"
    )
endif()

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/manifest/VkLayer_OpenFrameGen.json.in"
    "${OFG_LAYER_OUTPUT_DIR}/VkLayer_OpenFrameGen.json"
    @ONLY
)
