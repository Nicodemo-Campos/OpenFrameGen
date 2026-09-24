if(NOT DEFINED INPUT OR NOT DEFINED OUTPUT)
    message(FATAL_ERROR "embed_spirv.cmake requires INPUT and OUTPUT")
endif()

file(READ "${INPUT}" SPIRV_HEX HEX)
string(LENGTH "${SPIRV_HEX}" HEX_LENGTH)

if(HEX_LENGTH EQUAL 0)
    message(FATAL_ERROR "SPIR-V input is empty: ${INPUT}")
endif()

math(EXPR BYTE_COUNT "${HEX_LENGTH} / 2")
math(EXPR LAST_BYTE "${BYTE_COUNT} - 1")

set(CONTENT "#pragma once\n\n#include <cstddef>\n#include <cstdint>\n\nnamespace ofg::vulkan::generated {\n\nalignas(4) inline constexpr std::uint8_t kPassthroughSpirv[] = {\n    ")

foreach(INDEX RANGE 0 ${LAST_BYTE})
    math(EXPR OFFSET "${INDEX} * 2")
    string(SUBSTRING "${SPIRV_HEX}" ${OFFSET} 2 BYTE_HEX)
    string(APPEND CONTENT "0x${BYTE_HEX}")

    if(NOT INDEX EQUAL LAST_BYTE)
        string(APPEND CONTENT ", ")
    endif()

    math(EXPR COLUMN "${INDEX} % 12")
    if(COLUMN EQUAL 11 AND NOT INDEX EQUAL LAST_BYTE)
        string(APPEND CONTENT "\n    ")
    endif()
endforeach()

string(APPEND CONTENT "\n};\n\ninline constexpr std::size_t kPassthroughSpirvSize = sizeof(kPassthroughSpirv);\n\n} // namespace ofg::vulkan::generated\n")
file(WRITE "${OUTPUT}" "${CONTENT}")
