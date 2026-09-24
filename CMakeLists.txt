cmake_minimum_required(VERSION 3.20)

project(
    OpenFrameGen
    VERSION 0.1.0
    DESCRIPTION "Free and open-source cross-platform frame generation framework"
    LANGUAGES CXX
)

option(OFG_BUILD_CLI "Build the OpenFrameGen command-line interface" ON)
option(OFG_BUILD_TESTS "Build OpenFrameGen tests" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

add_subdirectory(core)

if(OFG_BUILD_CLI)
    add_subdirectory(frontends/cli)
endif()

if(OFG_BUILD_TESTS)
    include(CTest)
    enable_testing()
    add_subdirectory(tests)
endif()
