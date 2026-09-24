add_executable(ofg
    src/main.cpp
)

target_link_libraries(ofg
    PRIVATE
        OpenFrameGen::Core
)

target_compile_features(ofg PRIVATE cxx_std_20)

if(MSVC)
    target_compile_options(ofg PRIVATE /W4 /permissive-)
else()
    target_compile_options(ofg PRIVATE -Wall -Wextra -Wpedantic)
endif()
