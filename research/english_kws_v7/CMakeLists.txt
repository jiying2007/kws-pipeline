cmake_minimum_required(VERSION 3.16)
project(EnglishKwsV7 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

add_library(EnglishKwsV7 STATIC
    v7_ctc.c
    v7_kws_stream.c
)
target_include_directories(EnglishKwsV7 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(EnglishKwsV7 PRIVATE
    -Wall -Wextra -Werror
    -ffp-contract=off
    -fno-fast-math
)

include(CTest)
if(BUILD_TESTING)
    add_executable(test_v7_ctc_state test_v7_ctc_state.c)
    target_link_libraries(test_v7_ctc_state PRIVATE EnglishKwsV7)
    target_compile_options(test_v7_ctc_state PRIVATE -Wall -Wextra -Werror)
    add_test(NAME v7_ctc_candidate_identity COMMAND test_v7_ctc_state)
endif()
