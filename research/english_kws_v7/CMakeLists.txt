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
if(UNIX)
    target_link_libraries(EnglishKwsV7 PUBLIC m)
endif()
target_compile_options(EnglishKwsV7 PRIVATE
    -Wall -Wextra -Werror
    -ffp-contract=off
    -fno-fast-math
)

add_executable(v7_c11_dump_logits v7_c11_dump_logits.c)
target_link_libraries(v7_c11_dump_logits PRIVATE EnglishKwsV7)
target_compile_options(v7_c11_dump_logits PRIVATE
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
    add_executable(test_v7_model_load test_v7_model_load.c)
    target_link_libraries(test_v7_model_load PRIVATE EnglishKwsV7)
    target_compile_options(test_v7_model_load PRIVATE -Wall -Wextra -Werror)
    add_test(NAME v7_model_pack_candidate_identity COMMAND test_v7_model_load)
    add_executable(test_v7_stream_frames test_v7_stream_frames.c)
    target_link_libraries(test_v7_stream_frames PRIVATE EnglishKwsV7)
    target_compile_options(test_v7_stream_frames PRIVATE -Wall -Wextra -Werror)
    add_test(NAME v7_stream_frame_geometry COMMAND test_v7_stream_frames)
endif()
