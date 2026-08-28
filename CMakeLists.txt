cmake_minimum_required(VERSION 3.16)
project(kws_pipeline VERSION 0.1.0 LANGUAGES C)

option(KWS_BUILD_TESTS "Build hosted tests" ON)
option(KWS_BUILD_BENCH "Build hosted benchmark" ON)
option(KWS_STRICT "Enable strict warnings" ON)

add_library(kws_pipeline STATIC
  src/kws.c
  src/frontend.c
  src/model.c
  src/decoder.c
  src/keyword_pack.c
)
add_library(KwsPipeline::core ALIAS kws_pipeline)

target_include_directories(kws_pipeline
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_compile_features(kws_pipeline PUBLIC c_std_11)
target_link_libraries(kws_pipeline PUBLIC m)

if(KWS_STRICT)
  if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(kws_pipeline PRIVATE -Wall -Wextra -Wpedantic -Werror -Wconversion -Wshadow)
  endif()
endif()

if(KWS_BUILD_TESTS)
  enable_testing()
  add_executable(kws_tests tests/test_kws.c)
  target_link_libraries(kws_tests PRIVATE KwsPipeline::core)
  add_executable(kws_keyword_pack_tests tests/test_keyword_pack.c)
  target_link_libraries(kws_keyword_pack_tests PRIVATE KwsPipeline::core)
  if(KWS_STRICT AND CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(kws_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_compile_options(kws_keyword_pack_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
  endif()
  add_test(NAME kws_tests COMMAND kws_tests)
  add_test(NAME kws_keyword_pack_tests COMMAND kws_keyword_pack_tests)
endif()

if(KWS_BUILD_BENCH)
  add_executable(kws_bench bench/bench_kws.c)
  target_link_libraries(kws_bench PRIVATE KwsPipeline::core)
endif()
