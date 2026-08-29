cmake_minimum_required(VERSION 3.16)
project(kws_pipeline VERSION 0.2.0 LANGUAGES C)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(KWS_BUILD_TESTS "Build hosted tests" ON)
option(KWS_BUILD_BENCH "Build hosted benchmark" ON)
option(KWS_BUILD_TOOLS "Build hosted evaluation tools" ON)
option(KWS_BUILD_FUZZ "Build Clang/libFuzzer parser targets" OFF)
option(KWS_STRICT "Enable strict warnings" ON)

set(KWS_STRICT_OPTIONS -Wall -Wextra -Wpedantic -Werror -Wconversion -Wshadow)

add_library(kws_pipeline STATIC
  src/kws.c
  src/frontend.c
  src/model.c
  src/decoder.c
  src/keyword_pack.c
)
add_library(KwsPipeline::core ALIAS kws_pipeline)
set_target_properties(kws_pipeline PROPERTIES EXPORT_NAME core OUTPUT_NAME kws_pipeline)

target_include_directories(kws_pipeline
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_compile_features(kws_pipeline PUBLIC c_std_11)
target_link_libraries(kws_pipeline PUBLIC m)

if(KWS_STRICT AND CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(kws_pipeline PRIVATE ${KWS_STRICT_OPTIONS})
endif()

if(KWS_BUILD_TESTS)
  enable_testing()

  add_executable(kws_tests tests/test_kws.c)
  target_link_libraries(kws_tests PRIVATE KwsPipeline::core)

  add_executable(kws_keyword_pack_tests tests/test_keyword_pack.c)
  target_link_libraries(kws_keyword_pack_tests PRIVATE KwsPipeline::core)

  add_executable(kws_decoder_tests tests/test_decoder.c)
  target_include_directories(kws_decoder_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(kws_decoder_tests PRIVATE KwsPipeline::core)

  if(KWS_STRICT AND CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(kws_tests PRIVATE ${KWS_STRICT_OPTIONS})
    target_compile_options(kws_keyword_pack_tests PRIVATE ${KWS_STRICT_OPTIONS})
    target_compile_options(kws_decoder_tests PRIVATE ${KWS_STRICT_OPTIONS})
  endif()

  add_test(NAME kws_tests COMMAND kws_tests)
  add_test(NAME kws_keyword_pack_tests COMMAND kws_keyword_pack_tests)
  add_test(NAME kws_decoder_tests COMMAND kws_decoder_tests)
endif()

if(KWS_BUILD_BENCH)
  add_executable(kws_bench bench/bench_kws.c)
  target_link_libraries(kws_bench PRIVATE KwsPipeline::core)
  if(KWS_STRICT AND CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(kws_bench PRIVATE ${KWS_STRICT_OPTIONS})
  endif()
endif()

if(KWS_BUILD_TOOLS)
  add_library(kws_tool_io STATIC tools/tool_io.c tools/sha256.c)
  target_include_directories(kws_tool_io
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/tools
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  target_compile_features(kws_tool_io PUBLIC c_std_11)

  add_executable(kws_wav tools/kws_wav.c)
  target_link_libraries(kws_wav PRIVATE KwsPipeline::core kws_tool_io)

  add_executable(kws_raw_stream tools/kws_raw_stream.c)
  target_link_libraries(kws_raw_stream PRIVATE KwsPipeline::core kws_tool_io)

  add_executable(kws_board_bench tools/kws_board_bench.c)
  target_link_libraries(kws_board_bench PRIVATE KwsPipeline::core kws_tool_io)

  add_executable(kws_feature_dump tools/kws_feature_dump.c)
  target_include_directories(kws_feature_dump PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(kws_feature_dump PRIVATE KwsPipeline::core kws_tool_io)

  if(KWS_STRICT AND CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(kws_tool_io PRIVATE ${KWS_STRICT_OPTIONS})
    target_compile_options(kws_wav PRIVATE ${KWS_STRICT_OPTIONS})
    target_compile_options(kws_raw_stream PRIVATE ${KWS_STRICT_OPTIONS})
    target_compile_options(kws_board_bench PRIVATE ${KWS_STRICT_OPTIONS})
    target_compile_options(kws_feature_dump PRIVATE ${KWS_STRICT_OPTIONS})
  endif()
endif()

if(KWS_BUILD_FUZZ)
  if(NOT CMAKE_C_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR "KWS_BUILD_FUZZ requires Clang/libFuzzer")
  endif()

  add_executable(kws_fuzz_model fuzz/fuzz_model.c)
  target_link_libraries(kws_fuzz_model PRIVATE KwsPipeline::core)
  target_compile_options(kws_fuzz_model PRIVATE -fsanitize=fuzzer-no-link)
  target_link_options(kws_fuzz_model PRIVATE -fsanitize=fuzzer)

  add_executable(kws_fuzz_keyword_pack fuzz/fuzz_keyword_pack.c)
  target_link_libraries(kws_fuzz_keyword_pack PRIVATE KwsPipeline::core)
  target_compile_options(kws_fuzz_keyword_pack PRIVATE -fsanitize=fuzzer-no-link)
  target_link_options(kws_fuzz_keyword_pack PRIVATE -fsanitize=fuzzer)

  if(KWS_STRICT)
    target_compile_options(kws_fuzz_model PRIVATE ${KWS_STRICT_OPTIONS})
    target_compile_options(kws_fuzz_keyword_pack PRIVATE ${KWS_STRICT_OPTIONS})
  endif()
endif()

install(TARGETS kws_pipeline
  EXPORT KwsPipelineTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

configure_package_config_file(
  cmake/KwsPipelineConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/KwsPipelineConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/KwsPipeline
)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/KwsPipelineConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY ExactVersion
)
configure_file(
  cmake/kws-pipeline.pc.in
  ${CMAKE_CURRENT_BINARY_DIR}/kws-pipeline.pc
  @ONLY
)

install(EXPORT KwsPipelineTargets
  FILE KwsPipelineTargets.cmake
  NAMESPACE KwsPipeline::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/KwsPipeline
)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/KwsPipelineConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/KwsPipelineConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/KwsPipeline
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/kws-pipeline.pc
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
)