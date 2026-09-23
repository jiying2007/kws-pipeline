cmake_minimum_required(VERSION 3.16)
project(kws_pipeline VERSION 0.3.0 LANGUAGES C)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(KWS_BUILD_TESTS "Build hosted tests" ON)
option(KWS_BUILD_BENCH "Build hosted benchmark" ON)
option(KWS_BUILD_TOOLS "Build hosted evaluation tools" ON)
option(KWS_BUILD_FUZZ "Build Clang/libFuzzer parser targets" OFF)
option(KWS_STRICT "Enable strict warnings" ON)

# -Wcast-qual keeps the "model tensors are read-only zero-copy views into the
# caller's blob" contract compiler-enforced rather than conventional. The tensor
# pointers are const, but a single const-removing cast would let the runtime
# write into the caller's model blob silently; without this flag nothing reports
# it and -Werror has nothing to escalate.
set(KWS_STRICT_OPTIONS -Wall -Wextra -Wpedantic -Werror -Wconversion -Wshadow -Wcast-qual)

set(KWS_SOURCE_REVISION "" CACHE STRING "Exact source revision embedded in build identity")
if(KWS_SOURCE_REVISION STREQUAL "")
  execute_process(
    COMMAND git rev-parse HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE KWS_SOURCE_REVISION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE KWS_GIT_REVISION_RESULT
  )
  if(NOT KWS_GIT_REVISION_RESULT EQUAL 0 OR KWS_SOURCE_REVISION STREQUAL "")
    set(KWS_SOURCE_REVISION "unknown")
  else()
    execute_process(
      COMMAND git status --porcelain --untracked-files=no
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      OUTPUT_VARIABLE KWS_GIT_DIRTY
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NOT KWS_GIT_DIRTY STREQUAL "")
      string(APPEND KWS_SOURCE_REVISION "-dirty")
    endif()
  endif()
endif()
execute_process(
  COMMAND ${CMAKE_C_COMPILER} -dumpmachine
  OUTPUT_VARIABLE KWS_TARGET_TRIPLE
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE KWS_TARGET_RESULT
)
if(NOT KWS_TARGET_RESULT EQUAL 0 OR KWS_TARGET_TRIPLE STREQUAL "")
  set(KWS_TARGET_TRIPLE "${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
endif()
if(CMAKE_BUILD_TYPE)
  set(KWS_BUILD_TYPE_NAME "${CMAKE_BUILD_TYPE}")
else()
  set(KWS_BUILD_TYPE_NAME "unspecified")
endif()
string(CONCAT KWS_CONFIG_IDENTITY
  "tests=${KWS_BUILD_TESTS}\nbench=${KWS_BUILD_BENCH}\ntools=${KWS_BUILD_TOOLS}\n"
  "strict=${KWS_STRICT}\ncompiler=${CMAKE_C_COMPILER_ID}-${CMAKE_C_COMPILER_VERSION}\n"
  "target=${KWS_TARGET_TRIPLE}\nbuild_type=${KWS_BUILD_TYPE_NAME}\nflags=${CMAKE_C_FLAGS}\n")
string(SHA256 KWS_CONFIG_DIGEST "${KWS_CONFIG_IDENTITY}")
configure_file(cmake/kws_build_config.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/generated/kws_build_config.h @ONLY)

# configs/parameter-contract.json is the single source of truth for every
# tunable parameter and its validation range.  src/*.c include the generated
# header, so a contract edit that breaks a bound fails the build instead of
# shipping.  The python tools read the same JSON directly.
set(KWS_PARAMETER_CONTRACT
    "${CMAKE_CURRENT_SOURCE_DIR}/configs/parameter-contract.json")
set(KWS_PARAMETER_GENERATOR
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/gen_parameter_limits.py")
if(NOT EXISTS "${KWS_PARAMETER_CONTRACT}")
  message(FATAL_ERROR "missing parameter contract: ${KWS_PARAMETER_CONTRACT}")
endif()
if(NOT EXISTS "${KWS_PARAMETER_GENERATOR}")
  message(FATAL_ERROR "missing parameter generator: ${KWS_PARAMETER_GENERATOR}")
endif()
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
  set(KWS_PARAMETER_PYTHON "${Python3_EXECUTABLE}")
else()
  find_program(KWS_PARAMETER_PYTHON NAMES python3 python)
endif()
if(NOT KWS_PARAMETER_PYTHON)
  message(FATAL_ERROR
    "python3 is required to generate kws_parameter_limits.h from the contract")
endif()
execute_process(
  COMMAND "${KWS_PARAMETER_PYTHON}" "${KWS_PARAMETER_GENERATOR}"
          "${KWS_PARAMETER_CONTRACT}" "${CMAKE_CURRENT_BINARY_DIR}/generated"
  RESULT_VARIABLE KWS_PARAMETER_GENERATE_RESULT
  ERROR_VARIABLE KWS_PARAMETER_GENERATE_ERROR
)
if(NOT KWS_PARAMETER_GENERATE_RESULT EQUAL 0)
  message(FATAL_ERROR
    "parameter contract generation failed: ${KWS_PARAMETER_GENERATE_ERROR}")
endif()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
             "${KWS_PARAMETER_CONTRACT}" "${KWS_PARAMETER_GENERATOR}")

add_library(kws_pipeline STATIC
  src/kws.c
  src/build_info.c
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
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_BINARY_DIR}/generated
)
target_compile_features(kws_pipeline PUBLIC c_std_11)
target_link_libraries(kws_pipeline PUBLIC m)

if(KWS_STRICT AND CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(kws_pipeline PRIVATE ${KWS_STRICT_OPTIONS})
endif()

if(KWS_BUILD_TESTS OR KWS_BUILD_TOOLS)
  add_library(kws_trace_io STATIC tools/kws_trace_io.c)
  target_include_directories(kws_trace_io
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/tools
  )
  target_compile_features(kws_trace_io PUBLIC c_std_11)
  target_link_libraries(kws_trace_io PUBLIC m)
  if(KWS_STRICT AND CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(kws_trace_io PRIVATE ${KWS_STRICT_OPTIONS})
  endif()
endif()

if(KWS_BUILD_TESTS)
  enable_testing()

  add_executable(kws_tests tests/test_kws.c)
  target_include_directories(kws_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(kws_tests PRIVATE KwsPipeline::core)

  add_executable(kws_keyword_pack_tests tests/test_keyword_pack.c)
  target_link_libraries(kws_keyword_pack_tests PRIVATE KwsPipeline::core)

  add_executable(kws_decoder_tests tests/test_decoder.c)
  target_include_directories(kws_decoder_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(kws_decoder_tests PRIVATE KwsPipeline::core)

  add_executable(kws_decoder_retention_tests tests/test_decoder_retention.c)
  target_include_directories(kws_decoder_retention_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(kws_decoder_retention_tests PRIVATE KwsPipeline::core)

  add_executable(kws_trace_io_tests tests/test_trace_io.c)
  target_link_libraries(kws_trace_io_tests PRIVATE kws_trace_io)

  if(KWS_STRICT AND CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(kws_tests PRIVATE ${KWS_STRICT_OPTIONS})
    target_compile_options(kws_keyword_pack_tests PRIVATE ${KWS_STRICT_OPTIONS})
    target_compile_options(kws_decoder_tests PRIVATE ${KWS_STRICT_OPTIONS})
    target_compile_options(kws_decoder_retention_tests PRIVATE ${KWS_STRICT_OPTIONS})
    target_compile_options(kws_trace_io_tests PRIVATE ${KWS_STRICT_OPTIONS})
  endif()

  add_test(NAME kws_tests COMMAND kws_tests)
  add_test(NAME kws_keyword_pack_tests COMMAND kws_keyword_pack_tests)
  add_test(NAME kws_decoder_tests COMMAND kws_decoder_tests)
  add_test(NAME kws_decoder_retention_tests COMMAND kws_decoder_retention_tests)
  add_test(NAME kws_trace_io_tests COMMAND kws_trace_io_tests)
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

  add_executable(kws_wav_gru tools/kws_wav_gru.c)
  target_include_directories(kws_wav_gru PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(kws_wav_gru PRIVATE KwsPipeline::core kws_tool_io)

  add_executable(kws_raw_stream tools/kws_raw_stream.c)
  target_link_libraries(kws_raw_stream PRIVATE KwsPipeline::core kws_tool_io)

  add_executable(kws_board_bench tools/kws_board_bench.c)
  target_link_libraries(kws_board_bench PRIVATE KwsPipeline::core kws_tool_io)

  add_executable(kws_feature_dump tools/kws_feature_dump.c)
  target_include_directories(kws_feature_dump PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(kws_feature_dump PRIVATE KwsPipeline::core kws_tool_io)

  add_executable(kws_posterior_dump tools/kws_posterior_dump.c)
  target_include_directories(kws_posterior_dump PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(kws_posterior_dump PRIVATE KwsPipeline::core kws_tool_io kws_trace_io)

  add_executable(kws_decoder_replay tools/kws_decoder_replay.c)
  target_include_directories(kws_decoder_replay PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(kws_decoder_replay PRIVATE KwsPipeline::core kws_tool_io kws_trace_io)

  if(KWS_STRICT AND CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(kws_tool_io PRIVATE ${KWS_STRICT_OPTIONS})
    target_compile_options(kws_wav PRIVATE ${KWS_STRICT_OPTIONS})
    target_compile_options(kws_wav_gru PRIVATE ${KWS_STRICT_OPTIONS})
    target_compile_options(kws_raw_stream PRIVATE ${KWS_STRICT_OPTIONS})
    target_compile_options(kws_board_bench PRIVATE ${KWS_STRICT_OPTIONS})
    target_compile_options(kws_feature_dump PRIVATE ${KWS_STRICT_OPTIONS})
    target_compile_options(kws_posterior_dump PRIVATE ${KWS_STRICT_OPTIONS})
    target_compile_options(kws_decoder_replay PRIVATE ${KWS_STRICT_OPTIONS})
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
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
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
