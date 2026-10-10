cmake_minimum_required(VERSION 3.16)
project(kws_pipeline VERSION 0.4.0 LANGUAGES C)

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

set(KWS_CORE_SOURCES
  src/kws.c
  src/build_info.c
  src/frontend.c
  src/model.c
  src/decoder.c
  src/keyword_pack.c
)
add_library(kws_pipeline STATIC ${KWS_CORE_SOURCES})
add_library(KwsPipeline::core ALIAS kws_pipeline)
set_target_properties(kws_pipeline PROPERTIES EXPORT_NAME core OUTPUT_NAME kws_pipeline)

function(kws_configure_core target)
  target_include_directories(${target}
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${CMAKE_CURRENT_BINARY_DIR}/generated/$<CONFIG>
      ${CMAKE_CURRENT_BINARY_DIR}/generated
  )
  target_compile_features(${target} PUBLIC c_std_11)
  target_link_libraries(${target} PUBLIC m)

  if(KWS_STRICT AND CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target} PRIVATE ${KWS_STRICT_OPTIONS})
  endif()
endfunction()
kws_configure_core(kws_pipeline)

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
  target_include_directories(kws_decoder_replay PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_BINARY_DIR}/generated
  )
  target_link_libraries(kws_decoder_replay PRIVATE KwsPipeline::core kws_tool_io kws_trace_io)

  add_executable(kws_decoder_path_replay tools/kws_decoder_path_replay.c)
  target_include_directories(kws_decoder_path_replay PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_BINARY_DIR}/generated
  )
  target_link_libraries(kws_decoder_path_replay PRIVATE KwsPipeline::core kws_tool_io kws_trace_io)

  add_executable(kws_trace_slice tools/kws_trace_slice.c)
  target_link_libraries(kws_trace_slice PRIVATE kws_trace_io)

  if(KWS_STRICT AND CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(kws_tool_io PRIVATE ${KWS_STRICT_OPTIONS})
    target_compile_options(kws_wav PRIVATE ${KWS_STRICT_OPTIONS})
    target_compile_options(kws_raw_stream PRIVATE ${KWS_STRICT_OPTIONS})
    target_compile_options(kws_board_bench PRIVATE ${KWS_STRICT_OPTIONS})
    target_compile_options(kws_feature_dump PRIVATE ${KWS_STRICT_OPTIONS})
    target_compile_options(kws_posterior_dump PRIVATE ${KWS_STRICT_OPTIONS})
    target_compile_options(kws_decoder_replay PRIVATE ${KWS_STRICT_OPTIONS})
    target_compile_options(kws_decoder_path_replay PRIVATE ${KWS_STRICT_OPTIONS})
    target_compile_options(kws_trace_slice PRIVATE ${KWS_STRICT_OPTIONS})
  endif()

  if(KWS_BUILD_TESTS)
    add_test(NAME kws_trace_slice_self_test COMMAND kws_trace_slice --self-test)
  endif()
endif()

if(KWS_BUILD_FUZZ)
  if(NOT CMAKE_C_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR "KWS_BUILD_FUZZ requires Clang/libFuzzer")
  endif()

  # Compile a separate copy of the code under test. Instrumenting only the
  # harness gives libFuzzer no feedback from parser branches. Keep all fuzzer
  # flags private to this uninstalled library and its drivers, so ordinary
  # executables and the installed SDK never acquire a libFuzzer dependency.
  add_library(kws_fuzz_core STATIC ${KWS_CORE_SOURCES})
  kws_configure_core(kws_fuzz_core)
  target_compile_options(kws_fuzz_core PRIVATE -fsanitize=fuzzer-no-link)

  add_executable(kws_fuzz_model fuzz/fuzz_model.c)
  target_link_libraries(kws_fuzz_model PRIVATE kws_fuzz_core)
  target_compile_options(kws_fuzz_model PRIVATE -fsanitize=fuzzer-no-link)
  target_link_options(kws_fuzz_model PRIVATE -fsanitize=fuzzer)

  add_executable(kws_fuzz_keyword_pack fuzz/fuzz_keyword_pack.c)
  target_link_libraries(kws_fuzz_keyword_pack PRIVATE kws_fuzz_core)
  target_compile_options(kws_fuzz_keyword_pack PRIVATE -fsanitize=fuzzer-no-link)
  target_link_options(kws_fuzz_keyword_pack PRIVATE -fsanitize=fuzzer)

  if(KWS_STRICT)
    target_compile_options(kws_fuzz_model PRIVATE ${KWS_STRICT_OPTIONS})
    target_compile_options(kws_fuzz_keyword_pack PRIVATE ${KWS_STRICT_OPTIONS})
  endif()
endif()

# Collect target settings only after all core and hosted targets are defined.
include(cmake/KwsBuildIdentity.cmake)

# Git state changes do not reliably trigger CMake regeneration (and may involve
# a worktree gitfile, packed refs, or only tracked contents). Check on every core
# build, but configure_file preserves the header timestamp when identity is
# unchanged, so no-op builds do not compile or relink the library.
set(KWS_SOURCE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
set(KWS_SOURCE_REVISION_OVERRIDE "${KWS_SOURCE_REVISION}")
set(KWS_SOURCE_REVISION_HEADER
    "${CMAKE_CURRENT_BINARY_DIR}/generated/kws_source_revision.h")
include(cmake/KwsSourceRevision.cmake)
if(KWS_SOURCE_REVISION STREQUAL "")
  add_custom_target(kws_source_identity
    COMMAND "${CMAKE_COMMAND}"
      "-DKWS_SOURCE_DIRECTORY=${KWS_SOURCE_DIRECTORY}"
      "-DKWS_SOURCE_REVISION_HEADER=${KWS_SOURCE_REVISION_HEADER}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/KwsSourceRevision.cmake"
    BYPRODUCTS "${KWS_SOURCE_REVISION_HEADER}"
    VERBATIM
  )
  add_dependencies(kws_pipeline kws_source_identity)
  if(TARGET kws_fuzz_core)
    add_dependencies(kws_fuzz_core kws_source_identity)
  endif()
endif()

# Both SDK discovery formats promise prefix relocation. Absolute directories or
# parent traversal would escape that prefix, including with --install --prefix
# or DESTDIR, so reject these layouts instead of publishing a misleading .pc.
foreach(directory IN ITEMS CMAKE_INSTALL_LIBDIR CMAKE_INSTALL_INCLUDEDIR)
  if(IS_ABSOLUTE "${${directory}}" OR "${${directory}}" STREQUAL "" OR
     "${${directory}}" MATCHES "(^|/)\\.\\.(/|$)")
    message(FATAL_ERROR
      "${directory} must be a nonempty prefix-relative directory without '..' for a relocatable SDK")
  endif()
endforeach()
file(RELATIVE_PATH KWS_PKGCONFIG_PREFIX_RELATIVE
  "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/pkgconfig"
  "${CMAKE_INSTALL_PREFIX}")
string(REGEX REPLACE "/$" "" KWS_PKGCONFIG_PREFIX_RELATIVE "${KWS_PKGCONFIG_PREFIX_RELATIVE}")

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
