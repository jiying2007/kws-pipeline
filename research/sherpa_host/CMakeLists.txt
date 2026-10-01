cmake_minimum_required(VERSION 3.20)
project(sherpa_onnx_kws_root_subset LANGUAGES C CXX)

# Integration glue follows the existing kws-pipeline Apache-2.0 license.
# See LICENSE and PROVENANCE.md; retained third-party code keeps its licenses.
# This is a frozen Linux host candidate. Configuring or building requires its
# own approved execution lock; source preparation has not executed CMake.
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "This reviewed candidate has only Linux host compile contracts")
endif()
if(NOT CMAKE_C_COMPILER_ID STREQUAL "GNU" OR NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  message(FATAL_ERROR "This candidate retains the existing A GNU compiler options")
endif()
if(CMAKE_CONFIGURATION_TYPES)
  message(FATAL_ERROR "Use a single-configuration generator for the frozen contract")
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Frozen A configuration" FORCE)
endif()
if(NOT CMAKE_BUILD_TYPE STREQUAL "Release")
  message(FATAL_ERROR "Only the A Release compile contract is represented")
endif()

# All compile options are carried by the private targets below, exactly once.
# Clear project-local implicit configuration flags to avoid duplicate -O3,
# standards, or fast-math propagation. Do not introduce LTO, GC or CPU flags.
set(CMAKE_C_FLAGS "")
set(CMAKE_CXX_FLAGS "")
set(CMAKE_C_FLAGS_RELEASE "")
set(CMAKE_CXX_FLAGS_RELEASE "")
set(CMAKE_SHARED_LINKER_FLAGS "")
set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "")
set(CMAKE_EXE_LINKER_FLAGS "")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "")
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION FALSE)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(KWS_SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")

set(KWS_ORT_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/../lib/libonnxruntime.so"
    CACHE FILEPATH "Existing A ORT provider, unchanged; never downloaded or loaded by configure")
if(NOT EXISTS "${KWS_ORT_LIBRARY}")
  message(FATAL_ERROR "Supply the independently verified existing A ORT library")
endif()
file(SHA256 "${KWS_ORT_LIBRARY}" KWS_ORT_ACTUAL_SHA256)
if(NOT KWS_ORT_ACTUAL_SHA256 STREQUAL "4b3607aebd1784b26b6f9b20e4bd974c7ab8287043e4d095cb7d2cb40b5e566e")
  message(FATAL_ERROR "ORT bytes differ from the locked A provider")
endif()
add_library(kws-ort SHARED IMPORTED)
set_target_properties(kws-ort PROPERTIES IMPORTED_LOCATION "${KWS_ORT_LIBRARY}")

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/LockedTargets.cmake")

# Static archive order mirrors the selected subset of A. No FST/Kaldifst archive
# targets are imported; their observed header dependencies remain in vendor/.
# This candidate link is unproven until separately authorized and inspected.
target_link_libraries(sherpa-onnx-kws-c-api PRIVATE
  sherpa-onnx-kws-core
  kws-private-fbank
  kws-private-kissfft
  kws-private-kaldi-decoder
  kws-private-sentencepiece
  pthread
  kws-ort
  m
  dl
  "-pthread"
)
target_link_options(sherpa-onnx-kws-c-api PRIVATE
  "-O3"
  "-DNDEBUG"
  "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/cmake/kws-exports.lds"
  "-Wl,-Map=${CMAKE_CURRENT_BINARY_DIR}/libsherpa-onnx-kws-c-api.so.map"
)
set_target_properties(sherpa-onnx-kws-c-api PROPERTIES
  OUTPUT_NAME "sherpa-onnx-kws-c-api"
  NO_SONAME FALSE
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib"
  INSTALL_RPATH "$ORIGIN"
  INSTALL_RPATH_USE_LINK_PATH FALSE
  BUILD_WITH_INSTALL_RPATH TRUE
  LINK_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/kws-exports.lds"
)

# The CLI uses the frozen helpers and the original project public header.
# This is one additional C++ translation unit; no fake-test target is added.
add_executable(kws
  "${CMAKE_CURRENT_SOURCE_DIR}/cli/src/kws.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/cli/src/audio.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/cli/src/json.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/cli/src/options.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/cli/src/profile.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/cli/src/session.hpp"
)
target_include_directories(kws PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/cli/src"
  "${CMAKE_CURRENT_SOURCE_DIR}/source"
)
target_compile_options(kws PRIVATE "-O2" "-std=c++17")
target_link_libraries(kws PRIVATE sherpa-onnx-kws-c-api)
set_target_properties(kws PROPERTIES
  OUTPUT_NAME "kws"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
  INSTALL_RPATH "$ORIGIN/../lib"
  INSTALL_RPATH_USE_LINK_PATH FALSE
  BUILD_WITH_INSTALL_RPATH TRUE
  INTERPROCEDURAL_OPTIMIZATION FALSE
  CXX_SCAN_FOR_MODULES FALSE
)

# Every project source/header copy is under this exact common root. Apply only
# to these locally compiled C/C++ targets, never the imported prebuilt ORT.
# Preserve the original per-TU math flags in LockedTargets.cmake unchanged.
foreach(kws_target IN ITEMS
  kws-private-kaldi-decoder
  kws-private-fbank
  kws-private-kissfft
  sherpa-onnx-kws-core
  kws-private-sentencepiece
  sherpa-onnx-kws-c-api
  kws
)
  target_compile_options(${kws_target} PRIVATE
    "$<$<COMPILE_LANGUAGE:C,CXX>:-fmacro-prefix-map=${CMAKE_CURRENT_SOURCE_DIR}=.>"
  )
endforeach()

# No install/download/test/model execution is part of this preparation.
