cmake_minimum_required(VERSION 3.18)
project(HeyMero43 VERSION 0.1.0 LANGUAGES C)

# Opt-in single-word Hey Mero frozen 43K stream library.
# Model pack must be separately SHA authenticated by the embedding app.
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
option(HEY_MERO43_BUILD_TESTS "Build deterministic synthetic source tests" OFF)
option(HEY_MERO43_BUILD_HOST_TOOL "Build host-only PCM trace CLI" OFF)

add_library(heym43_stream STATIC mero43_stream.c mero_ctc.c)
add_library(HeyMero43::stream ALIAS heym43_stream)
set_target_properties(heym43_stream PROPERTIES
  OUTPUT_NAME heym43_stream EXPORT_NAME stream POSITION_INDEPENDENT_CODE OFF)
target_compile_features(heym43_stream PUBLIC c_std_11)
target_include_directories(heym43_stream PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/heym43>)
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(heym43_stream PRIVATE
    -Wall -Wextra -Werror -pedantic -fno-fast-math -ffp-contract=off)
endif()
if(UNIX)
  target_link_libraries(heym43_stream PUBLIC m)
endif()

if(HEY_MERO43_BUILD_TESTS)
  enable_testing()
  add_executable(heym43_synthetic_test test_mero43_stream.c)
  target_link_libraries(heym43_synthetic_test PRIVATE HeyMero43::stream)
  add_test(NAME heym43_synthetic_test COMMAND heym43_synthetic_test)
endif()
if(HEY_MERO43_BUILD_HOST_TOOL)
  add_executable(heym43_host_cli mero43_host_cli.c)
  target_link_libraries(heym43_host_cli PRIVATE HeyMero43::stream)
  install(TARGETS heym43_host_cli RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
install(TARGETS heym43_stream EXPORT HeyMero43Targets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES mero43_stream.h mero_ctc.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/heym43)
install(EXPORT HeyMero43Targets FILE HeyMero43Targets.cmake
  NAMESPACE HeyMero43:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/HeyMero43)
configure_package_config_file(
  cmake/HeyMero43Config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/HeyMero43Config.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/HeyMero43)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/HeyMero43ConfigVersion.cmake
  VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/HeyMero43Config.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/HeyMero43ConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/HeyMero43)
