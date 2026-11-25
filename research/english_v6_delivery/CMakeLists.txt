cmake_minimum_required(VERSION 3.18)
project(HeyMero43DeliveryLoadCheck LANGUAGES C)

# The caller first authenticates the installed SDK and supplies its exact
# configuration directory. This consumer cannot fall back to SDK source files
# or another package found in the user's registry or system search paths.
if(NOT DEFINED HeyMero43_DIR OR NOT IS_ABSOLUTE "${HeyMero43_DIR}" OR
   NOT EXISTS "${HeyMero43_DIR}/HeyMero43Config.cmake")
  message(FATAL_ERROR
    "Set absolute -DHeyMero43_DIR to the authenticated installed SDK's lib/cmake/HeyMero43 directory")
endif()
get_filename_component(_heym43_expected_config
  "${HeyMero43_DIR}/HeyMero43Config.cmake" REALPATH)
find_package(HeyMero43 0.1 CONFIG REQUIRED
  PATHS "${HeyMero43_DIR}" NO_DEFAULT_PATH)
get_filename_component(_heym43_actual_config "${HeyMero43_CONFIG}" REALPATH)
if(NOT _heym43_actual_config STREQUAL _heym43_expected_config)
  message(FATAL_ERROR "The package configuration differs from the authenticated SDK directory")
endif()
if(NOT TARGET HeyMero43::stream)
  message(FATAL_ERROR "The installed SDK did not export HeyMero43::stream")
endif()
get_target_property(_heym43_imported HeyMero43::stream IMPORTED)
if(NOT _heym43_imported)
  message(FATAL_ERROR "HeyMero43::stream must be an imported installed SDK target")
endif()

add_executable(heym43_delivery_load_check loader_check.c)
target_compile_features(heym43_delivery_load_check PRIVATE c_std_11)
set_target_properties(heym43_delivery_load_check PROPERTIES C_EXTENSIONS OFF)
target_link_libraries(heym43_delivery_load_check PRIVATE HeyMero43::stream)
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(heym43_delivery_load_check PRIVATE
    -Wall -Wextra -Werror -pedantic -fno-fast-math -ffp-contract=off)
endif()
