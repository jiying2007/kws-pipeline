# A configuration fingerprint, not an executable/content hash. Keep the fields
# explicit, ordered and location-independent. Hash each value separately so a
# newline/semicolon in a flag cannot change record boundaries.
function(kws_identity_append key value)
  string(REPLACE "${CMAKE_CURRENT_BINARY_DIR}" "<KWS_BUILD_DIR>" value "${value}")
  string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}" "<KWS_SOURCE_DIR>" value "${value}")
  string(SHA256 value_digest "${value}")
  set(KWS_CONFIG_IDENTITY "${KWS_CONFIG_IDENTITY}${key}=${value_digest}\n" PARENT_SCOPE)
endfunction()

# A cross compiler's configured target overrides its host/default -dumpmachine.
if(CMAKE_C_COMPILER_TARGET)
  set(KWS_TARGET_TRIPLE "${CMAKE_C_COMPILER_TARGET}")
else()
  execute_process(
    COMMAND "${CMAKE_C_COMPILER}" -dumpmachine
    OUTPUT_VARIABLE KWS_TARGET_TRIPLE
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE KWS_TARGET_RESULT
  )
  if(NOT KWS_TARGET_RESULT EQUAL 0 OR KWS_TARGET_TRIPLE STREQUAL "")
    set(KWS_TARGET_TRIPLE "${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
  endif()
endif()

set(KWS_TOOLCHAIN_DIGEST "")
if(CMAKE_TOOLCHAIN_FILE)
  # Use bytes, not the toolchain's checkout/install location. Included toolchain
  # files are represented by their resulting settings, not recursively hashed.
  file(SHA256 "${CMAKE_TOOLCHAIN_FILE}" KWS_TOOLCHAIN_DIGEST)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_TOOLCHAIN_FILE}")
endif()

get_property(KWS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(KWS_MULTI_CONFIG)
  set(KWS_IDENTITY_CONFIGS ${CMAKE_CONFIGURATION_TYPES})
elseif(CMAKE_BUILD_TYPE)
  set(KWS_IDENTITY_CONFIGS "${CMAKE_BUILD_TYPE}")
else()
  set(KWS_IDENTITY_CONFIGS "__kws_unspecified")
endif()
get_property(KWS_IDENTITY_TARGETS DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
list(SORT KWS_IDENTITY_TARGETS)

foreach(config IN LISTS KWS_IDENTITY_CONFIGS)
  if(config STREQUAL "__kws_unspecified")
    set(config "")
    set(KWS_BUILD_TYPE_NAME "unspecified")
  else()
    set(KWS_BUILD_TYPE_NAME "${config}")
  endif()
  string(TOUPPER "${config}" config_upper)
  set(KWS_CONFIG_IDENTITY "kws-build-config-v2\n")
  kws_identity_append(build_type "${KWS_BUILD_TYPE_NAME}")
  kws_identity_append(target_triple "${KWS_TARGET_TRIPLE}")
  kws_identity_append(toolchain_sha256 "${KWS_TOOLCHAIN_DIGEST}")
  foreach(variable IN ITEMS
      KWS_BUILD_TESTS KWS_BUILD_BENCH KWS_BUILD_TOOLS KWS_BUILD_FUZZ KWS_STRICT
      CMAKE_VERSION CMAKE_GENERATOR CMAKE_GENERATOR_PLATFORM CMAKE_GENERATOR_TOOLSET
      CMAKE_SYSTEM_NAME CMAKE_SYSTEM_VERSION CMAKE_SYSTEM_PROCESSOR CMAKE_SIZEOF_VOID_P
      CMAKE_C_COMPILER_ID CMAKE_C_COMPILER_VERSION CMAKE_C_COMPILER_FRONTEND_VARIANT
      CMAKE_C_SIMULATE_ID CMAKE_C_SIMULATE_VERSION CMAKE_C_COMPILER_ARCHITECTURE_ID
      CMAKE_C_COMPILER_TARGET CMAKE_C_COMPILER_EXTERNAL_TOOLCHAIN CMAKE_C_COMPILER_ARG1
      CMAKE_SYSROOT CMAKE_SYSROOT_COMPILE CMAKE_SYSROOT_LINK
      CMAKE_OSX_ARCHITECTURES CMAKE_OSX_DEPLOYMENT_TARGET CMAKE_OSX_SYSROOT
      CMAKE_C_FLAGS CMAKE_EXE_LINKER_FLAGS CMAKE_STATIC_LINKER_FLAGS)
    kws_identity_append("${variable}" "${${variable}}")
  endforeach()
  foreach(variable IN ITEMS CMAKE_C_FLAGS CMAKE_EXE_LINKER_FLAGS CMAKE_STATIC_LINKER_FLAGS)
    # Release uses both CMAKE_C_FLAGS and CMAKE_C_FLAGS_RELEASE, in that order.
    kws_identity_append("${variable}_${config_upper}" "${${variable}_${config_upper}}")
  endforeach()
  foreach(property IN ITEMS COMPILE_OPTIONS COMPILE_DEFINITIONS LINK_OPTIONS LINK_DIRECTORIES)
    get_directory_property(value "${property}")
    kws_identity_append("directory.${property}" "${value}")
  endforeach()
  foreach(target IN LISTS KWS_IDENTITY_TARGETS)
    foreach(property IN ITEMS
        TYPE C_STANDARD C_STANDARD_REQUIRED C_EXTENSIONS
        COMPILE_FEATURES COMPILE_FLAGS COMPILE_OPTIONS COMPILE_DEFINITIONS INCLUDE_DIRECTORIES
        LINK_FLAGS LINK_OPTIONS LINK_LIBRARIES LINK_DIRECTORIES STATIC_LIBRARY_OPTIONS
        INTERFACE_COMPILE_FEATURES INTERFACE_COMPILE_OPTIONS INTERFACE_COMPILE_DEFINITIONS
        INTERFACE_LINK_LIBRARIES INTERFACE_LINK_OPTIONS
        POSITION_INDEPENDENT_CODE INTERPROCEDURAL_OPTIMIZATION MSVC_RUNTIME_LIBRARY
        C_VISIBILITY_PRESET
        "COMPILE_DEFINITIONS_${config_upper}" "LINK_FLAGS_${config_upper}"
        "INTERPROCEDURAL_OPTIMIZATION_${config_upper}")
      get_target_property(value "${target}" "${property}")
      if(value STREQUAL "value-NOTFOUND")
        set(value "")
      endif()
      # Preserve ordered declarations, including generator expressions. Together
      # with the selected config these identify the project's target settings;
      # this deliberately does not claim to hash expanded compiler argv/bytes.
      kws_identity_append("${target}.${property}" "${value}")
    endforeach()
  endforeach()
  string(SHA256 KWS_CONFIG_DIGEST "${KWS_CONFIG_IDENTITY}")
  set(header_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")
  if(KWS_MULTI_CONFIG)
    string(APPEND header_dir "/${config}")
  endif()
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/kws_build_config.h.in"
                 "${header_dir}/kws_build_config.h" @ONLY)
endforeach()
