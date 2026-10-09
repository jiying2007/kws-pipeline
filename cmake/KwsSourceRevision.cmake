# Runs at configure time and before every core build. Only this small generated
# header depends on Git state; configure_file writes it only when bytes change.
set(KWS_BUILD_SOURCE_REVISION "${KWS_SOURCE_REVISION_OVERRIDE}")
if(KWS_BUILD_SOURCE_REVISION STREQUAL "")
  set(KWS_BUILD_SOURCE_REVISION "unknown")
  # Never associate an exported source tree with an unrelated enclosing repo.
  # A .git directory or gitfile at this project's root is required; git itself
  # resolves linked worktrees, detached HEADs, packed refs and submodules.
  if(EXISTS "${KWS_SOURCE_DIRECTORY}/.git")
    find_program(KWS_GIT_EXECUTABLE NAMES git)
    if(KWS_GIT_EXECUTABLE)
      # Do not let a caller's Git plumbing environment redirect source identity.
      set(git_command "${CMAKE_COMMAND}" -E env
        --unset=GIT_DIR --unset=GIT_WORK_TREE --unset=GIT_INDEX_FILE
        --unset=GIT_COMMON_DIR --unset=GIT_OBJECT_DIRECTORY
        --unset=GIT_ALTERNATE_OBJECT_DIRECTORIES
        "${KWS_GIT_EXECUTABLE}" -C "${KWS_SOURCE_DIRECTORY}")
      execute_process(COMMAND ${git_command} rev-parse --show-toplevel
        OUTPUT_VARIABLE git_root OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET RESULT_VARIABLE git_root_result)
      get_filename_component(source_root "${KWS_SOURCE_DIRECTORY}" REALPATH)
      get_filename_component(git_root "${git_root}" REALPATH)
      if(git_root_result EQUAL 0 AND git_root STREQUAL source_root)
        execute_process(COMMAND ${git_command} rev-parse --verify HEAD
          OUTPUT_VARIABLE revision OUTPUT_STRIP_TRAILING_WHITESPACE
          ERROR_QUIET RESULT_VARIABLE revision_result)
        execute_process(COMMAND ${git_command} status --porcelain --untracked-files=no
          OUTPUT_VARIABLE dirty OUTPUT_STRIP_TRAILING_WHITESPACE
          ERROR_QUIET RESULT_VARIABLE dirty_result)
        if(revision_result EQUAL 0 AND dirty_result EQUAL 0 AND
           NOT revision STREQUAL "")
          set(KWS_BUILD_SOURCE_REVISION "${revision}")
          if(NOT dirty STREQUAL "")
            string(APPEND KWS_BUILD_SOURCE_REVISION "-dirty")
          endif()
        endif()
      endif()
    endif()
  endif()
endif()
# Overrides are caller-supplied labels, not Git verification. Preserve their
# value as a C string without allowing quotes or backslashes to break the SDK.
string(REPLACE "\\" "\\\\" KWS_BUILD_SOURCE_REVISION "${KWS_BUILD_SOURCE_REVISION}")
string(REPLACE "\"" "\\\"" KWS_BUILD_SOURCE_REVISION "${KWS_BUILD_SOURCE_REVISION}")
string(REPLACE "\n" "\\n" KWS_BUILD_SOURCE_REVISION "${KWS_BUILD_SOURCE_REVISION}")
string(REPLACE "\r" "\\r" KWS_BUILD_SOURCE_REVISION "${KWS_BUILD_SOURCE_REVISION}")
configure_file("${CMAKE_CURRENT_LIST_DIR}/kws_source_revision.h.in"
               "${KWS_SOURCE_REVISION_HEADER}" @ONLY)
