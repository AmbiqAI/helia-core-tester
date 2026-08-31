set(_NSX_HELPERS_DIR "${CMAKE_CURRENT_LIST_DIR}")

# App-side NSX helpers consumed by nsx_app_bootstrap.cmake.
#
# Not part of the vendored modules/nsx-ambiq-sdk package (that SDK provides
# only the low-level SoC/board/module CMake contract documented in its own
# cmake/README.md); this file is the thin, app-owned glue that resolves a
# "consolidated SDK bundle" module name to its on-disk directory and wires the
# SEGGER J-Link flash/reset custom targets. Deliberately local/uncommitted
# scaffolding historically -- see docs/perf-stream-expansion-progress.md.

# nsx_module_dir_for_name(<name> <out_var>)
#
# Resolve the source directory for NSX module <name> (e.g. "nsx-cmsis-core"):
#   1. An explicit NSX_APP_MODULE_DIR_<name-with-underscores> override, if set
#      (see CMakeLists.txt's "consolidated SDK bundle" overlay loop, which
#      redirects every module under modules/nsx-ambiq-sdk/modules/<name>
#      rather than the flat default below).
#   2. Each directory in NSX_APP_PROJECT_DIRS, tried as "<dir>/modules/<name>".
#   3. The flat default "modules/<name>" (a plain, non-bundled module tree).
# All results are returned relative to NSX_APP_ROOT.
function(nsx_module_dir_for_name name out_var)
    string(REPLACE "-" "_" _nsx_mod_var "${name}")
    if(DEFINED NSX_APP_MODULE_DIR_${_nsx_mod_var})
        set(${out_var} "${NSX_APP_MODULE_DIR_${_nsx_mod_var}}" PARENT_SCOPE)
        return()
    endif()
    foreach(_nsx_project_dir IN LISTS NSX_APP_PROJECT_DIRS)
        if(IS_DIRECTORY "${NSX_APP_ROOT}/${_nsx_project_dir}/modules/${name}")
            set(${out_var} "${_nsx_project_dir}/modules/${name}" PARENT_SCOPE)
            return()
        endif()
    endforeach()
    set(${out_var} "modules/${name}" PARENT_SCOPE)
endfunction()

# nsx_add_segger_targets(<target>)
#
# Registers <target>_flash and <target>_reset custom targets that drive
# SEGGER JLinkExe against the board's already-published SoC facts
# (NSX_SEGGER_DEVICE / NSX_SEGGER_IF_SPEED / NSX_SEGGER_PF_ADDR, set by
# cmake/socs/facts/<skew>.cmake via nsx_load_soc_facts() -- see board.cmake's
# include chain). NSX_JLINK_SERIAL (optional, set via -DNSX_JLINK_SERIAL=...)
# disambiguates which probe to use when more than one is connected.
#
# CONFIRMED (2026-08-05): this device/if/speed/load-address combination
# matches the known-working JLink invocation for Apollo510 -- see
# cmake/nsx/segger/templates/flash_cmds.jlink.in's header comment.
function(nsx_add_segger_targets target)
    if(NOT DEFINED NSX_SEGGER_DEVICE OR NOT DEFINED NSX_SEGGER_IF_SPEED OR NOT DEFINED NSX_SEGGER_PF_ADDR)
        message(FATAL_ERROR
            "nsx_add_segger_targets(${target}): NSX_SEGGER_DEVICE/NSX_SEGGER_IF_SPEED/"
            "NSX_SEGGER_PF_ADDR must be published first (nsx_load_soc_facts() via board.cmake).")
    endif()

    find_program(NSX_JLINK_EXE JLinkExe)

    get_target_property(_nsx_rtd ${target} RUNTIME_OUTPUT_DIRECTORY)
    if(NOT _nsx_rtd)
        set(_nsx_rtd "${CMAKE_CURRENT_BINARY_DIR}")
    endif()
    set(NSX_JLINK_BIN_FILE "${_nsx_rtd}/${target}.bin")

    set(_nsx_segger_dir "${CMAKE_CURRENT_BINARY_DIR}/_nsx_segger/${target}")
    file(MAKE_DIRECTORY "${_nsx_segger_dir}")
    configure_file(
        "${_NSX_HELPERS_DIR}/segger/templates/flash_cmds.jlink.in"
        "${_nsx_segger_dir}/flash_cmds.jlink"
        @ONLY)
    configure_file(
        "${_NSX_HELPERS_DIR}/segger/templates/reset_cmds.jlink.in"
        "${_nsx_segger_dir}/reset_cmds.jlink"
        @ONLY)

    set(_nsx_jlink_common_args
        -device "${NSX_SEGGER_DEVICE}"
        -if SWD
        -speed "${NSX_SEGGER_IF_SPEED}"
        -ExitOnError 1
        -NoGui 1
        -AutoConnect 1)
    if(DEFINED NSX_JLINK_SERIAL)
        list(APPEND _nsx_jlink_common_args -USB "${NSX_JLINK_SERIAL}")
    endif()

    add_custom_target(${target}_flash
        COMMAND "${NSX_JLINK_EXE}" ${_nsx_jlink_common_args}
                -CommanderScript "${_nsx_segger_dir}/flash_cmds.jlink"
        DEPENDS ${target}
        COMMENT "Flashing ${target} via SEGGER J-Link (${NSX_SEGGER_DEVICE})"
        VERBATIM)

    add_custom_target(${target}_reset
        COMMAND "${NSX_JLINK_EXE}" ${_nsx_jlink_common_args}
                -CommanderScript "${_nsx_segger_dir}/reset_cmds.jlink"
        COMMENT "Resetting ${target}'s board via SEGGER J-Link"
        VERBATIM)

    add_custom_target(${target}_view
        COMMENT "No SWO/RTT viewer wired yet for ${target} -- use the perf-stream CLI's RTT session instead."
        VERBATIM)
endfunction()
