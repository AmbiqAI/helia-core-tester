# App-side NSX bootstrap glue, consumed by this repo's top-level CMakeLists.txt
# for HELIA_HARDWARE_BUILD=ON.
#
# Not part of the vendored modules/nsx-ambiq-sdk package. That SDK exposes only
# the low-level SoC/board/module CMake contract documented in its own
# cmake/README.md (toolchain -> SoC family -> SoC skew -> board layers, plus
# each module's own nsx_require_target()/FATAL_ERROR dependency checks) --
# there is no vendor-provided top-level "add these N modules for board X"
# orchestration function. This file is that app-owned orchestration layer.
# Historically hand-written and kept local/uncommitted per
# docs/perf-stream-expansion-progress.md; reconstructed here from the SDK's
# own documented contract plus this repo's own callers (CMakeLists.txt,
# helia_core_tester/perf_stream/cli.py) rather than from a lost original.

include("${CMAKE_CURRENT_LIST_DIR}/nsx_helpers.cmake")

# nsx_bootstrap_app(APP_ROOT <dir> BOARD <name> MODULES <name> [<name>...])
#
# Resolves NSX_ROOT/NSX_CMAKE_DIR/NSX_AMBIQSUITE_ROOT/NSX_SDK_PROVIDER, brings
# in the board descriptor (which transitively loads the SoC descriptor, SoC
# facts, and toolchain-flags helpers -- see modules/nsx-ambiq-sdk/cmake/
# README.md's Selection Order), then add_subdirectory()s each requested
# module in the caller-given (dependency) order.
macro(nsx_bootstrap_app)
    cmake_parse_arguments(NSX_BOOTSTRAP "" "APP_ROOT;BOARD" "MODULES" ${ARGN})
    if(NOT NSX_BOOTSTRAP_APP_ROOT)
        message(FATAL_ERROR "nsx_bootstrap_app: APP_ROOT is required.")
    endif()
    if(NOT NSX_BOOTSTRAP_BOARD)
        message(FATAL_ERROR "nsx_bootstrap_app: BOARD is required.")
    endif()

    set(NSX_APP_ROOT "${NSX_BOOTSTRAP_APP_ROOT}")

    # NSX_ROOT is the "consolidated SDK bundle" package root (the actual
    # modules/nsx-ambiq-sdk checkout), not the consuming app's own root --
    # cmake/socs/<skew>.cmake resolves its startup/linker-script sources as
    # "${NSX_ROOT}/modules/nsx-core/src/<skew>/...". NSX_APP_PROJECT_DIRS is
    # set by the caller (CMakeLists.txt) to the same bundle-relative path;
    # take its first entry as the bundle root.
    list(GET NSX_APP_PROJECT_DIRS 0 _nsx_bundle_dir)
    set(NSX_ROOT "${NSX_APP_ROOT}/${_nsx_bundle_dir}")
    set(NSX_CMAKE_DIR "${NSX_ROOT}/cmake")
    if(NOT EXISTS "${NSX_CMAKE_DIR}/nsx_toolchain_flags.cmake")
        message(FATAL_ERROR
            "nsx_bootstrap_app: ${NSX_CMAKE_DIR} does not look like an nsx-ambiq-sdk "
            "checkout (missing nsx_toolchain_flags.cmake).")
    endif()

    # Only the ambiqsuite SDK provider is currently wired up.
    set(NSX_SDK_PROVIDER "ambiqsuite")

    nsx_module_dir_for_name("nsx-ambiqsuite" _nsx_ambiqsuite_module_dir)
    set(NSX_AMBIQSUITE_ROOT "${NSX_APP_ROOT}/${_nsx_ambiqsuite_module_dir}/sdk")
    if(NOT IS_DIRECTORY "${NSX_AMBIQSUITE_ROOT}")
        message(FATAL_ERROR "nsx_bootstrap_app: NSX_AMBIQSUITE_ROOT does not exist: ${NSX_AMBIQSUITE_ROOT}")
    endif()

    # Read the provider's own manifest instead of hardcoding the version, so
    # it can't silently drift from whatever SDK bundle is actually vendored.
    set(_nsx_ambiqsuite_manifest "${NSX_APP_ROOT}/${_nsx_ambiqsuite_module_dir}/nsx-module.yaml")
    file(STRINGS "${_nsx_ambiqsuite_manifest}" _nsx_version_line REGEX "^  version:")
    string(REGEX REPLACE "^  version: *\"?([^\"]*)\"?$" "\\1" NSX_AMBIQSUITE_VERSION "${_nsx_version_line}")
    if(NOT NSX_AMBIQSUITE_VERSION)
        message(FATAL_ERROR "nsx_bootstrap_app: could not parse version from ${_nsx_ambiqsuite_manifest}")
    endif()

    # Pulls in cmake/socs/<skew>.cmake (via NSX_SEGGER_PF_ADDR ... 's
    # cmake/socs/facts/<skew>.cmake -> nsx_load_soc_facts()), which defines
    # nsx::board_flags, nsx::soc, nsx::soc_flags, NSX_STARTUP_SOURCE,
    # NSX_SYSTEM_SOURCE, NSX_LINKER_SCRIPT, and the NSX_SEGGER_* J-Link facts
    # nsx_add_segger_targets() (nsx_helpers.cmake) consumes later.
    include("${NSX_APP_ROOT}/boards/${NSX_BOOTSTRAP_BOARD}/board.cmake")

    foreach(_nsx_module_name IN LISTS NSX_BOOTSTRAP_MODULES)
        nsx_module_dir_for_name("${_nsx_module_name}" _nsx_module_dir)
        string(REPLACE "-" "_" _nsx_module_target "${_nsx_module_name}")
        add_subdirectory("${NSX_APP_ROOT}/${_nsx_module_dir}" "${CMAKE_BINARY_DIR}/_nsx/${_nsx_module_target}")
    endforeach()
endmacro()

# nsx_finalize_app(<target>)
#
# Post-processes a linked firmware target for real hardware: emits a raw
# <target>.bin next to the .elf (arm-none-eabi-objcopy, via the active
# toolchain file's CMAKE_OBJCOPY) and wires the <target>_flash/<target>_reset
# SEGGER J-Link custom targets (nsx_add_segger_targets(), nsx_helpers.cmake).
function(nsx_finalize_app target)
    if(NOT CMAKE_OBJCOPY)
        message(FATAL_ERROR "nsx_finalize_app(${target}): CMAKE_OBJCOPY is not set by the active toolchain file.")
    endif()
    if(NOT DEFINED NSX_LINKER_SCRIPT)
        message(FATAL_ERROR "nsx_finalize_app(${target}): NSX_LINKER_SCRIPT is not set (expected from board.cmake's SoC descriptor).")
    endif()
    # The SDK resolves and validates NSX_LINKER_SCRIPT (nsx_select_linker_script(),
    # nsx_assert_file_exists() in cmake/socs/<skew>.cmake) but never applies it --
    # unlike the FVP path's own explicit `-T ${LINK_FILE}` in this repo's
    # CMakeLists.txt, that's left to the app/target integration layer.
    target_link_options(${target} PRIVATE -T "${NSX_LINKER_SCRIPT}")
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND "${CMAKE_OBJCOPY}" -O binary "$<TARGET_FILE:${target}>" "$<TARGET_FILE_DIR:${target}>/${target}.bin"
        COMMENT "Generating ${target}.bin"
        VERBATIM)
    nsx_add_segger_targets(${target})
endfunction()
