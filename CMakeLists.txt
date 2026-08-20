cmake_minimum_required(VERSION 3.15.6)

# Build bare-metal tests for Arm FVP Corstone-300 with optional gcov coverage
project(helia_aot_fvp_tests C)
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
set(CMAKE_EXECUTABLE_SUFFIX ".elf")

set(CMSIS_PATH "artifacts/downloads/CMSIS_5" CACHE PATH "Path to CMSIS (CMSIS_5 checkout)")
set(CMSIS_NN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." CACHE PATH "ns-cmsis-nn repo root")
set(FVP_CORSTONE_300_PATH "${CMAKE_CURRENT_SOURCE_DIR}/Corstone-300" CACHE PATH "FVP startup/retarget files")
set(GENERATED_TESTS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/artifacts/generated_tests" CACHE PATH "Path to generated tests")
set(NEURALSPOTX_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../neuralspotx" CACHE PATH "Path to neuralspotx repo root")
set(TARGET_CPU "" CACHE STRING "Target CPU override (e.g. cortex-m55)")
option(ENABLE_COVERAGE "Enable GCC freestanding coverage for ns-cmsis-nn" OFF)
option(ENABLE_COVERAGE_MVE_FLOAT "Enable float MVE paths during coverage builds" OFF)
option(HELIA_BUILD_GENERATED_TESTS "Build descriptor-generated standalone test executables" ON)
option(HELIA_BUILD_UNIVERSAL_SIZE_PROBE "Build the phase-0 universal firmware size probe" OFF)
option(HELIA_BUILD_PERF_STREAM_BENCHMARK_SERVER "Build the streaming benchmark-server firmware skeleton" OFF)

# --benchmark support: instruments generated standalone test harnesses (see
# assets/templates/common/standalone/benchmark.j2) to split each op's
# init/precompute from the timed kernel call and report DWT cycle counts.
# Pass `-DHELIA_BENCHMARK_MODE=ON [-DHELIA_BENCHMARK_WARMUP_RUNS=<n>
# -DHELIA_BENCHMARK_MEASURED_RUNS=<n>]` directly to CMake (or via
# `build_and_run_fvp.py --cmake-def ...`) to enable; otherwise
# benchmark.j2's `#ifdef HELIA_BENCHMARK_MODE` block is compiled out and the
# harness behaves exactly as before (single-shot correctness run).
option(HELIA_BENCHMARK_MODE "Enable DWT-based op-level cycle benchmarking in generated test harnesses" OFF)
set(HELIA_BENCHMARK_WARMUP_RUNS "3" CACHE STRING "Untimed warmup iterations per op when HELIA_BENCHMARK_MODE is ON")
set(HELIA_BENCHMARK_MEASURED_RUNS "10" CACHE STRING "Timed measured iterations per op when HELIA_BENCHMARK_MODE is ON")

# --hardware support: build generated test executables against real Apollo510
# hardware (the vendored NSX board bring-up under boards/apollo510_evb +
# modules/nsx-ambiq-sdk) instead of the FVP/ethos-u-core-platform toolchain.
# See scripts/test_s4_conv_benchmark.sh --hardware for the driving script.
#
# UPDATE (2026-08-04): configure + cross-compile build verified end-to-end
# in a sandbox (arm-none-eabi-gcc 14.3.1, no real board attached) for both
# cortex-m55 and cortex-m55+nomve variants of convolve_generic_s4 -- produces
# a linked .elf and .bin via nsx_finalize_app(). The NSX_MODULES list below
# was derived from reading each vendored module's own nsx_require_target()/
# FATAL_ERROR dependency checks (no existing nsx_bootstrap_app() caller was
# found anywhere in this tree to use as a reference) and iteratively fixed
# against real `cmake --build` errors (added nsx-interrupt, linked
# nsx::core + nsx::perf). STILL UNVERIFIED: flashing via SEGGER J-Link and
# SWO capture on real Apollo510 silicon -- please report back any errors
# from those steps so they can be corrected.
option(HELIA_HARDWARE_BUILD "Build generated tests for real Apollo510 hardware (NSX bring-up) instead of the FVP toolchain" OFF)
set(HELIA_HARDWARE_BOARD "apollo510_evb" CACHE STRING "NSX board name for HELIA_HARDWARE_BUILD")

if((HELIA_BUILD_UNIVERSAL_SIZE_PROBE OR HELIA_BUILD_PERF_STREAM_BENCHMARK_SERVER) AND NOT HELIA_HARDWARE_BUILD)
  message(FATAL_ERROR "Streaming perf-stream firmware targets require HELIA_HARDWARE_BUILD=ON")
endif()

if(HELIA_HARDWARE_BUILD)
  set(NSX_APP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
  set(NSX_AMBIQ_SDK_DIR "modules/nsx-ambiq-sdk")
  if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${NSX_AMBIQ_SDK_DIR}/modules")
    message(FATAL_ERROR
      "HELIA_HARDWARE_BUILD=ON but ${NSX_AMBIQ_SDK_DIR}/modules is missing. "
      "This consolidated SDK bundle is vendored/fetched separately (not "
      "tracked in git) -- ensure it has been bootstrapped before configuring.")
  endif()

  # "Consolidated SDK bundle" overlay (see nsx_app_bootstrap.cmake's
  # nsx_module_dir_for_name() comment): every module below actually lives
  # under modules/nsx-ambiq-sdk/modules/<name>, not the flat modules/<name>
  # nsx_bootstrap_app() defaults to, so redirect each one explicitly.
  set(NSX_APP_PROJECT_DIRS "${NSX_AMBIQ_SDK_DIR}")
  file(GLOB _nsx_bundle_module_dirs RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/${NSX_AMBIQ_SDK_DIR}/modules"
       "${CMAKE_CURRENT_SOURCE_DIR}/${NSX_AMBIQ_SDK_DIR}/modules/*")
  foreach(_nsx_mod_name ${_nsx_bundle_module_dirs})
    if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${NSX_AMBIQ_SDK_DIR}/modules/${_nsx_mod_name}")
      string(REPLACE "-" "_" _nsx_mod_var "${_nsx_mod_name}")
      set(NSX_APP_MODULE_DIR_${_nsx_mod_var} "${NSX_AMBIQ_SDK_DIR}/modules/${_nsx_mod_name}")
    endif()
  endforeach()

  # Dependency order derived from each module's own nsx_require_target()/
  # FATAL_ERROR checks (modules/nsx-ambiq-sdk/modules/<name>/CMakeLists.txt):
  #   nsx-cmsis-core  : no deps (INTERFACE target nsx::cmsis_core)
  #   nsx-ambiqsuite  : no deps (defines nsx_sdk_ambiqsuite / NSX_SELECTED_SDK_TARGET)
  #   nsx-ambiq-hal   : needs nsx::board_flags + NSX_AMBIQSUITE_ROOT + NSX_AMBIQ_* HAL paths (board.cmake)
  #   nsx-ambiq-bsp   : needs nsx_ambiq_hal + nsx::board_flags + NSX_AMBIQ_BSP_* paths (board.cmake)
  #   nsx-soc-hal     : needs nsx_ambiq_hal + nsx_ambiq_bsp
  #   nsx-cmsis-startup: needs nsx::board_flags + nsx::cmsis_core + NSX_AMBIQSUITE_ROOT
  #   nsx-core        : needs nsx::soc_hal
  #   nsx-interrupt   : needs nsx::soc_hal (NSX_SOC_FAMILY); required before
  #                     nsx-uart per configure-time check in
  #                     nsx_toolchain_flags.cmake ("nsx::interrupt must be
  #                     defined before nsx-uart is added") -- found via an
  #                     actual `cmake -S . -B ... -DHELIA_HARDWARE_BUILD=ON`
  #                     configure attempt, not just static reading.
  #   nsx-uart        : needs nsx_core + nsx::interrupt (physical UART; not
  #                     currently used by benchmark_hw.j2, which uses
  #                     ITM/SWO via am_util_stdio, but kept available for
  #                     future use)
  #   nsx-perf        : needs nsx::soc_hal + nsx::core (not currently used by
  #                     benchmark_hw.j2, which reads pmu_armv8.h directly;
  #                     kept available for future use)
  set(NSX_MODULES
    nsx-cmsis-core
    nsx-ambiqsuite
    nsx-ambiq-hal
    nsx-ambiq-bsp
    nsx-soc-hal
    nsx-cmsis-startup
    nsx-core
    nsx-interrupt
    nsx-uart
    nsx-perf
  )

  include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/nsx/nsx_app_bootstrap.cmake")
  nsx_bootstrap_app(APP_ROOT "${NSX_APP_ROOT}" BOARD "${HELIA_HARDWARE_BOARD}" MODULES ${NSX_MODULES})

  # VERIFIED (2026-08-04, host arm-none-eabi-gcc 14.3.1 cross-build, no real
  # hardware): configured + built both cortex-m55 (MVE) and cortex-m55+nomve
  # (DSP-only) variants of convolve_generic_s4 end-to-end (compile, link,
  # .bin generation via nsx_finalize_app()). Confirmed with
  # `arm-none-eabi-objdump -d <elf> | grep -ciE "vldrw|vmulh|vaddva|vctp"`
  # that the +nomve build has 0 MVE instructions vs. 70 in the MVE build --
  # so NSX's toolchain flags do NOT clobber the global -mcpu. Still
  # unverified: everything downstream of `.bin` generation -- flashing via
  # SEGGER J-Link and SWO capture require a real Apollo510 board, which is
  # not available in this sandbox.
endif()

function(_strip_all_optimizations)
  foreach(lang C CXX ASM)
    foreach(var
      CMAKE_${lang}_FLAGS
      CMAKE_${lang}_FLAGS_RELEASE
      CMAKE_${lang}_FLAGS_RELWITHDEBINFO
      CMAKE_${lang}_FLAGS_MINSIZEREL
      CMAKE_${lang}_FLAGS_DEBUG
    )
      if(DEFINED ${var})
        set(_tmp "${${var}}")
        string(REGEX REPLACE "([ ;]|^)-Ofast([ ;]|$)" " " _tmp "${_tmp}")
        string(REGEX REPLACE "([ ;]|^)-O[0-3s]([ ;]|$)" " " _tmp "${_tmp}")
        string(REGEX REPLACE "[ ]+" " " _tmp "${_tmp}")
        string(STRIP "${_tmp}" _tmp)
        set(${var} "${_tmp}" CACHE STRING "" FORCE)
      endif()
    endforeach()
  endforeach()
endfunction()


# Write .gcda to the build tree so each config keeps its own artifacts
# set(COVERAGE_DIR "${CMAKE_BINARY_DIR}/coverage" CACHE PATH "Where .gcda is written (-fprofile-dir)")

if(NOT EXISTS "${CMSIS_PATH}/CMSIS/Core/Include/cmsis_compiler.h")
  message(FATAL_ERROR "CMSIS_PATH does not look valid: ${CMSIS_PATH}")
endif()
if(DEFINED TARGET_CPU AND NOT TARGET_CPU STREQUAL "")
  set(CMAKE_SYSTEM_PROCESSOR "${TARGET_CPU}")
endif()
if(NOT CMAKE_SYSTEM_PROCESSOR)
  message(FATAL_ERROR "CMAKE_SYSTEM_PROCESSOR not set (e.g. cortex-m55|cortex-m33|cortex-m4)")
endif()
if(ENABLE_COVERAGE AND NOT CMAKE_C_COMPILER_ID STREQUAL "GNU")
  message(FATAL_ERROR "ENABLE_COVERAGE requires GCC (arm-none-eabi-gcc)")
endif()

# ---------- Global warnings ----------
add_compile_options(-Wimplicit-function-declaration -Wunused-variable -Wunused-function -Wno-redundant-decls -Wvla)

# ---------- CPU / FPU flags (compile + link) ----------
# Apply once globally; all targets are Cortex-M
add_compile_options(-mcpu=${CMAKE_SYSTEM_PROCESSOR} -mthumb)
add_link_options(   -mcpu=${CMAKE_SYSTEM_PROCESSOR} -mthumb)

# Note: CMAKE_SYSTEM_PROCESSOR may carry a GCC multi-lib suffix (e.g.
# "cortex-m55+nomve", used to force-disable the MVE code path on the same
# physical Cortex-M55/Apollo510 silicon so it can be benchmarked against the
# MVE path -- see docs on HELIA_BENCHMARK_MODE above). Match with the
# "cpu(\\+|$)" pattern rather than STREQUAL everywhere below so such variants
# still resolve to the correct base-CPU behavior.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^cortex-m4(\\+|$)")
  add_compile_options(-mfloat-abi=hard -mfpu=fpv4-sp-d16)
  add_link_options(   -mfloat-abi=hard -mfpu=fpv4-sp-d16)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^cortex-m7(\\+|$)")
  add_compile_options(-mfloat-abi=hard -mfpu=fpv5-d16)
  add_link_options(   -mfloat-abi=hard -mfpu=fpv5-d16)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^cortex-m33(\\+|$)" OR CMAKE_SYSTEM_PROCESSOR MATCHES "^cortex-m55(\\+|$)")
  add_compile_options(-mfloat-abi=hard -mfpu=fpv5-sp-d16)
  add_link_options(   -mfloat-abi=hard -mfpu=fpv5-sp-d16)
else()
  add_compile_options(-mfloat-abi=soft)
  add_link_options(   -mfloat-abi=soft)
endif()

# ---------- cmsis-nn subproject ----------
if(EXISTS "${CMSIS_NN_ROOT}/CMakeLists.txt")
  # Use a stable binary subdir name "cmsis-nn" (we need this exact name below)
  add_subdirectory("${CMSIS_NN_ROOT}" cmsis-nn)
else()
  message(FATAL_ERROR "cmsis-nn CMakeLists.txt not found at: ${CMSIS_NN_ROOT}")
endif()

if(ENABLE_COVERAGE)
  if(NOT TARGET cmsis-nn)
    message(FATAL_ERROR "ENABLE_COVERAGE requested but target cmsis-nn was not created")
  endif()
  if(ENABLE_COVERAGE_MVE_FLOAT AND NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^cortex-m55(\\+|$)")
    message(FATAL_ERROR "ENABLE_COVERAGE_MVE_FLOAT is only supported for cortex-m55")
  endif()
  # Instrument ns-cmsis-nn only; keep harness sources uninstrumented.
  target_compile_options(cmsis-nn PRIVATE --coverage -fprofile-info-section -O0 -g)
  if(NOT ENABLE_COVERAGE_MVE_FLOAT)
    # Avoid inline MVE asm paths that can fail register allocation under gcov instrumentation.
    target_compile_definitions(cmsis-nn PRIVATE ARM_MATH_AUTOVECTORIZE)
  else()
    # Float MVE coverage requested. ARM_MATH_AUTOVECTORIZE keeps ARM_MATH_MVEI defined
    # (integer inline asm with a C fallback) but suppresses ARM_MATH_MVEF, so a
    # target-wide define would erase the float MVE paths we want to measure. Scope the
    # define to the non-float (integer) sources only: their register-heavy inline asm
    # falls back to C (avoiding gcov register-allocation failures), while the float
    # sources keep ARM_MATH_MVEF and compile their real MVE paths for coverage.
    if(CMAKE_VERSION VERSION_LESS "3.18")
      message(FATAL_ERROR
        "ENABLE_COVERAGE_MVE_FLOAT requires CMake >= 3.18 "
        "(set_source_files_properties TARGET_DIRECTORY); found ${CMAKE_VERSION}")
    endif()
    get_target_property(_nn_srcs cmsis-nn SOURCES)
    set(_nn_int_srcs "")
    foreach(_s IN LISTS _nn_srcs)
      # Classify by filename suffix; anchor to end so float-output integer-input
      # files (e.g. arm_dequantize_s8_f32.c) are correctly treated as float.
      if(NOT _s MATCHES "_(f16|f32|fp16|flt)\\.c$")
        list(APPEND _nn_int_srcs "${_s}")
      endif()
    endforeach()
    if(_nn_int_srcs)
      set_source_files_properties(${_nn_int_srcs}
        TARGET_DIRECTORY cmsis-nn
        PROPERTIES COMPILE_DEFINITIONS ARM_MATH_AUTOVECTORIZE)
    endif()
  endif()
endif()

# ---------- FVP retarget + startup ----------
set(RETARGET_SRCS
  ${FVP_CORSTONE_300_PATH}/retarget.c
  ${FVP_CORSTONE_300_PATH}/uart.c
)
if(ENABLE_COVERAGE)
  list(APPEND RETARGET_SRCS ${FVP_CORSTONE_300_PATH}/gcov_stream.c)
endif()
add_library(retarget STATIC ${RETARGET_SRCS})
target_compile_definitions(retarget PUBLIC USING_FVP_CORSTONE_300)
target_compile_definitions(retarget PRIVATE ENABLE_RETARGET_SYSCALLS)
if(ENABLE_COVERAGE)
  target_compile_definitions(retarget PRIVATE ENABLE_GCOV_COVERAGE)
endif()

# Map cortex-m* to ARMCM* for CMSIS startup. Strip any GCC multi-lib suffix
# (e.g. the trailing "+nomve" in "cortex-m55+nomve", used to force-disable
# MVE on the same physical Cortex-M55/Apollo510 silicon) before matching, so
# the device header/startup file selection is keyed on the base CPU only --
# ARM_CPU is about which physical chip's CMSIS device pack to use, not which
# instruction-set extensions the compiler emits for it.
string(REGEX REPLACE "\\+.*$" "" _cpu_base ${CMAKE_SYSTEM_PROCESSOR})
string(REGEX REPLACE "^cortex-m([0-9]+)$" "ARMCM\\1" ARM_CPU ${_cpu_base})
if(_cpu_base STREQUAL "cortex-m33")
  set(ARM_FEATURES "_DSP_FP")
elseif(_cpu_base STREQUAL "cortex-m4")
  set(ARM_FEATURES "_FP")
elseif(_cpu_base STREQUAL "cortex-m7")
  set(ARM_FEATURES "_DP")
else()
  set(ARM_FEATURES "")
endif()

if(NOT HELIA_HARDWARE_BUILD)
  # FVP toolchain path: bare CMSIS device startup/system sources + the
  # Corstone-300 linker script. Skipped for HELIA_HARDWARE_BUILD, which gets
  # its startup/linker script from the NSX board bring-up instead (nsx-core /
  # nsx-cmsis-startup modules, wired via nsx_bootstrap_app()/nsx_finalize_app()
  # above/below).
  add_library(cmsis_startup STATIC)
  target_sources(cmsis_startup PRIVATE
    ${CMSIS_PATH}/Device/ARM/${ARM_CPU}/Source/startup_${ARM_CPU}.c
    ${CMSIS_PATH}/Device/ARM/${ARM_CPU}/Source/system_${ARM_CPU}.c)
  target_include_directories(cmsis_startup PUBLIC
    ${CMSIS_PATH}/Device/ARM/${ARM_CPU}/Include
    ${CMSIS_PATH}/CMSIS/Core/Include)
  target_compile_options(cmsis_startup INTERFACE -include${ARM_CPU}${ARM_FEATURES}.h)
  target_compile_definitions(cmsis_startup PRIVATE ${ARM_CPU}${ARM_FEATURES})

  # Linker script (GNU)
  set(LINK_FILE ${FVP_CORSTONE_300_PATH}/linker.ld)
endif()

if(HELIA_BUILD_UNIVERSAL_SIZE_PROBE OR HELIA_BUILD_PERF_STREAM_BENCHMARK_SERVER)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)

  set(HELIA_KERNEL_SYMBOL_REFS_DIR "${CMAKE_BINARY_DIR}/perf_stream")
  set(HELIA_KERNEL_SYMBOL_REFS_INC "${HELIA_KERNEL_SYMBOL_REFS_DIR}/kernel_symbol_refs.inc")
  set(HELIA_KERNEL_SYMBOL_REF_ARGS
    --cmsis-nn-root "${CMSIS_NN_ROOT}"
    --archive "$<TARGET_FILE:cmsis-nn>"
    --output "${HELIA_KERNEL_SYMBOL_REFS_INC}")
  if(ARM_NN_ENABLE_F32)
    list(APPEND HELIA_KERNEL_SYMBOL_REF_ARGS --enable-f32)
  endif()
  if(ARM_NN_ENABLE_F16)
    list(APPEND HELIA_KERNEL_SYMBOL_REF_ARGS --enable-f16)
  endif()

  add_custom_command(
    OUTPUT "${HELIA_KERNEL_SYMBOL_REFS_INC}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${HELIA_KERNEL_SYMBOL_REFS_DIR}"
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_kernel_symbol_refs.py"
            ${HELIA_KERNEL_SYMBOL_REF_ARGS}
    DEPENDS
      cmsis-nn
      "${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_kernel_symbol_refs.py"
      "${CMSIS_NN_ROOT}/Include/arm_nnfunctions.h"
      "${CMSIS_NN_ROOT}/Include/arm_nnfunctions_flt.h"
    VERBATIM)

  add_custom_target(hct_kernel_symbol_refs DEPENDS "${HELIA_KERNEL_SYMBOL_REFS_INC}")

  function(_hct_link_perf_stream_target target_name)
    add_dependencies(${target_name} hct_kernel_symbol_refs)
    target_include_directories(${target_name} PRIVATE
      "${HELIA_KERNEL_SYMBOL_REFS_DIR}"
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/perf_stream"
      "${CMSIS_PATH}/CMSIS/Core/Include")
    target_compile_definitions(${target_name} PUBLIC HELIA_HARDWARE_BUILD)
    target_compile_options(${target_name} PRIVATE -Wno-cast-function-type)

    if(HELIA_HARDWARE_BUILD)
      target_link_libraries(${target_name} PRIVATE cmsis-nn nsx::board nsx::core nsx::perf)
      nsx_finalize_app(${target_name})
    else()
      target_link_options(${target_name} PRIVATE -T ${LINK_FILE})
      target_link_libraries(${target_name} PRIVATE cmsis-nn retarget cmsis_startup)
    endif()
  endfunction()

  if(HELIA_BUILD_UNIVERSAL_SIZE_PROBE)
    add_executable(hct_universal_size_probe
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/perf_stream/universal_size_probe.c")
    set_target_properties(hct_universal_size_probe PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/probe")
    _hct_link_perf_stream_target(hct_universal_size_probe)
  endif()

  if(HELIA_BUILD_PERF_STREAM_BENCHMARK_SERVER)
    set(HCT_RTT_BUILD_ROOT "${CMAKE_BINARY_DIR}/perf_stream/rtt")
    file(MAKE_DIRECTORY "${HCT_RTT_BUILD_ROOT}/RTT" "${HCT_RTT_BUILD_ROOT}/Config")
    configure_file("${NEURALSPOTX_ROOT}/examples/coremark/src/rtt/SEGGER_RTT.c" "${HCT_RTT_BUILD_ROOT}/RTT/SEGGER_RTT.c" COPYONLY)
    configure_file("${NEURALSPOTX_ROOT}/examples/coremark/src/rtt/SEGGER_RTT.h" "${HCT_RTT_BUILD_ROOT}/RTT/SEGGER_RTT.h" COPYONLY)
    configure_file("${NEURALSPOTX_ROOT}/examples/coremark/src/rtt/Config/SEGGER_RTT_Conf.h" "${HCT_RTT_BUILD_ROOT}/Config/SEGGER_RTT_Conf.h" COPYONLY)

    add_executable(hct_benchmark_server
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/perf_stream/benchmark_server_main.c"
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/perf_stream/hctp_protocol.c"
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/perf_stream/benchmark_server_catalog.c"
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/perf_stream/benchmark_server_messages.c"
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/perf_stream/benchmark_server_adapter.c"
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/perf_stream/benchmark_server_session.c"
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/perf_stream/benchmark_server_transport_rtt.c"
      "${HCT_RTT_BUILD_ROOT}/RTT/SEGGER_RTT.c")
    set_target_properties(hct_benchmark_server PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/perf_stream")
    target_include_directories(hct_benchmark_server PRIVATE
      "${HCT_RTT_BUILD_ROOT}/RTT"
      "${HCT_RTT_BUILD_ROOT}/Config")
    target_compile_definitions(hct_benchmark_server PRIVATE
      HCT_BENCHMARK_SERVER_BOARD_ID="${HELIA_HARDWARE_BOARD}"
      HCT_BENCHMARK_SERVER_TARGET_CPU="${TARGET_CPU}")
    _hct_link_perf_stream_target(hct_benchmark_server)
    add_custom_command(TARGET hct_benchmark_server POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE_DIR:hct_benchmark_server>/hct_benchmark_server.bin"
        "${CMAKE_BINARY_DIR}/hct_benchmark_server.bin"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:hct_benchmark_server>"
        "${CMAKE_BINARY_DIR}/hct_benchmark_server.elf")
  endif()
endif()

# ---------- Shared generated-test runtime ----------
# Runtime/validation logic common to every generated test .c file, compiled
# once here instead of being re-emitted by Jinja into each generated file.
add_library(helia_test_runtime STATIC src/test_runtime/helia_test_runtime.c)
target_include_directories(helia_test_runtime PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_compile_definitions(helia_test_runtime PUBLIC USING_FVP_CORSTONE_300)

# ---------- Helpers ----------
function(_normalize_target_name out_var in_path)
  get_filename_component(_tgt "${in_path}" NAME)
  set(${out_var} "${_tgt}" PARENT_SCOPE)
endfunction()

# ---------- Discover & build tests ----------
if(HELIA_BUILD_GENERATED_TESTS)
  # Load generated test directories from manifest-generated CMake file
  set(GENERATED_TEST_DIRS)
  set(GENERATED_TESTS_CMAKE "${GENERATED_TESTS_DIR}/tests.cmake")
  if(EXISTS "${GENERATED_TESTS_CMAKE}")
    include("${GENERATED_TESTS_CMAKE}")
    list(REMOVE_DUPLICATES GENERATED_TEST_DIRS)
  else()
    message(FATAL_ERROR "Missing generated tests list: ${GENERATED_TESTS_CMAKE}. Run `uv run helia_core_tester generate` first.")
  endif()

  set(GENERATED_TESTS_MANIFEST "${GENERATED_TESTS_DIR}/manifest.json")
  if(NOT EXISTS "${GENERATED_TESTS_MANIFEST}")
    message(WARNING "Missing generated tests manifest: ${GENERATED_TESTS_MANIFEST}")
  endif()

  foreach(TEST_DIR_REL ${GENERATED_TEST_DIRS})
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${TEST_DIR_REL}")
      message(FATAL_ERROR "Generated test directory missing: ${TEST_DIR_REL}")
    endif()
    file(GLOB TEST_SRCS_CHECK "${CMAKE_CURRENT_SOURCE_DIR}/${TEST_DIR_REL}/*.c")
    if(TEST_SRCS_CHECK STREQUAL "")
      message(WARNING "No generated C sources found in ${TEST_DIR_REL}")
    endif()
  endforeach()

  set(GENERATED_TEST_TARGETS)
  foreach(TEST_DIR_REL ${GENERATED_TEST_DIRS})
    get_filename_component(TEST_DIR_ABS "${TEST_DIR_REL}" ABSOLUTE)
    _normalize_target_name(TGT_NAME "${TEST_DIR_REL}")
    get_filename_component(TEST_PARENT_DIR "${TEST_DIR_REL}" DIRECTORY)
    get_filename_component(FAMILY_NAME "${TEST_PARENT_DIR}" NAME)

    # Find generated C files in this test directory (e.g., basic_s8_conv2d.c)
    file(GLOB TEST_SRCS "${TEST_DIR_ABS}/*_*.c")
    # Filter out any unwanted files if needed
    list(FILTER TEST_SRCS EXCLUDE REGEX ".*test_runner\\.c$")
    
    if(TEST_SRCS STREQUAL "")
      message(WARNING "No generated C sources found in ${TEST_DIR_REL}; skipping")
      continue()
    endif()
    
    set(SRCS ${TEST_SRCS})

    add_executable(${TGT_NAME} ${SRCS})
    set_target_properties(${TGT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests/${FAMILY_NAME}")

    # Includes (generated API + local cmsis.h + neuralSPOT bits used by tests)
    file(GLOB NEURALSPOT_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/modules/neuralSPOT/neuralspot/ns-*/includes")
    target_include_directories(${TGT_NAME} PRIVATE
      "${TEST_DIR_ABS}/includes"
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      ${NEURALSPOT_INCLUDE_DIRS}
      "${CMAKE_CURRENT_SOURCE_DIR}/modules/neuralSPOT/extern/AmbiqSuite/R5.1.0_rc27/utils"
      "${CMAKE_CURRENT_SOURCE_DIR}/modules/neuralSPOT/extern/AmbiqSuite/R5.1.0_rc27/boards/apollo510_evb/bsp"
      "${CMAKE_CURRENT_SOURCE_DIR}/modules/neuralSPOT/extern/AmbiqSuite/R5.1.0_rc27/CMSIS/ARM/Include"
      "${CMAKE_CURRENT_SOURCE_DIR}/modules/neuralSPOT/extern/AmbiqSuite/R5.1.0_rc27/CMSIS/AmbiqMicro/Include"
      "${CMAKE_CURRENT_SOURCE_DIR}/modules/neuralSPOT/extern/CMSIS/CMSIS-DSP-1.16.2/Include")

    target_compile_definitions(${TGT_NAME} PUBLIC $<IF:$<BOOL:${HELIA_HARDWARE_BUILD}>,HELIA_HARDWARE_BUILD,USING_FVP_CORSTONE_300>)
    target_compile_options(${TGT_NAME} PRIVATE -Wno-switch-default -Wno-unused-parameter -Wno-cast-align)

    if(HELIA_BENCHMARK_MODE)
      # See benchmark.j2 / the HELIA_BENCHMARK_MODE option comment above: these
      # defines gate and parameterize the generated harness's DWT cycle-count
      # instrumentation.
      target_compile_definitions(${TGT_NAME} PRIVATE
        HELIA_BENCHMARK_MODE
        HELIA_BENCHMARK_WARMUP_RUNS=${HELIA_BENCHMARK_WARMUP_RUNS}
        HELIA_BENCHMARK_MEASURED_RUNS=${HELIA_BENCHMARK_MEASURED_RUNS})
    endif()

    # Linker script + libs
    if(HELIA_HARDWARE_BUILD)
      # Real Apollo510 hardware: NSX board target supplies startup, system init,
      # and linker script; nsx_finalize_app() (called below) generates the .bin
      # and wires the <target>_flash/_reset/_view SEGGER J-Link CMake targets.
      if(ENABLE_COVERAGE)
        target_link_options(${TGT_NAME} PRIVATE --coverage)
      endif()
      target_link_libraries(${TGT_NAME} PRIVATE cmsis-nn nsx::board nsx::core nsx::perf)
      nsx_finalize_app(${TGT_NAME})
    else()
      target_link_options(${TGT_NAME} PRIVATE -T ${LINK_FILE})
      if(ENABLE_COVERAGE)
        # Pull in libgcov runtime symbols and emit gcov metadata in the final ELF.
        target_link_options(${TGT_NAME} PRIVATE --coverage)
      endif()
      target_link_libraries(${TGT_NAME} PRIVATE cmsis-nn retarget cmsis_startup helia_test_runtime)
    endif()

    list(APPEND GENERATED_TEST_TARGETS ${TGT_NAME})
  endforeach()

  if(GENERATED_TEST_TARGETS)
    add_custom_target(generated_tests ALL DEPENDS ${GENERATED_TEST_TARGETS})
  endif()
endif()

# ---------- Summary ----------
message(STATUS "  CMSIS_PATH             : ${CMSIS_PATH}")
message(STATUS "  CMSIS_NN_ROOT          : ${CMSIS_NN_ROOT}")
message(STATUS "  FVP_CORSTONE_300_PATH  : ${FVP_CORSTONE_300_PATH}")
message(STATUS "  TARGET CPU             : ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "  Coverage               : ${ENABLE_COVERAGE}")
message(STATUS "  Tests dir              : ${CMAKE_BINARY_DIR}/tests/<family>")
