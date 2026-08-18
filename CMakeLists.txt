cmake_minimum_required(VERSION 3.15.6)

# Build bare-metal tests for Arm FVP Corstone-300 with optional gcov coverage
project(helia_aot_fvp_tests C)
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
set(CMAKE_EXECUTABLE_SUFFIX ".elf")

set(CMSIS_PATH "artifacts/downloads/CMSIS_5" CACHE PATH "Path to CMSIS (CMSIS_5 checkout)")
set(CMSIS_NN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." CACHE PATH "ns-cmsis-nn repo root")
set(FVP_CORSTONE_300_PATH "${CMAKE_CURRENT_SOURCE_DIR}/Corstone-300" CACHE PATH "FVP startup/retarget files")
set(GENERATED_TESTS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/artifacts/generated_tests" CACHE PATH "Path to generated tests")
option(ENABLE_COVERAGE "Enable GCC freestanding coverage for ns-cmsis-nn" OFF)
option(ENABLE_COVERAGE_MVE_FLOAT "Enable float MVE paths during coverage builds" OFF)

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

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "cortex-m4")
  add_compile_options(-mfloat-abi=hard -mfpu=fpv4-sp-d16)
  add_link_options(   -mfloat-abi=hard -mfpu=fpv4-sp-d16)
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "cortex-m7")
  add_compile_options(-mfloat-abi=hard -mfpu=fpv5-d16)
  add_link_options(   -mfloat-abi=hard -mfpu=fpv5-d16)
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "cortex-m33" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "cortex-m55")
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
  if(ENABLE_COVERAGE_MVE_FLOAT AND NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "cortex-m55")
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

# Map cortex-m* to ARMCM* for CMSIS startup
string(REGEX REPLACE "^cortex-m([0-9]+)$" "ARMCM\\1" ARM_CPU ${CMAKE_SYSTEM_PROCESSOR})
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "cortex-m33")
  set(ARM_FEATURES "_DSP_FP")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "cortex-m4")
  set(ARM_FEATURES "_FP")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "cortex-m7")
  set(ARM_FEATURES "_DP")
else()
  set(ARM_FEATURES "")
endif()

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

  target_compile_definitions(${TGT_NAME} PUBLIC USING_FVP_CORSTONE_300)
  target_compile_options(${TGT_NAME} PRIVATE -Wno-switch-default -Wno-unused-parameter -Wno-cast-align)

  # Linker script + libs
  target_link_options(${TGT_NAME} PRIVATE -T ${LINK_FILE})
  if(ENABLE_COVERAGE)
    # Pull in libgcov runtime symbols and emit gcov metadata in the final ELF.
    target_link_options(${TGT_NAME} PRIVATE --coverage)
  endif()
  target_link_libraries(${TGT_NAME} PRIVATE cmsis-nn retarget cmsis_startup helia_test_runtime)

  list(APPEND GENERATED_TEST_TARGETS ${TGT_NAME})
endforeach()

if(GENERATED_TEST_TARGETS)
  add_custom_target(generated_tests ALL DEPENDS ${GENERATED_TEST_TARGETS})
endif()

# ---------- Summary ----------
message(STATUS "  CMSIS_PATH             : ${CMSIS_PATH}")
message(STATUS "  CMSIS_NN_ROOT          : ${CMSIS_NN_ROOT}")
message(STATUS "  FVP_CORSTONE_300_PATH  : ${FVP_CORSTONE_300_PATH}")
message(STATUS "  TARGET CPU             : ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "  Coverage               : ${ENABLE_COVERAGE}")
message(STATUS "  Tests dir              : ${CMAKE_BINARY_DIR}/tests/<family>")
