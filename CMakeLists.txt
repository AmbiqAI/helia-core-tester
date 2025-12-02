cmake_minimum_required(VERSION 3.15.6)

# Build bare-metal tests for Arm FVP Corstone-300 with optional gcov coverage
project(helia_aot_fvp_tests C)
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
set(CMAKE_EXECUTABLE_SUFFIX ".elf")

# ---------- Options & paths ----------
# option(ENABLE_COVERAGE "Enable code coverage" OFF)
set(CMSIS_PATH "downloads/CMSIS_5" CACHE PATH "Path to CMSIS (CMSIS_5 checkout)")
set(CMSIS_NN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/modules/ns-cmsis-nn" CACHE PATH "ns-cmsis-nn repo root")
set(FVP_CORSTONE_300_PATH "${CMAKE_CURRENT_SOURCE_DIR}/Corstone-300" CACHE PATH "FVP startup/retarget files")

# Write .gcda to the build tree so each config keeps its own artifacts
# set(COVERAGE_DIR "${CMAKE_BINARY_DIR}/coverage" CACHE PATH "Where .gcda is written (-fprofile-dir)")

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

# # ---------- Optimization & coverage ----------
# if(ENABLE_COVERAGE)
#   message(STATUS "Coverage: ON (O0, --coverage, -fprofile-dir=${COVERAGE_DIR})")
#   file(MAKE_DIRECTORY "${COVERAGE_DIR}")

#   # 1) Nuke any -O* already present globally
#   _strip_all_optimizations()

#   # 2) Coverage-friendly compile + link flags (parent dir)
#   # Note: -fomit-frame-pointer is needed even with -O0 to free registers for inline assembly
#   add_compile_options(-O0 -g -fno-lto -fprofile-arcs -ftest-coverage -fprofile-dir=${COVERAGE_DIR})
#   add_link_options(--coverage)

#   # 3) Ensure we don't mix syscall specs
#   # The toolchain file adds --specs=nosys.specs via add_link_options(), which propagates to all targets
#   # We need to override it by adding rdimon.specs, which should come later and take precedence
#   # However, we also strip from cache variables as a safety measure
#   foreach(v CMAKE_EXE_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS CMAKE_MODULE_LINKER_FLAGS)
#     if(DEFINED ${v} AND "${${v}}" MATCHES "nosys\\.specs")
#       string(REPLACE "--specs=nosys.specs" "" _lf_clean "${${v}}")
#       string(STRIP "${_lf_clean}" _lf_clean)
#       set(${v} "${_lf_clean}" CACHE STRING "" FORCE)
#     endif()
#   endforeach()
#   # Add rdimon.specs - linker processes specs in order, so this should override nosys.specs
#   # But we also strip at target level to be safe
#   add_link_options(--specs=rdimon.specs -Wl,--no-warn-rwx-segments)

# else()
#   message(STATUS "Coverage: OFF (use -DENABLE_COVERAGE=ON to enable)")
#   add_compile_options(-fomit-frame-pointer -Ofast)
#   if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
#     add_link_options(--specs=nosys.specs)
#   endif()
# endif()



# ---------- cmsis-nn subproject ----------
if(EXISTS "${CMSIS_NN_ROOT}/CMakeLists.txt")
  # Use a stable binary subdir name "cmsis-nn" (we need this exact name below)
  add_subdirectory("${CMSIS_NN_ROOT}" cmsis-nn)
else()
  message(FATAL_ERROR "cmsis-nn CMakeLists.txt not found at: ${CMSIS_NN_ROOT}")
endif()

# # If coverage is ON, forcibly append -O0 to the *cmsis-nn* target to fix GCC 14.2.1 ICE
# # The ICE happens in pass_lower_subreg3 when optimizing ARM intrinsics, so we force -O0
# if(ENABLE_COVERAGE)
#   # cmsis-nn target exists after add_subdirectory(...).
#   # Appending here ensures -O0 comes *after* whatever the subproject added (fixes GCC ICE).
#   set_property(TARGET cmsis-nn APPEND PROPERTY COMPILE_OPTIONS -O0 -fno-lto)
# endif()

# ---------- Unity (tiny static lib) ----------
include(FetchContent)
FetchContent_Declare(unity GIT_REPOSITORY https://github.com/ThrowTheSwitch/Unity.git GIT_TAG v2.6.0 GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(unity)
add_library(unity_fetch STATIC ${unity_SOURCE_DIR}/src/unity.c)
 target_include_directories(unity_fetch PUBLIC ${unity_SOURCE_DIR}/src)

# ---------- FVP retarget + startup ----------
add_library(retarget STATIC ${FVP_CORSTONE_300_PATH}/retarget.c ${FVP_CORSTONE_300_PATH}/uart.c)
target_compile_definitions(retarget PUBLIC USING_FVP_CORSTONE_300)
# if(ENABLE_COVERAGE)
  # Semihosting (rdimon) provides syscalls; do NOT compile your own stubs.
# else()
target_compile_definitions(retarget PRIVATE ENABLE_RETARGET_SYSCALLS)
# endif()

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

# ---------- Helpers ----------
function(_normalize_target_name out_var in_path)
  string(REGEX REPLACE "^GeneratedTests/([^/]+).*" "\\1" _tgt "${in_path}")
  set(${out_var} "${_tgt}" PARENT_SCOPE)
endfunction()

# ---------- Discover & build tests ----------
file(GLOB_RECURSE GENERATED_TEST_RUNNERS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "GeneratedTests/*/*/*/test_runner.c"
  "GeneratedTests/*/*/test_runner.c")

if(NOT GENERATED_TEST_RUNNERS)
  message(WARNING "No GeneratedTests/.../test_runner.c found. Did you run the generator?")
endif()

set(GENERATED_TEST_TARGETS)
foreach(TR_REL ${GENERATED_TEST_RUNNERS})
  get_filename_component(TEST_DIR_REL "${TR_REL}" DIRECTORY)
  get_filename_component(TEST_DIR_ABS "${TEST_DIR_REL}" ABSOLUTE)
  _normalize_target_name(TGT_NAME "${TEST_DIR_REL}")

  file(GLOB MODULE_SRCS "${TEST_DIR_ABS}/src/*.c")
  set(SRCS "${TEST_DIR_ABS}/test_runner.c" ${MODULE_SRCS})
  if(SRCS STREQUAL "")
    message(WARNING "No sources in ${TEST_DIR_REL}; skipping")
    continue()
  endif()

  add_executable(${TGT_NAME} ${SRCS})
  set_target_properties(${TGT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")

  # Includes (generated API + local cmsis.h + neuralSPOT bits used by tests)
  file(GLOB NEURALSPOT_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/modules/neuralSPOT/neuralspot/ns-*/includes-api")
  target_include_directories(${TGT_NAME} PRIVATE
    "${TEST_DIR_ABS}/includes-api"
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
  target_link_libraries(${TGT_NAME} PRIVATE unity_fetch cmsis-nn retarget cmsis_startup)

  # if(ENABLE_COVERAGE)
  #   target_compile_definitions(${TGT_NAME} PRIVATE ENABLE_COVERAGE)
  #   target_link_libraries(${TGT_NAME} PRIVATE gcov)
    
  #   # Strip nosys.specs from inherited link options and force rdimon.specs
  #   # The toolchain file adds --specs=nosys.specs globally via add_link_options(), which conflicts with rdimon
  #   # We need to get all link options (including inherited ones) and filter out nosys.specs
  #   get_target_property(_link_opts ${TGT_NAME} LINK_OPTIONS)
  #   get_target_property(_intf_link_opts ${TGT_NAME} INTERFACE_LINK_OPTIONS)
    
  #   # Combine all link options and filter out nosys.specs
  #   set(_all_link_opts)
  #   if(_link_opts)
  #     list(APPEND _all_link_opts ${_link_opts})
  #   endif()
  #   if(_intf_link_opts)
  #     list(APPEND _all_link_opts ${_intf_link_opts})
  #   endif()
    
  #   # Filter out nosys.specs
  #   set(_filtered_opts)
  #   foreach(opt IN LISTS _all_link_opts)
  #     if(NOT "${opt}" MATCHES "nosys\\.specs")
  #       list(APPEND _filtered_opts "${opt}")
  #     endif()
  #   endforeach()
    
  #   # Set the filtered options back
  #   if(_filtered_opts)
  #     set_target_properties(${TGT_NAME} PROPERTIES LINK_OPTIONS "${_filtered_opts}")
  #   else()
  #     set_target_properties(${TGT_NAME} PROPERTIES LINK_OPTIONS "")
  #   endif()
  #   set_target_properties(${TGT_NAME} PROPERTIES INTERFACE_LINK_OPTIONS "")
    
  #   # Force rdimon.specs (must come after removing nosys.specs)
  #   target_link_options(${TGT_NAME} PRIVATE --specs=rdimon.specs)
  # endif()

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
# message(STATUS "  Coverage               : ${ENABLE_COVERAGE}")
message(STATUS "  Tests dir              : ${CMAKE_BINARY_DIR}/tests")
