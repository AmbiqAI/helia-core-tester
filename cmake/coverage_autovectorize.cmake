# Coverage builds: which ns-cmsis-nn sources keep ARM_MATH_AUTOVECTORIZE.
#
# ARM_MATH_AUTOVECTORIZE swaps the MVE inline-asm integer paths for C fallbacks (avoiding
# gcov register-allocation failures at -O0) and suppresses ARM_MATH_MVEF, so it erases every
# MVE path it reaches. Float sources keep their MVE paths under ENABLE_COVERAGE_MVE_FLOAT,
# integer sources under ENABLE_COVERAGE_MVE_INT. arm_nn_mat_mul_core_4x_s8.c keeps the define
# in every mode: its inline-asm operand constraints cannot be met at -O0.
function(helia_coverage_autovectorize_sources out_var)
  set(_always arm_nn_mat_mul_core_4x_s8.c)
  set(_selected "")
  foreach(_s IN LISTS ARGN)
    get_filename_component(_name "${_s}" NAME)
    # A float token anywhere in the file name marks a float source, so float-input
    # integer-output files (arm_quantize_f32_s8.c) and integer-input float-output files
    # (arm_dequantize_s8_f32.c) are both treated as float.
    if(_name MATCHES "_(f16|f32|fp16|flt)(_|\\.c$)")
      if(NOT ENABLE_COVERAGE_MVE_FLOAT)
        list(APPEND _selected "${_s}")
      endif()
    elseif(NOT ENABLE_COVERAGE_MVE_INT OR _name IN_LIST _always)
      list(APPEND _selected "${_s}")
    endif()
  endforeach()
  set(${out_var} "${_selected}" PARENT_SCOPE)
endfunction()
