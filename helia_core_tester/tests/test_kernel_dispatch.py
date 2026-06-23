from helia_core_tester.generation.kernel_dispatch import (
    resolve_convolve_kernel,
    resolve_depthwise_conv_kernel,
    resolve_fully_connected_kernel,
)


def test_convolve_dispatch_uses_cpu_specific_buffer_api():
    m55 = resolve_convolve_kernel("S8", "S8", "cortex-m55")
    m4 = resolve_convolve_kernel("S8", "S8", "cortex-m4")
    m0 = resolve_convolve_kernel("S8", "S8", "cortex-m0")
    s16 = resolve_convolve_kernel("S16", "S8", "cortex-m0")

    assert m55["kernel_get_buffer_size_fn"].endswith("_mve")
    assert m4["kernel_get_buffer_size_fn"].endswith("_dsp")
    assert m0["kernel_get_buffer_size_fn"] == "arm_convolve_wrapper_s8_get_buffer_size"
    assert m55["call_style"] == "m55"
    assert m4["call_style"] == "baseline"
    assert m0["weight_c_type"] == "int8_t"
    assert s16["weight_c_type"] == "int8_t"


def test_depthwise_conv_dispatch_uses_cpu_specific_buffer_api():
    m55 = resolve_depthwise_conv_kernel("S16", "S8", "m55")
    m4 = resolve_depthwise_conv_kernel("S16", "S8", "m4")
    s8 = resolve_depthwise_conv_kernel("S8", "S8", "cortex-m0")

    assert m55["kernel_get_buffer_size_fn"].endswith("_mve")
    assert m4["kernel_get_buffer_size_fn"].endswith("_dsp")
    assert m55["weight_c_type"] == "int8_t"
    assert s8["weight_c_type"] == "int8_t"


def test_fully_connected_dispatch_uses_cpu_specific_buffer_api():
    m55 = resolve_fully_connected_kernel("S8", "S8", "cortex-m55")
    m0 = resolve_fully_connected_kernel("S8", "S8", "cortex-m0")

    assert m55["kernel_get_buffer_size_fn"].endswith("_mve")
    assert m0["kernel_get_buffer_size_fn"] == "arm_fully_connected_s8_get_buffer_size"


def test_fully_connected_float_dispatch_exposes_layout_and_params_type():
    f32 = resolve_fully_connected_kernel("FP32", "FP32", "cortex-m55")
    f16 = resolve_fully_connected_kernel("FP16", "FP16", "cortex-m55")

    assert f32["kernel_fn"] == "arm_fully_connected_f32"
    assert f32["kernel_get_buffer_size_fn"] == "arm_fully_connected_f32_get_buffer_size"
    assert f32["layout"] == "ARM_NN_LAYOUT_NHWC"
    assert f32["fc_params_type"] == "cmsis_nn_fc_params_f32"

    assert f16["kernel_fn"] == "arm_fully_connected_f16"
    assert f16["kernel_get_buffer_size_fn"] == "arm_fully_connected_f16_get_buffer_size"
    assert f16["layout"] == "ARM_NN_LAYOUT_NHWC"
    assert f16["fc_params_type"] == "cmsis_nn_fc_params_f16"
