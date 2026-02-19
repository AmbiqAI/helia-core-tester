from helia_core_tester.generation.kernel_dispatch import (
    resolve_conv2d_kernel,
    resolve_depthwise_conv2d_kernel,
    resolve_fully_connected_kernel,
)


def test_conv2d_dispatch_uses_cpu_specific_buffer_api():
    m55 = resolve_conv2d_kernel("S8", "S8", "cortex-m55")
    m4 = resolve_conv2d_kernel("S8", "S8", "cortex-m4")
    m0 = resolve_conv2d_kernel("S8", "S8", "cortex-m0")

    assert m55["kernel_get_buffer_size_fn"].endswith("_mve")
    assert m4["kernel_get_buffer_size_fn"].endswith("_dsp")
    assert m0["kernel_get_buffer_size_fn"] == "arm_convolve_wrapper_s8_get_buffer_size"
    assert m55["call_style"] == "m55"
    assert m4["call_style"] == "baseline"


def test_depthwise_dispatch_uses_cpu_specific_buffer_api():
    m55 = resolve_depthwise_conv2d_kernel("S16", "S8", "m55")
    m4 = resolve_depthwise_conv2d_kernel("S16", "S8", "m4")

    assert m55["kernel_get_buffer_size_fn"].endswith("_mve")
    assert m4["kernel_get_buffer_size_fn"].endswith("_dsp")


def test_fully_connected_dispatch_uses_cpu_specific_buffer_api():
    m55 = resolve_fully_connected_kernel("S8", "S8", "cortex-m55")
    m0 = resolve_fully_connected_kernel("S8", "S8", "cortex-m0")

    assert m55["kernel_get_buffer_size_fn"].endswith("_mve")
    assert m0["kernel_get_buffer_size_fn"] == "arm_fully_connected_s8_get_buffer_size"
