from __future__ import annotations

from pathlib import Path

import jinja2

from helia_core_tester.generation.ops.BroadcastFunctions.broadcast_to import OpBroadcastTo
from helia_core_tester.generation.ops.DynamicUpdateSliceFunctions.dynamic_update_slice import OpDynamicUpdateSlice
from helia_core_tester.generation.utils.template_context import TemplateContextBuilder


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _templates_root() -> Path:
    return _repo_root() / "assets" / "templates"


def _render(template_name: str, context: dict[str, object]) -> str:
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(_templates_root())),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    render_context = TemplateContextBuilder.build_validation_context(template_name, context)
    return env.get_template(template_name).render(**render_context)


def test_all_c_templates_use_standalone_harness_contract() -> None:
    template_paths = sorted(_templates_root().glob("**/*.c.j2"))
    assert template_paths

    for path in template_paths:
        text = path.read_text()
        assert '#include "unity.h"' not in text, path
        assert "void setUp(void)" not in text, path
        assert "void tearDown(void)" not in text, path
        assert "void test_" not in text, path
        assert "TEST_ASSERT" not in text, path
        assert '{% include "common/standalone/runtime_common.j2" %}' in text, path
        assert '{% include "common/standalone/main.j2" %}' in text, path
        assert "int32_t {{ prefix }}_{{ name }}_test_case_run(void)" in text, path
        assert (
            "HELIA_VALIDATE_STATUS(" in text
            or "HELIA_VALIDATE_EXPECTED_STATUS(" in text
        ), path
        assert "HELIA_VALIDATE_OUTPUTS(" in text, path
        assert "HELIA_VALIDATE_RETURN_FAILURES(" in text, path


def test_all_c_templates_keep_inline_validation_out_of_templates() -> None:
    template_paths = sorted(_templates_root().glob("**/*.c.j2"))
    assert template_paths

    for path in template_paths:
        text = path.read_text()
        assert "Mismatch[" not in text, path
        assert 'printf("%d Failures' not in text, path
        assert "compare_output(" not in text, path

        marker = "int32_t {{ prefix }}_{{ name }}_test_case_run(void)"
        start = text.find(marker)
        assert start >= 0, path
        end = text.find('{% include "common/standalone/main.j2" %}', start)
        assert end > start, path
        test_case_run = text[start:end]
        assert "if (status !=" not in test_case_run, path


def test_rendered_templates_use_shared_validation_helpers() -> None:
    rendered = {
        "relu": _render(
            "ActivationFunctions/relu/relu.c.j2",
            {
                "name": "relu_smoke",
                "prefix": "relu",
                "input_dtype": "int8_t",
                "output_dtype": "int8_t",
                "output_size": 4,
                "input_offset": 0,
                "output_offset": 0,
                "output_mult": 1,
                "output_shift": 0,
                "kernel_fn": "arm_relu_s8",
            },
        ),
        "comparison": _render(
            "ComparisonFunctions/comparison/comparison.c.j2",
            {
                "name": "comparison_smoke",
                "prefix": "comparison",
                "input_dtype": "int8_t",
                "output_size": 4,
                "kernel_fn": "arm_equal_s8",
                "input_1_offset": 0,
                "input_1_mult": 1,
                "input_1_shift": 0,
                "input_2_offset": 0,
                "input_2_mult": 1,
                "input_2_shift": 0,
                "left_shift": 0,
            },
        ),
        "argmax": _render(
            "BasicMathFunctions/argmax/argmax.c.j2",
            {
                "name": "argmax_smoke",
                "prefix": "argmax",
                "input_dtype": "int8_t",
                "output_dtype": "int32_t",
                "output_size": 4,
                "kernel_fn": "arm_argmax_s8",
            },
        ),
        "dequantize": _render(
            "QuantizationFunctions/dequantize/dequantize.c.j2",
            {
                "name": "dequantize_smoke",
                "prefix": "dequantize",
                "input_size": 4,
                "zero_point": 0,
                "scale": 0.125,
                "input_data_array": "    0",
                "expected_output_array": "    0.000000f",
                "input_dtype": "int8_t",
                "output_dtype": "float",
                "kernel_fn": "arm_dequantize_s8_f32",
                "has_activation": False,
                "activation_type": "NONE",
            },
        ),
        "split": _render(
            "ConcatenationFunctions/split/split.c.j2",
            {
                "name": "split_smoke",
                "prefix": "split",
                "input_dtype": "int8_t",
                "output_dtype": "int8_t",
                "kernel_fn": "arm_split_s8",
                "input_dims_count": 4,
                "axis": 3,
                "num_splits": 2,
                "outputs": [
                    {"name": "split_smoke_out0", "size": 4},
                    {"name": "split_smoke_out1", "size": 4},
                ],
            },
        ),
    }

    for name, text in rendered.items():
        assert '#include "unity.h"' not in text, name
        assert "void setUp(void)" not in text, name
        assert "void tearDown(void)" not in text, name
        assert "void test_" not in text, name
        assert "helia_test_platform_init();" in text, name
        assert "helia_test_finish(failures);" in text, name
        assert "int main(void)" in text, name
        assert "_test_case_run(void)" in text, name
        assert "HELIA_VALIDATE_STATUS(" in text, name
        assert "HELIA_VALIDATE_OUTPUTS(" in text, name
        assert "HELIA_VALIDATE_RETURN_FAILURES(" in text, name

    assert "TOLERANT_INT" in rendered["relu"]
    assert "BOOL" in rendered["comparison"]
    assert "EXACT_INT" in rendered["argmax"]
    assert "FLOAT" in rendered["dequantize"]
    assert "split_smoke_out0_output" in rendered["split"]
    assert "split_smoke_out1_output" in rendered["split"]


def test_basic_math_float_templates_render_preformatted_activation_literals() -> None:
    add_text = _render(
        "BasicMathFunctions/add/add.c.j2",
        {
            "name": "add_float_default_f32",
            "prefix": "add_float_default_f32",
            "input_dtype": "float",
            "output_dtype": "float",
            "float_kernel": True,
            "kernel_fn": "arm_elementwise_add_f32",
            "block_size": 128,
            "out_activation_min_literal": "-1.0e+30f",
            "out_activation_max_literal": "1.0e+30f",
            "output_dims": {"n": 1, "h": 4, "w": 4, "c": 8},
        },
    )
    mul_text = _render(
        "BasicMathFunctions/mul/mul.c.j2",
        {
            "name": "mul_float_default_f32",
            "prefix": "mul_float_default_f32",
            "input_dtype": "float",
            "output_dtype": "float",
            "float_kernel": True,
            "kernel_fn": "arm_elementwise_mul_f32",
            "block_size": 128,
            "out_activation_min_literal": "-1.0e+30f",
            "out_activation_max_literal": "1.0e+30f",
            "output_dims": {"n": 1, "h": 4, "w": 4, "c": 8},
        },
    )
    activation_text = _render(
        "ActivationFunctions/nn_activation_float/nn_activation_float.c.j2",
        {
            "name": "nn_activation_float_leaky_relu_f32",
            "prefix": "nn_activation_float_leaky_relu_f32",
            "input_dtype": "float",
            "output_dtype": "float",
            "kernel_fn": "arm_nn_activation_f32",
            "size": 4,
            "activation_symbol": "ARM_NN_FLT_ACT_LEAKY_RELU",
            "act_param_literal": "0.125f",
        },
    )

    for text in (add_text, mul_text):
        assert "1000000000000000019884624838656f" not in text
        assert "-1.0e+30f" in text
        assert "1.0e+30f" in text

    assert "0.125f" in activation_text
    assert "{{ act_param }}f" not in activation_text


def test_pooling_float_header_templates_render_public_float_params() -> None:
    avg_text = _render(
        "PoolingFunctions/avg_pool/avg_pool.h.j2",
        {
            "name": "avg_pool_float_default_f32",
            "prefix": "avg_pool_float_default_f32",
            "input_dtype": "float",
            "output_dtype": "float",
            "pool_params_type": "cmsis_nn_pool_params_f32",
            "float_kernel": True,
            "pool_activation_min_literal": "-1.0e+30f",
            "pool_activation_max_literal": "1.0e+30f",
            "pool_params": {
                "stride_w": 2,
                "stride_h": 2,
                "pad_w": 0,
                "pad_h": 0,
                "activation_min": -1.0e30,
                "activation_max": 1.0e30,
            },
            "input_dims": {"n": 1, "h": 6, "w": 6, "c": 3},
            "filter_dims": {"n": 1, "h": 2, "w": 2, "c": 1},
            "output_dims": {"n": 1, "h": 3, "w": 3, "c": 3},
            "input_data_array": "    0.0f",
            "expected_output_array": "    0.0f",
        },
    )
    max_text = _render(
        "PoolingFunctions/max_pool/max_pool.h.j2",
        {
            "name": "max_pool_float_default_f32",
            "prefix": "max_pool_float_default_f32",
            "input_dtype": "float",
            "output_dtype": "float",
            "pool_params_type": "cmsis_nn_pool_params_f32",
            "float_kernel": True,
            "pool_activation_min_literal": "-1.0e+30f",
            "pool_activation_max_literal": "1.0e+30f",
            "pool_params": {
                "stride_w": 2,
                "stride_h": 2,
                "pad_w": 0,
                "pad_h": 0,
                "activation_min": -1.0e30,
                "activation_max": 1.0e30,
            },
            "input_dims": {"n": 1, "h": 6, "w": 6, "c": 3},
            "filter_dims": {"n": 1, "h": 2, "w": 2, "c": 1},
            "output_dims": {"n": 1, "h": 3, "w": 3, "c": 3},
            "input_data_array": "    0.0f",
            "expected_output_array": "    0.0f",
        },
    )

    assert "cmsis_nn_pool_params_f32" in avg_text
    assert "cmsis_nn_pool_params_f32" in max_text
    for text in (avg_text, max_text):
        assert "-128f" not in text
        assert "127f" not in text
        assert "-1.0e+30f" in text
        assert "1.0e+30f" in text


def test_complex_float_templates_render_public_f32_signatures() -> None:
    conv_h = _render(
        "ConvolutionFunctions/convolve/convolve.h.j2",
        {
            "name": "convolve_float_default_f32",
            "prefix": "convolve_float_default_f32",
            "input_dims": {"n": 1, "h": 6, "w": 6, "c": 3},
            "filter_dims": {"n": 5, "h": 3, "w": 3, "c": 3},
            "output_dims": {"n": 1, "h": 6, "w": 6, "c": 5},
            "conv_params": {"stride_w": 1, "stride_h": 1, "dilation_w": 1, "dilation_h": 1, "pad_w": 1, "pad_h": 1},
            "weights_array": "    0.0f",
            "biases_array": "    0.0f",
            "has_biases": True,
            "input_data_array": "    0.0f",
            "expected_output_array": "    0.0f",
            "input_dtype": "float",
            "output_dtype": "float",
            "weight_dtype": "float",
            "bias_dtype": "float",
            "float_kernel": True,
            "conv_params_type": "cmsis_nn_conv_params_f32",
            "conv_activation_min_literal": "-1.0e+30f",
            "conv_activation_max_literal": "1.0e+30f",
        },
    )
    fc_h = _render(
        "FullyConnectedFunctions/fully_connected/fully_connected.h.j2",
        {
            "name": "fully_connected_float_default_f32",
            "prefix": "fully_connected_float_default_f32",
            "input_dims": {"n": 1, "h": 1, "w": 1, "c": 12},
            "filter_dims": {"n": 12, "h": 1, "w": 1, "c": 5},
            "output_dims": {"n": 1, "h": 1, "w": 1, "c": 5},
            "fc_params": {},
            "weights_array": "    0.0f",
            "biases_array": "    0.0f",
            "has_biases": True,
            "input_data_array": "    0.0f",
            "expected_output_array": "    0.0f",
            "input_dtype": "float",
            "output_dtype": "float",
            "weight_dtype": "float",
            "bias_dtype": "float",
            "kernel_fn": "arm_fully_connected_f32",
            "kernel_get_buffer_size_fn": "arm_fully_connected_f32_get_buffer_size",
            "buffer_size_max": 1024,
            "has_weight_sum": False,
            "weight_sum_array": "",
            "float_kernel": True,
            "fc_params_type": "cmsis_nn_fc_params_f32",
            "fc_activation_min_literal": "-1.0e+30f",
            "fc_activation_max_literal": "1.0e+30f",
        },
    )
    fc_c = _render(
        "FullyConnectedFunctions/fully_connected/fully_connected.c.j2",
        {
            "name": "fully_connected_float_default_f32",
            "prefix": "fully_connected_float_default_f32",
            "input_dims": {"n": 1, "h": 1, "w": 1, "c": 12},
            "filter_dims": {"n": 12, "h": 1, "w": 1, "c": 5},
            "output_dims": {"n": 1, "h": 1, "w": 1, "c": 5},
            "fc_params": {},
            "weights_array": "    0.0f",
            "biases_array": "    0.0f",
            "has_biases": True,
            "input_data_array": "    0.0f",
            "expected_output_array": "    0.0f",
            "input_dtype": "float",
            "output_dtype": "float",
            "weight_dtype": "float",
            "bias_dtype": "float",
            "kernel_fn": "arm_fully_connected_f32",
            "kernel_get_buffer_size_fn": "arm_fully_connected_f32_get_buffer_size",
            "buffer_size_max": 1024,
            "has_weight_sum": False,
            "weight_sum_array": "",
            "float_kernel": True,
            "fc_params_type": "cmsis_nn_fc_params_f32",
            "fc_activation_min_literal": "-1.0e+30f",
            "fc_activation_max_literal": "1.0e+30f",
        },
    )
    bmm_c = _render(
        "FullyConnectedFunctions/batch_matmul/batch_matmul.c.j2",
        {
            "name": "batch_matmul_float_default_f32",
            "prefix": "batch_matmul_float_default_f32",
            "input_lhs_dims": {"n": 1, "h": 1, "w": 4, "c": 3},
            "input_rhs_dims": {"n": 1, "h": 1, "w": 3, "c": 2},
            "output_dims": {"n": 1, "h": 1, "w": 4, "c": 2},
            "bmm_params": {"adj_x": False, "adj_y": False},
            "input_lhs_array": "    0.0f",
            "input_rhs_array": "    0.0f",
            "expected_output_array": "    0.0f",
            "input_dtype": "float",
            "input_rhs_dtype": "float",
            "output_dtype": "float",
            "kernel_fn": "arm_batch_matmul_f32",
            "kernel_get_buffer_size_fn": "arm_batch_matmul_f32_get_buffer_size",
            "buffer_size_max": 1024,
            "float_kernel": True,
            "bmm_params_type": "cmsis_nn_bmm_params_f32",
            "bmm_activation_min_literal": "-1.0e+30f",
            "bmm_activation_max_literal": "1.0e+30f",
        },
    )
    tconv_c = _render(
        "ConvolutionFunctions/transpose_conv/transpose_conv.c.j2",
        {
            "name": "transpose_conv_float_default_f32",
            "prefix": "transpose_conv_float_default_f32",
            "input_dims": {"n": 1, "h": 4, "w": 4, "c": 2},
            "filter_dims": {"n": 3, "h": 3, "w": 3, "c": 2},
            "output_dims": {"n": 1, "h": 8, "w": 8, "c": 3},
            "transpose_conv_params": {"stride_w": 2, "stride_h": 2, "dilation_w": 1, "dilation_h": 1, "pad_w": 1, "pad_h": 1, "pad_offset_w": 1, "pad_offset_h": 1},
            "weights_array": "    0.0f",
            "biases_array": "    0.0f",
            "has_biases": True,
            "has_weight_sum": False,
            "weight_sum_size": 0,
            "input_data_array": "    0.0f",
            "expected_output_array": "    0.0f",
            "input_dtype": "float",
            "output_dtype": "float",
            "weight_dtype": "float",
            "bias_dtype": "float",
            "kernel_fn": "arm_transpose_conv_f32",
            "kernel_get_buffer_size_fn": "arm_transpose_conv_f32_get_buffer_size",
            "buffer_size_max": 1024,
            "reverse_conv_ctx_size": 1024,
            "float_kernel": True,
            "kernel_layout": "ARM_NN_LAYOUT_NHWC",
            "transpose_conv_params_type": "cmsis_nn_transpose_conv_params_f32",
            "transpose_activation_min_literal": "-1.0e+30f",
            "transpose_activation_max_literal": "1.0e+30f",
        },
    )

    assert "cmsis_nn_conv_params_f32" in conv_h
    assert "ARM_NN_WEIGHT_FORMAT_STANDARD" in conv_h
    assert "arm_fully_connected_f32" in fc_c
    assert "cmsis_nn_fc_params_f32" in fc_h
    assert "arm_batch_matmul_f32" in bmm_c
    assert "arm_batch_matmul_f32_get_buffer_size" in bmm_c
    assert "arm_transpose_conv_f32" in tconv_c
    assert "ARM_NN_LAYOUT_NHWC" in tconv_c


def test_s16_conv_templates_render_int8_weights_for_public_wrapper_signatures() -> None:
    conv_h = _render(
        "ConvolutionFunctions/convolve/convolve.h.j2",
        {
            "name": "convolve_int16xint8xint32_case_04_s16",
            "prefix": "convolve_int16xint8xint32_case_04_s16",
            "input_dims": {"n": 1, "h": 32, "w": 32, "c": 2},
            "filter_dims": {"n": 2, "h": 2, "w": 2, "c": 2},
            "output_dims": {"n": 1, "h": 30, "w": 30, "c": 2},
            "conv_params": {"input_offset": 0, "output_offset": 0, "stride_w": 1, "stride_h": 1, "dilation_w": 2, "dilation_h": 2, "pad_w": 0, "pad_h": 0, "activation_min": -32768, "activation_max": 32767},
            "quant_params": {"per_channel": False, "multiplier": 1, "shift": 0},
            "weights_array": "    1",
            "biases_array": "    0",
            "has_biases": True,
            "input_data_array": "    0",
            "expected_output_array": "    0",
            "input_dtype": "int16_t",
            "output_dtype": "int16_t",
            "weight_dtype": "int8_t",
            "bias_dtype": "int64_t",
            "kernel_fn": "arm_convolve_wrapper_s16",
            "kernel_get_buffer_size_fn": "arm_convolve_wrapper_s16_get_buffer_size",
            "buffer_size_max": 1024,
        },
    )
    conv_c = _render(
        "ConvolutionFunctions/convolve/convolve.c.j2",
        {
            "name": "convolve_int16xint8xint32_case_04_s16",
            "prefix": "convolve_int16xint8xint32_case_04_s16",
            "input_dims": {"n": 1, "h": 32, "w": 32, "c": 2},
            "filter_dims": {"n": 2, "h": 2, "w": 2, "c": 2},
            "output_dims": {"n": 1, "h": 30, "w": 30, "c": 2},
            "conv_params": {"input_offset": 0, "output_offset": 0, "stride_w": 1, "stride_h": 1, "dilation_w": 2, "dilation_h": 2, "pad_w": 0, "pad_h": 0, "activation_min": -32768, "activation_max": 32767},
            "quant_params": {"per_channel": False, "multiplier": 1, "shift": 0},
            "weights_array": "    1",
            "biases_array": "    0",
            "has_biases": True,
            "input_data_array": "    0",
            "expected_output_array": "    0",
            "input_dtype": "int16_t",
            "output_dtype": "int16_t",
            "weight_dtype": "int8_t",
            "bias_dtype": "int64_t",
            "kernel_fn": "arm_convolve_wrapper_s16",
            "kernel_get_buffer_size_fn": "arm_convolve_wrapper_s16_get_buffer_size",
            "buffer_size_max": 1024,
        },
    )
    dw_h = _render(
        "ConvolutionFunctions/depthwise_conv/depthwise_conv.h.j2",
        {
            "name": "depthwise_conv_s16",
            "prefix": "depthwise_conv_s16",
            "input_dims": {"n": 1, "h": 8, "w": 8, "c": 4},
            "filter_dims": {"n": 1, "h": 3, "w": 3, "c": 4},
            "output_dims": {"n": 1, "h": 6, "w": 6, "c": 4},
            "dw_conv_params": {"input_offset": 0, "output_offset": 0, "ch_mult": 1, "stride_w": 1, "stride_h": 1, "dilation_w": 1, "dilation_h": 1, "pad_w": 0, "pad_h": 0, "activation_min": -32768, "activation_max": 32767},
            "quant_params": {"per_channel": False, "multiplier": 1, "shift": 0},
            "weights_array": "    1",
            "biases_array": "    0",
            "has_biases": True,
            "weight_sum_array": "",
            "has_weight_sum": False,
            "input_data_array": "    0",
            "expected_output_array": "    0",
            "input_dtype": "int16_t",
            "output_dtype": "int16_t",
            "weight_dtype": "int8_t",
            "bias_dtype": "int64_t",
            "kernel_fn": "arm_depthwise_conv_wrapper_s16",
            "kernel_get_buffer_size_fn": "arm_depthwise_conv_wrapper_s16_get_buffer_size",
            "buffer_size_max": 1024,
        },
    )
    dw_c = _render(
        "ConvolutionFunctions/depthwise_conv/depthwise_conv.c.j2",
        {
            "name": "depthwise_conv_s16",
            "prefix": "depthwise_conv_s16",
            "input_dims": {"n": 1, "h": 8, "w": 8, "c": 4},
            "filter_dims": {"n": 1, "h": 3, "w": 3, "c": 4},
            "output_dims": {"n": 1, "h": 6, "w": 6, "c": 4},
            "dw_conv_params": {"input_offset": 0, "output_offset": 0, "ch_mult": 1, "stride_w": 1, "stride_h": 1, "dilation_w": 1, "dilation_h": 1, "pad_w": 0, "pad_h": 0, "activation_min": -32768, "activation_max": 32767},
            "quant_params": {"per_channel": False, "multiplier": 1, "shift": 0},
            "weights_array": "    1",
            "biases_array": "    0",
            "has_biases": True,
            "weight_sum_array": "",
            "has_weight_sum": False,
            "input_data_array": "    0",
            "expected_output_array": "    0",
            "input_dtype": "int16_t",
            "output_dtype": "int16_t",
            "weight_dtype": "int8_t",
            "bias_dtype": "int64_t",
            "kernel_fn": "arm_depthwise_conv_wrapper_s16",
            "kernel_get_buffer_size_fn": "arm_depthwise_conv_wrapper_s16_get_buffer_size",
            "buffer_size_max": 1024,
        },
    )

    assert "static const int8_t convolve_int16xint8xint32_case_04_s16_weights[]" in conv_h
    assert "static const int64_t convolve_int16xint8xint32_case_04_s16_biases[]" in conv_h
    assert "static const int16_t convolve_int16xint8xint32_case_04_s16_weights[]" not in conv_h
    assert "arm_convolve_wrapper_s16" in conv_c

    assert "static const int8_t depthwise_conv_s16_weights[]" in dw_h
    assert "static const int64_t depthwise_conv_s16_biases[]" in dw_h
    assert "static const int16_t depthwise_conv_s16_weights[]" not in dw_h
    assert "arm_depthwise_conv_wrapper_s16" in dw_c


def test_gather_nd_invalid_status_render_uses_expected_status_helper() -> None:
    text = _render(
        "GatherFunctions/gather_nd/gather_nd.c.j2",
        {
            "name": "gather_nd_invalid_smoke",
            "prefix": "gather_nd",
            "input_dtype": "int8_t",
            "output_dtype": "int8_t",
            "kernel_fn": "arm_gather_nd_s8",
            "params_rank_test": 3,
            "indices_rank_test": 2,
            "batch_dims_test": 0,
            "output_size": 4,
            "expected_status": "ARM_CMSIS_NN_ARG_ERROR",
        },
    )

    assert "HELIA_VALIDATE_EXPECTED_STATUS(" in text
    assert "ARM_CMSIS_NN_ARG_ERROR" in text
    assert "{{ name }}_expected_output" not in text


def test_transpose_invalid_status_render_uses_expected_status_helper() -> None:
    text = _render(
        "TransposeFunctions/transpose/transpose.c.j2",
        {
            "name": "transpose_invalid_smoke",
            "prefix": "transpose",
            "input_dtype": "int8_t",
            "output_dtype": "int8_t",
            "kernel_fn": "arm_transpose_s8",
            "expected_status": "ARM_CMSIS_NN_ARG_ERROR",
            "input_dims": {"n": 2, "h": 4, "w": 3, "c": 1},
            "output_dims": {"n": 2, "h": 4, "w": 3, "c": 1},
            "num_dims": 3,
            "permutation_array": "    0, 1, 3",
        },
    )

    assert "HELIA_VALIDATE_EXPECTED_STATUS(" in text
    assert "ARM_CMSIS_NN_ARG_ERROR" in text
    assert "{{ name }}_expected_output" not in text


def test_transpose_header_float_uses_inline_perm_array() -> None:
    text = _render(
        "TransposeFunctions/transpose/transpose.h.j2",
        {
            "name": "transpose_float_smoke",
            "prefix": "transpose",
            "input_dims": {"n": 1, "h": 2, "w": 3, "c": 4},
            "output_dims": {"n": 1, "h": 3, "w": 2, "c": 4},
            "num_dims": 4,
            "permutation_array": "    0, 2, 1, 3",
            "input_data_array": "    0.0f",
            "expected_output_array": "    0.0f",
            "input_dtype": "float",
            "output_dtype": "float",
            "transpose_params_type": "cmsis_nn_transpose_params_f32",
            "float_kernel": True,
        },
    )

    assert "static const int32_t transpose_float_smoke_perm[]" in text
    assert ".perm = {" in text
    assert ".permutations =" not in text


def test_transpose_header_int_uses_permutations_pointer() -> None:
    text = _render(
        "TransposeFunctions/transpose/transpose.h.j2",
        {
            "name": "transpose_int_smoke",
            "prefix": "transpose",
            "input_dims": {"n": 1, "h": 2, "w": 3, "c": 4},
            "output_dims": {"n": 1, "h": 3, "w": 2, "c": 4},
            "num_dims": 4,
            "permutation_array": "    0, 2, 1, 3",
            "input_data_array": "    0",
            "expected_output_array": "    0",
            "input_dtype": "int8_t",
            "output_dtype": "int8_t",
            "transpose_params_type": "cmsis_nn_transpose_params",
            "float_kernel": False,
        },
    )

    assert "static const uint32_t transpose_int_smoke_permutations[]" in text
    assert ".permutations = transpose_int_smoke_permutations" in text
    assert ".perm = {" not in text


def test_rsqrt_invalid_status_render_uses_expected_status_helper() -> None:
    text = _render(
        "BasicMathFunctions/rsqrt/rsqrt.c.j2",
        {
            "name": "rsqrt_invalid_smoke",
            "prefix": "rsqrt",
            "call_style": "per_op",
            "input_dtype": "int16_t",
            "output_dtype": "int16_t",
            "kernel_fn": "arm_rsqrt_s16_per_op",
            "expected_status": "ARM_CMSIS_NN_ARG_ERROR",
            "input_dims": {"n": 1, "h": 1, "w": 4, "c": 1},
            "output_dims": {"n": 1, "h": 1, "w": 4, "c": 1},
            "input_offset": 0,
            "output_offset": 0,
            "out_activation_min": -32768,
            "out_activation_max": 32767,
            "block_size": 4,
            "rsqrt_lut_array": "    32767",
            "lut_dtype": "int16_t",
        },
    )

    assert "HELIA_VALIDATE_EXPECTED_STATUS(" in text
    assert "ARM_CMSIS_NN_ARG_ERROR" in text
    assert "{{ name }}_expected_output" not in text


def test_broadcast_to_invalid_status_render_uses_expected_status_helper() -> None:
    text = _render(
        "BroadcastFunctions/broadcast_to/broadcast_to.c.j2",
        {
            "name": "broadcast_invalid_smoke",
            "prefix": "broadcast",
            "c_type": "int8_t",
            "kernel_fn": "arm_broadcast_to_s8",
            "output_size": 4,
            "expected_status": "ARM_CMSIS_NN_ARG_ERROR",
            "input_arg": "NULL",
            "params_arg": "&broadcast_invalid_smoke_params",
            "output_arg": "broadcast_invalid_smoke_output",
        },
    )

    assert "HELIA_VALIDATE_EXPECTED_STATUS(" in text
    assert "ARM_CMSIS_NN_ARG_ERROR" in text
    assert "NULL," in text
    assert "broadcast_invalid_smoke_expected_output" not in text


def test_dynamic_update_slice_invalid_status_render_uses_expected_status_helper() -> None:
    text = _render(
        "DynamicUpdateSliceFunctions/dynamic_update_slice/dynamic_update_slice.c.j2",
        {
            "name": "dynamic_update_slice_invalid_smoke",
            "prefix": "dynamic_update_slice",
            "c_type": "int8_t",
            "kernel_fn": "arm_dynamic_update_slice_s8",
            "operand_size": 20,
            "expected_status": "ARM_CMSIS_NN_ARG_ERROR",
            "operand_arg": "dynamic_update_slice_invalid_smoke_operand",
            "update_arg": "dynamic_update_slice_invalid_smoke_update",
            "start_indices_arg": "NULL",
            "params_arg": "&dynamic_update_slice_invalid_smoke_params",
            "output_arg": "dynamic_update_slice_invalid_smoke_output",
        },
    )

    assert "HELIA_VALIDATE_EXPECTED_STATUS(" in text
    assert "ARM_CMSIS_NN_ARG_ERROR" in text
    assert "NULL," in text
    assert "dynamic_update_slice_invalid_smoke_expected_output" not in text


def test_broadcast_to_expected_error_generation_renders_rank_override(tmp_path: Path) -> None:
    desc = {
        "operator": "BroadcastTo",
        "name": "broadcast_to_rank9_smoke",
        "activation_dtype": "S8",
        "weight_dtype": "S8",
        "activation": "NONE",
        "expected_status": "ARM_CMSIS_NN_ARG_ERROR",
        "hint": {"call_style": "per_tensor", "extras": {"params_rank": 9}},
        "input_shape": [1],
        "output_shape": [1],
    }
    op = OpBroadcastTo(desc, seed=1)

    assert op.allow_no_tflite()
    op.generate_c_files(tmp_path)

    header = (tmp_path / "includes" / "broadcast_to_rank9_smoke_broadcast_to.h").read_text()
    source = (tmp_path / "broadcast_to_rank9_smoke_broadcast_to.c").read_text()
    assert ".rank = 9" in header
    assert "ARM_CMSIS_NN_ARG_ERROR" in source
    assert "HELIA_VALIDATE_RETURN_FAILURES(0)" in source


def test_dynamic_update_slice_expected_error_generation_renders_rank_override(tmp_path: Path) -> None:
    desc = {
        "operator": "DynamicUpdateSlice",
        "name": "dynamic_update_slice_rank0_smoke",
        "activation_dtype": "S8",
        "weight_dtype": "S8",
        "activation": "NONE",
        "expected_status": "ARM_CMSIS_NN_ARG_ERROR",
        "hint": {"call_style": "per_tensor", "extras": {"params_rank": 0}},
        "operand_shape": [4, 5],
        "update_shape": [2, 3],
        "start_indices": [1, 1],
    }
    op = OpDynamicUpdateSlice(desc, seed=1)

    assert op.allow_no_tflite()
    op.generate_c_files(tmp_path)

    header = (tmp_path / "includes" / "dynamic_update_slice_rank0_smoke_dynamic_update_slice.h").read_text()
    source = (tmp_path / "dynamic_update_slice_rank0_smoke_dynamic_update_slice.c").read_text()
    assert ".rank = 0" in header
    assert "ARM_CMSIS_NN_ARG_ERROR" in source
    assert "HELIA_VALIDATE_RETURN_FAILURES(0)" in source


def test_lstm_and_svdf_keep_specialized_shared_validation_contracts() -> None:
    lstm = (
        _templates_root()
        / "LSTMFunctions"
        / "lstm_unidirectional"
        / "lstm_unidirectional.c.j2"
    ).read_text()
    svdf = (_templates_root() / "SVDFunctions" / "svdf" / "svdf.c.j2").read_text()

    assert '{{ validation_report_limit | default(8) }}' in lstm
    assert "HELIA_VALIDATE_OUTPUTS(" in lstm

    assert '{{ validation_report_limit | default(8) }}' in svdf
    assert "HELIA_VALIDATE_SCALAR_EQ_INT(" in svdf
    assert "HELIA_VALIDATE_OUTPUTS(" in svdf


def test_quantize_and_dequantize_render_only_requested_validation_helpers() -> None:
    quantize = _render(
        "QuantizationFunctions/quantize/quantize.c.j2",
        {
            "name": "quantize_smoke",
            "prefix": "quantize",
            "input_size": 4,
            "zero_point": 0,
            "scale": 0.125,
            "input_data_array": "    0.000000f",
            "expected_output_array": "    0",
            "input_dtype": "float",
            "output_dtype": "int8_t",
            "kernel_fn": "arm_quantize_f32_s8",
            "has_activation": False,
            "activation_kernel_fn": None,
            "activation_type": "NONE",
            "comparison_tolerance": 1,
            "validation_helpers": ["tolerant_int"],
        },
    )
    dequantize = _render(
        "QuantizationFunctions/dequantize/dequantize.c.j2",
        {
            "name": "dequantize_smoke",
            "prefix": "dequantize",
            "input_size": 4,
            "zero_point": 0,
            "scale": 0.125,
            "input_data_array": "    0",
            "expected_output_array": "    0.000000f",
            "input_dtype": "int8_t",
            "output_dtype": "float",
            "kernel_fn": "arm_dequantize_s8_f32",
            "has_activation": False,
            "activation_type": "NONE",
            "comparison_atol": 0.01,
            "comparison_rtol": 0.001,
            "validation_helpers": ["float"],
        },
    )

    assert "#define HELIA_VALIDATE_TOLERANT_INTS" in quantize
    assert "#define HELIA_VALIDATE_FLOATS" not in quantize
    assert "#define HELIA_VALIDATE_EXACT_INTS" not in quantize
    assert "#define HELIA_VALIDATE_BOOLEANS" not in quantize
    assert "TOLERANT_INT" in quantize

    assert "#define HELIA_VALIDATE_FLOATS" in dequantize
    assert "#define HELIA_VALIDATE_TOLERANT_INTS" not in dequantize
    assert "#define HELIA_VALIDATE_EXACT_INTS" not in dequantize
    assert "#define HELIA_VALIDATE_BOOLEANS" not in dequantize
    assert "FLOAT" in dequantize


def test_top_level_cmake_no_longer_uses_unity() -> None:
    text = (_repo_root() / "CMakeLists.txt").read_text()
    assert "unity_fetch" not in text
    assert "ThrowTheSwitch/Unity" not in text
    assert "FetchContent_Declare(unity" not in text
    assert "target_link_libraries(${TGT_NAME} PRIVATE cmsis-nn retarget cmsis_startup)" in text
