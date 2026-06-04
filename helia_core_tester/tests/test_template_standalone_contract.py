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
