"""Rsqrt, Reshape and Squeeze render through the generic harness (iteration 1b, G15e): Rsqrt binds
its call style's scalars and LUT by name; the copies are one void kernel call over the block."""

from __future__ import annotations

import re

import pytest

from helia_core_tester.generation.ops._shared.copy_pool import copy_argument_pool
from helia_core_tester.generation.ops.BasicMathFunctions.rsqrt import rsqrt_argument_pool
from helia_core_tester.tests.harness_render import render_pool

D = {"n": 1, "h": 1, "w": 5, "c": 1}


def _call(source: str, fn: str) -> list[str]:
    body = re.sub(r"/\*.*?\*/", "", re.search(rf"{re.escape(fn)}\((.*?)\);", source, flags=re.S).group(1))
    return [a.strip() for a in body.split(",")]


def rsqrt_context(style: str = "per_op", **overrides) -> dict:
    universal = style == "universal"
    context = {"name": "rs", "call_style": style, "input_dims": D, "output_dims": D, "input_offset": 3,
               "output_offset": -4, "out_mult": 1073741824, "out_shift": -2, "needs_rescale": 1,
               "out_activation_min": -32768, "out_activation_max": 32767, "block_size": 5, "input_data_array": "    1",
               "expected_output_array": "    2", "input_dtype": "int16_t", "output_dtype": "int16_t",
               "kernel_fn": f"arm_rsqrt_s16_{style}", "lut_dtype": "int32_t" if universal else "int16_t",
               "rsqrt_lut_array": "    7", "expected_status": "ARM_CMSIS_NN_SUCCESS", "use_batch_harness": False}
    context.update(overrides)
    return context


def _rsqrt(context):
    return render_pool(context, rsqrt_argument_pool(context), stem="rsqrt",
                       validation_key="BasicMathFunctions/rsqrt/rsqrt.c.j2", label="Rsqrt")


def test_per_op_rsqrt_binds_offsets_range_size_and_lut() -> None:
    header, source = _rsqrt(rsqrt_context())
    assert _call(source, "return arm_rsqrt_s16_per_op") == ["input", "3", "output", "-4", "-32768", "32767", "5",
                                                            "rs_rsqrt_lut"]
    assert "static const int16_t rs_rsqrt_lut[] = {\n    7\n};" in header
    assert "HELIA_VALIDATE_STATUS(\"Rsqrt\", status);" in source and "#define RS_OUTPUT_SIZE (1 * 1 * 5 * 1)" in source


def test_universal_rsqrt_adds_the_requantisation() -> None:
    header, source = _rsqrt(rsqrt_context("universal"))
    assert _call(source, "return arm_rsqrt_s16_universal") == ["input", "3", "output", "-4", "1073741824", "-2", "1",
                                                               "-32768", "32767", "5", "rs_rsqrt_lut"]
    assert "static const int32_t rs_rsqrt_lut[] = {" in header


def test_per_op_rsqrt_ignores_the_requantisation_scalars() -> None:
    pool = rsqrt_argument_pool(rsqrt_context())
    assert not {"out_mult", "out_shift", "needs_rescale"} & set(pool.values)


def test_unknown_rsqrt_style_is_refused_by_name() -> None:
    with pytest.raises(ValueError, match="call_style 'vector' is not one of"):
        rsqrt_argument_pool(rsqrt_context(call_style="vector"))


def copy_context(kernel_fn: str = "arm_reshape_s8", ctype: str = "int8_t", total: int = 6, **overrides) -> dict:
    context = {"name": "cp", "kernel_fn": kernel_fn, "input_dims": {"n": 1, "h": 2, "w": 3, "c": 1},
               "output_dims": {"n": 1, "h": 1, "w": 6, "c": 1}, "total_size": total, "input_data_array": "    1",
               "expected_output_array": "    1", "input_dtype": ctype, "output_dtype": ctype,
               "use_batch_harness": False}
    context.update(overrides)
    return context


@pytest.mark.parametrize("stem, key, label", [("reshape", "ReshapeFunctions/reshape/reshape.c.j2", "Reshape"),
                                              ("squeeze", "TesterExtensions/squeeze/squeeze.c.j2", "Squeeze")])
def test_copies_call_the_void_kernel_once_and_report_success(stem: str, key: str, label: str) -> None:
    context = copy_context()
    _, source = render_pool(context, copy_argument_pool(context), stem=stem, validation_key=key, label=label)
    run = source[source.index("cp_run("):source.index("cp_test_case_run(void)")]
    assert _call(run, "arm_reshape_s8") == ["input", "output", "6"] and "return arm_reshape_s8(" not in run
    assert "return ARM_CMSIS_NN_SUCCESS;" in run and "#define CP_OUTPUT_SIZE (1 * 1 * 6 * 1)" in source
    assert f'HELIA_VALIDATE_STATUS("{label}", status);' in source


def test_float_copy_binds_the_float_kernel() -> None:
    context = copy_context("arm_reshape_f32", "float")
    _, source = render_pool(context, copy_argument_pool(context), stem="reshape",
                            validation_key="ReshapeFunctions/reshape/reshape.c.j2", label="Reshape")
    assert _call(source, "arm_reshape_f32") == ["input", "output", "6"]


def test_an_empty_copy_is_allowed_but_a_negative_one_is_not() -> None:
    assert copy_argument_pool(copy_context(total=0)).values["total_size"] == "0"
    with pytest.raises(ValueError, match="negative element count"):
        copy_argument_pool(copy_context(total=-1))
