"""SVDF renders through the generic harness (iteration 1b, G16b): its body stays a Jinja fragment
(scratch contexts, in-place state, sizer probes) and the kernel call goes through `_run`, which
takes the contexts, params and state the fragment built as pointer parameters."""

from __future__ import annotations

import copy
import re
from pathlib import Path

import pytest

from helia_core_tester.generation.harness import HarnessError, fragment
from helia_core_tester.generation.ops.SVDFunctions.svdf import OpSVDF, svdf_argument_pool
from helia_core_tester.generation.test_ops import default_seed_for_case
from helia_core_tester.tests.test_svdf_ctx_sizers import _base_descriptor, _repo_root
from helia_core_tester.generation.io.descriptors import load_all_descriptors


@pytest.fixture(scope="module")
def descriptors() -> dict[str, dict]:
    loaded = load_all_descriptors(str(_repo_root() / "assets" / "descriptors"))
    return {desc["name"]: desc for desc in loaded if desc["operator"] == "SVDF"}


def _emit(tmp_path: Path, desc: dict) -> tuple[str, str]:
    out = tmp_path / desc["name"]
    out.mkdir()
    OpSVDF(desc, default_seed_for_case(desc["name"]), target_cpu="cortex-m55").generate_c_files(out)
    return ((out / "includes" / f"{desc['name']}_svdf.h").read_text(), (out / f"{desc['name']}_svdf.c").read_text())


def _run_signature(source: str, name: str) -> list[str]:
    text = re.search(rf"int32_t {name}_run\((.*?)\) \{{", source, flags=re.S).group(1)
    return [" ".join(part.split()) for part in text.split(",")]


def _bound_call(source: str, kernel: str) -> list[str]:
    body = re.sub(r"/\*.*?\*/", "", re.search(rf"return {kernel}\((.*?)\);", source, flags=re.S).group(1))
    return [a.strip() for a in body.split(",")]


def test_s8_run_takes_the_contexts_params_and_state_the_fragment_built(tmp_path, descriptors) -> None:
    desc = _base_descriptor(descriptors, "arm_svdf_s8")
    n = desc["name"]
    header, source = _emit(tmp_path, desc)
    assert _run_signature(source, n) == [
        "const int8_t* __restrict input", "int8_t* __restrict output", "const cmsis_nn_context * ctx",
        "const cmsis_nn_context * input_ctx", "const cmsis_nn_context * output_ctx",
        "const cmsis_nn_svdf_params * svdf_params", "const cmsis_nn_per_tensor_quant_params * input_quant_params",
        "const cmsis_nn_per_tensor_quant_params * output_quant_params", "int8_t * state_data"]
    assert _bound_call(source, "arm_svdf_s8") == [
        "ctx", "input_ctx", "output_ctx", "svdf_params", "input_quant_params", "output_quant_params",
        f"&{n}_input_dims", "input", f"&{n}_state_dims", "state_data", f"&{n}_weights_feature_dims",
        f"{n}_weights_feature", f"&{n}_weights_time_dims", f"{n}_weights_time", f"&{n}_bias_dims", f"{n}_bias",
        f"&{n}_output_dims", "output"]
    # The harness declares no context or output of its own: the fragment owns them.
    assert f"static cmsis_nn_context {n}_ctx" not in source and f"{n.upper()}_OUTPUT_SIZE" not in source
    assert source.index(f"int32_t {n}_run(") < source.index("static int32_t run_svdf(void)")
    assert f"{n}_run({n}_input_data, {n}_output,\n" in source and "&ctx, &input_ctx, &output_ctx, &svdf_params," in source
    assert f"#define {n.upper()}_RANK" in header and f"static const int8_t {n}_output_ref[]" in header


def test_state_s16_kernel_takes_no_kernel_sum_context(tmp_path, descriptors) -> None:
    desc = _base_descriptor(descriptors, "arm_svdf_state_s16_s8")
    _, source = _emit(tmp_path, desc)
    signature = _run_signature(source, desc["name"])
    assert "const cmsis_nn_context * ctx" not in signature and signature[-1] == "int16_t * state_data"
    assert _bound_call(source, "arm_svdf_state_s16_s8")[:3] == ["input_ctx", "output_ctx", "svdf_params"]


@pytest.mark.parametrize("kernel, dtype", [("arm_svdf_f32", "float"), ("arm_svdf_f16", "float16_t")])
def test_float_run_passes_the_header_params_and_steps_through_the_sequence(tmp_path, descriptors, kernel, dtype) -> None:
    desc = _base_descriptor(descriptors, kernel)
    n = desc["name"]
    _, source = _emit(tmp_path, desc)
    assert _run_signature(source, n)[-2:] == [f"const cmsis_nn_svdf_params_{kernel[-3:]} * svdf_params",
                                              f"{dtype} * state_data"]
    assert f"status = {n}_run(step_input, {n}_output, &ctx, &input_ctx, &output_ctx, &{n}_svdf_params," in source
    assert "input_quant_params" not in source
    assert (tmp_path / n / f"{n}_svdf.sidecar.json").exists()


@pytest.mark.parametrize("case, marker", [
    ("svdf_fault_null_ctx_buf_s8", ".buf = NULL,"),
    ("svdf_fault_null_input_ctx_buf_s8", ".buf = NULL,"),
    ("svdf_fault_state_s16_null_input_ctx_buf_s8", ".buf = NULL,"),
])
def test_int_fault_cases_call_through_run_and_expect_the_rejection(tmp_path, descriptors, case, marker) -> None:
    desc = copy.deepcopy(descriptors[case])
    _, source = _emit(tmp_path, desc)
    assert marker in source and f"return {case}_run({case}_input_data, {case}_output," in source
    assert "HELIA_VALIDATE_EXPECTED_STATUS(" in source and f"HELIA_GUARD_CHECK_UNTOUCHED({case}_output" in source
    assert "HELIA_VALIDATE_OUTPUTS(" not in source


def test_an_unknown_variant_is_refused() -> None:
    with pytest.raises(ValueError, match="SVDF variant 'svdf_stream' is not one of"):
        svdf_argument_pool({"name": "x"}, "svdf_stream")


@pytest.mark.parametrize("path, macro", [("../secrets.txt", "header"), ("a/b.j2", "2bad"), ("a b.j2", "header"),
                                         ("a/b.c", "header")])
def test_fragment_refuses_what_is_not_a_template_path_and_macro(path: str, macro: str) -> None:
    with pytest.raises(HarnessError, match="is not a template path and a macro name"):
        fragment(path, macro)


def test_fragment_imports_the_macro_with_the_case_context() -> None:
    assert fragment("SVDFunctions/svdf/svdf.fragment.j2", "header") == (
        '{% from "SVDFunctions/svdf/svdf.fragment.j2" import header with context %}{{ header() }}')
