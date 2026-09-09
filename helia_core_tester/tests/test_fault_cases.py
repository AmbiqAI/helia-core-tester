"""Fault/rejection cases for the non-recurrent operator families (issue #72).

Covers the shared `fault:` / `expected_status:` plumbing in OperationBase and
the per-family `<op>_fault.c.j2` templates: an unknown kind is rejected with a
clear error, `expected_status` defaults to SUCCESS, the GRU/LSTM contexts the
mechanism was lifted from are unchanged, and every rendered fault case asserts
the kernel status without ever validating output.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from helia_core_tester.core.discovery import find_descriptors_dir
from helia_core_tester.generation.io.descriptors import load_all_descriptors
from helia_core_tester.generation.ops.ConvolutionFunctions.convolve import OpConvolve
from helia_core_tester.generation.ops.LSTMFunctions.gru_unidirectional import OpGRUUnidirectional
from helia_core_tester.generation.ops.PoolingFunctions.avg_pool import OpAvgPool
import helia_core_tester.generation.test_ops as generation_module

CPU = "cortex-m55"

# (case name, op suffix, source substring the fault must plant in the C file)
FAULT_CASES = [
    ("convolve_fault_null_ctx_buf_s8", "convolve", ".buf = NULL"),
    ("convolve_fault_null_ctx_buf_1xn_s8", "convolve", ".buf = NULL"),
    ("convolve_fault_null_ctx_buf_s16", "convolve", ".buf = NULL"),
    ("convolve_fault_null_ctx_buf_s4", "convolve", ".buf = NULL"),
    ("convolve_fault_null_weight_sum_ctx_s8", "convolve", ".buf = NULL"),
    ("convolve_fault_zero_stride_s8", "convolve", "stride.w = 0"),
    ("convolve_fault_channel_group_mismatch_s8", "convolve", "input_dims.c = "),
    ("convolve_fault_null_input_f32", "convolve", "*input_arg = NULL"),
    ("convolve_fault_null_output_f16", "convolve", "*output_arg = NULL"),
    ("convolve_fault_invalid_layout_f32", "convolve", "(arm_nn_tensor_layout)(ARM_NN_LAYOUT_NHWC + 1)"),
    ("depthwise_conv_fault_null_ctx_buf_s8", "depthwise_conv", ".buf = NULL"),
    ("depthwise_conv_fault_null_ctx_buf_s4", "depthwise_conv", ".buf = NULL"),
    ("depthwise_conv_fault_null_weight_sum_ctx_s8", "depthwise_conv", ".buf = NULL"),
    ("depthwise_conv_fault_channel_mismatch_s16", "depthwise_conv", "output_dims.c = "),
    ("depthwise_conv_fault_null_input_f32", "depthwise_conv", "*input_arg = NULL"),
    ("depthwise_conv_fault_invalid_layout_f16", "depthwise_conv", "(arm_nn_tensor_layout)(ARM_NN_LAYOUT_NHWC + 1)"),
    ("transpose_conv_fault_null_ctx_buf_s8", "transpose_conv", ".buf = NULL"),
    ("transpose_conv_fault_nonunit_dilation_s8", "transpose_conv", "dilation.w = 2"),
    ("transpose_conv_fault_null_reverse_conv_ctx_buf_s8", "transpose_conv", ".buf = NULL"),
    ("transpose_conv_fault_null_weight_sum_ctx_s8", "transpose_conv", ".buf = NULL"),
    ("transpose_conv_fault_null_output_f32", "transpose_conv", "*output_arg = NULL"),
    ("fully_connected_fault_null_ctx_buf_s8", "fully_connected", ".buf = NULL"),
    ("fully_connected_fault_null_ctx_buf_s16", "fully_connected", ".buf = NULL"),
    ("fully_connected_fault_small_ctx_size_s16", "fully_connected", ".size = 1"),
    ("fully_connected_fault_filter_n_mismatch_f32", "fully_connected", "filter_dims.n = "),
    ("fully_connected_fault_invalid_layout_f16", "fully_connected", "(arm_nn_tensor_layout)(ARM_NN_LAYOUT_NHWC + 1)"),
    ("batch_matmul_fault_null_ctx_buf_s8", "batch_matmul", ".buf = NULL"),
    ("batch_matmul_fault_small_ctx_size_s8", "batch_matmul", ".size = 1"),
    ("batch_matmul_fault_negative_dim_s8", "batch_matmul", "input_rhs_dims.w = -1"),
    ("batch_matmul_fault_null_input_f32", "batch_matmul", "*input_lhs_arg = NULL"),
    ("batch_matmul_fault_packed_rhs_adjoint_f16", "batch_matmul", "ARM_NN_WEIGHT_FORMAT_NT_N_PACKED"),
    ("avg_pool_fault_zero_dim_s8", "avg_pool", "input_dims.n = 0"),
    ("avg_pool_fault_negative_dim_s16", "avg_pool", "input_dims.n = -1"),
    ("avg_pool_fault_null_input_f32", "avg_pool", "*input_arg = NULL"),
    ("max_pool_fault_zero_dim_s16", "max_pool", "input_dims.n = 0"),
    ("max_pool_fault_negative_dim_f16", "max_pool", "input_dims.n = -1"),
    ("max_pool_fault_null_output_f32", "max_pool", "*output_arg = NULL"),
    ("svdf_fault_null_ctx_buf_s8", "svdf", ".buf = NULL"),
    ("svdf_fault_null_input_ctx_buf_s8", "svdf", ".buf = NULL"),
    ("svdf_fault_state_s16_null_output_ctx_buf_s8", "svdf", ".buf = NULL"),
    ("svdf_fault_null_input_f32", "svdf", "*input_arg = NULL"),
    ("svdf_fault_small_output_ctx_size_f16", "svdf", ".size = 1"),
    ("svdf_fault_zero_rank_f32", "svdf", "svdf_params.rank = 0"),
]

GRU_LSTM_CASES = [
    ("gru_unidirectional_error_null_input_f32", "gru_unidirectional", "null_input"),
    ("gru_unidirectional_float_null_buffers_reset_after_f32", "gru_unidirectional", "null_buffers"),
    ("lstm_unidirectional_error_null_params_f16", "lstm_unidirectional", "null_params"),
]


def _descriptors() -> dict[str, dict]:
    return {desc["name"]: desc for desc in load_all_descriptors(str(find_descriptors_dir()))}


@pytest.fixture(scope="module")
def descriptors() -> dict[str, dict]:
    return _descriptors()


@pytest.fixture(scope="module")
def rendered(tmp_path_factory: pytest.TempPathFactory, descriptors: dict[str, dict]) -> dict[str, Path]:
    out_dir = tmp_path_factory.mktemp("fault_cases")
    emitted: dict[str, Path] = {}
    for case_name, _suffix, _marker in FAULT_CASES + GRU_LSTM_CASES:
        desc = descriptors[case_name]
        generation_module.generate_test(desc, str(out_dir), cpu=CPU)
        emitted[case_name] = out_dir / desc["_family"] / case_name
    return emitted


def _test_case_run_body(source: str, case_name: str) -> str:
    match = re.search(
        rf"int32_t {re.escape(case_name)}_test_case_run\(void\)\s*\{{(.*?)\n\}}", source, re.DOTALL
    )
    assert match, f"{case_name}: no test_case_run body"
    return match.group(1)


def test_unknown_fault_kind_is_rejected_with_the_known_list() -> None:
    op = OpAvgPool(
        {
            "operator": "AvgPool",
            "name": "avg_pool_fault_bogus_s8",
            "activation_dtype": "S8",
            "input_shape": [1, 4, 4, 2],
            "pool_size": [2, 2],
            "strides": [2, 2],
            "padding": "VALID",
            "fault": "bogus",
            "expected_status": "ARM_CMSIS_NN_ARG_ERROR",
        },
        seed=1,
        target_cpu=CPU,
    )
    with pytest.raises(ValueError, match=r"Unsupported fault 'bogus' for AvgPool .*known kinds are \['negative_dim'"):
        op.fault_kind()


def test_operator_without_fault_support_names_that_in_the_error() -> None:
    class OpNoFaults(OpAvgPool):
        FAULT_KINDS = ()

    op = OpNoFaults({"operator": "AvgPool", "name": "x_s8", "fault": "zero_dim"}, seed=1, target_cpu=CPU)
    with pytest.raises(ValueError, match="this operator has no fault cases"):
        op.fault_kind()


def test_expected_status_defaults_to_success_and_rejects_unknown_tokens() -> None:
    op = OpConvolve({"operator": "Convolve", "name": "convolve_plain_s8"}, seed=1, target_cpu=CPU)
    assert op.fault_kind() is None
    assert op.expected_status() == "ARM_CMSIS_NN_SUCCESS"
    assert op.fault_context() == {}

    op = OpConvolve(
        {"operator": "Convolve", "name": "convolve_plain_s8", "fault": "null_ctx_buf"}, seed=1, target_cpu=CPU
    )
    assert op.fault_context() == {"fault": "null_ctx_buf", "expected_status": "ARM_CMSIS_NN_SUCCESS"}

    op = OpConvolve(
        {"operator": "Convolve", "name": "convolve_plain_s8", "expected_status": "ARM_CMSIS_NN_MAYBE"},
        seed=1,
        target_cpu=CPU,
    )
    with pytest.raises(ValueError, match="Unsupported expected_status 'ARM_CMSIS_NN_MAYBE'"):
        op.expected_status()


def test_required_capabilities_normalises_scalar_and_list_forms() -> None:
    base = {"operator": "Convolve", "name": "convolve_plain_s8"}
    assert OpConvolve(base, seed=1, target_cpu=CPU).required_capabilities() == ()
    assert OpConvolve({**base, "required_capabilities": "MVE"}, seed=1, target_cpu=CPU).required_capabilities() == ("mve",)
    assert OpConvolve({**base, "required_capabilities": ["dsp", " mve "]}, seed=1, target_cpu=CPU).required_capabilities() == (
        "dsp",
        "mve",
    )


def test_gru_fault_kinds_still_reject_unknown_kinds(descriptors: dict[str, dict]) -> None:
    desc = dict(descriptors["gru_unidirectional_error_null_input_f32"])
    desc["fault"] = "bogus"
    op = OpGRUUnidirectional(desc, seed=1, target_cpu=CPU)
    with pytest.raises(ValueError, match="Unsupported fault 'bogus' for GRUUnidirectional"):
        op.fault_kind()


@pytest.mark.parametrize(("case_name", "op_suffix", "kind"), GRU_LSTM_CASES)
def test_gru_lstm_fault_context_is_unchanged(
    rendered: dict[str, Path], descriptors: dict[str, dict], case_name: str, op_suffix: str, kind: str
) -> None:
    case_dir = rendered[case_name]
    sidecar = json.loads((case_dir / f"{case_name}_{op_suffix}.sidecar.json").read_text())
    expected_status = descriptors[case_name].get("expected_status", "ARM_CMSIS_NN_SUCCESS")
    assert sidecar["scalars"]["fault"] == kind
    assert sidecar["scalars"]["expected_status"] == expected_status
    source = (case_dir / f"{case_name}_{op_suffix}.c").read_text()
    assert "HELIA_VALIDATE_EXPECTED_STATUS(" in source
    assert expected_status in source


def test_fault_descriptors_all_expect_arg_error_and_follow_the_naming_contract(descriptors: dict[str, dict]) -> None:
    families = {
        "ConvolutionFunctions",
        "FullyConnectedFunctions",
        "PoolingFunctions",
        "SVDFunctions",
    }
    seen = 0
    for desc in descriptors.values():
        if desc.get("_family") not in families or "fault" not in desc:
            continue
        seen += 1
        assert desc["expected_status"] == "ARM_CMSIS_NN_ARG_ERROR", desc["name"]
        assert re.fullmatch(r"[a-z0-9_]+_fault_[a-z0-9_]+_(s8|s16|s4|f16|f32)", desc["name"]), desc["name"]
        assert desc["fault"] in desc["name"], desc["name"]
    assert seen >= len(FAULT_CASES)


@pytest.mark.parametrize(("case_name", "op_suffix", "marker"), FAULT_CASES)
def test_rendered_fault_case_asserts_status_and_never_validates_output(
    rendered: dict[str, Path], descriptors: dict[str, dict], case_name: str, op_suffix: str, marker: str
) -> None:
    case_dir = rendered[case_name]
    source = (case_dir / f"{case_name}_{op_suffix}.c").read_text()
    body = _test_case_run_body(source, case_name)

    assert "HELIA_VALIDATE_EXPECTED_STATUS(" in body
    assert "ARM_CMSIS_NN_ARG_ERROR" in body
    assert "HELIA_VALIDATE_RETURN_FAILURES(0)" in body
    assert "HELIA_VALIDATE_OUTPUTS" not in source
    assert "HELIA_VALIDATE_STATUS(" not in source
    assert marker in source, f"{case_name}: fault substitution {marker!r} missing"
    assert f"Fault mode: {descriptors[case_name]['fault']}." in source


@pytest.mark.parametrize(
    ("case_name", "op_suffix", "absent_static"),
    [
        ("convolve_fault_null_output_f32", "convolve", "_output["),
        ("depthwise_conv_fault_null_output_f16", "depthwise_conv", "_output["),
        ("transpose_conv_fault_null_ctx_buf_s8", "transpose_conv", "_buffer["),
        ("batch_matmul_fault_null_ctx_buf_s8", "batch_matmul", "_buffer["),
        ("svdf_fault_null_input_ctx_buf_s8", "svdf", "_scratch_input["),
    ],
)
def test_null_substituted_buffers_are_not_declared(
    rendered: dict[str, Path], descriptors: dict[str, dict], case_name: str, op_suffix: str, absent_static: str
) -> None:
    if case_name not in rendered:
        desc = descriptors[case_name]
        out_dir = rendered[next(iter(rendered))].parents[1]
        generation_module.generate_test(desc, str(out_dir), cpu=CPU)
        rendered[case_name] = out_dir / desc["_family"] / case_name
    source = (rendered[case_name] / f"{case_name}_{op_suffix}.c").read_text()
    assert f"{case_name}{absent_static}" not in source


def test_hardware_bridge_skips_fault_cases_with_a_clear_reason(tmp_path: Path) -> None:
    from helia_core_tester.perf_stream.generated_test_bridge import (
        GeneratedTestCase,
        UnsupportedGeneratedTestError,
        build_case_bundle_from_generated_test,
    )

    case = GeneratedTestCase(
        name="convolve_fault_null_ctx_buf_s8",
        cpu=CPU,
        family="ConvolutionFunctions",
        directory=tmp_path,
        descriptor={"operator": "Convolve", "name": "convolve_fault_null_ctx_buf_s8", "fault": "null_ctx_buf"},
    )
    with pytest.raises(UnsupportedGeneratedTestError, match=r"fault case \(null_ctx_buf\) asserts a kernel status only"):
        build_case_bundle_from_generated_test(tmp_path, case, require_fvp_pass=False)
