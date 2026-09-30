"""Requantize, BatchNorm and Softmax render through the generic harness (iteration 1b, G14): a
kernel that returns void is called as a statement and the case reports success, and the s16
softmax binds its LUT pair through a params struct."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from helia_core_tester.contract.bind import ContractBindError
from helia_core_tester.contract.ir import STATUS_PRESENT, ContractSet, FunctionDecl, ParamDecl
from helia_core_tester.generation.harness import ArgumentPool, plan_harness
from helia_core_tester.generation.ops.NNSupportFunctions.batch_norm import batch_norm_argument_pool
from helia_core_tester.generation.ops.NNSupportFunctions.requantize import requantize_argument_pool
from helia_core_tester.generation.ops.SoftmaxFunctions.softmax import softmax_argument_pool
from helia_core_tester.generation.ops.SoftmaxFunctions.softmax_luts import EXP_LUT, ONE_BY_ONE_LUT
from helia_core_tester.tests.harness_render import render_pool

D = {"n": 1, "h": 1, "w": 4, "c": 3}


def _call(source: str, fn: str) -> list[str]:
    body = re.sub(r"/\*.*?\*/", "", re.search(rf"{re.escape(fn)}\((.*?)\);", source, flags=re.S).group(1))
    return [a.strip() for a in body.split(",")]


def _run_body(source: str) -> str:
    return source[source.index("_run("):source.index("_test_case_run(void)")]


def softmax_context(kernel_fn: str = "arm_softmax_s8", ctype: str = "int8_t", out: str = "int8_t", **overrides) -> dict:
    context = {"name": "sm", "kernel_fn": kernel_fn, "input_dims": D, "output_dims": D, "num_rows": 4, "row_size": 3,
               "mult": 1073741824, "shift": 19, "diff_min": -3968, "input_data_array": "    1",
               "expected_output_array": "    2", "input_dtype": ctype, "output_dtype": out,
               "uses_lut": False, "float_kernel": False, "use_batch_harness": False}
    context.update(overrides)
    return context


def _softmax(context: dict) -> tuple[str, str]:
    return render_pool(context, softmax_argument_pool(context), stem="softmax",
                       validation_key="SoftmaxFunctions/softmax/softmax.c.j2", label="Softmax")


@pytest.mark.parametrize("kernel_fn, out", [("arm_softmax_s8", "int8_t"), ("arm_softmax_s8_s16", "int16_t")])
def test_void_softmax_is_called_as_a_statement_and_reports_success(kernel_fn: str, out: str) -> None:
    _, source = _softmax(softmax_context(kernel_fn, out=out))
    run = _run_body(source)
    assert f"return {kernel_fn}(" not in run
    assert re.search(rf"\n    {kernel_fn}\([^;]*\);\n    return ARM_CMSIS_NN_SUCCESS;", run)
    assert _call(run, kernel_fn) == ["input", "4", "3", "1073741824", "19", "-3968", "output"]


def test_s16_softmax_binds_its_lut_pair() -> None:
    context = softmax_context("arm_softmax_s16", "int16_t", "int16_t", uses_lut=True)
    header, source = _softmax(context)
    assert _call(source, "return arm_softmax_s16") == ["input", "4", "3", "1073741824", "19", "&sm_softmax_params",
                                                       "output"]
    assert "static const int16_t sm_exp_lut[513] = {" in header
    assert "static const int16_t sm_one_by_one_lut[513] = {" in header
    assert re.search(r"cmsis_nn_softmax_lut_s16 sm_softmax_params = \{\s*\.exp_lut = sm_exp_lut,\s*"
                     r"\.one_by_one_lut = sm_one_by_one_lut", header)


@pytest.mark.parametrize("table", [EXP_LUT, ONE_BY_ONE_LUT])
def test_luts_hold_513_entries_ending_at_the_table_limits(table: str) -> None:
    entries = [int(v) for v in table.replace(",", " ").split()]
    assert len(entries) == 513 and all(-32768 <= v <= 32767 for v in entries)
    assert entries[-1] in (32767, 16384)


def test_int_softmax_without_luts_declares_none() -> None:
    header, _ = _softmax(softmax_context())
    assert "_lut" not in header and "softmax_params" not in header


def test_float_softmax_takes_only_rows_and_returns_its_status() -> None:
    context = softmax_context("arm_softmax_f32", "float", "float", float_kernel=True)
    pool = softmax_argument_pool(context)
    assert not {"mult", "shift", "diff_min"} & set(pool.values)
    _, source = _softmax(context)
    assert _call(source, "return arm_softmax_f32") == ["input", "4", "3", "output"]
    assert "ARM_CMSIS_NN_SUCCESS;" not in _run_body(source)


def requantize_context(kernel_fn: str = "arm_requantize_s8_s8", ctype: str = "int8_t") -> dict:
    return {"name": "rq", "kernel_fn": kernel_fn, "input_size": 7, "input_shape_array": "    1, 7",
            "input_data_array": "    1", "expected_output_array": "    2", "input_dtype": ctype, "output_dtype": ctype,
            "effective_scale_multiplier": 1518500250, "effective_scale_shift": -3, "input_zeropoint": 4,
            "output_zeropoint": -5, "use_batch_harness": False}


@pytest.mark.parametrize("kernel_fn, ctype", [("arm_requantize_s8_s8", "int8_t"), ("arm_requantize_s16_s16", "int16_t")])
def test_requantize_binds_its_scalars_in_prototype_order(kernel_fn: str, ctype: str) -> None:
    context = requantize_context(kernel_fn, ctype)
    header, source = render_pool(context, requantize_argument_pool(context), stem="requantize",
                                 validation_key="NNSupportFunctions/requantize/requantize.c.j2", label="Requantize")
    assert _call(source, f"return {kernel_fn}") == ["input", "output", "7", "1518500250", "-3", "4", "-5"]
    assert "static const int32_t rq_input_shape[] = {\n    1, 7\n};" in header
    assert "#define RQ_OUTPUT_SIZE 7" in source and "cmsis_nn_context" not in source


def test_requantize_without_a_zeropoint_is_refused_by_name() -> None:
    context = requantize_context()
    pool = requantize_argument_pool(context)
    values = {k: v for k, v in pool.values.items() if k != "output_zeropoint"}
    with pytest.raises(ContractBindError, match=r"output_zeropoint"):
        render_pool(context, ArgumentPool(**{**pool.__dict__, "values": values}), stem="requantize",
                    validation_key="NNSupportFunctions/requantize/requantize.c.j2", label="Requantize")


def test_batch_norm_passes_scale_bias_dims_and_layout() -> None:
    context = {"name": "bn", "kernel_fn": "arm_batch_norm_f32", "input_dims": D, "input_data_array": "    1.0f",
               "scale_array": "    2.0f", "bias_array": "    3.0f", "expected_output_array": "    5.0f",
               "layout": "ARM_NN_LAYOUT_NHWC", "input_dtype": "float", "output_dtype": "float",
               "use_batch_harness": False}
    header, source = render_pool(context, batch_norm_argument_pool(context), stem="batch_norm",
                                 validation_key="NNSupportFunctions/batch_norm/batch_norm.c.j2", label="BatchNorm")
    assert _call(source, "return arm_batch_norm_f32") == ["input", "output", "bn_scale", "bn_bias", "&bn_input_dims",
                                                          "ARM_NN_LAYOUT_NHWC"]
    assert "static const float bn_scale[] = {\n    2.0f\n};" in header and "bn_output_dims" not in header
    assert "#define BN_OUTPUT_SIZE (1 * 1 * 4 * 3)" in source


def _decl(name: str, returns: str) -> FunctionDecl:
    return FunctionDecl(name=name, header="Include/arm_nnfunctions.h", line=1, guards=(), returns=returns,
                        params=(ParamDecl("input", "const int8_t *", "in"), ParamDecl("output", "int8_t *", "out")))


@pytest.mark.parametrize("returns, void", [("void", True), ("arm_cmsis_nn_status", False), ("int32_t", False)])
def test_plan_marks_only_void_kernels(returns: str, void: bool) -> None:
    decl = _decl("arm_fx_s8", returns)
    contracts = ContractSet(status=STATUS_PRESENT, root=Path("."), path=None, functions={decl.name: decl})
    pool = ArgumentPool(name="x", values={}, scratch_buffer=False)
    assert plan_harness(pool, kernel_fn="arm_fx_s8", sizer_fn=None, scratch_bytes=0,
                        contracts=contracts).void_return is void



def test_a_void_kernel_benchmarks_as_a_statement() -> None:
    from dataclasses import replace

    context = softmax_context()
    _, source = render_pool(context, replace(softmax_argument_pool(context), benchmark=True), stem="softmax",
                            validation_key="SoftmaxFunctions/softmax/softmax.c.j2", label="Softmax")
    bench = source[source.index("sm_bench_op(void)"):]
    bench = bench[:bench.index("\n}\n")]
    assert re.search(r"\n    arm_softmax_s8\([^;]*sm_output /\* output \*/\n    \);\n    return ARM_CMSIS_NN_SUCCESS;", bench)


def test_a_void_kernel_refuses_a_fault_a_rule_check_and_an_expected_error() -> None:
    from helia_core_tester.generation.harness import FaultEdit, HarnessError, RuleCheck

    decl = _decl("arm_fx_void", "void")
    contracts = ContractSet(status=STATUS_PRESENT, root=Path("."), path=None, functions={decl.name: decl})
    for field in ({"fault": FaultEdit("x", setup="    x;")}, {"checks": (RuleCheck("arm_fx_void", 1, "r"),)}):
        with pytest.raises(HarnessError, match="returns void, so a fault edit or rule check has no status"):
            plan_harness(ArgumentPool(name="x", values={}, scratch_buffer=False, benchmark=False, **field),
                         kernel_fn="arm_fx_void", sizer_fn=None, scratch_bytes=0, contracts=contracts)
    from helia_core_tester.generation.ops.SoftmaxFunctions.softmax import OpSoftmax

    op = OpSoftmax({"operator": "Softmax", "name": "sm", "activation_dtype": "S8",
                    "tensor_dtypes": {"input": "S8", "output": "S8"}}, seed=0, target_cpu="cortex-m55")
    context = softmax_context(expected_status="ARM_CMSIS_NN_ARG_ERROR")
    with pytest.raises(HarnessError, match="returns void, so expected_status ARM_CMSIS_NN_ARG_ERROR can never"):
        op.render_harness_files(Path(__file__).parent, stem="softmax", context=context,
                                pool=softmax_argument_pool(context),
                                validation_key="SoftmaxFunctions/softmax/softmax.c.j2", label="Softmax")
