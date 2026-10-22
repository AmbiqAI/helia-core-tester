"""DepthwiseConv renders through the generic harness: its kernel, scratch query, weight-sum
provider and planar rule are bound from the depthwise ArgumentPool (iteration 1b, G6)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from helia_core_tester.contract.ir import ContractError, ContractSet, FunctionDecl, ParamDecl, STATUS_PRESENT
from helia_core_tester.contract.render import ContractRenderError, load_current_contracts
from helia_core_tester.generation.harness import HarnessError
from helia_core_tester.generation.kernel_dispatch import DEPTHWISE_CONV_S8_PLANAR_RULE
from helia_core_tester.generation.ops.ConvolutionFunctions.depthwise_conv import depthwise_argument_pool
from helia_core_tester.tests.harness_render import depthwise_context, render_depthwise

WRAPPER_S8 = ("arm_depthwise_conv_wrapper_s8", "arm_depthwise_conv_wrapper_s8_get_buffer_size")
DIMS4 = ["&dw_case_dw_conv_params", "&dw_case_input_dims", "&dw_case_filter_dims", "&dw_case_output_dims"]


@pytest.fixture
def contracts() -> ContractSet:
    loaded = load_current_contracts()
    assert loaded.present, "the conftest fallback always provides the bound operators' contract"
    return loaded


def _strip(text: str) -> str:
    return re.sub(r"/\*.*?\*/|//[^\n]*", " ", text, flags=re.S)


def _kernel_args(rendered: str, kernel_fn: str) -> list[str]:
    call = re.search(rf"return {re.escape(kernel_fn)}\((.*?)\);", _strip(rendered), flags=re.S)
    assert call, f"{kernel_fn} is not called"
    return [a.strip() for a in call.group(1).split(",")]


def _sizer(rendered: str) -> tuple[str, list[str]] | None:
    call = re.search(r"required_buffer_size = (arm_\w+)\((.*?)\);", _strip(rendered), flags=re.S)
    return (call.group(1), [a.strip() for a in call.group(2).split(",") if a.strip()]) if call else None


@pytest.mark.parametrize("kernel_fn, sizer, flags, expected_call, expected_sizer", [
    (*WRAPPER_S8, {"weight_sum": True},
     ["&dw_case_ctx", "&dw_case_weight_sum_ctx", "&dw_case_dw_conv_params", "&dw_case_quant_params",
      "&dw_case_input_dims", "input", "&dw_case_filter_dims", "dw_case_weights", "&dw_case_bias_dims",
      "dw_case_biases", "&dw_case_output_dims", "output"], DIMS4),
    ("arm_depthwise_conv_wrapper_s16", "arm_depthwise_conv_wrapper_s16_get_buffer_size", {"bias_dtype": "int64_t"},
     ["&dw_case_ctx", "&dw_case_dw_conv_params", "&dw_case_quant_params", "&dw_case_input_dims", "input",
      "&dw_case_filter_dims", "dw_case_weights", "&dw_case_bias_dims", "(const int64_t*)dw_case_biases",
      "&dw_case_output_dims", "output"], DIMS4),
    ("arm_depthwise_conv_s8_opt_3x3", "arm_depthwise_conv_s8_opt_get_buffer_size", {"weight_sum": True},
     None, ["&dw_case_input_dims", "&dw_case_filter_dims"]),
    ("arm_depthwise_conv_f32", "arm_depthwise_conv_f32_get_buffer_size", {"float_kernel": True},
     ["&dw_case_ctx", "&dw_case_dw_conv_params", "&dw_case_input_dims", "input", "&dw_case_filter_dims",
      "dw_case_weights", "&dw_case_bias_dims", "dw_case_biases", "&dw_case_output_dims", "output",
      "ARM_NN_LAYOUT_NHWC"], DIMS4 + ["ARM_NN_LAYOUT_NHWC"]),
])
def test_table_kernels_bind_as_they_were_written(contracts, kernel_fn, sizer, flags, expected_call, expected_sizer) -> None:
    _, rendered = render_depthwise(depthwise_context(kernel_fn, sizer, **flags), contracts=contracts)
    if expected_call is not None:
        assert _kernel_args(rendered, kernel_fn) == expected_call
    assert _sizer(rendered) == (sizer, expected_sizer)
    assert rendered.count(f"__typeof__({kernel_fn})") == 1


def test_arm_depthwise_conv_s16_directly_with_no_scratch(contracts) -> None:
    _, rendered = render_depthwise(depthwise_context("arm_depthwise_conv_s16", None, scratch=0, bias_dtype="int64_t"),
                                   contracts=contracts)
    assert _kernel_args(rendered, "arm_depthwise_conv_s16")[8] == "(const int64_t*)dw_case_biases"
    assert _sizer(rendered) is None and "int32_t required_buffer_size = 0;" in rendered
    assert "HELIA_VALIDATE_SIZER" not in rendered


def test_legacy_parameter_names_bind_through_aliases(contracts) -> None:
    _, rendered = render_depthwise(depthwise_context("arm_depthwise_conv_s4", None, scratch=0), contracts=contracts)
    assert "input, /* input */" in rendered and "dw_case_weights, /* kernel */" in rendered
    assert "dw_case_biases, /* bias */" in rendered and "output /* output */" in rendered
    assert _kernel_args(rendered, "arm_depthwise_conv_s4")[0] == "&dw_case_ctx"
    assert "weight_sum_ctx" not in _strip(rendered)


def test_no_bias_passes_null_but_keeps_bias_dims(contracts) -> None:
    header, rendered = render_depthwise(depthwise_context(
        "arm_depthwise_conv_fast_s16", "arm_depthwise_conv_fast_s16_get_buffer_size", has_biases=False),
        contracts=contracts)
    assert _kernel_args(rendered, "arm_depthwise_conv_fast_s16")[7:9] == ["&dw_case_bias_dims", "NULL"]
    assert "static const cmsis_nn_dims dw_case_bias_dims = {\n    .n = 1,\n    .h = 1,\n    .w = 1,\n    .c = 4\n};" in rendered
    assert "static const int32_t* dw_case_biases = NULL;" in header
    assert _sizer(rendered) == ("arm_depthwise_conv_fast_s16_get_buffer_size",
                                ["&dw_case_input_dims", "&dw_case_filter_dims"])


def test_weight_sum_provider_computes_at_runtime_and_falls_back_to_the_header(contracts) -> None:
    header, rendered = render_depthwise(depthwise_context(*WRAPPER_S8, weight_sum=True), contracts=contracts)
    assert "static const int32_t dw_case_weight_sum[] = {\n    0, 0, 0, 0\n};" in header
    assert "static cmsis_nn_context dw_case_weight_sum_ctx;" in rendered
    # The runtime buffer is sized by a literal element count, with no #define of its own.
    assert re.search(r"int32_t body\[4\];\s*uint8_t tail\[HELIA_GUARD_BYTES\];\s*} dw_case_weight_sum_runtime_guard;", rendered)
    assert "WEIGHT_SUM_BUFFER_SIZE" not in rendered
    setup = _strip(rendered)
    assert re.search(r"arm_depthwise_convolve_weight_sum\(\s*dw_case_weight_sum_runtime,\s*NULL,\s*dw_case_weights,", setup)
    assert "dw_case_weight_sum_ctx.buf = (uint8_t *)dw_case_weight_sum;" in setup
    assert "dw_case_weight_sum_ctx.size = 4 * sizeof(int32_t);" in setup
    assert 'HELIA_GUARD_CHECK(dw_case_weight_sum_runtime, "Depthwise Conv weight_sum", failures);' in rendered


def test_weight_sum_provider_is_dropped_for_a_prototype_without_it(contracts) -> None:
    header, rendered = render_depthwise(depthwise_context(
        "arm_depthwise_conv_s8", None, scratch=0, weight_sum=True), contracts=contracts)
    assert "dw_case_weight_sum_runtime" not in rendered and "arm_depthwise_convolve_weight_sum" not in rendered
    assert "dw_case_weight_sum[]" in header  # the precomputed data stays; it is data, not a call


def test_a_wrapper_that_needs_weight_sums_without_them_is_refused(contracts) -> None:
    with pytest.raises(ContractRenderError, match=r"cannot supply \['weight_sum_ctx"):
        render_depthwise(depthwise_context(*WRAPPER_S8, weight_sum=False), contracts=contracts)


@pytest.mark.parametrize("supported, expected", [(True, 1), (False, 0)])
def test_planar_rule_is_bound_from_the_pool_and_checked(contracts, supported: bool, expected: int) -> None:
    _, rendered = render_depthwise(depthwise_context(
        "arm_depthwise_conv_s8_opt_planar", "arm_depthwise_conv_s8_opt_get_buffer_size", weight_sum=True,
        planar_supported=supported, planar_rule_fn=DEPTHWISE_CONV_S8_PLANAR_RULE), contracts=contracts)
    text = _strip(rendered)
    call = re.search(rf"int32_t planar_supported = {DEPTHWISE_CONV_S8_PLANAR_RULE}\((.*?)\);", text, flags=re.S)
    assert call and [a.strip() for a in call.group(1).split(",")] == DIMS4
    assert f"if (planar_supported != {expected}) {{" in rendered
    assert text.index("int failures = 0;") < call.start() < text.index("HELIA_GUARD_CHECK(dw_case_buffer")


def test_no_planar_expectation_renders_no_rule_check(contracts) -> None:
    _, rendered = render_depthwise(depthwise_context(*WRAPPER_S8, weight_sum=True, planar_supported=None,
                                                     planar_rule_fn=DEPTHWISE_CONV_S8_PLANAR_RULE), contracts=contracts)
    assert DEPTHWISE_CONV_S8_PLANAR_RULE not in rendered


def test_force_no_scratch_hands_the_kernel_no_buffer(contracts) -> None:
    _, rendered = render_depthwise(depthwise_context(
        "arm_depthwise_conv_f32", "arm_depthwise_conv_f32_get_buffer_size", float_kernel=True, force_no_scratch=True),
        contracts=contracts)
    assert "dw_case_ctx.buf = NULL;" in rendered and "dw_case_ctx.size = 0;" in rendered
    assert "dw_case_ctx.buf = dw_case_buffer;" not in rendered
    assert "HELIA_GUARD_ARM(dw_case_buffer, true" in rendered  # still armed, so the unused scratch is still checked


def test_depthwise_harness_carries_no_benchmark_path(contracts) -> None:
    _, rendered = render_depthwise(depthwise_context(*WRAPPER_S8, weight_sum=True), contracts=contracts)
    assert "HELIA_BENCHMARK_MODE" not in rendered and "_bench_op" not in rendered


def test_a_prototype_the_pool_cannot_satisfy_is_refused(tmp_path: Path) -> None:
    def param(name: str, c_type: str) -> ParamDecl:
        return ParamDecl(name, c_type, "in")

    kernel = FunctionDecl(name="arm_fx_depthwise_s8", header="Include/arm_nnfunctions.h", line=1, guards=(),
                          returns="arm_cmsis_nn_status",
                          params=(param("ctx", "const cmsis_nn_context *"), param("lut", "const int16_t *")))
    rule = FunctionDecl(name="arm_fx_rule", header="Include/arm_nnfunctions.h", line=1, guards=(), returns="int32_t",
                        params=(param("lut", "const int16_t *"),))
    contracts = ContractSet(status=STATUS_PRESENT, root=tmp_path, path=tmp_path / "x.json",
                            functions={d.name: d for d in (kernel, rule)})
    with pytest.raises(ContractRenderError, match=r"cannot supply \['lut \(const int16_t \*\)'\]"):
        render_depthwise(depthwise_context("arm_fx_depthwise_s8", None, scratch=0), contracts=contracts)
    with pytest.raises(ContractError, match="arm_fx_missing_get_buffer_size: not in the kernel contract"):
        render_depthwise(depthwise_context("arm_fx_depthwise_s8", "arm_fx_missing_get_buffer_size"), contracts=contracts)
    with pytest.raises(ContractError, match="arm_fx_missing_rule: not in the kernel contract"):
        render_depthwise(depthwise_context("arm_depthwise_conv_s4", None, scratch=0, planar_supported=True,
                                           planar_rule_fn="arm_fx_missing_rule"), contracts=load_current_contracts())
    with pytest.raises(ContractRenderError, match=r"arm_fx_rule.*cannot supply \['lut"):
        render_depthwise(depthwise_context("arm_fx_depthwise_s8", None, scratch=0, planar_supported=False,
                                           planar_rule_fn="arm_fx_rule"), contracts=contracts)


def test_pool_rejects_a_rule_result_that_shadows_a_declaration() -> None:
    pool = depthwise_argument_pool(depthwise_context(*WRAPPER_S8, planar_supported=True,
                                                     planar_rule_fn=DEPTHWISE_CONV_S8_PLANAR_RULE, weight_sum=True))
    assert [c.result_var for c in pool.checks] == ["planar_supported"] and pool.benchmark is False
    with pytest.raises(HarnessError, match="rule check variable 'dw_case_weights' is not a free C identifier"):
        type(pool)(**{**pool.__dict__, "checks": (type(pool.checks[0])(DEPTHWISE_CONV_S8_PLANAR_RULE, 1,
                                                                       "dw_case_weights"),)}).validate()
