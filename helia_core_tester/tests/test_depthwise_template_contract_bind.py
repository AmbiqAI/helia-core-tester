"""depthwise_conv.c.j2 binds its kernel and scratch-query calls from one pool (iteration 1b, G4)."""

from __future__ import annotations

import re
from pathlib import Path

import jinja2
import pytest

from helia_core_tester.contract.ir import ContractError, ContractSet, FunctionDecl, ParamDecl, STATUS_PRESENT
from helia_core_tester.contract.render import ContractRenderError, contract_globals, load_current_contracts
from helia_core_tester.core.discovery import find_tester_templates_dir
from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

TEMPLATE = "ConvolutionFunctions/depthwise_conv/depthwise_conv.c.j2"


def _environment(contracts: ContractSet) -> jinja2.Environment:
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(find_tester_templates_dir())),
                             trim_blocks=True, lstrip_blocks=True)
    env.globals.update(contract_globals(lambda: contracts))
    return env


@pytest.fixture
def contracts() -> ContractSet:
    loaded = load_current_contracts()
    assert loaded.present, "the conftest fallback always provides the bound operators' contract"
    return loaded


def _context(kernel_fn: str, sizer: str | None, *, float_kernel: bool = False, has_biases: bool = True,
             weight_sum: bool = False, needs_layout: bool = False, scratch: int | None = None) -> dict:
    dims = {"n": 1, "h": 2, "w": 2, "c": 4}
    ctype = "float" if float_kernel else "int8_t"
    context = {
        "name": "dw_case", "kernel_fn": kernel_fn, "kernel_get_buffer_size_fn": sizer, "float_kernel": float_kernel,
        "has_biases": has_biases, "takes_weight_sum_ctx": weight_sum, "has_weight_sum": weight_sum,
        "kernel_needs_layout": needs_layout, "buffer_size_needs_layout": needs_layout,
        "kernel_layout": "ARM_NN_LAYOUT_NHWC", "entry_scratch_bytes": scratch, "buffer_size_max": 64,
        "output_dims": dims, "filter_dims": dims, "input_dims": dims, "input_dtype": ctype, "output_dtype": ctype,
        "bias_dtype": "float" if float_kernel else "int32_t", "use_batch_harness": False,
        "validation_mode_token": "FLOAT" if float_kernel else "TOLERANT_INT", "validation_tolerance": 1,
        "validation_atol": 1e-3, "validation_rtol": 1e-3, "validation_report_limit": 20, "supports_benchmark": True,
    }
    return TemplateContextBuilder.build_validation_context(TEMPLATE, context)


def _strip(text: str) -> str:
    return re.sub(r"/\*.*?\*/|//[^\n]*", " ", text, flags=re.S)


def _kernel_args(rendered: str, kernel_fn: str) -> list[str]:
    call = re.search(rf"kernel_status = {re.escape(kernel_fn)}\((.*?)\);", _strip(rendered), flags=re.S)
    assert call, f"{kernel_fn} is not called"
    return [a.strip() for a in call.group(1).split(",")]


def _sizer(rendered: str) -> tuple[str, list[str]] | None:
    call = re.search(r"required_buffer_size = (arm_\w+)\((.*?)\);", _strip(rendered), flags=re.S)
    return (call.group(1), [a.strip() for a in call.group(2).split(",") if a.strip()]) if call else None


DIMS4 = ["&dw_case_dw_conv_params", "&dw_case_input_dims", "&dw_case_filter_dims", "&dw_case_output_dims"]


@pytest.mark.parametrize("kernel_fn, sizer, flags, expected_call, expected_sizer", [
    ("arm_depthwise_conv_wrapper_s8", "arm_depthwise_conv_wrapper_s8_get_buffer_size", {"weight_sum": True},
     ["&dw_case_ctx", "&dw_case_weight_sum_ctx", "&dw_case_dw_conv_params", "&dw_case_quant_params",
      "&dw_case_input_dims", "input", "&dw_case_filter_dims", "dw_case_weights", "&dw_case_bias_dims",
      "dw_case_biases", "&dw_case_output_dims", "output"], DIMS4),
    ("arm_depthwise_conv_wrapper_s16", "arm_depthwise_conv_wrapper_s16_get_buffer_size", {},
     ["&dw_case_ctx", "&dw_case_dw_conv_params", "&dw_case_quant_params", "&dw_case_input_dims", "input",
      "&dw_case_filter_dims", "dw_case_weights", "&dw_case_bias_dims", "(const int64_t*)dw_case_biases",
      "&dw_case_output_dims", "output"], DIMS4),
    ("arm_depthwise_conv_s8_opt_3x3", "arm_depthwise_conv_s8_opt_get_buffer_size", {"weight_sum": True},
     None, ["&dw_case_input_dims", "&dw_case_filter_dims"]),
    ("arm_depthwise_conv_f32", "arm_depthwise_conv_f32_get_buffer_size", {"float_kernel": True, "needs_layout": True},
     ["&dw_case_ctx", "&dw_case_dw_conv_params", "&dw_case_input_dims", "input", "&dw_case_filter_dims",
      "dw_case_weights", "&dw_case_bias_dims", "dw_case_biases", "&dw_case_output_dims", "output",
      "ARM_NN_LAYOUT_NHWC"], DIMS4 + ["ARM_NN_LAYOUT_NHWC"]),
])
def test_table_kernels_bind_as_they_were_written(contracts, kernel_fn, sizer, flags, expected_call, expected_sizer) -> None:
    rendered = _environment(contracts).get_template(TEMPLATE).render(**_context(kernel_fn, sizer, **flags))
    if expected_call is not None:
        assert _kernel_args(rendered, kernel_fn) == expected_call
    assert _sizer(rendered) == (sizer, expected_sizer)
    assert rendered.count(f"__typeof__({kernel_fn})") == 1


def test_arm_depthwise_conv_s16_directly_with_no_scratch(contracts) -> None:
    rendered = _environment(contracts).get_template(TEMPLATE).render(
        **_context("arm_depthwise_conv_s16", None, scratch=0))
    assert _kernel_args(rendered, "arm_depthwise_conv_s16")[8] == "(const int64_t*)dw_case_biases"
    assert _sizer(rendered) is None and "int32_t required_buffer_size = 0;" in rendered
    assert "HELIA_VALIDATE_SIZER" not in rendered


def test_legacy_parameter_names_bind_through_aliases(contracts) -> None:
    rendered = _environment(contracts).get_template(TEMPLATE).render(
        **_context("arm_depthwise_conv_s4", None, scratch=0))
    text = _strip(rendered)
    assert "input, /* input */" in rendered and "dw_case_weights, /* kernel */" in rendered
    assert "dw_case_biases, /* bias */" in rendered and "output /* output */" in rendered
    assert _kernel_args(rendered, "arm_depthwise_conv_s4")[0] == "&dw_case_ctx" and "weight_sum_ctx" not in text


def test_no_bias_passes_null_but_keeps_bias_dims(contracts) -> None:
    rendered = _environment(contracts).get_template(TEMPLATE).render(
        **_context("arm_depthwise_conv_fast_s16", "arm_depthwise_conv_fast_s16_get_buffer_size", has_biases=False))
    args = _kernel_args(rendered, "arm_depthwise_conv_fast_s16")
    assert args[7:9] == ["&dw_case_bias_dims", "NULL"]
    assert _sizer(rendered) == ("arm_depthwise_conv_fast_s16_get_buffer_size",
                                ["&dw_case_input_dims", "&dw_case_filter_dims"])


def test_a_prototype_the_pool_cannot_satisfy_is_refused(tmp_path: Path) -> None:
    def param(name: str, c_type: str) -> ParamDecl:
        return ParamDecl(name, c_type, "in")

    kernel = FunctionDecl(name="arm_fx_depthwise_s8", header="Include/arm_nnfunctions.h", line=1, guards=(),
                          returns="arm_cmsis_nn_status",
                          params=(param("ctx", "const cmsis_nn_context *"), param("lut", "const int16_t *")))
    contracts = ContractSet(status=STATUS_PRESENT, root=tmp_path, path=tmp_path / "x.json", functions={kernel.name: kernel})
    with pytest.raises(ContractRenderError, match=r"cannot supply \['lut \(const int16_t \*\)'\]"):
        _environment(contracts).get_template(TEMPLATE).render(**_context("arm_fx_depthwise_s8", None, scratch=0))
    with pytest.raises(ContractError, match="arm_fx_missing_get_buffer_size: not in the kernel contract"):
        _environment(contracts).get_template(TEMPLATE).render(
            **_context("arm_fx_depthwise_s8", "arm_fx_missing_get_buffer_size"))
