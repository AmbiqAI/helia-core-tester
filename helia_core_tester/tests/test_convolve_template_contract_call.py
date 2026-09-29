"""convolve.c.j2 renders its kernel call from the contract export (H6 pilot)."""

from __future__ import annotations

import os
import re
from pathlib import Path

import jinja2
import pytest

from helia_core_tester.contract.ir import ContractError, ContractSet, FunctionDecl, ParamDecl, STATUS_PRESENT, load_contract_set
from helia_core_tester.contract.render import ContractRenderError, contract_globals
from helia_core_tester.core.discovery import find_tester_templates_dir
from helia_core_tester.generation.utils.temp_sizer_probe import resolve_cmsis_nn_root
from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

TEMPLATE = "ConvolutionFunctions/convolve/convolve.c.j2"


def _environment(contracts: ContractSet) -> jinja2.Environment:
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(find_tester_templates_dir())),
                             trim_blocks=True, lstrip_blocks=True)
    env.globals.update(contract_globals(lambda: contracts))
    return env


def _context(kernel_fn: str, *, float_kernel: bool, has_biases: bool = True, needs_layout: bool = False,
             entry_family: str | None = None) -> dict:
    dims = {"n": 1, "h": 2, "w": 2, "c": 4}
    # The flags convolve.py derives: weight sums for the s8 wrapper and the direct
    # arm_convolve_s8 family, which also takes an upscale_dims argument.
    context = {
        "name": "conv_case", "kernel_fn": kernel_fn,
        "kernel_get_buffer_size_fn": "arm_convolve_s8_get_buffer_size" if entry_family == "convolve_s8" else kernel_fn + "_get_buffer_size",
        "entry_family": entry_family,
        "conv_s8_weight_sum": kernel_fn == "arm_convolve_wrapper_s8" or entry_family == "convolve_s8",
        "bias_is_struct": kernel_fn == "arm_convolve_wrapper_s16", "entry_scratch_bytes": None,
        "float_kernel": float_kernel, "has_biases": has_biases, "kernel_needs_layout": needs_layout,
        "buffer_size_needs_layout": needs_layout, "kernel_layout": "ARM_NN_LAYOUT_NHWC",
        "buffer_size_max": 64, "output_dims": dims, "filter_dims": dims, "input_dims": dims,
        "input_dtype": "float" if float_kernel else "int8_t", "output_dtype": "float" if float_kernel else "int8_t",
        "bias_dtype": "float" if float_kernel else "int32_t", "use_batch_harness": False,
        "validation_mode_token": "FLOAT" if float_kernel else "TOLERANT_INT", "validation_tolerance": 1,
        "validation_atol": 1e-3, "validation_rtol": 1e-3, "validation_report_limit": 20, "supports_benchmark": True,
    }
    return TemplateContextBuilder.build_validation_context(TEMPLATE, context)


def _call_args(rendered: str, kernel_fn: str) -> list[list[str]]:
    """Every argument list of `kernel_fn(...)` in the render, comments stripped."""
    text = re.sub(r"/\*.*?\*/|//[^\n]*", " ", rendered, flags=re.S)
    calls = re.findall(rf"return {re.escape(kernel_fn)}\((.*?)\);", text, flags=re.S)
    return [[arg.strip() for arg in call.split(",")] for call in calls]


@pytest.fixture
def contracts() -> ContractSet:
    loaded = load_contract_set(resolve_cmsis_nn_root())
    if not loaded.present:
        if os.environ.get("HELIA_CORE_TESTER_REQUIRE_CONTRACT"):
            pytest.fail("HELIA_CORE_TESTER_REQUIRE_CONTRACT is set but no kernel contract was found")
        pytest.skip("no ns-cmsis-nn checkout with a kernel contract")
    return loaded


@pytest.mark.parametrize(
    "kernel_fn, float_kernel, needs_layout, expected",
    [
        ("arm_convolve_wrapper_s8", False, False,
         ["&conv_case_ctx", "&conv_case_weight_sum_ctx", "&conv_case_conv_params", "&conv_case_quant_params",
          "&conv_case_input_dims", "input", "&conv_case_filter_dims", "conv_case_weights", "&conv_case_bias_dims",
          "conv_case_biases", "&conv_case_output_dims", "output"]),
        ("arm_convolve_wrapper_s4", False, False,
         ["&conv_case_ctx", "&conv_case_conv_params", "&conv_case_quant_params", "&conv_case_input_dims", "input",
          "&conv_case_filter_dims", "conv_case_weights", "&conv_case_bias_dims", "conv_case_biases",
          "&conv_case_output_dims", "output"]),
        ("arm_convolve_wrapper_s16", False, False,
         ["&conv_case_ctx", "&conv_case_conv_params", "&conv_case_quant_params", "&conv_case_input_dims", "input",
          "&conv_case_filter_dims", "conv_case_weights", "&conv_case_bias_dims", "&conv_case_bias_data",
          "&conv_case_output_dims", "output"]),
        ("arm_convolve_wrapper_f32", True, False,
         ["&conv_case_ctx", "&conv_case_conv_params", "&conv_case_input_dims", "input", "&conv_case_filter_dims",
          "conv_case_weights", "&conv_case_bias_dims", "conv_case_biases", "&conv_case_output_dims", "output"]),
        ("arm_convolve_1x1_f16", True, True,
         ["&conv_case_ctx", "&conv_case_conv_params", "&conv_case_input_dims", "input", "&conv_case_filter_dims",
          "conv_case_weights", "&conv_case_bias_dims", "conv_case_biases", "&conv_case_output_dims", "output",
          "ARM_NN_LAYOUT_NHWC"]),
    ],
)
def test_call_follows_the_prototype(contracts: ContractSet, kernel_fn: str, float_kernel: bool,
                                    needs_layout: bool, expected: list[str]) -> None:
    rendered = _environment(contracts).get_template(TEMPLATE).render(
        **_context(kernel_fn, float_kernel=float_kernel, needs_layout=needs_layout))
    calls = _call_args(rendered, kernel_fn)
    assert len(calls) == 2, "the correctness path and the benchmark path both call the kernel"
    assert calls[0] == expected
    assert calls[1] == [a.replace("input", "conv_case_input").replace("output", "conv_case_output") if a in ("input", "output") else a for a in expected]
    assert [p.name for p in contracts.require(kernel_fn).params] == \
        [n for n in ["ctx", "weight_sum_ctx", "conv_params", "quant_params", "input_dims", "input_data", "filter_dims",
                     "filter_data", "bias_dims", "bias_data", "output_dims", "output_data", "layout"]
         if n in {p.name for p in contracts.require(kernel_fn).params}]
    assert rendered.count(f"__typeof__({kernel_fn})") == 1, "one file-scope parity assert"


@pytest.mark.parametrize("kernel_fn", ["arm_convolve_s8", "arm_convolve_s8_small_cin", "arm_convolve_s8_3x3_c16_s1"])
def test_direct_s8_entries_get_weight_sums_and_a_null_upscale(contracts: ContractSet, kernel_fn: str) -> None:
    if contracts.find(kernel_fn) is None:
        pytest.skip(f"{kernel_fn} is not in this checkout's contract")
    rendered = _environment(contracts).get_template(TEMPLATE).render(
        **_context(kernel_fn, float_kernel=False, entry_family="convolve_s8"))
    calls = _call_args(rendered, kernel_fn)
    assert len(calls) == 2
    assert calls[0] == ["&conv_case_ctx", "&conv_case_weight_sum_ctx", "&conv_case_conv_params",
                        "&conv_case_quant_params", "&conv_case_input_dims", "input", "&conv_case_filter_dims",
                        "conv_case_weights", "&conv_case_bias_dims", "conv_case_biases", "NULL",
                        "&conv_case_output_dims", "output"]
    assert "arm_convolve_weight_sum(" in rendered


def test_no_bias_passes_null_for_both_bias_arguments(contracts: ContractSet) -> None:
    rendered = _environment(contracts).get_template(TEMPLATE).render(
        **_context("arm_convolve_wrapper_s8", float_kernel=False, has_biases=False))
    assert _call_args(rendered, "arm_convolve_wrapper_s8")[0][8:10] == ["NULL", "NULL"]


def test_template_refuses_a_prototype_it_cannot_satisfy(tmp_path: Path) -> None:
    """Binding is by name, so a checkout whose arm_convolve_wrapper_s8 lost weight_sum_ctx
    gets a call without it rather than a shifted one; a kernel the contract does not
    declare still refuses to render."""
    param = lambda name, c_type, direction: ParamDecl(name, c_type, direction)
    decl = FunctionDecl(
        name="arm_convolve_wrapper_s8", header="Include/arm_nnfunctions.h", line=1, guards=(),
        returns="arm_cmsis_nn_status",
        params=(param("ctx", "const cmsis_nn_context *", "in,out"), param("conv_params", "const cmsis_nn_conv_params *", "in"),
                param("quant_params", "const cmsis_nn_per_channel_quant_params *", "in"),
                param("input_dims", "const cmsis_nn_dims *", "in"), param("input_data", "const int8_t *", "in"),
                param("filter_dims", "const cmsis_nn_dims *", "in"), param("filter_data", "const int8_t *", "in"),
                param("bias_dims", "const cmsis_nn_dims *", "in"), param("bias_data", "const int32_t *", "in"),
                param("output_dims", "const cmsis_nn_dims *", "in"), param("output_data", "int8_t *", "out")),
    )
    sizer = FunctionDecl(
        name="arm_convolve_wrapper_s8_get_buffer_size", header="Include/arm_nnfunctions.h", line=2, guards=(),
        returns="int32_t", params=tuple(param(n, "const cmsis_nn_dims *", "in") for n in ("input_dims", "filter_dims")))
    contracts = ContractSet(status=STATUS_PRESENT, root=tmp_path, path=tmp_path / "x.json",
                            functions={decl.name: decl, sizer.name: sizer})
    # The pool offers weight_sum_ctx, but this prototype does not take it: binding simply
    # leaves it out, so the call still renders for the parameters the prototype does take.
    rendered = _environment(contracts).get_template(TEMPLATE).render(**_context("arm_convolve_wrapper_s8", float_kernel=False))
    assert "&conv_case_weight_sum_ctx, /* weight_sum_ctx */" not in rendered
    with pytest.raises(ContractError, match=r"arm_convolve_wrapper_s4: not in the kernel contract"):
        _environment(contracts).get_template(TEMPLATE).render(**_context("arm_convolve_wrapper_s4", float_kernel=False))


def _sizer_args(rendered: str) -> list[tuple[str, list[str]]]:
    text = re.sub(r"/\*.*?\*/|//[^\n]*", " ", rendered, flags=re.S)
    return [(fn, [a.strip() for a in args.split(",") if a.strip()])
            for fn, args in re.findall(r"required_buffer_size = (arm_\w+)\((.*?)\);", text, flags=re.S)]


@pytest.mark.parametrize("sizer, expected", [
    ("arm_convolve_wrapper_s8_get_buffer_size",
     ["&conv_case_conv_params", "&conv_case_input_dims", "&conv_case_filter_dims", "&conv_case_output_dims"]),
    ("arm_convolve_s8_get_buffer_size", ["&conv_case_input_dims", "&conv_case_filter_dims"]),
    ("arm_convolve_f32_get_buffer_size",
     ["&conv_case_conv_params", "&conv_case_input_dims", "&conv_case_filter_dims", "&conv_case_output_dims",
      "ARM_NN_LAYOUT_NHWC"]),
])
def test_the_sizer_is_bound_from_its_prototype(contracts: ContractSet, sizer: str, expected: list[str]) -> None:
    float_kernel = sizer.startswith("arm_convolve_f32")
    kernel = "arm_convolve_f32" if float_kernel else "arm_convolve_wrapper_s8"
    context = _context(kernel, float_kernel=float_kernel, needs_layout=float_kernel)
    context["kernel_get_buffer_size_fn"] = sizer
    rendered = _environment(contracts).get_template(TEMPLATE).render(**context)
    sizers = _sizer_args(rendered)
    assert sizers and all(call == (sizer, expected) for call in sizers)


def test_contract_entry_with_a_struct_bias_and_its_own_sizer(contracts: ContractSet) -> None:
    context = _context("arm_convolve_s16", float_kernel=False, entry_family="contract")
    context.update(bias_is_struct=True, kernel_get_buffer_size_fn="arm_convolve_s16_get_buffer_size")
    rendered = _environment(contracts).get_template(TEMPLATE).render(**context)
    assert _call_args(rendered, "arm_convolve_s16")[0] == [
        "&conv_case_ctx", "&conv_case_conv_params", "&conv_case_quant_params", "&conv_case_input_dims", "input",
        "&conv_case_filter_dims", "conv_case_weights", "&conv_case_bias_dims", "&conv_case_bias_data",
        "&conv_case_output_dims", "output"]
    assert "static const cmsis_nn_bias_data conv_case_bias_data" in rendered
    assert _sizer_args(rendered)[0] == ("arm_convolve_s16_get_buffer_size",
                                        ["&conv_case_input_dims", "&conv_case_filter_dims"])


def test_contract_entry_that_takes_weight_sums(contracts: ContractSet) -> None:
    context = _context("arm_convolve_1x1_s8_fast", float_kernel=False, entry_family="contract")
    context.update(conv_s8_weight_sum=True, kernel_get_buffer_size_fn="arm_convolve_1x1_s8_fast_get_buffer_size")
    rendered = _environment(contracts).get_template(TEMPLATE).render(**context)
    assert _call_args(rendered, "arm_convolve_1x1_s8_fast")[0][:2] == ["&conv_case_ctx", "&conv_case_weight_sum_ctx"]
    assert "arm_convolve_weight_sum(" in rendered
    assert _sizer_args(rendered)[0] == ("arm_convolve_1x1_s8_fast_get_buffer_size", ["&conv_case_input_dims"])


def test_entry_scratch_renders_no_sizer_call(contracts: ContractSet) -> None:
    context = _context("arm_convolve_s16", float_kernel=False, entry_family="contract")
    context.update(bias_is_struct=True, kernel_get_buffer_size_fn=None, entry_scratch_bytes=0)
    rendered = _environment(contracts).get_template(TEMPLATE).render(**context)
    assert "int32_t required_buffer_size = 0;" in rendered
    assert "HELIA_VALIDATE_SIZER" not in rendered and not _sizer_args(rendered)


def test_a_prototype_the_pool_cannot_satisfy_is_refused(tmp_path: Path) -> None:
    def param(name: str, c_type: str) -> ParamDecl:
        return ParamDecl(name, c_type, "in")

    decl = FunctionDecl(
        name="arm_fx_convolve_s8", header="Include/arm_nnfunctions.h", line=1, guards=(), returns="arm_cmsis_nn_status",
        params=(param("ctx", "const cmsis_nn_context *"), param("lut", "const int16_t *"),
                param("output_data", "int8_t *")),
    )
    sizer = FunctionDecl(name="arm_fx_convolve_s8_get_buffer_size", header="Include/arm_nnfunctions.h", line=2,
                         guards=(), returns="int32_t", params=(param("input_dims", "const cmsis_nn_dims *"),))
    contracts = ContractSet(status=STATUS_PRESENT, root=tmp_path, path=tmp_path / "x.json",
                            functions={decl.name: decl, sizer.name: sizer})
    context = _context("arm_fx_convolve_s8", float_kernel=False, entry_family="contract")
    with pytest.raises(ContractRenderError, match=r"cannot supply \['lut \(const int16_t \*\)'\]"):
        _environment(contracts).get_template(TEMPLATE).render(**context)
