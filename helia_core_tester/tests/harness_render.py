"""Render cases of the harness-rendered operators from a context dict (test helper)."""

from __future__ import annotations

from typing import Any, Optional

import jinja2

from helia_core_tester.contract.ir import ContractSet
from helia_core_tester.contract.render import contract_globals, load_current_contracts
from helia_core_tester.core.discovery import find_tester_templates_dir
from helia_core_tester.generation.harness import plan_harness, render_declaration
from helia_core_tester.generation.harness import ArgumentPool
from helia_core_tester.generation.ops.ConvolutionFunctions.convolve import convolve_argument_pool
from helia_core_tester.generation.ops.ConvolutionFunctions.depthwise_conv import depthwise_argument_pool
from helia_core_tester.generation.ops.ConvolutionFunctions.transpose_conv import (
    TRANSPOSE_CONV_VALIDATION_KEY,
    transpose_conv_argument_pool,
)
from helia_core_tester.generation.ops._shared.pool_base import pool_argument_pool
from helia_core_tester.generation.ops.FullyConnectedFunctions.batch_matmul import BMM_VALIDATION_KEY, bmm_argument_pool
from helia_core_tester.generation.ops.FullyConnectedFunctions.fully_connected import (
    FC_VALIDATION_KEY,
    fc_argument_pool,
    fc_sizer,
)
from helia_core_tester.generation.ops._shared.base import OperationBase
from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

CONVOLVE_VALIDATION_KEY = "ConvolutionFunctions/convolve/convolve.c.j2"
DEPTHWISE_VALIDATION_KEY = "ConvolutionFunctions/depthwise_conv/depthwise_conv.c.j2"


def convolve_context(kernel_fn: str, *, float_kernel: bool = False, has_biases: bool = True,
                     sizer: Optional[str] = "default", entry_family: Optional[str] = None,
                     scratch: Optional[int] = None, **overrides: Any) -> dict:
    dims = {"n": 1, "h": 2, "w": 2, "c": 4}
    ctype = "float" if float_kernel else "int8_t"
    context: dict[str, Any] = {
        "name": "conv_case", "kernel_fn": kernel_fn,
        "kernel_get_buffer_size_fn": kernel_fn + "_get_buffer_size" if sizer == "default" else sizer,
        "entry_family": entry_family, "entry_scratch_bytes": scratch, "float_kernel": float_kernel,
        "has_biases": has_biases, "kernel_layout": "ARM_NN_LAYOUT_NHWC", "buffer_size_max": 64,
        "input_dims": dims, "filter_dims": dims, "output_dims": dims,
        "conv_params": {"input_offset": 0, "output_offset": 0, "stride_w": 1, "stride_h": 1, "dilation_w": 1,
                        "dilation_h": 1, "pad_w": 0, "pad_h": 0, "activation_min": -128, "activation_max": 127},
        "quant_params": {"per_channel": False, "multiplier": 1, "shift": 0},
        "conv_params_type": "cmsis_nn_conv_params_f32" if float_kernel else "cmsis_nn_conv_params",
        "conv_activation_min_literal": "-1.0e+30f", "conv_activation_max_literal": "1.0e+30f",
        "weights_array": "    1", "biases_array": "    0", "input_data_array": "    0",
        "expected_output_array": "    0", "input_dtype": ctype, "output_dtype": ctype,
        "weight_dtype": ctype, "bias_dtype": "float" if float_kernel else "int32_t", "use_batch_harness": False,
    }
    context.update(overrides)
    return context


def depthwise_context(kernel_fn: str, sizer: Optional[str], *, float_kernel: bool = False, has_biases: bool = True,
                      weight_sum: bool = False, scratch: Optional[int] = None, **overrides: Any) -> dict:
    dims = {"n": 1, "h": 2, "w": 2, "c": 4}
    ctype = "float" if float_kernel else "int8_t"
    context: dict[str, Any] = {
        "name": "dw_case", "kernel_fn": kernel_fn, "kernel_get_buffer_size_fn": sizer, "float_kernel": float_kernel,
        "has_biases": has_biases, "takes_weight_sum_ctx": weight_sum, "has_weight_sum": weight_sum,
        "weight_sum_array": "    0, 0, 0, 0" if weight_sum else "", "kernel_layout": "ARM_NN_LAYOUT_NHWC",
        "entry_scratch_bytes": scratch, "buffer_size_max": 64, "input_dims": dims, "filter_dims": dims,
        "output_dims": dims,
        "dw_conv_params": {"input_offset": 0, "output_offset": 0, "ch_mult": 1, "stride_w": 1, "stride_h": 1,
                           "dilation_w": 1, "dilation_h": 1, "pad_w": 0, "pad_h": 0, "activation_min": -128,
                           "activation_max": 127},
        "quant_params": {"per_channel": False, "multiplier": 1, "shift": 0},
        "dw_activation_min_literal": "-1.0e+30f", "dw_activation_max_literal": "1.0e+30f",
        "weights_array": "    1", "biases_array": "    0", "input_data_array": "    0",
        "expected_output_array": "    0", "input_dtype": ctype, "output_dtype": ctype, "weight_dtype": ctype,
        "bias_dtype": "float" if float_kernel else "int32_t", "use_batch_harness": False,
    }
    if float_kernel:
        context["dw_conv_params_type"] = "cmsis_nn_dw_conv_params_f32"
    context.update(overrides)
    return context


def render_pool(context: dict, pool: ArgumentPool, *, stem: str, validation_key: str, label: str,
                contracts: Optional[ContractSet] = None, sizer_fn: Any = "context") -> tuple[str, str]:
    """(header, source) as OperationBase.render_harness_files writes them."""
    contracts = contracts if contracts is not None else load_current_contracts()
    # A private environment: the cached one is shared by every generation in the session, and a
    # test's contract must not leak into it.
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(find_tester_templates_dir())),
                             trim_blocks=True, lstrip_blocks=True)
    env.globals.update(contract_globals(lambda: contracts))
    header = env.get_template(OperationBase.HARNESS_HEADER).render(
        name=context["name"], header_declarations=[render_declaration(d) for d in pool.header])
    sizer = context.get("kernel_get_buffer_size_fn") if sizer_fn == "context" else sizer_fn
    plan = plan_harness(pool, kernel_fn=context["kernel_fn"], sizer_fn=sizer,
                        scratch_bytes=None if sizer else int(context.get("entry_scratch_bytes") or 0),
                        contracts=contracts)
    render_context = TemplateContextBuilder.build_validation_context(validation_key, dict(context))
    from helia_core_tester.generation.ops._shared.base import _render_pool_snippets

    render_context.update(harness=plan, pool=_render_pool_snippets(env, pool, render_context), header_name=f"{context['name']}_{stem}.h", harness_label=label,
                          harness_output_count=pool.output_count, harness_benchmark=pool.benchmark)
    return header, env.get_template(OperationBase.HARNESS_SOURCE).render(**render_context)


def render_convolve(context: dict, *, bias_is_struct: bool = False,
                    contracts: Optional[ContractSet] = None) -> tuple[str, str]:
    pool = convolve_argument_pool(context, has_biases=bool(context["has_biases"]), bias_is_struct=bias_is_struct)
    return render_pool(context, pool, stem="convolve", validation_key=CONVOLVE_VALIDATION_KEY, label="Convolution",
                       contracts=contracts)


def render_depthwise(context: dict, *, contracts: Optional[ContractSet] = None) -> tuple[str, str]:
    return render_pool(context, depthwise_argument_pool(context), stem="depthwise_conv",
                       validation_key=DEPTHWISE_VALIDATION_KEY, label="Depthwise convolution", contracts=contracts)


def render_fully_connected(context: dict, *, contracts: Optional[ContractSet] = None) -> tuple[str, str]:
    return render_pool(context, fc_argument_pool(context), stem="fully_connected", validation_key=FC_VALIDATION_KEY,
                       label="Fully connected", contracts=contracts, sizer_fn=fc_sizer(context))


def render_batch_matmul(context: dict, *, contracts: Optional[ContractSet] = None) -> tuple[str, str]:
    return render_pool(context, bmm_argument_pool(context), stem="batch_matmul", validation_key=BMM_VALIDATION_KEY,
                       label="Batch matmul", contracts=contracts)


def render_transpose_conv(context: dict, *, contracts: Optional[ContractSet] = None) -> tuple[str, str]:
    return render_pool(context, transpose_conv_argument_pool(context), stem="transpose_conv",
                       validation_key=TRANSPOSE_CONV_VALIDATION_KEY, label="Transpose convolution", contracts=contracts)


def render_pooling(context: dict, *, suffix: str = "avg_pool", contracts: Optional[ContractSet] = None) -> tuple[str, str]:
    return render_pool(context, pool_argument_pool(context), stem=suffix,
                       validation_key=f"PoolingFunctions/{suffix}/{suffix}.c.j2", label="Pooling", contracts=contracts)
