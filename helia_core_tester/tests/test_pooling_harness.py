"""Pooling renders through the generic harness (iteration 1b, G10): a case may have no scratch
buffer or no context at all, an alias kernel no public header declares takes its public twin's
prototype locally, `src`/`dst` bind as input/output, and the JSON sidecar is written from the
same validation context as before."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from helia_core_tester.contract.bind import bind
from helia_core_tester.contract.ir import STATUS_PRESENT, ContractSet, FunctionDecl, ParamDecl
from helia_core_tester.generation.harness import ArgumentPool, HarnessError, plan_harness
from helia_core_tester.generation.harness.plan import render_prototype
from helia_core_tester.generation.ops._shared.pool_base import UNDECLARED_ALIASES, pool_argument_pool, pool_fault
from helia_core_tester.tests.harness_render import render_pool, render_pooling


def pool_context(kernel_fn: str = "arm_avgpool_s8", sizer: str | None = "arm_avgpool_s8_get_buffer_size", *,
                 buffer_size_max: int = 32, float_kernel: bool = False, **overrides) -> dict:
    ctype = "float" if float_kernel else "int8_t"
    context = {
        "name": "pool_case", "kernel_fn": kernel_fn, "kernel_get_buffer_size_fn": sizer, "float_kernel": float_kernel,
        "input_dims": {"n": 1, "h": 4, "w": 4, "c": 3}, "filter_dims": {"n": 1, "h": 2, "w": 2, "c": 1},
        "output_dims": {"n": 1, "h": 2, "w": 2, "c": 3},
        "pool_params": {"stride_w": 2, "stride_h": 2, "pad_w": 0, "pad_h": 0, "activation_min": -128,
                        "activation_max": 127},
        "pool_params_type": "cmsis_nn_pool_params_f32" if float_kernel else "cmsis_nn_pool_params",
        "pool_activation_min_literal": "-1.0e+30f", "pool_activation_max_literal": "1.0e+30f",
        "input_data_array": "    0", "expected_output_array": "    0", "input_dtype": ctype, "output_dtype": ctype,
        "buffer_size_max": buffer_size_max, "use_batch_harness": False,
    }
    context.update(overrides)
    return context


def test_avgpool_sizes_scratch_from_struct_fields() -> None:
    header, source = render_pooling(pool_context())
    assert re.search(r"required_buffer_size = arm_avgpool_s8_get_buffer_size\(\s*pool_case_output_dims\.w, /\* dim_dst_width \*/"
                     r"\s*pool_case_input_dims\.c /\* ch_src \*/\s*\);", source)
    assert "pool_case_ctx.buf = pool_case_buffer;" in source
    assert ".stride.w = 2,\n    .stride.h = 2," in header and ".activation.min = -128," in header


def test_max_pool_has_neither_scratch_buffer_nor_required_size() -> None:
    _, source = render_pooling(pool_context("arm_max_pool_s8", None, buffer_size_max=0), suffix="max_pool")
    assert "pool_case_buffer" not in source and "BUFFER_SIZE_MAX" not in source
    assert "required_buffer_size" not in source
    assert "pool_case_ctx.buf = NULL;" in source and "pool_case_ctx.size = 0;" in source


def test_float_avgpool_keeps_its_guarded_buffer_but_hands_no_scratch() -> None:
    _, source = render_pooling(pool_context("arm_avg_pool_f32", None, float_kernel=True))
    assert "HELIA_GUARD_ARM(pool_case_buffer, true" in source and "pool_case_ctx.buf = NULL;" in source
    assert "required_buffer_size" not in source
    assert re.search(r"return arm_avg_pool_f32\([^;]*input, /\* src \*/[^;]*output /\* dst \*/", source, flags=re.S)


@pytest.mark.parametrize("alias, public", sorted(UNDECLARED_ALIASES.items()))
def test_undeclared_alias_takes_its_public_twins_prototype(alias: str, public: str) -> None:
    ftype = "float16_t" if alias.endswith("f16") else "float32_t"
    _, source = render_pooling(pool_context(alias, None, float_kernel=True))
    ptype = "cmsis_nn_pool_params_f16" if alias.endswith("f16") else "cmsis_nn_pool_params_f32"
    assert (f"arm_cmsis_nn_status {alias}(const cmsis_nn_context *ctx, const {ptype} *pool_params, "
            f"const cmsis_nn_dims *input_dims, const {ftype} *src, const cmsis_nn_dims *filter_dims, "
            f"const cmsis_nn_dims *output_dims, {ftype} *dst);") in source
    assert f"{public}'s prototype" in source
    assert "_Static_assert" not in source


@pytest.mark.parametrize("kind, marker", [
    ("zero_dim", "pool_case_fault_input_dims.n = 0;"),
    ("negative_dim", "pool_case_fault_input_dims.n = -1;"),
    ("null_input", "NULL, /* input_data */"),
])
def test_pooling_faults(kind: str, marker: str) -> None:
    context = pool_context()
    pool = pool_fault(pool_argument_pool(context), kind, context)
    context = {**context, "fault": kind, "expected_status": "ARM_CMSIS_NN_ARG_ERROR"}
    _, source = render_pool(context, pool, stem="avg_pool", validation_key="PoolingFunctions/avg_pool/avg_pool_fault.c.j2",
                            label="Pooling")
    assert marker in source
    # The scratch query keeps the passing dims.
    assert "pool_case_output_dims.w, /* dim_dst_width */" in source


def test_pooling_fault_without_an_edit_is_refused() -> None:
    context = pool_context()
    with pytest.raises(ValueError, match="no pooling fault edit for 'channel_mismatch'"):
        pool_fault(pool_argument_pool(context), "channel_mismatch", context)


def test_the_harness_writes_the_sidecar_from_the_validation_context(tmp_path: Path) -> None:
    from helia_core_tester.generation.ops.PoolingFunctions.avg_pool import OpAvgPool

    desc = {"operator": "AvgPool", "name": "pool_case", "activation_dtype": "S8",
            "tensor_dtypes": {"input": "S8", "output": "S8"}}
    op = OpAvgPool(desc, seed=0, target_cpu="cortex-m55")
    context = pool_context()
    op.render_harness_files(tmp_path, stem="avg_pool", context=dict(context), pool=pool_argument_pool(context),
                            validation_key="PoolingFunctions/avg_pool/avg_pool.c.j2", label="Pooling", sidecar=True)
    sidecar = json.loads((tmp_path / "pool_case_avg_pool.sidecar.json").read_text())
    assert sidecar["kernel_fn"] == "arm_avgpool_s8" and sidecar["op_suffix"] == "avg_pool"
    assert "harness" not in sidecar["scalars"] and "header_name" not in sidecar["scalars"]
    assert sidecar["scalars"]["buffer_size_max"] == 32 and "input_data_array" not in sidecar["scalars"]


# --- harness mechanics this family introduced --------------------------------------------------

def _decl(name: str, *params: str, returns: str = "arm_cmsis_nn_status") -> FunctionDecl:
    return FunctionDecl(name=name, header="Include/arm_nnfunctions.h", line=1, guards=(), returns=returns,
                        params=tuple(ParamDecl(p, "const int8_t *", "in") for p in params))


CTX = _decl("arm_fx_ctx_s8", "ctx", "src", "dst")
BARE = _decl("arm_fx_bare_s8", "input", "output")
SIZER = _decl("arm_fx_ctx_s8_get_buffer_size", "input_dims", returns="int32_t")
CONTRACTS = ContractSet(status=STATUS_PRESENT, root=Path("."), path=None,
                        functions={d.name: d for d in (CTX, BARE, SIZER)})


def _plan(kernel: str, sizer=None, **overrides):
    pool = ArgumentPool(**{"name": "x", "values": {"ctx": "&x_ctx", "input_dims": "&x_dims"}, **overrides})
    return plan_harness(pool, kernel_fn=kernel, sizer_fn=sizer, scratch_bytes=None if sizer else 0, contracts=CONTRACTS)


def test_a_kernel_without_a_context_gets_neither_context_nor_scratch() -> None:
    plan = _plan("arm_fx_bare_s8", scratch_buffer=False)
    assert not plan.uses_ctx and not plan.scratch_buffer
    context = {"name": "x", "kernel_fn": "arm_fx_bare_s8", "input_dtype": "int8_t", "output_dtype": "int8_t",
               "use_batch_harness": False}
    pool = ArgumentPool(name="x", values={}, scratch_buffer=False, output_count="(4)")
    _, source = render_pool(context, pool, stem="x", validation_key="PoolingFunctions/max_pool/max_pool.c.j2",
                            label="X", contracts=CONTRACTS, sizer_fn=None)
    assert "required_buffer_size" not in source and "cmsis_nn_context" not in source


@pytest.mark.parametrize("kernel, sizer, overrides, message", [
    ("arm_fx_ctx_s8", "arm_fx_ctx_s8_get_buffer_size", {"scratch_buffer": False, "no_scratch": True},
     "without a scratch buffer cannot query or claim scratch"),
    ("arm_fx_ctx_s8", None, {"scratch_buffer": False}, "set no_scratch so the context is empty"),
    ("arm_fx_bare_s8", None, {}, "takes no context, so the case has no use for a scratch buffer"),
    ("arm_fx_ctx_s8", None, {"prototype_from": "arm_fx_bare_s8"}, "is public; bind it from the contract"),
])
def test_scratch_and_context_are_refused_when_they_disagree(kernel, sizer, overrides, message) -> None:
    with pytest.raises(HarnessError, match=message):
        _plan(kernel, sizer, **overrides)


def test_context_setup_needs_a_scratch_buffer() -> None:
    with pytest.raises(HarnessError, match="context_setup needs the scratch buffer"):
        ArgumentPool(name="x", values={"ctx": "&x_ctx"}, scratch_buffer=False, context_setup="    x;").validate()


def test_local_prototype_renders_the_public_twin_under_the_alias() -> None:
    plan = _plan("arm_fx_ctx_s8_alias", scratch_buffer=True, no_scratch=True, prototype_from="arm_fx_ctx_s8")
    assert plan.local_prototype == ("arm_cmsis_nn_status arm_fx_ctx_s8_alias(const int8_t *ctx, const int8_t *src, "
                                    "const int8_t *dst);")
    assert render_prototype(_decl("arm_fx_void")) == "arm_cmsis_nn_status arm_fx_void(void);"
    assert "input, /* src */" in plan.run_call and "output /* dst */" in plan.run_call


def test_src_and_dst_bind_as_input_and_output() -> None:
    assert bind(CTX, {"ctx": "c", "input_data": "i", "output_data": "o"}) == {"ctx": "c", "src": "i", "dst": "o"}
