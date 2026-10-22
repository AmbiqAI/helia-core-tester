"""Contract conformance cases (Phase 2 T4a of AmbiqAI/ns-cmsis-nn#829).

A descriptor with `conformance: true` checks, on the target, that the library it is built
against agrees with the checkout's host planner (contract/planner.py) for the case's layer, on
every build the case is compiled under (`planner.build_targets`):

- a build the planner plans must run the layer to the expected result with a context of exactly
  the planned size (the harness sizes ctx with the target's own query and checks the red zones
  on both sides of it; the case also asserts `ctx.size` equals the plan);
- a wrapper's route, as the wrapper's own selector (Include/Internal/arm_nn_dispatch.h) computes
  it on the target, must equal the planned route;
- a build the planner refuses as unsupported must be one where the case expects
  ARM_CMSIS_NN_NO_IMPL_ERROR, and the reverse;
- a packed FC stream and its kernel sums built on the target must equal the host packer's;
- a fault case's context must be planned non-empty on every build, so a NULL `ctx->buf` is the
  argument error the case expects.

Disagreement found at generation fails the case; disagreement on the target fails its run.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping, Optional, Sequence

from helia_core_tester.contract import planner

AUTOVECTORIZE_BUILD = "defined(HELIA_CMSIS_NN_INT_AUTOVECTORIZE)"
NO_IMPL = "ARM_CMSIS_NN_NO_IMPL_ERROR"
SUCCESS = "ARM_CMSIS_NN_SUCCESS"
DISPATCH_HEADER = '"Internal/arm_nn_dispatch.h"'


class ConformanceError(ValueError):
    """A conformance case whose descriptor and the checkout's planner disagree."""


@dataclasses.dataclass(frozen=True)
class PlanRequest:
    """What an operator asks the planner for a conformance case.

    `selector` is the on-target call of the wrapper's route selector (None for a direct entry);
    `takes_ctx` says whether the call passes the case's `ctx`; `packed` carries the host packer's
    inputs for a packed FC case."""

    method: str
    layer: Mapping[str, Any]
    selector: Optional[str] = None
    takes_ctx: bool = True
    packed: Optional[Mapping[str, Any]] = None


def _expects_decline(desc: Mapping[str, Any], target) -> bool:
    status = str(desc.get("expected_status", SUCCESS)).strip()
    if status == NO_IMPL:
        return True
    return bool(desc.get("autovectorize_declines")) and target.autovectorize


def _build_plans(desc: Mapping[str, Any], cpu: str, request: PlanRequest) -> list[tuple[Any, Optional[Any]]]:
    """(target, plan or None when refused as unsupported) per build, checked against the descriptor."""
    package, _ = planner.current_planner()
    name = desc.get("name", "<unnamed>")
    plans = []
    for target in planner.build_targets(package, cpu):
        try:
            plan = planner.plan(request.method, target, **request.layer)
        except planner.PlanRefusedError as error:
            if error.status != "unsupported":
                raise ConformanceError(f"{name}: the planner refuses the layer on {_label(target)} ({error})") from None
            plan = None
        declines = _expects_decline(desc, target)
        if (plan is None) != declines:
            raise ConformanceError(
                f"{name}: on {_label(target)} the planner {'refuses' if plan is None else 'plans'} the layer, but "
                f"the case expects {'ARM_CMSIS_NN_NO_IMPL_ERROR' if declines else 'the full result'}")
        plans.append((target, plan))
    return plans


def _label(target) -> str:
    return target.isa + (" + ARM_MATH_AUTOVECTORIZE" if target.autovectorize else "")


def build_condition(target) -> str:
    """The preprocessor condition true exactly in a compilation of the case for `target`'s library build.
    Any other compilation (the host check's scalar build of an MVE case, say) asserts nothing."""
    if target.isa == "mve":
        autovec = AUTOVECTORIZE_BUILD if target.autovectorize else f"!{AUTOVECTORIZE_BUILD}"
        return f"defined(ARM_MATH_MVEI) && {autovec}"
    if target.isa == "dsp":
        return "defined(ARM_MATH_DSP) && !defined(ARM_MATH_MVEI)"
    return "!defined(ARM_MATH_DSP) && !defined(ARM_MATH_MVEI)"


def _per_build(plans: Sequence[tuple[Any, Optional[Any]]], render) -> list[str]:
    """`render(plan)` for each planned build, each under its build's condition."""
    lines = []
    for target, plan in plans:
        body = render(plan) if plan is not None else []
        if body:
            lines += [f"#if {build_condition(target)}", *body, "#endif"]
    return lines


def _c_array(values: Sequence[int]) -> str:
    return "{" + ", ".join(str(int(v)) for v in values) + "}"


def _signed_bytes(data: bytes) -> list[int]:
    return [b - 256 if b > 127 else b for b in data]


def apply(pool, context: Mapping[str, Any], desc: Mapping[str, Any], cpu: str):
    """The pool with the conformance checks of `context['conformance']` added; unchanged for any other case."""
    request: Optional[PlanRequest] = context.get("conformance")
    if not desc.get("conformance"):
        if request is not None:
            raise ConformanceError(f"{desc.get('name')}: a plan request on a case that is not a conformance case")
        return pool
    from helia_core_tester.generation.entry import apply_entry_manifest

    desc = apply_entry_manifest(desc)
    name = desc.get("name", "<unnamed>")
    if request is None:
        raise ConformanceError(f"{name}: conformance is not supported for this case "
                               f"({context.get('kernel_fn')}); the T4a pilot covers the s8 depthwise and FC "
                               "wrappers, the packed FC entry and s8 pooling")
    plans = _build_plans(desc, cpu, request)
    n, label = context["name"], f"{name} conformance"
    if desc.get("fault"):
        empty = [_label(t) for t, plan in plans if plan is not None and plan.ctx_bytes <= 0]
        if empty:
            raise ConformanceError(f"{name}: a {desc['fault']} case needs a planned context, but {empty} plan none")
        return pool

    def checks(plan) -> list[str]:
        lines = []
        if request.takes_ctx:
            lines.append(f'    HELIA_VALIDATE_SCALAR_EQ_INT("{label}", "ctx.size against the planned bytes", '
                         f"{int(plan.ctx_bytes)}, {n}_ctx.size);")
        if request.selector is not None:
            route_name = planner.route_name(plan)
            if route_name is None:
                raise ConformanceError(f"{name}: planned route {plan.route!r} has no route enum to check")
            lines.append(f'    HELIA_VALIDATE_SCALAR_EQ_INT("{label}", "{plan.route}", {route_name}, '
                         f"{request.selector});")
        return lines

    body = _per_build(plans, checks)
    header = ""
    if request.packed is not None:
        stream, sums = _host_packed(request.packed)
        header = "\n".join([
            "// The host packer's stream and kernel sums for this layer (ns_cmsis_nn_contracts).",
            f"static const int8_t {n}_host_packed[{len(stream)}] = {_c_array(_signed_bytes(stream))};",
            f"static const int32_t {n}_host_kernel_sum[{len(sums)}] = {_c_array(sums)};",
        ])

        def packed_checks(plan) -> list[str]:
            return [f"    HELIA_VALIDATE_OUTPUTS(EXACT_INT, (const int8_t *){n}_packed, {n}_host_packed, "
                    f"{len(stream)}, 0, 0.0f, 0.0f, 20, failures);",
                    f"    HELIA_VALIDATE_OUTPUTS(EXACT_INT, {n}_kernel_sum, {n}_host_kernel_sum, {len(sums)}, "
                    f"0, 0.0f, 0.0f, 20, failures);"]

        body += _per_build(plans, packed_checks)
    snippet = "\n".join(["    // Contract conformance: the target agrees with the checkout's planner.", *body])
    return dataclasses.replace(
        pool,
        extra_checks="\n".join(part for part in (pool.extra_checks, snippet) if part),
        includes=tuple(pool.includes) + ((DISPATCH_HEADER,) if request.selector is not None else ()),
        file_scope="\n".join(part for part in (pool.file_scope, header) if part),
    )


def _host_packed(packed: Mapping[str, Any]) -> tuple[bytes, list[int]]:
    """The stream and kernel sums the manifest's prepare steps build, run on the host by the library's own code."""
    package, impl = planner.current_planner()
    k, c = packed["filter_dims"][0], packed["filter_dims"][3]
    sums = planner.vector_sum_s8(packed["filter_data"], cols=k, rows=c, lhs_offset=packed["input_offset"],
                                 rhs_offset=packed["filter_offset"], bias=packed.get("bias"))
    try:
        stream = impl.pack_fully_connected_per_channel_packed_s8(
            packed["filter_data"], filter_dims=packed["filter_dims"], kernel_sums=sums,
            multipliers=packed["multipliers"], shifts=packed["shifts"])
    except package.PlanRefused as error:
        raise ConformanceError(f"the host packer refuses the layer ({error})") from None
    return stream, sums
