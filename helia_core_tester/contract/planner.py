"""The host planner of the ns-cmsis-nn checkout the tester reads (ns_cmsis_nn_contracts.open_planner).

The planner is the library's own dispatch and sizing code built for the host from the checkout's
`Planner/` sources, so a plan names the kernel a wrapper reaches and the bytes its context needs
for a target build exactly as the target library computes them. The tester asks it instead of
keeping Python copies of those rules.

A generated case is built for several library configurations of one CPU (the default build and
the ARM_MATH_AUTOVECTORIZE coverage builds), so a capacity the case declares takes the largest
answer over the builds of that CPU (`planned_ctx_bytes`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

from helia_core_tester.contract.ir import ContractError, load_contract_set, load_contracts_package
from helia_core_tester.core.cpu_targets import get_cpu_profile


class PlannerUnavailable(ContractError):
    """The checkout's planner cannot be built, loaded or trusted."""


class PlanRefusedError(ContractError):
    """The planner refused a layer; `status` is the planner's reason (shape, range, unsupported, ...)."""

    def __init__(self, status: str, message: str) -> None:
        super().__init__(message)
        self.status = status


_PLANNERS: dict[str, Any] = {}


def clear_planner_cache() -> None:
    _PLANNERS.clear()


def load_planner(cmsis_nn_root: Optional[Path]):
    """The planner of `cmsis_nn_root` for GCC builds; one per checkout and process."""
    contracts = load_contract_set(cmsis_nn_root)
    if not contracts.present:
        raise PlannerUnavailable(f"the ns-cmsis-nn checkout ({cmsis_nn_root or 'unresolved'}) has no kernel contract, "
                                 "so it has no planner; it needs an ns-cmsis-nn that ships ns_cmsis_nn_contracts "
                                 "(AmbiqAI/ns-cmsis-nn#889 or later)")
    root = Path(contracts.root)
    key = str(root.resolve())
    if key in _PLANNERS:
        return _PLANNERS[key]
    package = load_contracts_package(root)
    if not hasattr(package, "open_planner"):
        raise PlannerUnavailable(f"{root}: its ns_cmsis_nn_contracts has no open_planner")
    try:
        planner = package.open_planner(root, "gcc")
    except (package.PlannerError, package.ContractError, OSError, ValueError) as error:
        raise PlannerUnavailable(f"{root}: the host planner cannot be built or loaded ({error})") from None
    _PLANNERS[key] = (package, planner)
    return _PLANNERS[key]


def current_planner():
    """(package, planner) of the resolved checkout (render.resolve_cmsis_nn_root)."""
    from helia_core_tester.contract import render

    return load_planner(render.resolve_cmsis_nn_root())


def build_targets(package, cpu: Optional[str]) -> list:
    """The library builds a case for `cpu` is compiled under; every build when `cpu` is None."""
    if cpu is None:
        return [package.Target("scalar"), package.Target("dsp"), package.Target("mve"),
                package.Target("mve", autovectorize=True)]
    profile = get_cpu_profile(cpu)
    if profile.has_mve:
        return [package.Target("mve"), package.Target("mve", autovectorize=True)]
    return [package.Target("dsp" if profile.has_dsp else "scalar")]


def isa_target(package, mve: bool, dsp: bool = True):
    return package.Target("mve" if mve else "dsp" if dsp else "scalar")


def _method(planner, method: str) -> Callable:
    fn = getattr(planner, method, None)
    if fn is None:
        raise PlannerUnavailable(f"this checkout's planner has no {method}; it needs AmbiqAI/ns-cmsis-nn's P1e "
                                 "planner calls or later")
    return fn


def plan(method: str, target, **layer):
    """One plan of `method` (a Planner method name) for `target`, with refusals typed for the tester."""
    package, planner = current_planner()
    fn = _method(planner, method)
    try:
        return fn(target, **layer)
    except package.PlanRefused as error:
        raise PlanRefusedError(error.status, str(error)) from None
    except (ValueError, package.PlannerError) as error:
        raise PlanRefusedError("arguments", f"{method}: {error}") from None


def planned_ctx_bytes(method: str, cpu: Optional[str], **layer) -> int:
    """The largest ctx a `method` plan needs over the builds of `cpu`; a build that declines the layer
    needs none. Every build declining is an error: the case would size nothing it runs."""
    package, _ = current_planner()
    sizes = []
    refusals = []
    for target in build_targets(package, cpu):
        try:
            sizes.append(plan(method, target, **layer).ctx_bytes)
        except PlanRefusedError as error:
            if error.status != "unsupported":
                raise
            refusals.append(str(error))
    if not sizes:
        raise PlanRefusedError("unsupported", f"{method}: every build of {cpu or 'any CPU'} declines the layer "
                                              f"({'; '.join(refusals)})")
    return max(sizes)


def buffer_bytes(target, implementation_id: str, buffer_name: str, **layer) -> int:
    """The bytes of one manifest buffer on `target`, from the library's own size query for the target ISA.

    `layer` may hold every operand of the layer; only those the buffer's size query takes are passed on.
    """
    package, planner = current_planner()
    fn = _method(planner, "buffer_bytes")
    try:
        buffer = planner.implementations.require(implementation_id).buffer(buffer_name)
    except (package.ContractError, KeyError) as error:
        raise PlanRefusedError("arguments", f"{implementation_id}.{buffer_name}: {error}") from None
    roots = {ref.split(".", 1)[0] for ref in buffer.size_args.values()}
    missing = sorted(roots - set(layer))
    if missing:
        raise PlanRefusedError("arguments", f"{implementation_id}.{buffer_name}: no {missing} given")
    try:
        return fn(target, implementation_id, buffer_name, **{root: layer[root] for root in roots})
    except package.PlanRefused as error:
        raise PlanRefusedError(error.status, str(error)) from None
    except (ValueError, package.PlannerError) as error:
        raise PlanRefusedError("arguments", f"{implementation_id}.{buffer_name}: {error}") from None


def planned_buffer_bytes(implementation_id: str, buffer_name: str, cpu: Optional[str], **layer) -> int:
    """The largest size of a manifest buffer over the builds of `cpu`."""
    package, _ = current_planner()
    return max(buffer_bytes(target, implementation_id, buffer_name, **layer)
               for target in build_targets(package, cpu))


def vector_sum_s8(data, *, cols: int, rows: int, lhs_offset: int, rhs_offset: int, bias=None) -> list[int]:
    """arm_vector_sum_s8 run by the library's own code (per row: bias + lhs_offset * sum(data + rhs_offset))."""
    package, planner = current_planner()
    try:
        return planner.vector_sum_s8(data, cols=cols, rows=rows, lhs_offset=lhs_offset, rhs_offset=rhs_offset,
                                     bias=bias)
    except package.PlanRefused as error:
        raise PlanRefusedError(error.status, str(error)) from None
    except ValueError as error:
        raise PlanRefusedError("arguments", f"arm_vector_sum_s8: {error}") from None


def fc_packed_stream_bytes(filter_data, *, filter_dims, multipliers, shifts) -> int:
    """The length of the stream arm_fully_connected_per_channel_packed_s8_pack writes for this filter, from the
    library's own packer (the size depends only on the filter shape, whether or not the layer passes the gate)."""
    package, planner = current_planner()
    try:
        return len(planner.pack_fully_connected_per_channel_packed_s8(
            filter_data, filter_dims=filter_dims, kernel_sums=[0] * int(filter_dims[3]),
            multipliers=multipliers, shifts=shifts))
    except package.PlanRefused as error:
        raise PlanRefusedError(error.status, str(error)) from None
    except ValueError as error:
        raise PlanRefusedError("arguments", f"arm_fully_connected_per_channel_packed_s8_pack: {error}") from None
