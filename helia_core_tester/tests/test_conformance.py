"""Contract conformance (generation/conformance.py): the grid covers every T4a route on every FVP profile, and a
case whose expectation the planner contradicts fails at generation."""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import pytest

from helia_core_tester.contract import planner
from helia_core_tester.generation import conformance
from helia_core_tester.generation.conformance import ConformanceError, PlanRequest
from helia_core_tester.generation.harness import ArgumentPool
from helia_core_tester.generation.io.descriptors import load_all_descriptors
from helia_core_tester.generation.test_ops import generate_test
from helia_core_tester.tests.contract_fallback import FIXTURE_ROOT
from helia_core_tester.tests.contract_required import P1E_PLAN_FIELDS, P1dPlan, require_planner

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Every T4a implementation of the pilot manifest. The direct depthwise entries and arm_depthwise_conv_s8_opt's own
# planar/channelwise routes wait for planner calls the package does not have yet (T4b).
T4A = {"depthwise_conv_s8_to_conv", "depthwise_conv_3x3_s8", "depthwise_conv_s8_opt", "depthwise_conv_s8",
       "fully_connected_per_channel_s8", "fully_connected_s8", "fully_connected_per_channel_packed_s8",
       "avg_pool_s8", "max_pool_s8"}
T4B = {"depthwise_conv_s8_opt_3x3", "depthwise_conv_s8_opt_3x3_c64_s1", "depthwise_conv_s8_opt_planar",
       "depthwise_conv_s8_opt_channelwise"}
PROFILES = ("cortex-m0", "cortex-m4", "cortex-m55")
DW = dict(input_dims=(1, 12, 12, 8), filter_dims=(1, 5, 5, 8), output_dims=(1, 12, 12, 8), ch_mult=1,
          padding=(2, 2))


@pytest.fixture
def fixture_planner(monkeypatch: pytest.MonkeyPatch):
    from helia_core_tester.contract import render

    monkeypatch.setattr(render, "resolve_cmsis_nn_root", lambda: FIXTURE_ROOT)
    return planner.load_planner(FIXTURE_ROOT)


def _grid() -> list[dict]:
    return [d for d in load_all_descriptors(str(PROJECT_ROOT / "assets" / "descriptors")) if d.get("conformance")]


def _layer(desc: dict) -> tuple[str, dict]:
    """The planner request a grid case makes, read from its descriptor."""
    n, h, w, c = desc["input_shape"] if len(desc["input_shape"]) == 4 else (desc["input_shape"][0], 1, 1,
                                                                            desc["input_shape"][1])
    if desc["operator"] == "DepthwiseConv":
        kh, kw, _, mult = desc["filter_shape"]
        sh, sw = desc["strides"]
        dh, dw = desc.get("dilation", (1, 1))
        if desc["padding"] == "SAME":
            oh, ow = -(-h // sh), -(-w // sw)
            ph = max((oh - 1) * sh + (kh - 1) * dh + 1 - h, 0) // 2
            pw = max((ow - 1) * sw + (kw - 1) * dw + 1 - w, 0) // 2
        else:
            oh, ow = (h - (kh - 1) * dh - 1) // sh + 1, (w - (kw - 1) * dw - 1) // sw + 1
            ph = pw = 0
        return "depthwise_conv_wrapper_s8", dict(
            input_dims=(n, h, w, c), filter_dims=(1, kh, kw, c * mult), output_dims=(n, oh, ow, c * mult),
            ch_mult=mult, stride=(sh, sw), padding=(ph, pw), dilation=(dh, dw))
    if desc["operator"] == "FullyConnected":
        cout, k = desc["filter_shape"]
        if desc.get("entry") == "arm_fully_connected_per_channel_packed_s8":
            offset = (desc.get("hint", {}).get("extras") or {}).get("force_filter_offset", 0)
            return "fully_connected_per_channel_packed_s8", dict(
                filter_dims=(k, 1, 1, cout), filter_offset=offset, multipliers=[1 << 30] * cout, shifts=[-1] * cout)
        return "fully_connected_wrapper_s8", dict(
            input_dims=(n, 1, 1, k), filter_dims=(k, 1, 1, cout), output_dims=(n, 1, 1, cout),
            per_channel=not desc.get("hint", {}).get("force_per_tensor"))
    kh, kw = desc["pool_size"]
    sh, sw = desc["strides"]
    oh, ow = (-(-h // sh), -(-w // sw)) if desc["padding"] == "SAME" else ((h - kh) // sh + 1, (w - kw) // sw + 1)
    return "pool_s8", dict(max_pool=desc["operator"] == "MaxPool", input_dims=(n, h, w, c), kernel_hw=(kh, kw),
                           output_dims=(n, oh, ow, c), stride=(sh, sw), padding=(0, 0))


def _reached(package, desc: dict, cpu: str) -> set[str]:
    method, layer = _layer(desc)
    reached = set()
    for target in planner.build_targets(package, cpu):
        try:
            plan = planner.plan(method, target, **layer)
        except planner.PlanRefusedError:
            continue
        reached.add(plan.implementation.id)
    return reached


def test_the_grid_reaches_every_t4a_implementation_on_some_profile(fixture_planner) -> None:
    package, impl = fixture_planner
    covered = {cpu: set() for cpu in PROFILES}
    for desc in _grid():
        if desc.get("fault") or desc.get("expected_status"):
            continue
        for cpu in PROFILES:
            covered[cpu] |= _reached(package, desc, cpu)
    assert set().union(*covered.values()) == T4A
    assert set(impl.implementations.implementations) == T4A | T4B, "a manifest implementation is in neither suite"
    # Each wrapper route that exists on a build is reached there.
    assert {"depthwise_conv_s8_to_conv", "depthwise_conv_s8_opt", "depthwise_conv_s8"} <= covered["cortex-m55"]
    assert {"depthwise_conv_3x3_s8", "depthwise_conv_s8_opt", "depthwise_conv_s8"} <= covered["cortex-m4"]
    assert {"fully_connected_per_channel_s8", "fully_connected_s8", "avg_pool_s8",
            "max_pool_s8"} <= covered["cortex-m0"]
    assert "fully_connected_per_channel_packed_s8" in covered["cortex-m55"]


def test_the_grid_has_a_rejection_and_faults(fixture_planner) -> None:
    package, _ = fixture_planner
    rejected = [d for d in _grid() if d.get("expected_status") == "ARM_CMSIS_NN_NO_IMPL_ERROR"]
    assert rejected and all(not _reached(package, d, cpu) for d in rejected for cpu in PROFILES)
    faults = [d for d in _grid() if d.get("fault") == "null_ctx_buf"]
    assert {d["operator"] for d in faults} == {"DepthwiseConv", "FullyConnected"}


def _routes_checked(extra_checks: str) -> list[tuple[str, str]]:
    return re.findall(r'"([a-z0-9_]+:[a-z0-9_]+)", (\w+), sel\(\)\);', extra_checks)


def test_route_checks_name_the_packages_route_constants(fixture_planner) -> None:
    package, _ = fixture_planner
    header = (FIXTURE_ROOT / "Include" / "Internal" / "arm_nn_dispatch.h").read_text()
    constants = set(re.findall(r"\b(ARM_NN_\w+_ROUTE_\w+)\s*=", header))
    for method, layer, cpu in [("depthwise_conv_wrapper_s8", DW, "cortex-m55"),
                               ("depthwise_conv_wrapper_s8", dict(DW, filter_dims=(1, 3, 3, 8), padding=(1, 1)),
                                "cortex-m4"),
                               ("fully_connected_wrapper_s8", dict(input_dims=(1, 1, 1, 16), filter_dims=(16, 1, 1, 4),
                                                                   output_dims=(1, 1, 1, 4), per_channel=False),
                                "cortex-m0")]:
        pool = conformance.apply(_pool(), {"name": "x", "conformance": PlanRequest(method, layer, selector="sel()")},
                                 {"name": "x", "conformance": True}, cpu)
        plans = [planner.plan(method, t, **layer) for t in planner.build_targets(package, cpu)]
        assert _routes_checked(pool.extra_checks) == [(p.route, p.route_name) for p in plans]
        assert {p.route_name for p in plans} <= constants


def test_route_checks_take_the_name_from_the_plan_not_a_local_table(
    fixture_planner, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = planner.plan
    monkeypatch.setattr(planner, "plan", lambda *a, **k: dataclasses.replace(real(*a, **k),
                                                                             route_name="ARM_NN_RENAMED_ROUTE"))
    pool = conformance.apply(_pool(), {"name": "x", "conformance": PlanRequest("depthwise_conv_wrapper_s8", DW,
                                                                               selector="sel()")},
                             {"name": "x", "conformance": True}, "cortex-m55")
    assert _routes_checked(pool.extra_checks) == [("dw_wrapper_s8:opt", "ARM_NN_RENAMED_ROUTE")] * 2
    assert not hasattr(conformance, "ROUTE_ENUMS")


def test_a_package_whose_plans_name_no_route_is_unavailable(fixture_planner, monkeypatch: pytest.MonkeyPatch) -> None:
    real = planner.plan
    monkeypatch.setattr(planner, "plan", lambda *a, **k: P1dPlan(
        **{f.name: getattr(real(*a, **k), f.name) for f in dataclasses.fields(P1dPlan)}))
    with pytest.raises(planner.PlannerUnavailable, match=r"names no route constant \(Plan.route_name\)"):
        conformance.apply(_pool(), {"name": "x", "conformance": PlanRequest("depthwise_conv_wrapper_s8", DW,
                                                                            selector="sel()")},
                          {"name": "x", "conformance": True}, "cortex-m55")
    # A case without a selector checks no route, so it still renders on such a package.
    pool = conformance.apply(_pool(), {"name": "x", "conformance": PlanRequest("depthwise_conv_wrapper_s8", DW)},
                             {"name": "x", "conformance": True}, "cortex-m55")
    assert "planned bytes" in pool.extra_checks and not _routes_checked(pool.extra_checks)


@pytest.mark.parametrize("name", [d["name"] for d in _grid()])
@pytest.mark.parametrize("cpu", PROFILES)
def test_every_grid_case_generates(name: str, cpu: str, tmp_path: Path) -> None:
    require_planner(plan_fields=P1E_PLAN_FIELDS)
    desc = next(d for d in _grid() if d["name"] == name)
    from helia_core_tester.core.cpu_targets import missing_required_capabilities
    from helia_core_tester.generation.entry import apply_entry_manifest

    effective = apply_entry_manifest(desc)
    if missing_required_capabilities(cpu, effective.get("required_capabilities") or ()):
        pytest.skip(f"{name} needs {effective['required_capabilities']}")
    generate_test(effective, str(tmp_path), cpu=cpu)
    source = "".join(p.read_text() for p in tmp_path.rglob("*.c"))
    # A fault case is checked against the plan at generation; its run asserts only the argument error.
    assert ("Contract conformance" in source) is not bool(desc.get("fault"))
    if not desc.get("fault") and not desc.get("expected_status"):
        assert "planned bytes" in source or "_host_packed" in source


def _pool(n: str = "x") -> ArgumentPool:
    return ArgumentPool(name=n, values={})


def test_a_case_expecting_success_where_the_planner_refuses_fails_at_generation(fixture_planner) -> None:
    layer = dict(filter_dims=(13, 1, 1, 5), filter_offset=3, multipliers=[1 << 30] * 5, shifts=[-1] * 5)
    request = PlanRequest("fully_connected_per_channel_packed_s8", layer, takes_ctx=False)
    with pytest.raises(ConformanceError, match="the planner refuses the layer, but the case expects the full result"):
        conformance.apply(_pool(), {"name": "x", "conformance": request}, {"name": "x", "conformance": True},
                          "cortex-m55")


def test_a_case_expecting_a_decline_the_planner_plans_fails_at_generation(fixture_planner) -> None:
    request = PlanRequest("depthwise_conv_wrapper_s8", DW, selector="sel()")
    desc = {"name": "x", "conformance": True, "expected_status": "ARM_CMSIS_NN_NO_IMPL_ERROR"}
    with pytest.raises(ConformanceError, match="the planner plans the layer, but the case expects "
                                               "ARM_CMSIS_NN_NO_IMPL_ERROR"):
        conformance.apply(_pool(), {"name": "x", "conformance": request}, desc, "cortex-m4")


def test_a_fault_case_needs_a_planned_context(fixture_planner) -> None:
    three = dict(DW, filter_dims=(1, 3, 3, 8), padding=(1, 1))
    request = PlanRequest("depthwise_conv_wrapper_s8", three, selector="sel()")
    desc = {"name": "x", "conformance": True, "fault": "null_ctx_buf", "expected_status": "ARM_CMSIS_NN_ARG_ERROR"}
    with pytest.raises(ConformanceError, match="needs a planned context, but \\['dsp'\\] plan none"):
        conformance.apply(_pool(), {"name": "x", "conformance": request}, desc, "cortex-m4")
    assert conformance.apply(_pool(), {"name": "x", "conformance": PlanRequest("depthwise_conv_wrapper_s8", DW)},
                             desc, "cortex-m4") == _pool()


def test_a_shape_the_planner_refuses_fails_at_generation(fixture_planner) -> None:
    request = PlanRequest("depthwise_conv_wrapper_s8", dict(DW, output_dims=(2, 12, 12, 8)))
    with pytest.raises(ConformanceError, match="refuses the layer on mve"):
        conformance.apply(_pool(), {"name": "x", "conformance": request}, {"name": "x", "conformance": True},
                          "cortex-m55")


def test_unsupported_and_stray_requests_fail_closed(fixture_planner) -> None:
    with pytest.raises(ConformanceError, match="conformance is not supported for this case"):
        conformance.apply(_pool(), {"name": "x", "kernel_fn": "arm_convolve_wrapper_s8"},
                          {"name": "x", "conformance": True}, "cortex-m55")
    with pytest.raises(ConformanceError, match="not a conformance case"):
        conformance.apply(_pool(), {"name": "x", "conformance": PlanRequest("pool_s8", {})}, {"name": "x"},
                          "cortex-m55")
    assert conformance.apply(_pool(), {"name": "x"}, {"name": "x"}, "cortex-m55") == _pool()


def test_each_build_asserts_only_in_its_own_compilation(fixture_planner) -> None:
    request = PlanRequest("depthwise_conv_wrapper_s8", DW, selector="sel()")
    pool = conformance.apply(_pool(), {"name": "x", "conformance": request}, {"name": "x", "conformance": True},
                             "cortex-m55")
    guards = re.findall(r"#if (.+)", pool.extra_checks)
    assert guards == ["defined(ARM_MATH_MVEI) && !defined(HELIA_CMSIS_NN_INT_AUTOVECTORIZE)",
                      "defined(ARM_MATH_MVEI) && defined(HELIA_CMSIS_NN_INT_AUTOVECTORIZE)"]
    assert pool.extra_checks.count("sel()") == 2 and pool.includes == ('"Internal/arm_nn_dispatch.h"',)
    m4 = conformance.apply(_pool(), {"name": "x", "conformance": PlanRequest("depthwise_conv_wrapper_s8", DW)},
                           {"name": "x", "conformance": True}, "cortex-m4")
    assert re.findall(r"#if (.+)", m4.extra_checks) == ["defined(ARM_MATH_DSP) && !defined(ARM_MATH_MVEI)"]
    assert m4.includes == ()
    m0 = conformance.apply(_pool(), {"name": "x", "conformance": PlanRequest("depthwise_conv_wrapper_s8", DW)},
                           {"name": "x", "conformance": True}, "cortex-m0")
    assert re.findall(r"#if (.+)", m0.extra_checks) == ["!defined(ARM_MATH_DSP) && !defined(ARM_MATH_MVEI)"]


def test_a_build_that_declines_asserts_nothing(fixture_planner) -> None:
    layer = dict(filter_dims=(13, 1, 1, 5), filter_offset=0, multipliers=[1 << 30] * 5, shifts=[-1] * 5)
    packed = dict(filter_data=bytes(65), filter_dims=(13, 1, 1, 5), input_offset=1, filter_offset=0, bias=None,
                  multipliers=[1 << 30] * 5, shifts=[-1] * 5)
    pool = conformance.apply(_pool(), {"name": "x", "conformance": PlanRequest(
        "fully_connected_per_channel_packed_s8", layer, takes_ctx=False, packed=packed)},
        {"name": "x", "conformance": True, "autovectorize_declines": True}, "cortex-m55")
    assert re.findall(r"#if (.+)", pool.extra_checks) == [
        "defined(ARM_MATH_MVEI) && !defined(HELIA_CMSIS_NN_INT_AUTOVECTORIZE)"]
    assert "x_host_packed[224]" in pool.file_scope and "x_host_kernel_sum[5]" in pool.file_scope


def test_the_host_stream_is_written_as_signed_bytes() -> None:
    assert conformance._signed_bytes(bytes([0, 127, 128, 255])) == [0, 127, -128, -1]


def test_an_unknown_route_has_no_enum_to_check(fixture_planner, monkeypatch: pytest.MonkeyPatch) -> None:
    real = planner.plan
    monkeypatch.setattr(planner, "plan", lambda *a, **k: dataclasses.replace(real(*a, **k), route_name=None))
    with pytest.raises(ConformanceError, match="planned route 'dw_wrapper_s8:opt' has no route enum"):
        request = PlanRequest("depthwise_conv_wrapper_s8", DW, selector="sel()")
        conformance.apply(_pool(), {"name": "x", "conformance": request}, {"name": "x", "conformance": True},
                          "cortex-m55")


def test_the_plan_request_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        PlanRequest("pool_s8", {}).method = "x"
