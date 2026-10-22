"""contract/planner.py: the checkout's host planner, as generation asks it."""

from __future__ import annotations

import dataclasses
import shutil
from pathlib import Path

import pytest

from helia_core_tester.contract import planner
from helia_core_tester.contract.planner import PlannerUnavailable, PlanRefusedError
from helia_core_tester.tests.contract_fallback import FIXTURE_ROOT
from helia_core_tester.tests.contract_required import P1E_CALLS, P1E_PLAN_FIELDS, P1dPlan, require_planner

DW = dict(input_dims=(1, 8, 8, 16), filter_dims=(1, 3, 3, 16), output_dims=(1, 8, 8, 16), ch_mult=1, padding=(1, 1))


@pytest.fixture
def fixture_planner(monkeypatch: pytest.MonkeyPatch):
    from helia_core_tester.contract import render

    monkeypatch.setattr(render, "resolve_cmsis_nn_root", lambda: FIXTURE_ROOT)
    return planner.load_planner(FIXTURE_ROOT)


@pytest.mark.parametrize("cpu, builds", [
    ("cortex-m55", [("mve", False), ("mve", True)]),
    ("cortex-m55-dsp", [("dsp", False)]),
    ("cortex-m4", [("dsp", False)]),
    ("cortex-m0", [("scalar", False)]),
    (None, [("scalar", False), ("dsp", False), ("mve", False), ("mve", True)]),
])
def test_each_cpu_is_planned_for_the_builds_its_cases_compile_under(fixture_planner, cpu, builds) -> None:
    package, _ = fixture_planner
    assert [(t.isa, t.autovectorize) for t in planner.build_targets(package, cpu)] == builds


def test_the_planned_capacity_is_the_largest_build(fixture_planner) -> None:
    package, _ = fixture_planner
    per_build = {t.isa: planner.plan("depthwise_conv_wrapper_s8", t, **DW).ctx_bytes
                 for t in planner.build_targets(package, None)}
    assert planner.planned_ctx_bytes("depthwise_conv_wrapper_s8", None, **DW) == max(per_build.values())
    assert planner.planned_ctx_bytes("depthwise_conv_wrapper_s8", "cortex-m55", **DW) == per_build["mve"] > 0
    # Without MVE a 3x3 filter with pad 1 runs arm_depthwise_conv_3x3_s8, which takes no scratch.
    assert planner.planned_ctx_bytes("depthwise_conv_wrapper_s8", "cortex-m4", **DW) == per_build["dsp"] == 0


def test_a_build_that_declines_needs_nothing_and_every_build_declining_fails(fixture_planner) -> None:
    layer = dict(filter_dims=(13, 1, 1, 5), filter_offset=0, multipliers=[1 << 30] * 5, shifts=[-1] * 5)
    assert planner.planned_ctx_bytes("fully_connected_per_channel_packed_s8", "cortex-m55", **layer) == 0
    with pytest.raises(PlanRefusedError, match="every build of cortex-m4 declines") as error:
        planner.planned_ctx_bytes("fully_connected_per_channel_packed_s8", "cortex-m4", **layer)
    assert error.value.status == "unsupported"


@pytest.mark.parametrize("layer, status", [
    (dict(DW, output_dims=(2, 8, 8, 16)), "shape"),
    (dict(DW, input_dims=(1, 8, 8)), "arguments"),
    (dict(DW, input_dims=(1, 8, 8, 2**31)), "arguments"),
])
def test_a_refused_layer_is_a_typed_contract_error(fixture_planner, layer, status: str) -> None:
    package, _ = fixture_planner
    with pytest.raises(PlanRefusedError) as error:
        planner.plan("depthwise_conv_wrapper_s8", package.Target("mve"), **layer)
    assert error.value.status == status
    with pytest.raises(PlanRefusedError):
        planner.planned_ctx_bytes("depthwise_conv_wrapper_s8", "cortex-m55", **layer)


def test_vector_sum_runs_the_library(fixture_planner) -> None:
    assert planner.vector_sum_s8(bytes([1, 2, 3, 0xFF, 0xFE, 0xFD]), cols=3, rows=2, lhs_offset=2, rhs_offset=0,
                                 bias=[10, 20]) == [22, 8]
    with pytest.raises(PlanRefusedError, match="cols and rows must be positive"):
        planner.vector_sum_s8(b"", cols=0, rows=1, lhs_offset=0, rhs_offset=0)
    with pytest.raises(PlanRefusedError, match="data holds 2 values"):
        planner.vector_sum_s8(b"\x01\x02", cols=3, rows=1, lhs_offset=0, rhs_offset=0)


def test_the_packed_stream_size_is_the_packers(fixture_planner) -> None:
    assert planner.fc_packed_stream_bytes(bytes(13 * 5), filter_dims=(13, 1, 1, 5), multipliers=[1 << 30] * 5,
                                          shifts=[-1] * 5) == 2 * (4 * 16 + 48)
    with pytest.raises(PlanRefusedError, match="returned"):
        # A left shift is outside the packed path's quantization.
        planner.fc_packed_stream_bytes(bytes(13 * 5), filter_dims=(13, 1, 1, 5), multipliers=[1 << 30] * 5,
                                       shifts=[1] * 5)


def test_one_planner_per_checkout(fixture_planner) -> None:
    assert planner.load_planner(FIXTURE_ROOT) is fixture_planner


def test_a_checkout_without_a_contract_has_no_planner(tmp_path: Path) -> None:
    (tmp_path / "Include").mkdir()
    for root in (tmp_path, None):
        with pytest.raises(PlannerUnavailable, match="has no kernel contract, so it has no planner"):
            planner.load_planner(root)


def test_a_planner_that_cannot_build_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "ns-cmsis-nn"
    shutil.copytree(FIXTURE_ROOT, root)
    (root / "Planner" / "sources.txt").write_text("Planner/src/missing.c\n")
    planner.clear_planner_cache()
    with pytest.raises(PlannerUnavailable, match="cannot be built or loaded"):
        planner.load_planner(root)


@pytest.mark.parametrize("layer", [
    dict(DW, input_dims=(1, 65536, 65536, 1), filter_dims=(1, 1, 1, 1), output_dims=(1, 65536, 65536, 1),
         padding=(0, 0)),
    dict(DW, input_dims=(1, 1, 2**30, 4), filter_dims=(1, 1, 1, 4), output_dims=(1, 1, 2**30, 4), padding=(0, 0)),
])
def test_overflowing_dims_are_refused_not_truncated(fixture_planner, layer) -> None:
    package, _ = fixture_planner
    for target in planner.build_targets(package, None):
        with pytest.raises(PlanRefusedError) as error:
            planner.plan("depthwise_conv_wrapper_s8", target, **layer)
        assert error.value.status == "range", target


DW_3X3 = dict(input_dims=(1, 5, 7, 16), filter_dims=(1, 3, 3, 16), output_dims=(1, 5, 7, 16),
              dw_conv_params={"ch_mult": 1, "padding": (1, 1)})


def test_the_3x3_entries_scratch_is_their_size_query_on_every_build(fixture_planner) -> None:
    package, _ = fixture_planner
    # 16 groups of (3 x 52 + 32) bytes, a pad row of W x C bytes and 16 for alignment, alike on every build.
    want = 16 * (3 * 52 + 32) + 7 * 16 + 16
    for record in ("depthwise_conv_s8_opt_3x3", "depthwise_conv_s8_opt_3x3_c64_s1"):
        for target in planner.build_targets(package, None):
            assert planner.buffer_bytes(target, record, "scratch", **DW_3X3) == want
        assert planner.planned_buffer_bytes(record, "scratch", "cortex-m4", **DW_3X3) == want


def test_the_opt_entries_scratch_is_the_largest_build(fixture_planner) -> None:
    package, _ = fixture_planner
    per_build = {t.isa: planner.buffer_bytes(t, "depthwise_conv_s8_opt_channelwise", "scratch", **DW_3X3)
                 for t in planner.build_targets(package, None)}
    # MVE: 4 x 124 x 3 x 3 bytes; DSP: C x 3 x 3 int16; the plain build takes none.
    assert per_build == {"mve": 4 * 124 * 9, "dsp": 16 * 9 * 2, "scalar": 0}
    assert planner.planned_buffer_bytes("depthwise_conv_s8_opt_planar", "scratch", "cortex-m55", **DW_3X3) == 4464
    assert planner.planned_buffer_bytes("depthwise_conv_s8_opt_planar", "scratch", None, **DW_3X3) == 4464
    assert planner.planned_buffer_bytes("depthwise_conv_s8_opt_planar", "weight_sums", "cortex-m55", **DW_3X3) == 64
    assert planner.planned_buffer_bytes("depthwise_conv_s8_opt_planar", "weight_sums", "cortex-m4", **DW_3X3) == 0


@pytest.mark.parametrize("args, layer, status, match", [
    (("no_such_record", "scratch"), DW_3X3, "arguments", "no such implementation"),
    (("depthwise_conv_s8_opt_3x3", "no_such_buffer"), DW_3X3, "arguments", "no_such_buffer"),
    (("depthwise_conv_s8", "ctx"), DW_3X3, "arguments", "no size query"),
    (("depthwise_conv_s8_opt", "scratch"), dict(input_dims=(1, 4, 4, 8)), "arguments", "no \\['filter_dims'\\] given"),
    (("depthwise_conv_s8_opt_3x3", "scratch"), dict(input_dims=(1, 4, 4)), "arguments", "input_dims"),
    (("depthwise_conv_s8_opt_3x3", "scratch"), dict(input_dims=(1, 1, 1, 2**31 - 100)), "range", "refused"),
    (("depthwise_conv_s8_opt_3x3", "scratch"), dict(input_dims=(1, 0, 4, 8)), "shape", "refused"),
])
def test_a_refused_buffer_is_a_typed_contract_error(fixture_planner, args, layer, status: str, match: str) -> None:
    package, _ = fixture_planner
    with pytest.raises(PlanRefusedError, match=match) as error:
        planner.buffer_bytes(package.Target("mve"), *args, **layer)
    assert error.value.status == status


def test_the_opt_router_is_planned_against_ctx_size(fixture_planner) -> None:
    package, _ = fixture_planner
    layer = dict(input_dims=(1, 10, 10, 8), filter_dims=(1, 3, 3, 8), output_dims=(1, 8, 8, 8))
    # The plane is 10 x 10 + 32 bytes.
    mve = package.Target("mve")
    assert planner.plan("depthwise_conv_s8_opt", mve, ctx_size=132, **layer).route == "dw_opt_s8:planar"
    assert planner.plan("depthwise_conv_s8_opt", mve, ctx_size=0, **layer).route == "dw_opt_s8:channelwise"
    with pytest.raises(PlanRefusedError) as error:
        planner.plan("depthwise_conv_s8_opt", mve, ctx_size=131, **layer)
    assert error.value.status == "range"
    assert planner.plan("depthwise_conv_s8_opt", package.Target("dsp"), ctx_size=0, **layer).kernel == (
        "arm_depthwise_conv_s8_opt_channelwise")


def test_a_planner_without_the_call_is_unavailable(fixture_planner, monkeypatch: pytest.MonkeyPatch) -> None:
    package, loaded = fixture_planner
    monkeypatch.setattr(type(loaded), "buffer_bytes", property(lambda self: None), raising=False)
    with pytest.raises(PlannerUnavailable, match="has no buffer_bytes"):
        planner.buffer_bytes(package.Target("mve"), "depthwise_conv_s8_opt_3x3", "scratch", **DW_3X3)
    monkeypatch.setattr(type(loaded), "depthwise_conv_s8_opt", property(lambda self: None), raising=False)
    with pytest.raises(PlannerUnavailable, match="has no depthwise_conv_s8_opt"):
        planner.plan("depthwise_conv_s8_opt", package.Target("mve"), ctx_size=0, input_dims=(1, 8, 8, 8),
                     filter_dims=(1, 3, 3, 8), output_dims=(1, 6, 6, 8))


@pytest.fixture
def no_contract_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from helia_core_tester.contract import render

    (tmp_path / "Include").mkdir()
    (tmp_path / "Include" / "arm_nnfunctions.h").write_text("/* v7.35.1: no export, no package */\n")
    monkeypatch.setattr(render, "resolve_cmsis_nn_root", lambda: tmp_path)
    monkeypatch.delenv("HELIA_CORE_TESTER_REQUIRE_CONTRACT", raising=False)
    return tmp_path


@pytest.mark.parametrize("methods", [(), P1E_CALLS])
def test_a_tree_without_the_package_skips_planner_tests_with_the_reason(no_contract_tree: Path, methods) -> None:
    with pytest.raises(pytest.skip.Exception, match="needs the checkout's planner: .*has no kernel contract"):
        require_planner(*methods)
    with pytest.raises(PlannerUnavailable):
        planner.current_planner()


def test_a_tree_without_the_package_fails_planner_tests_where_the_leg_requires_one(
    no_contract_tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HELIA_CORE_TESTER_REQUIRE_CONTRACT", "1")
    with pytest.raises(pytest.fail.Exception, match="HELIA_CORE_TESTER_REQUIRE_CONTRACT is set but .*no kernel contract"):
        require_planner(*P1E_CALLS)


def test_a_planner_without_the_p1e_calls_skips_naming_them(fixture_planner, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HELIA_CORE_TESTER_REQUIRE_CONTRACT", raising=False)
    require_planner(*P1E_CALLS)
    _, loaded = fixture_planner
    monkeypatch.setattr(type(loaded), "buffer_bytes", property(lambda self: None), raising=False)
    with pytest.raises(pytest.skip.Exception, match="planner has no buffer_bytes$"):
        require_planner(*P1E_CALLS)
    require_planner("depthwise_conv_s8_opt")


def test_a_package_whose_plans_name_no_route_skips_naming_the_fields(
    fixture_planner, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HELIA_CORE_TESTER_REQUIRE_CONTRACT", raising=False)
    package, _ = fixture_planner
    require_planner(plan_fields=P1E_PLAN_FIELDS)
    monkeypatch.setattr(package, "Plan", P1dPlan)
    with pytest.raises(pytest.skip.Exception, match="planner has no Plan.route_value, Plan.route_name$"):
        require_planner(plan_fields=P1E_PLAN_FIELDS)
    require_planner(*P1E_CALLS)
    monkeypatch.setenv("HELIA_CORE_TESTER_REQUIRE_CONTRACT", "1")
    with pytest.raises(pytest.fail.Exception, match="REQUIRE_CONTRACT is set but .*has no Plan.route_value"):
        require_planner(plan_fields=P1E_PLAN_FIELDS)


def test_a_plan_without_a_route_name_is_unavailable_not_guessed(fixture_planner) -> None:
    package, _ = fixture_planner
    plan = planner.plan("depthwise_conv_wrapper_s8", package.Target("mve"), **DW)
    assert planner.route_name(plan) == plan.route_name and plan.route_name.startswith("ARM_NN_DW_CONV_S8_ROUTE_")
    old = P1dPlan(**{f.name: getattr(plan, f.name) for f in dataclasses.fields(P1dPlan)})
    with pytest.raises(PlannerUnavailable, match=r"names no route constant \(Plan.route_name\).*P1e"):
        planner.route_name(old)
    direct = planner.plan("fully_connected_per_channel_packed_s8", package.Target("mve"), filter_dims=(16, 1, 1, 4),
                          filter_offset=0, multipliers=[1 << 30] * 4, shifts=[-1] * 4)
    assert direct.route is None and planner.route_name(direct) is None
