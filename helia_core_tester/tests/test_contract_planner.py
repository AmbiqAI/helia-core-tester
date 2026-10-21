"""contract/planner.py: the checkout's host planner, as generation asks it."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from helia_core_tester.contract import planner
from helia_core_tester.contract.planner import PlannerUnavailable, PlanRefusedError
from helia_core_tester.tests.contract_fallback import FIXTURE_ROOT

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
