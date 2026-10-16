"""Hidden s8 TransposeConv shapes."""

from __future__ import annotations

import hashlib
from collections import Counter

import pytest
import yaml

import helia_core_tester.generation.random_shapes as rs

TC8 = ("TransposeConv", "S8")
SECRET = "tc8-test-secret-0123456789"


def _layer(case: dict) -> rs.Layer:
    kh, kw, cout, cin = case["filter_shape"]
    n, h, w, _ = case["input_shape"]
    return rs.Layer(h, w, cin, cout, kh, kw, *case["strides"], padding=case["padding"], transpose=True, n=n)


@pytest.mark.parametrize(("sh", "sw", "cin", "route"), [
    (1, 1, 17, rs.TC8_REVERSE), (2, 2, 64, rs.TC8_REVERSE), (2, 2, 16, rs.TC8_ROLLING),
    (3, 1, 64, rs.TC8_ROLLING), (1, 3, 17, rs.TC8_ROLLING), (1, 1, 1, rs.TC8_ROLLING),
])
def test_route_mirrors_wrapper(sh: int, sw: int, cin: int, route: str) -> None:
    assert rs.tc8_route(rs.Layer(4, 4, cin, 8, 3, 3, sh, sw, transpose=True)) == route


@pytest.mark.parametrize("cpu", ["cortex-m55", "cortex-m4"])
@pytest.mark.parametrize("seed", (0, 1, 7, 12345))
def test_tc8_shapes_fit_the_board(cpu: str, seed: int) -> None:
    workspace = rs.min_workspace(cpu)
    for case in rs.sample_cases(30, seed, cpu, (TC8,)):
        layer = _layer(case)
        assert case["name"].startswith(f"rs{seed}_tc8_") and case["expected_route"] == rs.tc8_route(layer)
        assert "required_kernel_symbols" not in case and case["weight_dtype"] == "S8"
        assert rs.footprint("TransposeConv", layer, "S8") <= workspace
        assert rs.layer_macs("TransposeConv", layer) <= rs.MAX_MACS


@pytest.mark.parametrize("seed", range(4))
def test_mixed_draws_split_routes(seed: int) -> None:
    mixed = [c for c in rs.sample_cases(40, seed, ops=(TC8,)) if "edge" not in c]
    routes = Counter(c["expected_route"] for c in mixed)
    assert routes[rs.TC8_REVERSE] == routes[rs.TC8_ROLLING]


@pytest.mark.parametrize(("n", "edges"), [(1, 1), (3, 2), (6, 4), (12, 8), (14, 9), (24, 9)])
def test_edge_share(n: int, edges: int) -> None:
    plan = rs._tc8_plan(n)
    assert len(plan) == n and plan[n - edges:] == [rs.TC8_EDGES[i % len(rs.TC8_EDGES)] for i in range(edges)]
    assert set(plan[:n - edges]) <= set(rs.TC8_ROUTES)


@pytest.mark.parametrize("seed", range(8))
def test_edge_categories_hold(seed: int) -> None:
    cases = [c for c in rs.sample_cases(24, seed, ops=(TC8,)) if "edge" in c]
    assert [c["edge"] for c in cases] == list(rs.TC8_EDGES)
    for case in cases:
        n, _, _, cin = case["input_shape"]
        kh, kw, cout, _ = case["filter_shape"]
        sh, sw = case["strides"]
        edge = case["edge"]
        assert (n == 2) == (edge == "batch2")
        if edge == "threshold":
            assert cin in (rs.TC8_REVERSE_MIN_CIN - 1, rs.TC8_REVERSE_MIN_CIN) and max(sh, sw) <= 2
        elif edge == "stride3":
            assert cin >= rs.TC8_REVERSE_MIN_CIN and 3 in (sh, sw)
            assert case["expected_route"] == rs.TC8_ROLLING
        elif edge == "decoder":
            assert (sh, sw) == (2, 2) and kh == kw and 2 <= kh <= 4
        elif edge == "clamp":
            assert -128 <= case["activation_min"] < 0 < case["activation_max"] <= 127
            assert case["activation"] == "NONE"
        elif edge == "gaps":
            assert kw < sw and kh <= sh
        elif edge == "overlap":
            assert min(kh, kw) >= 5 and max(sh, sw) <= 2
        elif edge == "odd_tail":
            assert cout % 2
        elif edge == "unit_tail":
            assert cin == cout == 1 and case["activation"] == "NONE"


@pytest.mark.parametrize("seed", range(6))
def test_fused_relu_has_no_clamp(seed: int) -> None:
    # The kernel clamp would override an unfused ReLU.
    for case in rs.sample_cases(40, seed, ops=(TC8,)):
        if case["activation"] != "NONE":
            assert "activation_min" not in case and "activation_max" not in case


@pytest.mark.parametrize("cpu", ["cortex-m55", "cortex-m4"])
def test_tc8_hidden_set_pinned(cpu: str) -> None:
    # Pinned when tc8 joined; no MVE routes.
    cases = rs.hidden_cases(8, SECRET.encode(), cpu, (TC8,))
    digest = hashlib.sha256(yaml.safe_dump_all(cases, sort_keys=False).encode()).hexdigest()
    assert digest == "2eaa99bfa58e6b1e5a1bba84fb8e98b64954160604dea17f7e7ee299088b80e5"
