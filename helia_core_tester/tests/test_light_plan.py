"""Light timing for untouched cases."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from helia_core_tester.hardware import code_graph, hardware_pipeline, score, session, session_runner, wrapper_route
from helia_core_tester.hardware.hardware_pipeline import StreamOptions


@dataclass
class _Case:
    case_id: str
    manifest: dict | None = None
    kernel_id: int = 1


@pytest.mark.parametrize(("light", "want"), [
    ({"a", "b", "d"}, {"a", "d"}),
    ({"a", "b", "c", "d"}, {"a", "b", "c", "d"}),
    ({"b", "c"}, {"b"}),
    (set(), set()),
])
def test_case_before_full_runs_full(light, want) -> None:
    assert session_runner.light_plan(["a", "b", "c", "d"], frozenset(light)) == want


def test_batches_split_where_plan_changes() -> None:
    cases = [_Case(i) for i in "abcd"]
    assert [c.case_id for c in session_runner.same_plan(cases, frozenset("ab"))] == ["a", "b"]
    assert [c.case_id for c in session_runner.same_plan(cases[2:], frozenset("ab"))] == ["c", "d"]


def test_light_plan_keeps_passes_and_cases() -> None:
    timing = {"warmups": 3, "samples": 20, "iterations_per_sample": 4, "min_cycles": 9, "max_iterations": 7}
    cases = [_Case("a", {"timing": timing}), _Case("b", {"timing": timing})]
    full = session.session_plan_for_bundles(cases, ())
    light = session.session_plan_for_bundles(cases, (), light=True)
    assert (full.warmups, full.samples, full.iterations_per_sample) == (3, 20, 4)
    assert (light.warmups, light.samples, light.iterations_per_sample) == (0, 1, 1)
    assert (light.cases, light.passes, light.min_cycles, light.max_iterations) == (full.cases, full.passes, 9, 7)


def _graph(**digests) -> dict:
    nodes = {name: {"digest": digest, "refs": refs} for name, (digest, refs) in digests.items()}
    return {"schema": code_graph.SCHEMA, "schema_version": code_graph.SCHEMA_VERSION, "nodes": nodes}


@pytest.fixture
def world(tmp_path: Path, monkeypatch):
    """wrap -> inner_a or inner_b; inner_b changed."""
    base = _graph(wrap=("w", ["inner_a", "inner_b"]), inner_a=("a", []), inner_b=("b", []))
    cand = _graph(wrap=("w", ["inner_a", "inner_b"]), inner_a=("a", []), inner_b=("B", []))
    graph = tmp_path / "graph.json"
    graph.write_text(json.dumps(base))
    rows = {"ca": {"inner_symbol": "inner_a"}, "cb": {"inner_symbol": "inner_b"}, "cw": {"inner_symbol": ""}}
    golden = SimpleNamespace(rows=rows, symbol=lambda case_id: "wrap")
    monkeypatch.setattr(code_graph, "code_graph", lambda build_dir: cand)
    monkeypatch.setattr(score, "load_bundle", lambda path: golden)
    monkeypatch.setattr(wrapper_route, "build_gate", lambda build_dir: True)
    routes = {}
    # The build routes like the golden run.
    monkeypatch.setattr(wrapper_route, "inner_symbol", lambda timed, manifest, gate: routes.get(
        manifest["id"], (rows.get(manifest["id"]) or {}).get("inner_symbol") or None))
    options = StreamOptions(golden_from=tmp_path, light_graph=graph)
    cases = [_Case(i, {"id": i}) for i in ("ca", "cb", "cw", "new")]
    return options, cases, routes, graph


def test_light_cases_are_untouched(world, tmp_path) -> None:
    options, cases, routes, _ = world
    # cw times the wrapper: reaches inner_b.
    assert hardware_pipeline.light_cases(tmp_path, options, cases) == {"ca"}


def test_candidate_route_counts(world, tmp_path) -> None:
    """A case the candidate reroutes stays full."""
    options, cases, routes, _ = world
    routes["ca"] = "inner_b"
    assert hardware_pipeline.light_cases(tmp_path, options, cases) == frozenset()


@pytest.mark.parametrize("trouble", ["no_graph", "bad_graph", "no_golden"])
def test_doubt_runs_full(world, tmp_path, monkeypatch, trouble) -> None:
    options, cases, _, graph = world
    if trouble == "no_graph":
        options = StreamOptions(golden_from=tmp_path)
    elif trouble == "bad_graph":
        graph.write_text("{")
    else:
        monkeypatch.setattr(score, "load_bundle", lambda path: (_ for _ in ()).throw(OSError("gone")))
    assert hardware_pipeline.light_cases(tmp_path, options, cases) == frozenset()
