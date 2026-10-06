"""Hidden cases join a hardware run."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from helia_core_tester.hardware import hardware_pipeline
from helia_core_tester.hardware.boards import resolve_board
from helia_core_tester.hardware.generated_test_bridge import discover_generated_tests
from helia_core_tester.hardware.hardware_pipeline import (
    HiddenSetError, StreamOptions, prepare_bundles, resolved_selection, run_hardware_pipeline,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BOARD = resolve_board("apollo510_evb")
COMMITMENT = "ab" * 32


def _hidden(root: Path, names=("h0123456789ab",), cpu: str = "cortex-m55") -> Path:
    """A minimal hidden root."""
    summary = root / "artifacts" / "random_shapes" / cpu / "summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(json.dumps({"seed_commitment": COMMITMENT, "cpu": cpu, "cases": len(names)}))
    for name in names:
        case = root / "artifacts" / "generated_tests" / "int" / cpu / "ConvolutionFunctions" / name
        case.mkdir(parents=True)
        (case / "descriptor.yaml").write_text(yaml.safe_dump({"name": name, "operator": "Convolve", "activation_dtype": "S8", "weight_dtype": "S8"}))
    return root


def _fake_bridge(monkeypatch, *, hidden_skips: int = 0) -> list:
    from helia_core_tester.hardware.case_bundle import CaseBundle

    calls = []

    def _build(project_root, *, tests_root=None, **kwargs):
        calls.append((tests_root, kwargs.get("fvp_gate")))
        name = "hidden" if tests_root else "public"
        bundle = CaseBundle(Path("."), Path("m.json"), {"case_id": name}, ())
        return [bundle], [(SimpleNamespace(name="x"), "why")] * (hidden_skips if tests_root else 0)

    monkeypatch.setattr("helia_core_tester.hardware.session_runner.build_generated_test_case_bundles", _build)
    return calls


def test_discover_reads_another_root(tmp_path: Path) -> None:
    root = _hidden(tmp_path / "h")
    found = discover_generated_tests(PROJECT_ROOT, tests_root=root)
    assert [c.name for c in found] == ["h0123456789ab"] and found[0].directory.is_relative_to(root)


def test_hidden_cases_join_and_are_marked(tmp_path: Path, monkeypatch) -> None:
    calls = _fake_bridge(monkeypatch)
    root = _hidden(tmp_path / "h")
    bundles, _ = prepare_bundles(PROJECT_ROOT, BOARD, StreamOptions(hidden_set=root, fvp_gate="strict"))
    assert calls == [(None, "strict"), (root, "off")]
    assert [(b.case_id, b.manifest.get("hidden")) for b in bundles] == [("public", None), ("hidden", True)]


def test_partial_hidden_set_refused(tmp_path: Path, monkeypatch) -> None:
    _fake_bridge(monkeypatch, hidden_skips=1)
    with pytest.raises(HiddenSetError, match="1 hidden case"):
        prepare_bundles(PROJECT_ROOT, BOARD, StreamOptions(hidden_set=_hidden(tmp_path / "h")))


def test_other_cpu_refused_before_board(tmp_path: Path, monkeypatch) -> None:
    root = _hidden(tmp_path / "h", cpu="cortex-m4")
    monkeypatch.setattr(hardware_pipeline, "flash_firmware", lambda *a, **k: pytest.fail("flashed"))
    with pytest.raises(HiddenSetError, match="No cortex-m55 hidden set"):
        run_hardware_pipeline(
            PROJECT_ROOT, BOARD, 1, options=StreamOptions(hidden_set=root), build_dir=tmp_path,
            skip_generate=True, skip_flash=True, echo=lambda _: None,
        )


def test_selection_records_commitment_only(tmp_path: Path) -> None:
    root = _hidden(tmp_path / "h", names=("h0123456789ab", "hba9876543210"))
    selection = resolved_selection(PROJECT_ROOT, BOARD, StreamOptions(hidden_set=root))
    assert selection["hidden_set"] == {"seed_commitment": COMMITMENT, "cases": 2}
    assert "hidden_set" not in resolved_selection(PROJECT_ROOT, BOARD, StreamOptions())


def test_case_summary_flags_hidden(tmp_path: Path) -> None:
    from helia_core_tester.hardware.case_bundle import build_abs_s8_case_bundle, hidden_bundle, load_case_bundle
    from helia_core_tester.hardware.fake_target import FakeTargetTransport
    from helia_core_tester.hardware.measurement import counter_passes_for_selection
    from helia_core_tester.hardware.result_bundle import write_result_bundle
    from helia_core_tester.hardware.session import HostSession

    def _load(case_id: str):
        built = build_abs_s8_case_bundle(PROJECT_ROOT, output_root=tmp_path, case_id=case_id)
        return load_case_bundle(built.manifest_path)

    public, hidden = _load("abs_public"), _load("abs_hidden")
    hidden = hidden_bundle(hidden)
    passes = counter_passes_for_selection({"cpu": "default"})
    result = HostSession(FakeTargetTransport(), counter_passes=passes).run_many([public, hidden])
    root = write_result_bundle(result, session_id="s", output_root=tmp_path, memory_report={}, kernel_catalog={})
    with (root / "case_summary.csv").open(encoding="utf-8", newline="") as handle:
        rows = {row["case_id"]: row["hidden"] for row in csv.DictReader(handle)}
    assert rows == {"abs_public": "false", "abs_hidden": "true"}
    cases = json.loads((root / "cases.json").read_text())
    assert [c["hidden"] for c in cases] == [False, True]
