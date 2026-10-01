"""Board matrix: path keying, board fit, summary."""

from __future__ import annotations

import csv
import fcntl
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from helia_core_tester.core.config import Config
from helia_core_tester.generation.io.descriptors import load_all_descriptors
from helia_core_tester.generation.test_ops import generate_test
from helia_core_tester.hardware.hardware_pipeline import generation_lock
from helia_core_tester.hardware.session_runner import build_generated_test_case_bundles
from helia_core_tester.scripts import board_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOW = datetime(2026, 10, 1, 12, 0, 0, tzinfo=timezone.utc)


def test_staged_cases_keyed_by_board(tmp_path: Path) -> None:
    desc = next(d for d in load_all_descriptors(str(PROJECT_ROOT / "assets" / "descriptors")) if d["name"] == "abs_default_s8")
    (tmp_path / "assets").symlink_to(PROJECT_ROOT / "assets")
    generate_test(desc, str(tmp_path / "artifacts" / "generated_tests" / "int" / "cortex-m55"), seed=Config.seed)

    def staged(board_id):
        bundles, _ = build_generated_test_case_bundles(
            tmp_path, family="BasicMathFunctions", name_filter="abs_default_s8", fvp_gate="off", board_id=board_id,
        )
        (bundle,) = bundles
        return bundle.root_dir.relative_to(tmp_path / "artifacts" / "stream_cases" / "int").parts[0]

    assert staged("apollo510_evb") == "apollo510_evb"
    assert staged("apollo330mP_evb") == "apollo330mP_evb"
    # No board keeps the CPU key.
    assert staged(None) == "cortex-m55"


def test_generation_lock_excludes_second_holder(tmp_path: Path) -> None:
    lock = tmp_path / "artifacts" / "generated_tests" / ".cortex-m55.lock"
    with generation_lock(tmp_path, "cortex-m55"):
        with lock.open("w") as other, pytest.raises(BlockingIOError):
            fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with lock.open("w") as other:
        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_dwt_board_falls_back_to_default() -> None:
    legs = board_matrix.plan_legs(["apollo510_evb:11", "apollo3p_evb"], ["mve:default"], "int", NOW)
    m55, m4 = legs
    assert m55.pmu_args == ("--pmu-counters", "mve:default") and m55.note is None
    assert m55.serial_no == 11 and m55.session_id == "apollo510_evb-20261001T120000Z"
    assert m4.pmu_args == () and "DWT cycles only" in m4.note
    assert not board_matrix.runs_parallel(legs)


def test_plan_refuses_bad_input() -> None:
    with pytest.raises(ValueError, match="given twice"):
        board_matrix.plan_legs(["apollo510_evb", "apollo510_evb"], [], "int", NOW)
    with pytest.raises(ValueError):
        board_matrix.plan_legs(["no_such_board"], [], "int", NOW)
    with pytest.raises(ValueError):
        board_matrix.plan_legs(["apollo510_evb"], ["bogus:all"], "int", NOW)


def test_parallel_needs_distinct_serials() -> None:
    legs = board_matrix.plan_legs(["apollo510_evb:1", "apollo330mP_evb:2"], [], "int", NOW)
    assert board_matrix.runs_parallel(legs)
    legs = board_matrix.plan_legs(["apollo510_evb:1", "apollo330mP_evb:1"], [], "int", NOW)
    assert not board_matrix.runs_parallel(legs)


def _write_bundle(root: Path, board: str, rows: list[dict], *, rejected: tuple[str, ...] = (), boot: dict | None = None) -> Path:
    root.mkdir(parents=True)
    failed = sum(row["comparison_passed"] == "false" for row in rows)
    manifest = {
        "session_id": root.name,
        "target": {"board": board, "cpu": "cortex-m55", "pmu_tier": "armv8m"},
        "firmware_build_id": f"hct-{board}",
        "build": {"kernels": {"ref": "v7.38.0", "commit": "82786f27ffafb228f629", "root": None}},
    }
    if boot is not None:
        manifest["boot"] = boot
    (root / "session_manifest.json").write_text(json.dumps(manifest))
    (root / "session_summary.json").write_text(json.dumps({
        "case_count": len(rows), "passed_cases": len(rows) - failed, "failed_cases": failed,
        "rejected_cases": list(rejected),
    }))
    fields = ["case_id", "comparison_passed", "median_cycles", *sorted({k for row in rows for k in row} - {"case_id", "comparison_passed", "median_cycles"})]
    with (root / "case_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)
    return root


def _fixture_bundles(tmp_path: Path) -> list[Path]:
    return [
        _write_bundle(tmp_path / "a510", "apollo510_evb", [
            {"case_id": "add_s8", "comparison_passed": "true", "median_cycles": "100.5", "ARM_PMU_MVE_INST_RETIRED": "40"},
            {"case_id": "only_510", "comparison_passed": "true", "median_cycles": "9"},
        ], boot={"status": 0, "core_clock_hz": 250_000_000}),
        _write_bundle(tmp_path / "a3p", "apollo3p_evb", [
            {"case_id": "add_s8", "comparison_passed": "false", "median_cycles": "300"},
            {"case_id": "conv_s8", "comparison_passed": "false", "median_cycles": "0"},
        ], rejected=("conv_s8",)),
    ]


def test_summary_from_fixture_bundles(tmp_path: Path) -> None:
    out = tmp_path / "out"
    status = board_matrix.main(["summarize", *map(str, _fixture_bundles(tmp_path)), "--out", str(out)])
    assert status == 1

    summary = json.loads((out / "board_matrix.json").read_text())
    assert summary["schema"] == "hct.hardware.board_matrix" and summary["schema_version"] == 1
    a510, a3p = summary["boards"]
    assert a510["golden"] == {"total": 2, "passed": 2, "failed": 0, "rejected": 0}
    assert a510["boot"] == {"status": 0, "core_clock_hz": 250_000_000}
    assert a3p["golden"] == {"total": 2, "passed": 0, "failed": 1, "rejected": 1}
    assert a3p["status"] == "failed" and a3p["boot"] == {"status": None, "core_clock_hz": None}
    assert summary["cases"] == [{
        "case_id": "add_s8",
        "median_cycles": {"apollo510_evb": 100.5, "apollo3p_evb": 300.0},
        "ARM_PMU_MVE_INST_RETIRED": {"apollo510_evb": 40.0, "apollo3p_evb": None},
    }]

    markdown = (out / "board_matrix.md").read_text()
    assert "| apollo510_evb | passed | 2 | 0 | 0 | 0 | 250 | hct-apollo510_evb | v7.38.0@82786f27ffaf |  |" in markdown
    assert "| add_s8 | 100.5 | 300 |" in markdown
    assert "| add_s8 | 40 | - |" in markdown


def test_summary_refuses_bad_bundles(tmp_path: Path) -> None:
    a = _write_bundle(tmp_path / "a", "apollo510_evb", [])
    b = _write_bundle(tmp_path / "b", "apollo510_evb", [])
    assert board_matrix.main(["summarize", str(a), str(b), "--out", str(tmp_path / "out")]) == 2
    assert board_matrix.main(["summarize", str(tmp_path), "--out", str(tmp_path / "out")]) == 2


def test_run_writes_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    commands = []

    def fake_run(command, cwd, stdout, stderr, env):
        assert "HPX_JLINK_SERIAL" not in env
        commands.append(command)
        board = command[command.index("--board") + 1]
        session = command[command.index("--session-id") + 1]
        if board == "apollo510_evb":
            _write_bundle(tmp_path / "artifacts" / "reports" / "hardware" / session, board, [
                {"case_id": "add_s8", "comparison_passed": "true", "median_cycles": "10"},
            ])
            return type("Done", (), {"returncode": 0})()
        return type("Done", (), {"returncode": 1})()

    monkeypatch.setenv("HPX_JLINK_SERIAL", "1")
    monkeypatch.setattr(board_matrix, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(board_matrix.subprocess, "run", fake_run)
    out = tmp_path / "out"
    status = board_matrix.main([
        "run", "--board", "apollo510_evb:1", "--board", "apollo3p_evb:2", "--limit", "2",
        "--pmu-counters", "mve:default", "--out", str(out), "--", "--fvp-gate", "off",
    ])
    assert status == 1
    m55, m4 = commands
    assert m55[-4:] == ["--pmu-counters", "mve:default", "--serial-no", "1"]
    assert "--pmu-counters" not in m4 and m4[-4:] == ["--fvp-gate", "off", "--serial-no", "2"]
    assert "--limit" in m55 and "2" in m55

    summary = json.loads((out / "board_matrix.json").read_text())
    a510, a3p = summary["boards"]
    assert a510["status"] == "passed" and a510["exit_code"] == 0
    assert a3p["status"] == "error" and a3p["bundle"] is None and "DWT" in a3p["note"]
    # Same keys on every row.
    assert a3p.keys() == a510.keys() and a3p["golden"] is None
    assert summary["selection"]["pmu_counters"] == ["mve:default"]
    assert [case["case_id"] for case in summary["cases"]] == ["add_s8"]
    # Errored boards stay in every metric map.
    assert summary["cases"][0]["median_cycles"] == {"apollo510_evb": 10.0, "apollo3p_evb": None}
    assert summary["cases"][0]["ARM_PMU_MVE_INST_RETIRED"] == {"apollo510_evb": None, "apollo3p_evb": None}
    assert (out / "logs" / "apollo3p_evb.log").is_file()


@pytest.mark.parametrize("extra", [["--board", "apollo3p_evb"], ["--session-id=x"], ["--serial-no", "9"], ["--pmu-counters", "cpu:all"], ["--build-dir", "b"], ["--build-dir=b"]])
def test_run_rejects_owned_options(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra: list[str]) -> None:
    monkeypatch.setattr(board_matrix, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(board_matrix.subprocess, "run", lambda *a, **k: pytest.fail("launched a leg"))
    assert board_matrix.main(["run", "--board", "apollo510_evb:1", "--out", str(tmp_path / "out"), "--", *extra]) == 2


def _break(bundle: Path, name: str, text: str) -> Path:
    (bundle / name).write_text(text)
    return bundle


@pytest.mark.parametrize("name,text", [
    ("session_summary.json", "{}"),
    ("session_summary.json", "{not json"),
    ("session_manifest.json", "[]"),
    ("case_summary.csv", ""),
    ("case_summary.csv", "median_cycles\n1\n"),
    ("session_summary.json", '{"case_count": 1, "passed_cases": 1, "failed_cases": 0, "rejected_cases": null}'),
    ("session_summary.json", '{"case_count": 1, "passed_cases": true, "failed_cases": 0, "rejected_cases": []}'),
    ("session_summary.json", '{"case_count": 1, "passed_cases": 1, "failed_cases": 1, "rejected_cases": []}'),
    ("session_summary.json", '{"case_count": 1, "passed_cases": 1, "failed_cases": 0, "rejected_cases": ["x"]}'),
    ("session_manifest.json", '{"target": {"board": "apollo510_evb"}, "boot": "up"}'),
    ("session_manifest.json", '{"target": {"board": "apollo510_evb"}, "build": {"kernels": "bad"}}'),
    ("case_summary.csv", "case_id,comparison_passed,median_cycles\nadd_s8,true,fast\n"),
    ("case_summary.csv", "case_id,comparison_passed,median_cycles\nadd_s8,true,nan\n"),
    ("case_summary.csv", "case_id,comparison_passed,median_cycles,ARM_PMU_MVE_INST_RETIRED\nadd_s8,true,1,inf\n"),
    ("session_manifest.json", '{"target": {"board": "apollo510_evb"}, "boot": {"core_clock_hz": NaN}}'),
])
def test_summarize_rejects_broken_bundle(tmp_path: Path, name: str, text: str) -> None:
    bundle = _break(_write_bundle(tmp_path / "b", "apollo510_evb", [
        {"case_id": "add_s8", "comparison_passed": "true", "median_cycles": "1"},
    ]), name, text)
    assert board_matrix.main(["summarize", str(bundle), "--out", str(tmp_path / "out")]) == 2


def test_run_reports_broken_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(command, cwd, stdout, stderr, env):
        session = command[command.index("--session-id") + 1]
        bundle = _write_bundle(tmp_path / "artifacts" / "reports" / "hardware" / session, "apollo510_evb", [])
        _break(bundle, "session_summary.json", "{}")
        return type("Done", (), {"returncode": 0})()

    monkeypatch.setattr(board_matrix, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(board_matrix.subprocess, "run", fake_run)
    out = tmp_path / "out"
    assert board_matrix.main(["run", "--board", "apollo510_evb", "--out", str(out)]) == 1
    (row,) = json.loads((out / "board_matrix.json").read_text())["boards"]
    assert row["status"] == "error" and row["bundle"] is None and "missing case_count" in row["note"]
    assert row.keys() == board_matrix.empty_entry("x").keys()


def test_run_survives_launch_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args, **kwargs):
        raise FileNotFoundError("no python")

    monkeypatch.setattr(board_matrix, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(board_matrix.subprocess, "run", fake_run)
    out = tmp_path / "out"
    assert board_matrix.main(["run", "--board", "apollo510_evb", "--out", str(out)]) == 1
    (row,) = json.loads((out / "board_matrix.json").read_text())["boards"]
    assert row["status"] == "error" and row["exit_code"] == 1
    assert "launch failed: no python" in (out / "logs" / "apollo510_evb.log").read_text()
