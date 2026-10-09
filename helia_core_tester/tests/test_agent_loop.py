from __future__ import annotations

import dataclasses
import json
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

from helia_core_tester.agent_loop import agent, judge, ledger
from helia_core_tester.agent_loop.config import Campaign, ConfigError, from_json, parse_campaign
from helia_core_tester.agent_loop.prompt import render_prompt, route_facts
from helia_core_tester.agent_loop.workspace import Workspace
from helia_core_tester.hardware.toolchain import toolchain_spec

BASE = {
    "name": "dw-s8", "board": "apollo330mP_evb", "target": {"op": "DepthwiseConv", "dtype": "S8"},
    "kernels": {"repo": "nn", "ref": "v7.40.0"}, "secrets_dir": "/secret/dw",
}


def _campaign(**over) -> Campaign:
    data = json.loads(json.dumps(BASE))
    for key, value in over.items():
        if key in ("op", "dtype", "case_ids"):
            data["target"][key] = value
        else:
            data[key] = value
    return parse_campaign(data, Path("/cfg"))


# --- config -----------------------------------------------------------------------------


def test_config_defaults_and_paths() -> None:
    c = _campaign()
    assert c.legs == ("tcm", "mram") and c.evals == 12 and c.repeats == 3 and c.hidden_shapes == 12
    assert c.kernels_repo == Path("/cfg/nn") and c.secrets_dir == Path("/secret/dw")
    assert c.bench_id == "apollo330mP_evb"
    assert from_json(json.loads(json.dumps(c.to_json()))) == c


@pytest.mark.parametrize("over, message", [
    ({"colour": "red"}, "unknown key"),
    ({"name": "Bad Name"}, "name"),
    ({"board": "nope"}, "board"),
    ({"legs": ["tcm", "tcm"]}, "legs"),
    ({"legs": ["sram"]}, "legs"),
    ({"legs": [{}]}, "legs"),
    ({"submit_deadline_s": 600}, "at most 570"),
    ({"board": "apollo3p_evb", "legs": ["mram"]}, "no cached MRAM"),
    ({"evals": 0}, "evals"),
    ({"evals": True}, "wrong type"),
    ({"cost_usd": -1}, "cost_usd"),
    ({"op": "FullyConnected"}, "hidden_shapes: No random shapes"),
    ({"op": "Convolve", "dtype": "S16"}, "or set 0"),
    ({"op": "Conv 2d"}, "one word"),
    ({"case_ids": ["ok", "a b"]}, "case_ids"),
    ({"size_evals": -1}, "size_evals: must be >= 0"),
    ({"size_evals": 12}, "size_evals: must be < evals"),
    ({"evals": 3, "size_evals": 3}, "size_evals: must be < evals"),
    ({"size_evals": True}, "wrong type"),
])
def test_config_rejects(over, message) -> None:
    with pytest.raises(ConfigError, match=message):
        _campaign(**over)


@pytest.mark.parametrize("over, budget", [
    ({}, 4), ({"evals": 6}, 2), ({"evals": 2}, 1), ({"evals": 1}, 0), ({"size_evals": 0}, 0),
    ({"evals": 3, "size_evals": 2}, 2),
])
def test_size_evals_budget(over, budget) -> None:
    c = _campaign(**over)
    assert c.size_budget == budget
    assert ("size_evals" in c.to_json()) == ("size_evals" in over)
    assert from_json(json.loads(json.dumps(c.to_json()))) == c


def test_saved_campaign_without_size_evals() -> None:
    saved = _campaign(evals=9).to_json()
    assert "size_evals" not in saved
    c = from_json(saved)
    assert c.size_evals is None and c.size_budget == 3 and c.to_json() == saved


def test_config_missing_key() -> None:
    data = {k: v for k, v in BASE.items() if k != "secrets_dir"}
    with pytest.raises(ConfigError, match="secrets_dir"):
        parse_campaign(data, Path("/cfg"))


def test_config_dw_s16_hidden() -> None:
    c = _campaign(dtype="S16", hidden_shapes=6)
    assert (c.op, c.dtype, c.hidden_shapes) == ("DepthwiseConv", "S16", 6)


def test_config_tc16_hidden() -> None:
    c = _campaign(op="TransposeConv", dtype="S16", hidden_shapes=6, legs=["tcm"])
    assert (c.op, c.dtype, c.hidden_shapes, c.legs) == ("TransposeConv", "S16", 6, ("tcm",))
    with pytest.raises(ConfigError, match="hidden_shapes: No random shapes"):
        _campaign(op="TransposeConv", dtype="S8", hidden_shapes=6)


def test_config_fc_without_hidden() -> None:
    c = _campaign(op="FullyConnected", hidden_shapes=0, legs=["tcm"])
    assert c.op == "FullyConnected" and c.legs == ("tcm",)


# --- ledger -----------------------------------------------------------------------------


def test_ledger_counter_ignores_rows(tmp_path: Path) -> None:
    led = ledger.Ledger(tmp_path / "ledger")
    assert [led.next_id(), led.next_id()] == ["001", "002"]
    led.append({"eval": "002", "charged": False, "infra": True})
    led.append({"eval": "003", "charged": True, "infra": False})
    assert led.next_id() == "003" and led.charged() == 1 and led.infra_streak() == 0
    led.append({"eval": "004", "charged": False, "infra": True})
    led.append({"eval": "005", "charged": False, "infra": True})
    assert led.infra_streak() == 2


@pytest.mark.parametrize("legs, wanted, overall", [
    ({"tcm": {"verdict": "pass"}, "mram": {"verdict": "pass"}}, ("tcm", "mram"), "pass"),
    ({"tcm": {"verdict": "pass"}, "mram": {"verdict": "fail"}}, ("tcm", "mram"), "fail"),
    ({"tcm": {"verdict": "no_gain"}, "mram": {"verdict": "pass"}}, ("tcm", "mram"), "no_gain"),
    ({"tcm": {"verdict": "rejected", "stage": "check"}}, ("tcm", "mram"), "rejected"),
    ({"tcm": {"verdict": "pass"}}, ("tcm", "mram"), "error"),
    ({"tcm": {"verdict": "weird"}}, ("tcm",), "error"),
])
def test_merge_legs(legs, wanted, overall) -> None:
    assert ledger.merge_legs(legs, wanted) == overall


def test_is_infra() -> None:
    assert ledger.is_infra(None)
    assert not ledger.is_infra({"verdict": "error", "stage": "run"})
    assert ledger.is_infra({"verdict": "refused", "stage": "tester"})
    assert not ledger.is_infra({"verdict": "refused", "stage": "check"})
    assert not ledger.is_infra({"verdict": "fail", "stage": "score"})


def _case(cid: str, touched, cycles: int, speedup: float) -> dict:
    return {"case_id": cid, "touched": touched, "baseline_cycles": int(cycles * speedup), "candidate_cycles": cycles,
            "speedup": speedup, "band_pct": 1.0, "cycles_per_mac_candidate": 1.5}


def test_leg_view_compact() -> None:
    cases = [_case(f"t{i}", True, 100 * i, 1.1) for i in range(1, 13)]
    cases += [_case("u1", False, 50, 0.97), _case("u2", False, 60, 1.02)]
    hints = [{"case_id": c["case_id"], "diagnosis": "x"} for c in cases]
    verdict = {"verdict": "pass", "stage": "score", "score": 0.1, "cases": cases, "hints": hints, "extra": "dropped",
               "families": {"depthwise": {"cases": 12, "geomean_speedup": 1.1, "weight": 9}}}
    view = ledger.leg_view(verdict, hints=True)
    assert [row[0] for row in view["cases"]] == [f"t{i}" for i in range(1, 13)]
    assert view["untouched_cases"] == {"count": 2, "speedup_min": 0.97, "speedup_max": 1.02}
    assert [h["case_id"] for h in view["hints"]] == [f"t{i}" for i in range(12, 2, -1)]
    assert "extra" not in view and view["families"] == {"depthwise": {
        "cases": 12, "geomean_speedup": 1.1, "regression": None, "untouched_cases": None, "untouched_geomean": None}}
    assert "hints" not in ledger.leg_view(verdict, hints=False)


# --- submit -----------------------------------------------------------------------------


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    w = Workspace(tmp_path / "ws")
    for tree in ("base", "agent", "submit/tree"):
        (w.root / tree / "Source").mkdir(parents=True)
        (w.root / tree / "Source" / "a.c").write_text("int a;\n")
    # The fake check skips staging.
    (w.submit_dir / "tree" / "Source" / "a.c").write_text("int a = 1;\n")
    camp = _campaign(evals=2, retries=1, max_infra_errors=3, secrets_dir=str(tmp_path / "secret"))
    w.save(camp, {"base_commit": "b" * 40, "tester_commit": "c" * 40})
    return w


class FakeBoard:
    """Answers each leg call from a script."""

    def __init__(self, outputs: list) -> None:
        self.outputs, self.calls = list(outputs), []

    def __call__(self, cmd, **kwargs):
        self.calls.append(cmd)
        out = self.outputs.pop(0)
        text = out if isinstance(out, str) else json.dumps(out)
        return subprocess.CompletedProcess(cmd, 0 if out else 1, stdout=text)


def _ok_check(ws, campaign, base, area, deadline=None):
    return True, {"check": {"ok": True}, "build": "ok", "code_size": {"delta_bytes": 64}}


def _leg(verdict: str, stage: str = "score") -> dict:
    return {"verdict": verdict, "stage": stage, "score": 0.01, "cases": [_case("c1", True, 90, 1.1)],
            "families": {"depthwise": {"geomean_speedup": 1.1}}}


def _submit(ws: Workspace, board: FakeBoard, capsys, checker=_ok_check) -> tuple[int, dict]:
    rc = judge.submit(ws, runner=board, checker=checker)
    return rc, json.loads(capsys.readouterr().out)


def test_submit_pass_charges_and_records(ws: Workspace, capsys) -> None:
    board = FakeBoard([_leg("pass"), _leg("pass")])
    rc, view = _submit(ws, board, capsys)
    assert rc == 0 and view["verdict"] == "pass" and view["evals_left"] == 1
    assert [Path(c[c.index("--baseline") + 1]).name for c in board.calls] == ["tcm", "mram"]
    assert {c[c.index("--kernels") + 1] for c in board.calls} == {str(ws.submit_dir / "tree")}
    assert "hints" in view["legs"]["tcm"] and "hints" not in view["legs"]["mram"]
    row = ledger.Ledger(ws.ledger).rows()[0]
    assert row["eval"] == "001" and row["charged"] and row["size_delta"] == 64
    assert b"int a = 1;" in (ws.ledger / "001.diff").read_bytes()
    assert json.loads((ws.results / "001.json").read_text()) == view


def test_submit_skips_mram_when_unscored(ws: Workspace, capsys) -> None:
    board = FakeBoard([{"verdict": "rejected", "stage": "objects", "findings": [{"rule": "x"}]}])
    rc, view = _submit(ws, board, capsys)
    assert rc == 3 and view["verdict"] == "rejected" and list(view["legs"]) == ["tcm"]
    assert len(board.calls) == 1 and ledger.Ledger(ws.ledger).charged() == 1


def test_submit_infra_is_free_and_retried(ws: Workspace, capsys, monkeypatch) -> None:
    monkeypatch.setattr(judge, "RETRY_PAUSE_S", 0)
    board = FakeBoard([_leg("pass"), "", ""])
    rc, view = _submit(ws, board, capsys)
    assert rc == 5 and view["verdict"] == "error" and "not charged" in view["note"]
    # No free scores from the tcm leg.
    assert view["evals_left"] == 2 and len(board.calls) == 3 and view["legs"] == {}
    row = ledger.Ledger(ws.ledger).rows()[0]
    assert row["infra"] and not row["charged"] and row["attempts"] == {"tcm": 1, "mram": 2}
    # Retry succeeds: one call more, charged.
    board = FakeBoard(["", _leg("pass"), _leg("no_gain")])
    rc, view = _submit(ws, board, capsys)
    assert rc == 4 and view["verdict"] == "no_gain" and view["evals_left"] == 1
    assert ledger.Ledger(ws.ledger).rows()[-1]["eval"] == "002"


def test_submit_candidate_errors_are_charged(ws: Workspace, capsys, monkeypatch) -> None:
    monkeypatch.setattr(judge, "RETRY_PAUSE_S", 0)
    crash = {"verdict": "error", "stage": "run", "reason": "hardware run exited 5"}
    board = FakeBoard([crash, crash])
    rc, view = _submit(ws, board, capsys)
    assert rc == 5 and view["evals_left"] == 1 and "faults or hangs" in view["note"] and len(board.calls) == 2
    hang = subprocess.CompletedProcess([], judge.TIMED_OUT, stdout="")
    rc, view = judge.submit(ws, runner=lambda cmd, **kw: hang, checker=_ok_check), json.loads(capsys.readouterr().out)
    assert view["evals_left"] == 0 and "timed out" in view["legs"]["tcm"]["reason"]


def test_charged_error_survives_busy_retry(ws: Workspace, capsys, monkeypatch) -> None:
    monkeypatch.setattr(judge, "RETRY_PAUSE_S", 0)
    crash = {"verdict": "error", "stage": "run", "reason": "hardware run exited 5"}
    board = FakeBoard([crash, ""])
    rc, view = _submit(ws, board, capsys)
    assert rc == 5 and view["evals_left"] == 1 and view["legs"]["tcm"]["reason"] == "hardware run exited 5"
    assert ledger.Ledger(ws.ledger).rows()[0]["charged"]


def test_run_bounded_kills_on_deadline() -> None:
    import time

    start = time.monotonic()
    with pytest.raises(judge.OutOfTime):
        judge.run_bounded(["sh", "-c", "sleep 30 & sleep 30"], time.monotonic() + 0.5)
    assert time.monotonic() - start < 5
    assert judge.run_bounded(["sh", "-c", "echo hi"], time.monotonic() + 5).stdout == "hi\n"


def test_run_check_times_out(ws: Workspace, monkeypatch) -> None:
    campaign, _ = ws.load()

    def slow(*args, **kwargs):
        raise judge.OutOfTime

    monkeypatch.setattr(judge, "_check_steps", slow)
    ok, out = judge.run_check(ws, campaign, "b" * 40, ws.check_dir, deadline=0.0)
    assert not ok and "timed out" in out["check"]["error"]


def test_submit_respects_deadline(ws: Workspace, capsys, monkeypatch) -> None:
    campaign, facts = ws.load()
    ws.save(Campaign(**{**campaign.__dict__, "submit_deadline_s": 120, "eval_timeout_s": 300}), facts)
    board = FakeBoard([_leg("pass"), _leg("pass")])
    _submit(ws, board, capsys)
    call = board.calls[0]
    lock_s, eval_s = int(call[call.index("--timeout") + 1]), int(call[call.index("timeout") + 1])
    assert eval_s <= 110 and lock_s + eval_s <= 120


def test_submit_precheck_is_free(ws: Workspace, capsys) -> None:
    def bad(ws, campaign, base, area, deadline=None):
        return False, {"check": {"ok": True}, "build": {"ok": False, "errors": ["a.c:1: error: x"]}}

    board = FakeBoard([])
    rc, view = _submit(ws, board, capsys, checker=bad)
    assert rc == 3 and view["verdict"] == "rejected" and view["evals_left"] == 2 and not board.calls
    assert view["build"]["errors"] == ["a.c:1: error: x"]


def test_submit_budget_spent(ws: Workspace, capsys) -> None:
    for _ in range(2):
        _submit(ws, FakeBoard([_leg("fail"), _leg("fail")]), capsys)
    board = FakeBoard([])
    rc, view = _submit(ws, board, capsys)
    assert rc == ledger.EXIT_BUDGET and view == {"verdict": "budget_spent", "exit_code": 6, "evals_left": 0}
    assert not board.calls and len(ledger.Ledger(ws.ledger).rows()) == 2


def test_submit_lock_busy_is_free(ws: Workspace, capsys, monkeypatch) -> None:
    import fcntl

    monkeypatch.setattr(judge, "MIN_EVAL_S", 120)
    campaign, facts = ws.load()
    ws.save(Campaign(**{**campaign.__dict__, "submit_deadline_s": 121}), facts)
    with (ws.root / ".submit.lock").open("a") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        board = FakeBoard([])
        rc, view = _submit(ws, board, capsys)
    assert rc == 5 and "not charged" in view["note"] and view["evals_left"] == 2 and not board.calls
    row = ledger.Ledger(ws.ledger).rows()[-1]
    assert row["infra"] and not row["charged"] and row["eval"] is None


def test_submit_stops_after_infra_streak(ws: Workspace, capsys) -> None:
    led = ledger.Ledger(ws.ledger)
    for n in range(3):
        led.append({"eval": f"x{n}", "charged": False, "infra": True})
    rc, view = _submit(ws, FakeBoard([]), capsys)
    assert rc == 5 and "Stop" in view["note"]


# --- patches ----------------------------------------------------------------------------


def _patch_tree(tmp_path: Path) -> Path:
    tree = tmp_path / "tree"
    (tree / "Source" / "Conv").mkdir(parents=True)
    (tree / "Source" / "Conv" / "a.c").write_text("one\ntwo\n")
    return tree


@pytest.mark.parametrize("old, new", [
    ("/home/u/proof/base", "/home/u/proof/agent"),
    ("base", "agent"),
])
def test_apply_patch_any_prefix(tmp_path: Path, old: str, new: str) -> None:
    tree = _patch_tree(tmp_path)
    diff = (f"diff -ruN {old}/Source/Conv/a.c {new}/Source/Conv/a.c\n"
            f"--- {old}/Source/Conv/a.c\t2026-10-07 16:44:59 -0400\n"
            f"+++ {new}/Source/Conv/a.c\t2026-10-07 20:23:32 -0400\n"
            "@@ -1,2 +1,2 @@\n one\n-two\n+three\n"
            f"diff -ruN {old}/Include/b.h {new}/Include/b.h\n"
            f"--- {old}/Include/b.h\t1970-01-01 00:00:00 +0000\n"
            f"+++ {new}/Include/b.h\t2026-10-07 20:23:32 -0400\n"
            "@@ -0,0 +1 @@\n+int b;\n")
    judge.apply_patch(tree, diff.encode())
    assert (tree / "Source" / "Conv" / "a.c").read_text() == "one\nthree\n"
    assert (tree / "Include" / "b.h").read_text() == "int b;\n"


def test_apply_patch_new_file_from_dev_null(tmp_path: Path) -> None:
    tree = _patch_tree(tmp_path)
    diff = b"diff --git a/Source/n.c b/Source/n.c\n--- /dev/null\n+++ b/Source/n.c\n@@ -0,0 +1 @@\n+int n;\n"
    judge.apply_patch(tree, diff)
    assert (tree / "Source" / "n.c").read_text() == "int n;\n"


def test_apply_patch_checks_every_header(tmp_path: Path) -> None:
    tree = _patch_tree(tmp_path)
    (tree / "nsx").mkdir()
    (tree / "nsx" / "x.txt").write_text("a\n")
    # Second header pair, no diff line.
    diff = (b"--- base/Source/Conv/a.c\n+++ agent/Source/Conv/a.c\n@@ -1,2 +1,2 @@\n one\n-two\n+three\n"
            b"--- a/nsx/x.txt\n+++ b/nsx/x.txt\n@@ -1 +1 @@\n-a\n+evil\n")
    with pytest.raises(ValueError, match="outside Source/Include"):
        judge.apply_patch(tree, diff)
    assert (tree / "nsx" / "x.txt").read_text() == "a\n" and "three" not in (tree / "Source" / "Conv" / "a.c").read_text()


@pytest.mark.parametrize("diff, message", [
    (b"--- a/Source/../nsx/x\n+++ b/Source/../nsx/x\n@@ -1 +1 @@\n-a\n+b\n", "outside"),
    (b"--- a/Source/a.c\n+++ b/Source/a.c\n@@ -1 +1 @@\n-a\n+b\n+c\n", "unexpected line"),
    (b"@@ -1 +1 @@\n-a\n+b\n", "before file headers"),
    (b"--- a/Source/a.c\n+++ b/Source/a.c\n@@ -1,2 +1,2 @@\n-a\n", "ends early"),
    (b"Only in agent/Source: x.c\n", "unexpected line"),
])
def test_clean_patch_rejects(diff: bytes, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        judge.clean_patch(diff)


def test_clean_patch_keeps_no_newline_marker() -> None:
    diff = (b"--- base/Source/a.c\n+++ agent/Source/a.c\n@@ -1 +1 @@\n-a\n\\ No newline at end of file\n"
            b"+b\n\\ No newline at end of file\n")
    assert judge.clean_patch(diff).count(b"\\ No newline") == 2


def test_apply_patch_refuses_other_trees(tmp_path: Path) -> None:
    diff = b"--- base/nsx/x.txt\n+++ agent/nsx/x.txt\n@@ -0,0 +1 @@\n+x\n"
    with pytest.raises(ValueError, match="outside Source/Include"):
        judge.apply_patch(_patch_tree(tmp_path), diff)


def test_tree_diff_round_trips(ws: Workspace, tmp_path: Path) -> None:
    diff = judge.tree_diff(ws, ws.submit_dir / "tree")
    copy = tmp_path / "copy"
    (copy / "Source").mkdir(parents=True)
    (copy / "Source" / "a.c").write_text("int a;\n")
    judge.apply_patch(copy, diff)
    assert (copy / "Source" / "a.c").read_text() == "int a = 1;\n"


def test_size_delta() -> None:
    out = judge.size_delta({"a.o": 100, "b.o": 50}, {"a.o": 120, "b.o": 50, "c.o": 8})
    assert out == {"kernel_text_bytes": [150, 178], "delta_bytes": 28, "changed_objects": {"a.o": [100, 120], "c.o": [0, 8]}}


# --- settings and wrappers --------------------------------------------------------------


def test_settings_use_absolute_rules(ws: Workspace) -> None:
    campaign, _ = ws.load()
    perms = agent.agent_settings(ws, campaign, extra_denies=[Path("/src/tester")])["permissions"]
    root = ws.root
    assert f"Read(/{root}/agent/**)" in perms["allow"]
    assert f"Edit(/{root}/agent/Source/**)" in perms["allow"]
    assert f"Bash({root}/bin/submit)" in perms["allow"] and f"Bash({root}/bin/disasm:*)" in perms["allow"]
    for name in ("ledger", "baselines", "tester", "base", "check", "submit", "logs"):
        assert f"Read(/{root}/{name}/**)" in perms["deny"]
    assert f"Read(/{root}/campaign.json)" in perms["deny"]
    assert f"Read(/{campaign.secrets_dir}/**)" in perms["deny"] and "Read(//src/tester/**)" in perms["deny"]
    assert all(rule.startswith(("Read(//", "Edit(//", "Bash(/")) for rule in perms["allow"])


def test_wrappers_are_exact(ws: Workspace) -> None:
    agent.write_wrappers(ws)
    for name in agent.WRAPPERS:
        path = ws.bin / name
        assert path.stat().st_mode & 0o111
        text = path.read_text()
        assert text.startswith("#!/bin/sh\n")
        assert f"agent-loop {name} --workspace {ws.root}" in text and str(ws.tester) in text
    assert (ws.bin / "disasm").read_text().rstrip().endswith('"$1"')
    assert '"$1"' not in (ws.bin / "submit").read_text()


# --- prompt -----------------------------------------------------------------------------

DW_ROWS = [
    {"case_id": "dw1", "timed_symbol": "arm_depthwise_conv_wrapper_s8", "inner_symbol": "arm_depthwise_conv_s8_opt",
     "cycles_per_mac": "3.5", "hidden": "false"},
    {"case_id": "dw2", "timed_symbol": "arm_depthwise_conv_wrapper_s8", "inner_symbol": "arm_depthwise_conv_s8_opt",
     "cycles_per_mac": "2.5", "hidden": "false"},
    {"case_id": "dw3", "timed_symbol": "arm_depthwise_conv_s8_opt_3x3", "inner_symbol": "", "cycles_per_mac": "2.1",
     "hidden": "false"},
    {"case_id": "h0123", "timed_symbol": "arm_depthwise_conv_wrapper_s8", "inner_symbol": "secret_route",
     "cycles_per_mac": "9.9", "hidden": "true"},
]
CONV_ROWS = [
    {"case_id": "c1", "timed_symbol": "arm_convolve_wrapper_s8", "inner_symbol": "arm_convolve_1x1_s8_fast",
     "cycles_per_mac": "0.6", "hidden": "false"},
    {"case_id": "c2", "timed_symbol": "arm_convolve_wrapper_s8", "inner_symbol": "arm_convolve_s8",
     "cycles_per_mac": "0.9", "hidden": "false"},
]
PATHS = {"submit": Path("/w/bin/submit"), "check": Path("/w/bin/check"), "disasm": Path("/w/bin/disasm"),
         "results": Path("/w/agent-results")}


def test_route_facts_skip_hidden() -> None:
    facts = route_facts(DW_ROWS, "cortex-m55")
    assert [(f["timed"], f["inner"], f["cases"], f["median"], f["best"]) for f in facts] == [
        ("arm_depthwise_conv_wrapper_s8", "arm_depthwise_conv_s8_opt", 2, 3.0, 2.5),
        ("arm_depthwise_conv_s8_opt_3x3", "", 1, 2.1, 2.1),
    ]
    assert facts[0]["ceiling"] == 0.5


def test_prompt_depthwise() -> None:
    text = render_prompt(_campaign(), DW_ROWS, PATHS)
    assert "s8 depthwise convolution faster on the Cortex-M55" in text
    assert "`arm_depthwise_conv_wrapper_s8` -> `arm_depthwise_conv_s8_opt`: 2 cases, median 3.0" in text
    assert "3 public, plus 1 hidden" in text and "secret_route" not in text and "h0123" not in text
    assert "run exactly `/w/bin/submit`" in text and "`/w/bin/disasm <function_name>`" in text
    assert "leg `mram`" in text and "you have 12 evaluations" in text and "/w/agent-results/NNN.json" in text
    assert "Depthwise has no reduction" in text and "previous attempt" not in text
    for keep in ("comparison_failed", "prepare_regression", "Code size.", "Tiling.", "Work plan:", "short summary"):
        assert keep in text


def test_prompt_convolve_has_no_depthwise() -> None:
    camp = _campaign(op="Convolve", legs=["tcm"], evals=8)
    diff = b"--- base/Source/a.c\n+++ agent/Source/ConvolutionFunctions/arm_convolve_s8.c\n@@ -1 +1 @@\n-a\n+b\n"
    camp = Campaign(**{**camp.__dict__, "start_notes": "- 1x1 path vectorized."})
    text = render_prompt(camp, CONV_ROWS, PATHS, start_diff=diff)
    assert "s8 convolution faster" in text and "depthwise" not in text.lower()
    assert "ceiling 0.125" in text and "leg `mram`" not in text and "you have 8 evaluations" in text
    assert "`Source/ConvolutionFunctions/arm_convolve_s8.c`" in text and "- 1x1 path vectorized." in text
    assert "previous attempt is already applied" in text


def test_prompt_transpose_conv() -> None:
    rows = [{"case_id": "a", "timed_symbol": "arm_transpose_conv_s16", "inner_symbol": "", "cycles_per_mac": "9.0"}]
    text = render_prompt(_campaign(op="TransposeConv", dtype="S16", legs=["tcm"]), rows, PATHS)
    assert "s16 transpose convolution faster" in text and "`arm_transpose_conv_s16`: 1 case" in text
    assert "depthwise" not in text.lower()


# --- selftest and status ----------------------------------------------------------------


def _tool(tid: str, name: str, arg: str) -> dict:
    key = "command" if name == "Bash" else "file_path"
    return {"type": "assistant", "message": {"content": [{"type": "tool_use", "id": tid, "name": name, "input": {key: arg}}]}}


def _result(tid: str, error: bool) -> dict:
    return {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": tid, "is_error": error}]}}


def test_judge_selftest() -> None:
    items = [{"tool": "Read", "arg": "/w/agent/a", "allow": True}, {"tool": "Read", "arg": "/w/ledger/l", "allow": False},
             {"tool": "Bash", "arg": "curl x", "allow": False}, {"tool": "Bash", "arg": "/w/bin/disasm f", "allow": True}]
    events = [_tool("1", "Read", "/w/agent/a"), _result("1", False), _tool("2", "Read", "/w/ledger/l"), _result("2", True),
              _tool("3", "Bash", "curl x"), _result("3", True),
              {"type": "result", "permission_denials": [{"tool_use_id": "2"}]}]
    out = {r["arg"]: (r["outcome"], r["ok"]) for r in agent.judge_selftest(items, events)}
    assert out == {"/w/agent/a": ("allowed", True), "/w/ledger/l": ("denied", True), "curl x": ("error", False),
                   "/w/bin/disasm f": ("not_attempted", False)}


def test_run_cost() -> None:
    assert agent.run_cost([{"type": "system"}]) == {"finished": False}
    done = agent.run_cost([{"type": "result", "subtype": "success", "total_cost_usd": 1.5, "num_turns": 9}])
    assert done == {"finished": True, "subtype": "success", "cost_usd": 1.5, "turns": 9, "denials": 0}


def test_launch_records_real_pid(ws: Workspace) -> None:
    ws.prompt.write_text("go")
    seen = {}

    class Proc:
        pid = 4242

    def popen(args, **kwargs):
        seen.update(args=args, **kwargs)
        return Proc()

    meta = agent.launch(ws, popen=popen)
    args = seen["args"]
    assert meta["pid"] == 4242 and json.loads(ws.run_meta.read_text())["pid"] == 4242
    assert seen["start_new_session"] and seen["cwd"] == ws.agent
    assert args[args.index("--max-budget-usd") + 1] == "25.00"
    assert args[args.index("--session-id") + 1] == meta["session_id"]
    assert args[args.index("--permission-mode") + 1] == "dontAsk" and "--no-session-persistence" not in args
    # Finished run: resume with the rest of the cap.
    Path(meta["log"]).write_text(json.dumps({"type": "result", "total_cost_usd": 10.0}) + "\n")
    Proc.pid = 1  # not a claude process
    resumed = agent.launch(ws, popen=popen, resume=True)
    args = seen["args"]
    assert args[args.index("--resume") + 1] == meta["session_id"] and "--session-id" not in args
    assert args[args.index("--max-budget-usd") + 1] == "15.00" and args[args.index("--settings") + 1] == str(ws.settings)
    assert args[args.index("--permission-mode") + 1] == "dontAsk" and "--strict-mcp-config" in args
    assert resumed["log"] != meta["log"] and resumed["logs"] == [meta["log"], resumed["log"]]
    # Status reads only the newest log.
    assert agent.status(ws, tail=0)["cost"] == {"finished": False} and agent.status(ws, tail=0)["spent_usd"] == 10.0


def test_commands_need_finished_init(ws: Workspace) -> None:
    from typer.testing import CliRunner

    from helia_core_tester.cli import app

    result = CliRunner().invoke(app, ["agent-loop", "status", "-w", str(ws.root)])
    assert result.exit_code == 2 and "not done" in result.output
    campaign, facts = ws.load()
    ws.save(campaign, {**facts, "ready": True})
    assert CliRunner().invoke(app, ["agent-loop", "status", "-w", str(ws.root)]).exit_code == 0


# --- init helpers -----------------------------------------------------------------------


def test_shallow_tree_rebuilds_partial(tmp_path: Path) -> None:
    from helia_core_tester.agent_loop import setup
    from helia_core_tester.tests.test_harness_lock import _git as git, _repo

    repo = _repo(tmp_path / "nn", {"Source/a.c": "int a;\n"})
    sha = git(repo, "rev-parse", "HEAD").strip()
    dest = tmp_path / "agent"
    dest.mkdir()
    (dest / "junk").write_text("half done")
    setup.shallow_tree(repo, sha, dest, tag=True)
    assert setup.tree_at(dest, sha, tag=True) and not (dest / "junk").exists()
    (dest / "Source" / "a.c").write_text("edited\n")
    assert not setup.tree_at(dest, sha, tag=True)


def test_unreadable_start_patch(tmp_path: Path, monkeypatch) -> None:
    from helia_core_tester.agent_loop import setup

    camp = _campaign(start_patch=str(tmp_path / "missing.diff"), secrets_dir=str(tmp_path / "s"))
    ws = Workspace(tmp_path / "ws")
    monkeypatch.setattr(setup, "pin_tester", lambda ws, sha: "c" * 40)
    monkeypatch.setattr(setup, "shallow_tree", lambda *a, **k: None)
    monkeypatch.setattr(setup, "_git", lambda *a, **k: "b" * 40)
    monkeypatch.setattr(setup, "check_paths", lambda ws, c: None)
    with pytest.raises(setup.InitError, match="Cannot read start_patch"):
        setup.init_workspace(ws, camp, echo=lambda _: None)


def test_rerun_init_clears_ready(tmp_path: Path, monkeypatch) -> None:
    from helia_core_tester.agent_loop import setup

    camp = _campaign(secrets_dir=str(tmp_path / "s"))
    ws = Workspace(tmp_path / "ws")
    ws.root.mkdir()
    ws.save(camp, {"ready": True, "tester_commit": "c" * 40})
    monkeypatch.setattr(setup, "check_paths", lambda ws, c: None)

    def broken(ws, sha):
        raise setup.InitError("pin failed")

    monkeypatch.setattr(setup, "pin_tester", broken)
    with pytest.raises(setup.InitError, match="pin failed"):
        setup.init_workspace(ws, camp, echo=lambda _: None)
    assert "ready" not in ws.load()[1]


def test_selftest_probes_are_unique(ws: Workspace) -> None:
    campaign, _ = ws.load()
    first = agent.probes(ws, campaign, "aaaa")
    second = agent.probes(ws, campaign, "bbbb")
    writes = [p["arg"] for p in first if p["tool"] == "Write"]
    assert all("aaaa" in w for w in writes)
    assert not set(writes) & {p["arg"] for p in second}


def test_hidden_set_targets_campaign_op(tmp_path: Path, monkeypatch) -> None:
    from helia_core_tester.agent_loop import setup

    camp = _campaign(op="DepthwiseConv", secrets_dir=str(tmp_path / "s"))
    ws = Workspace(tmp_path / "ws")
    ws.root.mkdir()
    ws.save(camp, {})
    calls = []
    monkeypatch.setattr(setup, "resolve_board", lambda _: type("B", (), {"cpu": "cortex-m55"}))

    def fake_run(cmd, log, env=None, **_):
        calls.append((cmd, env))
        ws.hidden_dir(camp).mkdir(parents=True)

    monkeypatch.setattr(setup, "_run", fake_run)
    setup.make_hidden(ws, camp, {})
    cmd, env = calls[0]
    assert cmd[cmd.index("--op") + 1] == "DepthwiseConv" and cmd[cmd.index("--dtype") + 1] == "S8"
    # The symbol probe reads the base tree.
    assert env == {"CMSIS_NN_ROOT": str(ws.base)}


def test_secrets_dir_needs_ownership(tmp_path: Path) -> None:
    from helia_core_tester.agent_loop import setup

    secrets_dir = tmp_path / "s"
    (secrets_dir / "hidden").mkdir(parents=True)
    (secrets_dir / "seed").write_text("x" * 64)
    (secrets_dir / "hidden" / "done").write_text("")
    camp = _campaign(secrets_dir=str(secrets_dir))
    ws = Workspace(tmp_path / "ws")
    ws.root.mkdir()
    ws.save(camp, {})
    # Old campaign's done marker: refused.
    with pytest.raises(setup.InitError, match="not empty"):
        setup.make_hidden(ws, camp, {})
    # Owner resumes; done is honoured.
    facts = {"secrets_owner": str(ws.root)}
    setup.make_hidden(ws, camp, facts)
    fresh = Workspace(tmp_path / "ws2")
    fresh.root.mkdir()
    camp2 = _campaign(secrets_dir=str(tmp_path / "s2"))
    facts2: dict = {}
    setup.claim_secrets(fresh, camp2, facts2)
    assert facts2["secrets_owner"] == str(fresh.root) and fresh.load()[1]["secrets_owner"] == str(fresh.root)
    assert oct((tmp_path / "s2").stat().st_mode & 0o777) == "0o700"


# --- toolchains -------------------------------------------------------------------------


@pytest.fixture
def atfe_root(tmp_path: Path, monkeypatch) -> Path:
    """A fake ATFE_ROOT with bin/clang."""
    clang = tmp_path / "atfe" / "bin" / "clang"
    clang.parent.mkdir(parents=True)
    clang.write_text("#!/bin/sh\necho 22.1.0\n")
    clang.chmod(0o755)
    monkeypatch.setenv("ATFE_ROOT", str(clang.parent.parent))
    return clang.parent.parent


def test_toolchains_default_keeps_gcc_files() -> None:
    c = _campaign()
    # Saved as before toolchains existed.
    assert c.toolchains == ("gcc",) and c.leg_names == ("tcm", "mram") and "toolchains" not in c.to_json()
    assert from_json(c.to_json()) == c


def test_saved_campaign_keeps_gcc(ws: Workspace) -> None:
    # Only the drift facts are new.
    ws.save(_campaign(), {"toolchains": {"gcc": None}})
    saved = json.loads(ws.state.read_text())
    assert "toolchains" not in saved["campaign"] and saved["toolchains"] == {"gcc": None}


def test_atfe_needs_atfe_root(tmp_path: Path, monkeypatch) -> None:
    clang = tmp_path / "clang"
    clang.write_text("#!/bin/sh\necho 18.0.0\n")
    clang.chmod(0o755)
    monkeypatch.delenv("ATFE_ROOT", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    spec = toolchain_spec("atfe")
    # PATH clang is not ATfE.
    assert spec.installed() is None
    with pytest.raises(FileNotFoundError, match="ATFE_ROOT"):
        spec.objdump()


def test_toolchains_expand_legs(atfe_root: Path) -> None:
    c = _campaign(toolchains=["gcc", "atfe"])
    assert c.leg_names == ("tcm", "mram", "tcm-atfe", "mram-atfe")
    assert [(r.placement, r.toolchain) for r in c.runs][2:] == [("tcm", "atfe"), ("mram", "atfe")]
    assert from_json(json.loads(json.dumps(c.to_json()))) == c
    assert _campaign(toolchains=["atfe"], legs=["tcm"]).leg_names == ("tcm-atfe",)


@pytest.mark.parametrize("value, message", [
    (["gcc", "gcc"], "toolchains"), (["icc"], "toolchains"), ([{}], "toolchains"), (["atfe"], "ATFE_ROOT"),
])
def test_toolchains_rejects(value, message, monkeypatch) -> None:
    monkeypatch.delenv("ATFE_ROOT", raising=False)
    with pytest.raises(ConfigError, match=message):
        _campaign(toolchains=value)


def test_size_ref_per_toolchain(ws: Workspace, atfe_root: Path, monkeypatch) -> None:
    from helia_core_tester.agent_loop import setup

    camp = _campaign(toolchains=["gcc", "atfe"])
    cmds = []
    monkeypatch.setattr(setup, "_run", lambda cmd, log, cwd=None: cmds.append((cmd, log)))
    monkeypatch.setattr(setup, "kernel_lib", lambda build: build)
    monkeypatch.setattr(setup, "object_sizes", lambda lib: {lib.name: 1})
    for toolchain in camp.toolchains:
        setup.make_size_ref(ws, camp, toolchain)
    # gcc gets no flag: older testers lack it.
    assert ["--toolchain" in c for c, _ in cmds] == [False, True] and cmds[1][0][-3] == "atfe"
    assert [Path(c[c.index("--build-dir") + 1]).name for c, _ in cmds] == ["size-ref-build", "size-ref-build-atfe"]
    assert [log.name for _, log in cmds] == ["size-ref.log", "size-ref-atfe.log"]
    assert json.loads(ws.size_ref_of("atfe").read_text()) == {"size-ref-build-atfe": 1}
    assert ws.size_ref_of("gcc") == ws.size_ref


def test_disasm_picks_objdump(ws: Workspace, atfe_root: Path, monkeypatch, capsys) -> None:
    _two_toolchains(ws)
    seen = []

    def objdump(tool, args):
        seen.append((tool, Path(args[-1]).parent.name))
        return "00000000 <arm_fn>:\n   0:\tbx lr\n\n"

    monkeypatch.setattr(judge, "run_binutil", objdump)
    monkeypatch.setattr(judge, "kernel_lib", lambda build: build / "lib.a")
    assert judge.disasm(ws, "arm_fn") == 0 and judge.disasm(ws, "arm_fn", "atfe") == 0
    assert seen == [("arm-none-eabi-objdump", "build"), (str(atfe_root / "bin" / "llvm-objdump"), "build-atfe")]
    assert "bx lr" in capsys.readouterr().out
    assert judge.disasm(ws, "arm_fn", "icc") == 2 and "gcc|atfe" in capsys.readouterr().out


def test_disasm_wrapper_args(ws: Workspace, tmp_path: Path) -> None:
    cmd = shlex.join([*ws.tester_cmd(), "agent-loop", "disasm", "--workspace", str(ws.root)])
    text = agent.wrapper_text(ws, "disasm").replace(cmd, "echo")
    script = tmp_path / "disasm"
    script.write_text(text)
    script.chmod(0o755)

    def run(*args: str) -> tuple[int, str]:
        proc = subprocess.run([str(script), *args], capture_output=True, text=True)
        return proc.returncode, proc.stdout.strip()

    assert run("fn") == (0, "-- fn")
    assert run("fn", "--toolchain", "atfe") == (0, "--toolchain atfe -- fn")
    assert run("--workspace=/x") == (0, "-- --workspace=/x")
    assert run("fn", "--workspace", "/x")[0] == 2


LEGS4 = ("tcm", "mram", "tcm-atfe", "mram-atfe")


def _replace(ws: Workspace, **over) -> None:
    campaign, facts = ws.load()
    ws.save(dataclasses.replace(campaign, **over), facts)


def _two_toolchains(ws: Workspace) -> None:
    _replace(ws, toolchains=("gcc", "atfe"))


def _keyed_check(ws, campaign, base, area, deadline=None):
    return True, {"check": {"ok": True}, "build": {"gcc": "ok", "atfe": "ok"},
                  "code_size": {"gcc": {"delta_bytes": 64}, "atfe": {"delta_bytes": 48}}}


def _leg_gain(verdict: str, gain: float) -> dict:
    return {**_leg(verdict), "families": {"depthwise": {"geomean_speedup": gain}}}


def test_submit_runs_every_toolchain_leg(ws: Workspace, atfe_root: Path, capsys) -> None:
    _two_toolchains(ws)
    board = FakeBoard([_leg_gain("pass", 1.2), _leg_gain("pass", 1.1), _leg_gain("pass", 1.05), _leg_gain("pass", 1.0)])
    rc, view = _submit(ws, board, capsys, checker=_keyed_check)
    assert rc == 0 and view["verdict"] == "pass"
    assert tuple(Path(c[c.index("--baseline") + 1]).name for c in board.calls) == LEGS4
    assert view["toolchains"]["gcc"] == {"geomean": round((1.2 * 1.1) ** 0.5, 4), "size_delta": 64}
    assert view["toolchains"]["atfe"] == {"geomean": round(1.05 ** 0.5, 4), "size_delta": 48}
    assert "hints" in view["legs"]["tcm"] and "hints" not in view["legs"]["tcm-atfe"]
    row = ledger.Ledger(ws.ledger).rows()[0]
    assert row["size_delta"] == {"gcc": 64, "atfe": 48} and list(row["legs"]) == list(LEGS4)
    assert row["toolchains"]["atfe"]["size_delta"] == 48
    info = agent.status(ws, tail=0)
    assert info["rows"][0]["toolchains"] == row["toolchains"]


def test_submit_atfe_failure_fails_eval(ws: Workspace, atfe_root: Path, capsys) -> None:
    _two_toolchains(ws)
    board = FakeBoard([_leg("pass"), _leg("pass"), _leg("fail"), _leg("pass")])
    rc, view = _submit(ws, board, capsys, checker=_keyed_check)
    assert rc == 1 and view["verdict"] == "fail" and view["legs"]["tcm-atfe"]["verdict"] == "fail"


def test_gcc_only_row_stays_flat(ws: Workspace, capsys) -> None:
    rc, view = _submit(ws, FakeBoard([_leg("pass"), _leg("pass")]), capsys)
    row = ledger.Ledger(ws.ledger).rows()[0]
    assert "toolchains" not in view and "toolchains" not in row and row["size_delta"] == 64


def test_submit_refuses_toolchain_drift(ws: Workspace, atfe_root: Path, capsys) -> None:
    _two_toolchains(ws)
    campaign, facts = ws.load()
    ws.save(campaign, {**facts, "toolchains": {"gcc": None, "atfe": {"name": "atfe", "version": "21.0.0"}}})
    board = FakeBoard([])
    rc, view = _submit(ws, board, capsys, checker=_keyed_check)
    assert rc == 5 and "Compiler changed" in view["note"] and not board.calls
    assert ledger.Ledger(ws.ledger).charged() == 0


def test_baseline_rerecords_on_new_compiler(ws: Workspace, atfe_root: Path, monkeypatch) -> None:
    from helia_core_tester.agent_loop import setup
    from helia_core_tester.agent_loop.config import Leg

    out = ws.baseline("tcm-atfe")
    bundle = out / "bundles" / "s1"
    bundle.mkdir(parents=True)
    (out / "baseline.json").write_text("{}")
    manifest: dict = {"build": {}}
    monkeypatch.setattr(setup, "read_baseline", lambda path: {"sessions": ["s1"]})
    leg = Leg("tcm-atfe", "tcm", "atfe")
    for version, stale in ((None, False), ("21.0.0", True), ("22.1.0", False)):
        manifest["build"]["toolchain"] = version and {"name": "atfe", "version": version}
        (bundle / "session_manifest.json").write_text(json.dumps(manifest))
        assert setup.baseline_stale(out, leg) == stale
    ws.size_ref_of("atfe").write_text("{}")
    cmds, lines = [], []
    monkeypatch.setattr(setup, "_run", lambda cmd, log, cwd=None: cmds.append(cmd))
    setup.make_baseline(ws, _campaign(toolchains=["gcc", "atfe"]), leg, lines.append)
    assert not cmds
    manifest["build"]["toolchain"]["version"] = "21.0.0"
    (bundle / "session_manifest.json").write_text(json.dumps(manifest))
    setup.make_baseline(ws, _campaign(toolchains=["gcc", "atfe"]), leg, lines.append)
    cmd = cmds[0]
    assert cmd[cmd.index("--toolchain") + 1] == "atfe" and cmd[cmd.index("--placement") + 1] == "tcm"
    assert "re-recording" in lines[0] and not bundle.exists() and not ws.size_ref_of("atfe").exists()


def test_settings_deny_every_size_ref(ws: Workspace, atfe_root: Path) -> None:
    camp = _campaign(toolchains=["gcc", "atfe"], secrets_dir="/secret/x")
    deny = agent.agent_settings(ws, camp)["permissions"]["deny"]
    for path in (ws.size_ref, ws.size_ref_of("atfe")):
        assert f"Read(/{path})" in deny
    assert f"Read(/{ws.size_build_of('atfe')}/**)" in deny
    probes = [p["arg"] for p in agent.probes(ws, camp, "t")]
    assert f"{ws.bin / 'disasm'} selftest_probe --toolchain atfe" in probes


def test_prompt_names_both_toolchains(atfe_root: Path) -> None:
    text = render_prompt(_campaign(toolchains=["gcc", "atfe"]), DW_ROWS, PATHS)
    assert "leg `tcm-atfe`" in text and "Built with ATfE clang." in text and "optimize(...)" in text
    assert "--toolchain atfe" in text
    plain = render_prompt(_campaign(), DW_ROWS, PATHS)
    assert "atfe" not in plain and "Built with" not in plain


def test_prompt_atfe_only(atfe_root: Path) -> None:
    text = render_prompt(_campaign(toolchains=["atfe"], legs=["tcm"]), DW_ROWS, PATHS)
    assert "Every leg builds with atfe" in text and "optimize(...)" in text and "Built with ATfE clang." in text
    assert "llvm-objdump syntax" in text and "--toolchain" not in text and "keyed by compiler" not in text


# --- size phase -------------------------------------------------------------------------

FIXTURES = Path(__file__).parent / "fixtures" / "agent_loop"


def _fixture_ws(tmp_path: Path, name: str, campaign: Campaign) -> Workspace:
    """Workspace holding a real campaign ledger."""
    w = Workspace(tmp_path / name)
    w.ledger.mkdir(parents=True)
    shutil.copy(FIXTURES / f"{name}-ledger.jsonl", w.ledger / "ledger.jsonl")
    w.save(campaign, {"base_commit": "b" * 40, "ready": True})
    return w


def _status_cli(w: Workspace, *extra: str) -> str:
    from typer.testing import CliRunner

    from helia_core_tester.cli import app

    result = CliRunner().invoke(app, ["agent-loop", "status", "-w", str(w.root), "--tail", "0", *extra])
    assert result.exit_code == 0, result.output
    return result.output


def test_prompt_two_phases(atfe_root: Path) -> None:
    text = render_prompt(_campaign(toolchains=["gcc", "atfe"]), DW_ROWS, PATHS)
    assert "Two phases (of your 12 evals, the last 4 are for size):" in text
    assert "Speed phase, while `evals_left` is above 4: keep chasing speed" in text
    assert "Size phase, once `evals_left` is 4 or less" in text and "no\n  plausible speed idea left" in text
    assert "`code_size` per compiler\n  (gcc and atfe)" in text and "bytes saved per compiler" in text
    assert "per compiler (gcc / atfe)" in text and "names the smallest pass" in text
    assert "in the size phase it is your main target" in text and "Never trade correctness" in text
    assert "After your first pass" not in text and "Spend the remaining evals" not in text
    plain = render_prompt(_campaign(evals=6), DW_ROWS, PATHS)
    assert "Two phases (of your 6 evals, the last 2 are for size):" in plain and "per compiler" not in plain


def test_prompt_no_size_phase() -> None:
    text = render_prompt(_campaign(size_evals=0), DW_ROWS, PATHS)
    assert "Two phases" not in text and "size phase" not in text and "Size phase" not in text
    assert "5. After a pass, keep chasing speed" in text and "keep each pass lean" in text


def test_pass_names_the_phase(ws: Workspace, capsys) -> None:
    rc, view = _submit(ws, FakeBoard([_leg("pass"), _leg("pass")]), capsys)
    assert rc == 0 and view["next"] == "next: shrink code (1 eval left); fastest pass 001 1.10x at +64 B"
    assert json.loads((ws.results / "001.json").read_text())["next"] == view["next"]
    # No evals left, no next.
    rc, view = _submit(ws, FakeBoard([_leg("fail"), _leg("pass")]), capsys)
    assert rc == 1 and "next" not in view


def test_speed_phase_then_size_phase(ws: Workspace, capsys) -> None:
    _replace(ws, evals=5, size_evals=2)
    rc, view = _submit(ws, FakeBoard([_leg("fail"), _leg("pass")]), capsys)
    assert rc == 1 and "next" not in view
    rc, view = _submit(ws, FakeBoard([_leg("pass"), _leg("pass")]), capsys)
    assert view["next"] == "next: keep chasing speed (1 eval until size phase); fastest pass 002 1.10x at +64 B"
    # Size phase: fails carry next too.
    rc, view = _submit(ws, FakeBoard([_leg("fail"), _leg("pass")]), capsys)
    assert rc == 1 and view["next"] == "next: shrink code (2 evals left); fastest pass 002 1.10x at +64 B"


def test_speed_fail_has_no_next(ws: Workspace, capsys) -> None:
    _replace(ws, evals=6, size_evals=2)
    _submit(ws, FakeBoard([_leg("pass"), _leg("pass")]), capsys)
    rc, view = _submit(ws, FakeBoard([_leg("fail"), _leg("pass")]), capsys)
    assert rc == 1 and view["evals_left"] == 4 and "next" not in view


def test_no_size_phase_note(ws: Workspace, capsys) -> None:
    _replace(ws, evals=3, size_evals=0)
    _, view = _submit(ws, FakeBoard([_leg("pass"), _leg("pass")]), capsys)
    assert view["next"] == "next: keep chasing speed (2 evals left); fastest pass 001 1.10x at +64 B"
    _, view = _submit(ws, FakeBoard([_leg("fail"), _leg("pass")]), capsys)
    assert "next" not in view


@pytest.mark.parametrize("left, size_evals, text", [
    (5, 4, "speed phase (1 eval until size phase)"),
    (4, 4, "size phase (4 evals left)"),
    (1, 4, "size phase (1 eval left)"),
    (12, 0, "speed phase (12 evals left)"),
    (0, 0, "speed phase (0 evals left)"),
])
def test_phase_text(left, size_evals, text) -> None:
    assert ledger.phase_text(left, size_evals) == text
    assert ledger.size_phase(left, size_evals) == text.startswith("size")


def test_size_tie_names_fastest_only() -> None:
    rows = [{"eval": e, "verdict": "pass", "size_delta": 64, "legs": {"tcm": {"geomean": {"f": gain}}}}
            for e, gain in (("003", 3.0353), ("004", 3.0357))]
    picks = ledger.passing_evals(rows, _campaign().runs)
    assert ledger.next_note(picks, 2, 2, True) == "next: shrink code (2 evals left); fastest pass 004 3.04x at +64 B"


def test_free_evals_have_no_next(ws: Workspace, capsys, monkeypatch) -> None:
    monkeypatch.setattr(judge, "RETRY_PAUSE_S", 0)
    # Size phase after the pass.
    _replace(ws, evals=4, size_evals=3)
    _submit(ws, FakeBoard([_leg("pass"), _leg("pass")]), capsys)
    rejected = lambda *a, **k: (False, {"check": {"ok": False}})  # noqa: E731
    rc, view = _submit(ws, FakeBoard([]), capsys, checker=rejected)
    assert view["verdict"] == "rejected" and "next" not in view
    rc, view = _submit(ws, FakeBoard([None, None]), capsys)
    assert view["verdict"] == "error" and "not charged" in view["note"] and "next" not in view


def test_pass_note_names_fastest_and_smallest(ws: Workspace, atfe_root: Path, capsys) -> None:
    _two_toolchains(ws)
    _replace(ws, evals=3)
    board = FakeBoard([_leg_gain("pass", 2.0)] * 4)
    _submit(ws, board, capsys, checker=_keyed_check)
    small = lambda *a, **k: (True, {"check": {"ok": True}, "build": {"gcc": "ok", "atfe": "ok"},  # noqa: E731
                                    "code_size": {"gcc": {"delta_bytes": -8}, "atfe": {"delta_bytes": 16}}})
    _, view = _submit(ws, FakeBoard([_leg_gain("pass", 1.5)] * 4), capsys, checker=small)
    assert view["next"] == ("next: shrink code (1 eval left); fastest pass 001 2.00x / 2.00x at +64 B / +48 B; "
                            "smallest pass 002 1.50x / 1.50x at -8 B / +16 B")


def test_status_tradeoff_two_toolchains(tmp_path: Path, atfe_root: Path) -> None:
    w = _fixture_ws(tmp_path, "dw-s16", _campaign(toolchains=["gcc", "atfe"]))
    passes = agent.status(w, tail=0)["passes"]
    assert [(p["eval"], p["fastest"], p["smallest"], p["pareto"]) for p in passes] == [
        ("002", False, False, True), ("006", True, True, True)]
    assert passes[1]["toolchains"] == {"gcc": {"geomean": 2.7768, "size_delta": 3088},
                                       "atfe": {"geomean": 3.4246, "size_delta": 2154}}
    assert passes[1]["diff"] == str(w.ledger / "006.diff")
    assert ledger.next_note(passes, 6, 4, True) == ("next: keep chasing speed (2 evals until size phase); "
                                                    "fastest pass 006 2.78x / 3.42x at +3,088 B / +2,154 B")
    assert ledger.next_note(passes, 4, 4, False) == ("next: shrink code (4 evals left); "
                                                     "fastest pass 006 2.78x / 3.42x at +3,088 B / +2,154 B")
    text = _status_cli(w)
    assert "dw-s8: 6/12 evals, speed phase (2 evals until size phase), agent" in text
    assert "  eval   gcc speed   gcc bytes  atfe speed  atfe bytes  marks" in text
    assert "  006        2.78x      +3,088       3.42x      +2,154  fastest smallest pareto" in text
    assert f"diff {w.ledger / '002.diff'}" in text
    assert json.loads(_status_cli(w, "--json"))["passes"] == passes


def test_status_tradeoff_one_toolchain(tmp_path: Path) -> None:
    w = _fixture_ws(tmp_path, "conv-s8", _campaign(op="Convolve"))
    assert agent.status(w, tail=0)["passes"] == [] and "passing evals" not in _status_cli(w)
    rows = [json.loads(line) for line in (w.ledger / "ledger.jsonl").read_text().splitlines()]
    # Pretend 004 and 007 passed.
    for row in rows:
        row["verdict"] = "pass" if row["eval"] in ("004", "007") else row["verdict"]
    (w.ledger / "ledger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    passes = agent.status(w, tail=0)["passes"]
    assert passes[0]["toolchains"] == {"gcc": {"geomean": 1.0829, "size_delta": 1700}}
    assert [(p["eval"], p["fastest"], p["smallest"], p["pareto"]) for p in passes] == [
        ("004", False, True, True), ("007", True, False, True)]
    text = _status_cli(w)
    assert "dw-s8: 8/12 evals, size phase (4 evals left), agent" in text
    info = json.loads(_status_cli(w, "--json"))
    assert info["phase"] == "size phase (4 evals left)" and info["size_evals"] == 4
    assert "  eval   gcc speed   gcc bytes  marks" in text and "  004        1.08x      +1,700  smallest pareto" in text


def test_dominated_pass_is_marked() -> None:
    runs = _campaign().runs
    rows = [{"eval": e, "verdict": "pass", "size_delta": size, "legs": {"tcm": {"geomean": {"f": gain}}}}
            for e, size, gain in (("001", 100, 1.5), ("002", 200, 1.2), ("003", None, 2.0))]
    picks = ledger.passing_evals(rows, runs)
    # Unknown size still ranks on speed.
    assert [(p["fastest"], p["smallest"], p["pareto"]) for p in picks] == [
        (False, True, True), (False, False, False), (True, False, False)]
    assert ledger.next_note(picks, 2, 4, False) == ("next: shrink code (2 evals left); fastest pass 003 2.00x at ? B; "
                                                    "smallest pass 001 1.50x at +100 B")
