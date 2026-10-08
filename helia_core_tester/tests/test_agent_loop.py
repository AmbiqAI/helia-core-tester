from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from helia_core_tester.agent_loop import agent, judge, ledger
from helia_core_tester.agent_loop.config import Campaign, ConfigError, from_json, parse_campaign
from helia_core_tester.agent_loop.prompt import render_prompt, route_facts
from helia_core_tester.agent_loop.workspace import Workspace

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
    ({"board": "apollo3p_evb", "legs": ["mram"]}, "no cached MRAM"),
    ({"evals": 0}, "evals"),
    ({"evals": True}, "wrong type"),
    ({"cost_usd": -1}, "cost_usd"),
    ({"op": "FullyConnected"}, "hidden_shapes"),
    ({"op": "Conv 2d"}, "one word"),
    ({"case_ids": ["ok", "a b"]}, "case_ids"),
])
def test_config_rejects(over, message) -> None:
    with pytest.raises(ConfigError, match=message):
        _campaign(**over)


def test_config_missing_key() -> None:
    data = {k: v for k, v in BASE.items() if k != "secrets_dir"}
    with pytest.raises(ConfigError, match="secrets_dir"):
        parse_campaign(data, Path("/cfg"))


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


def _ok_check(ws, campaign, base, area):
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


def test_submit_respects_deadline(ws: Workspace, capsys, monkeypatch) -> None:
    campaign, facts = ws.load()
    ws.save(Campaign(**{**campaign.__dict__, "submit_deadline_s": 120, "eval_timeout_s": 300}), facts)
    board = FakeBoard([_leg("pass"), _leg("pass")])
    _submit(ws, board, capsys)
    call = board.calls[0]
    lock_s, eval_s = int(call[call.index("--timeout") + 1]), int(call[call.index("timeout") + 1])
    assert eval_s <= 110 and lock_s + eval_s <= 120


def test_submit_precheck_is_free(ws: Workspace, capsys) -> None:
    def bad(ws, campaign, base, area):
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
    assert args[args.index("--max-budget-usd") + 1] == "25" and args[args.index("--session-id") + 1] == meta["session_id"]
    assert args[args.index("--permission-mode") + 1] == "dontAsk" and "--no-session-persistence" not in args
