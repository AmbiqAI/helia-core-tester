"""Contract tests for the nightly hardware workflow.

Mirrors heliaPROFILER's `tests/test_hardware_validation_workflow.py`: the
things a hardware CI workflow gets wrong are invisible until a bench run fails
at 05:00, so the runner labels, the runner-owned environment, the artifact
name and the shell steps' argument assembly are pinned here rather than
discovered on a red nightly. The shell steps are executed under the host's real
bash with `uv` stubbed out, so a quoting or `[[ ]]` mistake fails a unit test.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "hardware-nightly.yml"

# heliaPROFILER's hardware validation nightly. These jobs ask for the same
# physical boards on the same runners, and a runner takes one job at a time, so
# the two schedules must not coincide.
HPX_NIGHTLY_CRON = "0 9 * * *"

# The runner answers these from its own environment. A job cannot be told which
# probe it may open: with several probes on one bench an input is ambiguous,
# and a job must not be able to name another board's probe.
RUNNER_OWNED_INPUTS = {"serial_no", "jlink_serial", "jlink_serials", "board"}


@pytest.fixture(scope="module")
def workflow() -> dict[Any, Any]:
    return yaml.safe_load(WORKFLOW_PATH.read_text())


@pytest.fixture(scope="module")
def benchmark_job(workflow: dict[Any, Any]) -> dict[str, Any]:
    return workflow["jobs"]["benchmark"]


def _triggers(workflow: dict[Any, Any]) -> dict[str, Any]:
    # YAML 1.1 loaders (PyYAML) read the bare ``on`` key as boolean True;
    # YAML 1.2 loaders keep the string. Accept either.
    return workflow[True] if True in workflow else workflow["on"]


def _inputs(workflow: dict[Any, Any]) -> dict[str, Any]:
    return _triggers(workflow)["workflow_dispatch"]["inputs"]


def _step(job: dict[str, Any], name: str) -> dict[str, Any]:
    for step in job["steps"]:
        if step.get("name") == name:
            return step
    raise AssertionError(f"step {name!r} not found")


def _run_bash(script: str, env: dict[str, str], cwd: Path) -> subprocess.CompletedProcess:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("workflow script tests require bash")
    return subprocess.run(
        [bash, "--noprofile", "--norc", "-euo", "pipefail", "-c", script],
        env={"PATH": os.environ["PATH"], **env},
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _guard_env(tmp_path: Path, **overrides: str) -> dict[str, str]:
    env = {
        "HPX_BOARD": "apollo510_evb",
        "HPX_JLINK_SERIAL": "1160003180",
        "HCT_NIGHTLY_BOARD": "apollo510_evb",
        "HCT_NIGHTLY_SESSION_SUFFIX": "",
        "HCT_CACHE_DIR": str(tmp_path / "cache"),
        "GITHUB_RUN_ID": "987654",
        "GITHUB_ENV": str(tmp_path / "env"),
        "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
        "RUNNER_NAME": "nmysore-nuc-apollo510_evb",
    }
    env.update(overrides)
    return env


def _exported(tmp_path: Path) -> dict[str, str]:
    lines = (tmp_path / "env").read_text().splitlines()
    return dict(line.split("=", 1) for line in lines if "=" in line)


# --- triggers and inputs ----------------------------------------------------


def test_nightly_schedule_does_not_collide_with_the_profiler_nightly(
    workflow: dict[Any, Any],
) -> None:
    crons = [entry["cron"] for entry in _triggers(workflow)["schedule"]]
    assert crons == ["0 5 * * *"]
    assert HPX_NIGHTLY_CRON not in crons


def test_probe_identity_is_never_a_workflow_input(workflow: dict[Any, Any]) -> None:
    inputs = _inputs(workflow)
    assert RUNNER_OWNED_INPUTS.isdisjoint(inputs)
    assert set(inputs) == {"boards", "suite", "family", "limit", "session_suffix"}


def test_suite_input_matches_the_cli_and_defaults_to_both(workflow: dict[Any, Any]) -> None:
    from helia_core_tester.hardware.session_runner import canonical_suite

    suite = _inputs(workflow)["suite"]
    assert suite["default"] == "both"
    assert set(suite["options"]) == {"int", "float", "both"}
    # Every option the dispatch offers must be one the CLI accepts.
    for option in suite["options"]:
        assert canonical_suite(option) == option
    assert workflow["env"]["HCT_NIGHTLY_SUITE"] == "${{ inputs.suite || 'both' }}"


def test_empty_boards_input_means_the_whole_board_table(workflow: dict[Any, Any]) -> None:
    """The scheduled run passes no inputs, so an explicit default board list in
    the workflow would silently keep a newly added table row out of the
    nightly. The plan job resolves "" to every board instead."""
    assert _inputs(workflow)["boards"]["default"] == ""
    assert workflow["env"]["HCT_NIGHTLY_BOARDS"] == "${{ inputs.boards || '' }}"


def test_optional_narrowing_inputs_default_to_empty(workflow: dict[Any, Any]) -> None:
    inputs = _inputs(workflow)
    for name in ("family", "limit", "session_suffix"):
        assert inputs[name]["default"] == ""
        assert inputs[name].get("required", False) is False


# --- plan job ---------------------------------------------------------------


def test_plan_job_derives_the_matrix_from_the_board_table(workflow: dict[Any, Any]) -> None:
    plan = workflow["jobs"]["plan"]
    assert plan["outputs"]["boards"] == "${{ steps.matrix.outputs.boards }}"
    script = _step(plan, "Build the board matrix")["run"]
    assert "helia_core_tester/scripts/board_matrix.py" in script
    assert '--boards "${HCT_NIGHTLY_BOARDS}"' in script
    # --no-project keeps the hosted plan job off the project's dependency set.
    assert "--no-project" in script
    assert (REPO_ROOT / "helia_core_tester" / "scripts" / "board_matrix.py").is_file()


# --- job identity -----------------------------------------------------------


def test_one_job_per_board_pinned_by_board_label(
    workflow: dict[Any, Any], benchmark_job: dict[str, Any]
) -> None:
    assert benchmark_job["needs"] == "plan"
    assert benchmark_job["strategy"]["fail-fast"] is False
    assert benchmark_job["strategy"]["matrix"]["board"] == (
        "${{ fromJSON(needs.plan.outputs.boards) }}"
    )
    assert benchmark_job["runs-on"] == ["self-hosted", "hpx-hardware", "${{ matrix.board }}"]
    assert benchmark_job["env"]["HCT_NIGHTLY_BOARD"] == "${{ matrix.board }}"
    # Runner exclusivity is the only serialisation, exactly as in hpx's
    # workflow: a group keyed by board would throttle several runners of one
    # board type back to one job.
    assert "concurrency" not in benchmark_job
    assert "concurrency" not in workflow
    assert isinstance(benchmark_job["timeout-minutes"], int)


def test_uv_caches_live_outside_the_runners_read_only_home(workflow: dict[Any, Any]) -> None:
    """The runner service's HOME is a root-owned NixOS-managed directory."""
    for key in ("UV_CACHE_DIR", "UV_PYTHON_INSTALL_DIR", "HCT_CACHE_DIR"):
        assert workflow["env"][key].startswith("${{ github.workspace }}/../")


# --- the guard step ---------------------------------------------------------


def test_guard_runs_before_anything_touches_the_checkout(benchmark_job: dict[str, Any]) -> None:
    guard = _step(benchmark_job, "Resolve board and probe from the runner")
    assert benchmark_job["steps"][0] is guard
    script = guard["run"]
    assert '-z "${HPX_BOARD:-}" || -z "${HPX_JLINK_SERIAL:-}"' in script
    assert '"${HPX_BOARD}" != "${HCT_NIGHTLY_BOARD}"' in script
    # jq is part of the runner contract; a bench without it fails here rather
    # than an hour later in the summary step.
    assert "command -v jq" in script


@pytest.mark.parametrize(
    "missing", ["HPX_BOARD", "HPX_JLINK_SERIAL"]
)
def test_guard_fails_when_the_runner_exports_no_probe(
    benchmark_job: dict[str, Any], tmp_path: Path, missing: str
) -> None:
    env = _guard_env(tmp_path, **{missing: ""})
    result = _run_bash(_step(benchmark_job, "Resolve board and probe from the runner")["run"], env, tmp_path)
    assert result.returncode == 2
    assert "runner contract" in result.stderr


def test_guard_fails_when_the_runner_owns_another_board(
    benchmark_job: dict[str, Any], tmp_path: Path
) -> None:
    env = _guard_env(tmp_path, HPX_BOARD="apollo330mP_evb")
    result = _run_bash(_step(benchmark_job, "Resolve board and probe from the runner")["run"], env, tmp_path)
    assert result.returncode == 2
    assert "check the runner's labels" in result.stderr


def test_guard_derives_the_session_id_and_build_dir(
    benchmark_job: dict[str, Any], tmp_path: Path
) -> None:
    env = _guard_env(tmp_path)
    result = _run_bash(_step(benchmark_job, "Resolve board and probe from the runner")["run"], env, tmp_path)
    assert result.returncode == 0
    exported = _exported(tmp_path)
    assert exported["HCT_SESSION_ID"] == "nightly-987654-apollo510_evb"
    assert exported["HCT_BUILD_DIR"] == str(tmp_path / "cache" / "hardware" / "apollo510_evb")
    summary = (tmp_path / "summary").read_text()
    assert "1160003180" in summary and "nightly-987654-apollo510_evb" in summary


def test_guard_appends_a_session_suffix_and_refuses_a_path_unsafe_one(
    benchmark_job: dict[str, Any], tmp_path: Path
) -> None:
    script = _step(benchmark_job, "Resolve board and probe from the runner")["run"]
    ok = _run_bash(script, _guard_env(tmp_path, HCT_NIGHTLY_SESSION_SUFFIX="asm-off"), tmp_path)
    assert ok.returncode == 0
    assert _exported(tmp_path)["HCT_SESSION_ID"] == "nightly-987654-apollo510_evb-asm-off"
    # The session id is also the bundle directory name.
    bad = _run_bash(script, _guard_env(tmp_path, HCT_NIGHTLY_SESSION_SUFFIX="../escape"), tmp_path)
    assert bad.returncode == 2
    assert "session_suffix" in bad.stderr


def test_guard_points_python_at_the_hosts_ca_bundle(
    benchmark_job: dict[str, Any], tmp_path: Path
) -> None:
    """`hardware run` fetches ARM GCC over HTTPS from Python on a cold
    checkout, and the uv-managed interpreter's compiled-in certificate
    directory does not exist on the NixOS benches."""
    script = _step(benchmark_job, "Resolve board and probe from the runner")["run"]
    bundle = tmp_path / "ca-certificates.crt"
    bundle.write_text("")
    patched = script.replace("/etc/ssl/certs/ca-certificates.crt", str(bundle), 1)
    assert patched != script

    result = _run_bash(patched, _guard_env(tmp_path), tmp_path)
    assert result.returncode == 0, result.stderr
    assert _exported(tmp_path)["SSL_CERT_FILE"] == str(bundle)

    # A runner that already declares one keeps it.
    (tmp_path / "env").write_text("")
    result = _run_bash(patched, _guard_env(tmp_path, SSL_CERT_FILE="/runner/own.pem"), tmp_path)
    assert result.returncode == 0, result.stderr
    assert "SSL_CERT_FILE" not in _exported(tmp_path)


def test_the_download_cache_survives_the_checkouts_clean(
    benchmark_job: dict[str, Any], tmp_path: Path
) -> None:
    """ARM GCC and CMSIS_5 land in artifacts/downloads, which
    actions/checkout removes with every other ignored file."""
    from helia_core_tester.hardware.toolchain import DOWNLOADS_DIR

    step = _step(benchmark_job, "Link the download cache into the workspace")
    assert DOWNLOADS_DIR == "artifacts/downloads"
    assert DOWNLOADS_DIR in step["run"]
    # It must come after the checkout that would otherwise delete the link.
    names = [entry.get("name") for entry in benchmark_job["steps"]]
    assert names.index(step["name"]) > names.index("Checkout")

    env = {"HCT_CACHE_DIR": str(tmp_path / "cache")}
    stale = tmp_path / DOWNLOADS_DIR
    stale.mkdir(parents=True)
    (stale / "left-over").write_text("")
    result = _run_bash(step["run"], env, tmp_path)
    assert result.returncode == 0, result.stderr
    assert stale.is_symlink()
    assert stale.resolve() == (tmp_path / "cache" / "downloads").resolve()


def test_missing_flatc_is_reported_but_never_fails_the_job(
    benchmark_job: dict[str, Any], tmp_path: Path
) -> None:
    """flatc is reached only on generation's LSTM path, to rewrite a converted
    .tflite. Every step of that path is wrapped and falls back to the validated
    ns-cmsis-nn UnitTest reference vectors, which `hardware run` points
    generation at through CMSIS_NN_ROOT -- so LSTM cases are generated and
    streamed with or without it. Failing the job over flatc would take a whole
    board's nightly out for a build input it does not need."""
    script = _step(benchmark_job, "Resolve board and probe from the runner")["run"]
    empty_path = tmp_path / "nopath"
    empty_path.mkdir(exist_ok=True)
    for name in ("bash", "jq", "sed", "printf"):
        found = shutil.which(name)
        if found:
            (empty_path / name).symlink_to(found)
    env = _guard_env(tmp_path)
    env["PATH"] = str(empty_path)
    result = _run_bash(script, env, tmp_path)
    assert result.returncode == 0, result.stderr
    assert "flatc is not on the runner PATH" in result.stdout
    assert _exported(tmp_path)["HCT_FLATC_STATE"].startswith("absent")


# --- preflight and run steps ------------------------------------------------


def test_preflight_uses_the_runners_board_and_probe(benchmark_job: dict[str, Any]) -> None:
    doctor = _step(benchmark_job, "Check host dependencies")
    assert doctor["run"].strip() == "uv run helia_core_tester doctor"
    probes = _step(benchmark_job, "Check the runner can open its probe")
    assert '--board "${HCT_NIGHTLY_BOARD}"' in probes["run"]
    # `probes match` resolves the serial from $HPX_JLINK_SERIAL, which is
    # exactly the value the run step passes as --serial-no.
    assert "probes match" in probes["run"]


def test_preview_rejects_a_family_with_no_firmware_dispatch(
    benchmark_job: dict[str, Any], tmp_path: Path
) -> None:
    from helia_core_tester.hardware.generated_test_bridge import bridged_families

    script = _step(benchmark_job, "Preview the selection")["run"]
    families = bridged_families()
    assert families, "the bridge registers no families at all"
    ok = _run_bash(script, {"HCT_NIGHTLY_FAMILY": families[0], "HOME": os.environ["HOME"]}, REPO_ROOT)
    assert ok.returncode == 0, ok.stderr
    bad = _run_bash(script, {"HCT_NIGHTLY_FAMILY": "NotAFamily", "HOME": os.environ["HOME"]}, REPO_ROOT)
    assert bad.returncode == 2
    assert "would select no case" in bad.stderr


@pytest.mark.parametrize("family", ["", "BasicMathFunctions"])
@pytest.mark.parametrize("limit", ["", "8"])
def test_run_step_passes_the_runner_owned_identity_and_optional_narrowing(
    benchmark_job: dict[str, Any], tmp_path: Path, family: str, limit: str
) -> None:
    run_json = tmp_path / "run.json"
    env = {
        "HPX_BOARD": "apollo510_evb",
        "HPX_JLINK_SERIAL": "1160003180",
        "HCT_NIGHTLY_SUITE": "int",
        "HCT_NIGHTLY_FAMILY": family,
        "HCT_NIGHTLY_LIMIT": limit,
        "HCT_SESSION_ID": "nightly-987654-apollo510_evb",
        "HCT_BUILD_DIR": str(tmp_path / "bd"),
        "HCT_RUN_JSON": str(run_json),
        "ARGS_FILE": str(tmp_path / "args"),
    }
    # `uv run <cmd> ...` -> record the arguments, then write a plausible
    # document so the totals check has something to read.
    stub = (
        'uv() { shift; printf "%s\\n" "$@" > "${ARGS_FILE}"; '
        'printf \'{"totals":{"ran":3,"passed":3,"failed":0,"skipped":0}}\'; }\n'
    )
    result = _run_bash(stub + _step(benchmark_job, "Run the hardware kernel suite")["run"], env, tmp_path)
    assert result.returncode == 0, result.stderr
    args = (tmp_path / "args").read_text().splitlines()
    assert args[:3] == ["helia_core_tester", "hardware", "run"]
    assert args[args.index("--board") + 1] == "apollo510_evb"
    assert args[args.index("--serial-no") + 1] == "1160003180"
    assert args[args.index("--session-id") + 1] == "nightly-987654-apollo510_evb"
    assert args[args.index("--build-dir") + 1] == str(tmp_path / "bd")
    assert "--json" in args
    if family:
        assert args[args.index("--family") + 1] == family
    else:
        assert "--family" not in args
    if limit:
        assert args[args.index("--limit") + 1] == limit
    else:
        assert "--limit" not in args


def test_run_step_fails_when_the_selection_ran_no_case(
    benchmark_job: dict[str, Any], tmp_path: Path
) -> None:
    """A green job must mean cases ran, not that the selection was empty."""
    env = {
        "HPX_BOARD": "apollo510_evb",
        "HPX_JLINK_SERIAL": "1160003180",
        "HCT_NIGHTLY_SUITE": "int",
        "HCT_NIGHTLY_FAMILY": "",
        "HCT_NIGHTLY_LIMIT": "",
        "HCT_SESSION_ID": "s",
        "HCT_BUILD_DIR": str(tmp_path / "bd"),
        "HCT_RUN_JSON": str(tmp_path / "run.json"),
    }
    stub = 'uv() { printf \'{"totals":{"ran":0,"passed":0,"failed":0,"skipped":9}}\'; }\n'
    result = _run_bash(stub + _step(benchmark_job, "Run the hardware kernel suite")["run"], env, tmp_path)
    assert result.returncode == 2
    assert "selected no case on apollo510_evb" in result.stderr


def test_summary_reads_the_fields_the_cli_actually_writes(
    benchmark_job: dict[str, Any], tmp_path: Path
) -> None:
    """Pins the step summary against `build_json_summary`'s real document."""
    from helia_core_tester.hardware.run_summary import build_json_summary

    class _Comparison:
        passed = False

    class _Statistics:
        median_cycles = 1234.0
        valid_for_regression = True

    class _CaseBundle:
        case_id = "add_broadcast_s16_hw_generated"

    class _Case:
        comparison = _Comparison()
        statistics = _Statistics()
        case_bundle = _CaseBundle()

    class _Result:
        cases = [_Case()]

    class _Skipped:
        name = "lstm_unidirectional_s8_hw_generated"

    document = build_json_summary(
        _Result(),
        [(_Skipped(), "no hardware dispatch")],
        session_id="nightly-987654-apollo510_evb",
        board_id="apollo510_evb",
        bundle=tmp_path / "artifacts" / "reports" / "hardware" / "nightly-987654-apollo510_evb",
        timing={"total_s": 12.5},
        dependencies={
            "qualification": "qualified",
            "unqualified_reasons": [],
            "workspace": {"inputs": {"kernel_source": "ns-cmsis-nn@" + "7" * 40}},
        },
    )
    run_json = tmp_path / "run.json"
    run_json.write_text(json.dumps(document))
    summary = tmp_path / "summary"
    summary.write_text("")
    env = {
        "HCT_RUN_JSON": str(run_json),
        "HCT_NIGHTLY_BOARD": "apollo510_evb",
        "GITHUB_STEP_SUMMARY": str(summary),
    }
    result = _run_bash(_step(benchmark_job, "Summarise the run")["run"], env, tmp_path)
    assert result.returncode == 0, result.stderr
    text = summary.read_text()
    assert "1 ran, 0 passed, 1 failed, 1 skipped" in text
    assert "ns-cmsis-nn@" + "7" * 40 in text
    assert "`qualified`" in text
    assert "add_broadcast_s16_hw_generated" in text
    assert "unrecorded" not in text


def test_summary_survives_a_job_that_never_produced_a_document(
    benchmark_job: dict[str, Any], tmp_path: Path
) -> None:
    summary = tmp_path / "summary"
    summary.write_text("")
    env = {
        "HCT_RUN_JSON": str(tmp_path / "absent.json"),
        "HCT_NIGHTLY_BOARD": "apollo510_evb",
        "GITHUB_STEP_SUMMARY": str(summary),
    }
    result = _run_bash(_step(benchmark_job, "Summarise the run")["run"], env, tmp_path)
    assert result.returncode == 0, result.stderr
    assert "No run summary" in summary.read_text()


# --- artifacts --------------------------------------------------------------


def test_bundles_are_uploaded_per_board_even_when_the_run_failed(
    benchmark_job: dict[str, Any],
) -> None:
    upload = _step(benchmark_job, "Upload the result bundles")
    assert upload["if"] == "always()"
    assert upload["with"]["name"] == "hardware-kernels-${{ github.run_id }}-${{ matrix.board }}"
    # The upload runs on failure too, so re-running a failed board job meets
    # its own first-attempt artifact; without overwrite the re-run fails.
    assert upload["with"]["overwrite"] is True
    paths = [line for line in upload["with"]["path"].splitlines() if line.strip()]
    assert "artifacts/reports/hardware/" in paths
    assert _step(benchmark_job, "Summarise the run")["if"] == "always()"


def test_the_upload_path_is_where_the_cli_writes_its_bundles(tmp_path: Path) -> None:
    """Pins the artifact path against `result_bundle`'s own layout rather than
    a hard-coded string in the workflow that could drift from it."""
    from helia_core_tester.hardware.result_bundle import bundle_root_for

    root = bundle_root_for(tmp_path, "nightly-987654-apollo510_evb")
    assert root.relative_to(tmp_path).as_posix() == (
        "artifacts/reports/hardware/nightly-987654-apollo510_evb"
    )
