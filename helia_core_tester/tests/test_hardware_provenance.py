"""Dependency provenance derived from `nsx.lock`.

Everything here runs against a fake `nsx.lock` (parsed by neuralspotx's own
reader, so the shape under test is the real one) and a fake compile database --
no board, no build, no network.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from helia_core_tester.hardware import provenance
from helia_core_tester.hardware.boards import resolve_board
from helia_core_tester.hardware.case_bundle import build_abs_s8_case_bundle, load_case_bundle
from helia_core_tester.hardware.dependency_baseline import parse_baseline
from helia_core_tester.hardware.fake_target import FakeTargetTransport
from helia_core_tester.hardware.nsx_app import AppRender, KernelOptions, KernelSource, ModuleSpec
from helia_core_tester.hardware.result_bundle import write_result_bundle
from helia_core_tester.hardware.session import HostSession

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BOARD = resolve_board("apollo510_evb")

KERNEL_PIN = "1a" * 20
SDK_PIN = "2b" * 20
NSX_PIN = "3c" * 20
PMU_PIN = "4d" * 20
#: A real (if improbable) all-digit commit SHA: YAML reads it as an integer.
NUMERIC_COMMIT = "9" * 40

_BASELINE_DOC = {
    "schema": "hct.dependency-baseline",
    "schema_version": 1,
    "baseline_id": "test-baseline-0001",
    "projects": {
        "ns-cmsis-nn": {"url": "https://github.com/AmbiqAI/ns-cmsis-nn.git", "ref": KERNEL_PIN},
        "nsx-ambiq-sdk": {"url": "https://github.com/AmbiqAI/nsx-ambiq-sdk.git", "ref": SDK_PIN},
        "neuralspotx": {"url": "https://github.com/AmbiqAI/neuralspotx.git", "ref": NSX_PIN},
        "nsx-pmu-armv8m": {"url": "https://github.com/AmbiqAI/nsx-pmu-armv8m.git", "ref": PMU_PIN},
    },
}


def _baseline(path: Path | None = None):
    return parse_baseline(json.loads(json.dumps(_BASELINE_DOC)), path=path)


def _lock_text(*, kernel: str, sdk_commit: str = SDK_PIN) -> str:
    """A minimal but real nsx.lock: one git module per pinned project, the packaged
    board module, and whatever `kernel` says the kernels resolved to."""
    return f"""\
schema_version: 4
targets:
  {BOARD.nsx_board}:
    generated_at: '2026-09-18T00:00:00+00:00'
    nsx_tool:
      version: 0.2.0
    manifest:
      path: nsx.yml
      hash: sha256:{'b' * 64}
    target:
      board: {BOARD.nsx_board}
      soc: {BOARD.soc}
      toolchain: arm-none-eabi-gcc
    modules:
      nsx-ambiqsuite:
        project: nsx-ambiq-sdk
        kind: git
        constraint: {sdk_commit}
        resolved:
          url: https://github.com/AmbiqAI/nsx-ambiq-sdk.git
          commit: {sdk_commit}
          vendored_at: modules/nsx-ambiq-sdk
          content_hash: sha256:{'c' * 64}
          acquired_at: '2026-09-18T00:00:01+00:00'
      nsx-board-apollo510-evb:
        project: neuralspotx
        kind: packaged
        constraint: packaged
        resolved:
          tool_version: 0.2.0
          vendored_at: boards/{BOARD.nsx_board}
          content_hash: sha256:{'d' * 64}
          acquired_at: '2026-09-18T00:00:01+00:00'
{kernel}"""


def _git_kernel_module(commit: str = KERNEL_PIN, *, tag: str | None = None) -> str:
    tag_line = f"\n          tag: {tag}" if tag else ""
    return f"""\
      nsx-cmsis-nn:
        project: ns-cmsis-nn
        kind: git
        constraint: {commit}
        resolved:
          url: https://github.com/AmbiqAI/ns-cmsis-nn.git{tag_line}
          commit: {commit}
          vendored_at: modules/ns-cmsis-nn
          content_hash: sha256:{'e' * 64}
          acquired_at: '2026-09-18T00:00:01+00:00'
"""


def _local_kernel_module(path: Path) -> str:
    return f"""\
      nsx-cmsis-nn:
        project: ns-cmsis-nn
        kind: local
        constraint: path:{path}
        resolved:
          vendored_at: modules/ns-cmsis-nn
          content_hash: sha256:{'f' * 64}
          acquired_at: '2026-09-18T00:00:01+00:00'
"""


def _render(app_dir: Path, *, baseline=None, kernel_source: KernelSource | None = None) -> AppRender:
    """An AppRender with the fields provenance reads; the rendered text is arbitrary
    because only its digest matters here."""
    return AppRender(
        app_dir=app_dir,
        board=BOARD,
        modules=(
            ModuleSpec("nsx-ambiqsuite", "nsx-ambiq-sdk"),
            ModuleSpec("nsx-cmsis-nn", "ns-cmsis-nn"),
        ),
        kernel_source=kernel_source or KernelSource(ref=KERNEL_PIN),
        kernel_options=KernelOptions(),
        baseline=baseline or _baseline(),
        nsx_yml="# nsx.yml\n",
        modules_cmake="# modules.cmake\n",
        cmakelists="# CMakeLists.txt\n",
    )


def _build(tmp_path: Path, *, kernel: str, kernel_source: KernelSource | None = None, baseline=None, **kwargs) -> dict:
    app_dir = tmp_path / "bd" / "nsx_app"
    app_dir.mkdir(parents=True, exist_ok=True)
    (app_dir / "nsx.lock").write_text(_lock_text(kernel=kernel), encoding="utf-8")
    render = _render(app_dir, baseline=baseline, kernel_source=kernel_source)
    return provenance.build_provenance(
        render, repo_root=tmp_path, build_dir=tmp_path / "bd",
        lock_mode=provenance.LOCK_REUSED, **kwargs,
    )


def _git_repo(root: Path, *, dirty: bool = False) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "kernel.c").write_text("int main(void) { return 0; }\n")
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=root, check=True, env={**dict(__import__("os").environ), **env})
    if dirty:
        (root / "kernel.c").write_text("int main(void) { return 1; }\n")
    return root


# --- the block ----------------------------------------------------------------------


def test_block_is_written_in_the_hpx_shape(tmp_path: Path) -> None:
    document = _build(tmp_path, kernel=_git_kernel_module())

    assert document["schema"] == "hct.hardware.dependencies"
    # The four hpx `DependencyProvenance.to_dict()` sections, spelled the same way.
    assert {"workspace", "lock", "modules", "overrides", "qualification"} <= set(document)
    assert set(document["workspace"]) == {
        "schema_version", "fingerprint", "baseline_id", "baseline_fingerprint", "registry_hash", "inputs",
    }
    assert set(document["lock"]) == {
        "mode", "update_requested", "offline", "frozen_sync", "schema_version", "sha256", "manifest_hash",
    }
    assert document["lock"]["mode"] == "reused"
    assert document["lock"]["frozen_sync"] is True
    assert document["lock"]["schema_version"] == 4
    # Digests are hpx `ContentDigest` objects, with the "sha256:" the lock spells
    # stripped -- the algorithm is the field next to the value, not a prefix on it.
    assert document["lock"]["manifest_hash"] == {"algorithm": "sha256", "value": "b" * 64}
    assert document["lock"]["sha256"]["algorithm"] == "sha256"

    modules = {module["name"]: module for module in document["modules"]}
    assert [module["name"] for module in document["modules"]] == sorted(modules), "modules are sorted by name"
    assert set(modules["nsx-cmsis-nn"]) == {
        "name", "project", "kind", "requested_ref", "requested_tag",
        "peeled_commit", "content_hash", "url", "vendored_at",
    }
    assert modules["nsx-cmsis-nn"] == {
        "name": "nsx-cmsis-nn",
        "project": "ns-cmsis-nn",
        "kind": "git",
        "requested_ref": KERNEL_PIN,
        "requested_tag": None,
        "peeled_commit": KERNEL_PIN,
        "content_hash": {"algorithm": "sha256", "value": "e" * 64},
        "url": "https://github.com/AmbiqAI/ns-cmsis-nn.git",
        "vendored_at": "modules/ns-cmsis-nn",
    }
    # A packaged module has no commit or URL to record; its content hash is the
    # whole of its identity, and NSX's own kind is what says so.
    packaged = modules["nsx-board-apollo510-evb"]
    assert packaged["kind"] == "packaged" and packaged["peeled_commit"] is None and packaged["url"] is None
    assert packaged["content_hash"]["value"] == "d" * 64

    assert document["qualification"] == "qualified"
    assert document["unqualified_reasons"] == []
    assert document["overrides"] == []
    assert document["workspace"]["baseline_id"] == "test-baseline-0001"
    assert document["workspace"]["inputs"]["board"] == BOARD.id
    assert document["workspace"]["inputs"]["kernel_source"] == f"ns-cmsis-nn@{KERNEL_PIN}"


def test_a_requested_tag_is_carried_through(tmp_path: Path) -> None:
    document = _build(tmp_path, kernel=_git_kernel_module(tag="v7.23.0"))
    kernel = next(m for m in document["modules"] if m["name"] == "nsx-cmsis-nn")
    assert kernel["requested_tag"] == "v7.23.0" and kernel["peeled_commit"] == KERNEL_PIN


def test_a_lock_that_resolved_off_a_baseline_pin_is_not_qualified(tmp_path: Path) -> None:
    # NSX gives a packaged registry's module-level revision precedence over an
    # app's project-level override, so a manifest that asserts the pin is not
    # proof the lock resolved to it. The claim is only made after comparing.
    document = _build(tmp_path, kernel=_git_kernel_module("5e" * 20))

    assert document["qualification"] == "development-overrides"
    [reason] = document["unqualified_reasons"]
    assert "nsx-cmsis-nn" in reason and "5e" * 20 in reason and KERNEL_PIN in reason


def test_an_all_digit_commit_stays_a_string(tmp_path: Path) -> None:
    # YAML reads a 40-digit SHA as an integer; a peeled_commit that is a number
    # compares equal to nothing a reader will hold against it.
    document = _build(tmp_path, kernel=_git_kernel_module(NUMERIC_COMMIT))
    kernel = next(m for m in document["modules"] if m["name"] == "nsx-cmsis-nn")
    assert kernel["peeled_commit"] == NUMERIC_COMMIT and kernel["requested_ref"] == NUMERIC_COMMIT
    assert json.loads(json.dumps(document))["modules"] == document["modules"]


def test_a_path_override_is_recorded_and_disqualifies_the_build(tmp_path: Path) -> None:
    checkout = _git_repo(tmp_path / "ns-cmsis-nn")
    document = _build(
        tmp_path,
        kernel=_local_kernel_module(checkout),
        kernel_source=KernelSource(path=checkout),
    )

    assert document["qualification"] == "development-overrides"
    [override] = document["overrides"]
    assert override["scope"] == "module" and override["name"] == "nsx-cmsis-nn"
    assert override["mode"] == "path" and override["requested"] == str(checkout)
    # The content hash is the lock's, not a second digest computed here: one
    # identity for the tree, and it is the one NSX verified on sync.
    assert override["content_hash"] == {"algorithm": "sha256", "value": "f" * 64}
    # The tester's own addition: which commit the edited checkout was on.
    assert override["local_checkout"]["commit"] and override["local_checkout"]["dirty"] is False
    assert any("local path" in reason for reason in document["unqualified_reasons"])
    # The overridden project is excluded from the baseline comparison -- the
    # override *is* the intent -- so it is named once, as an override.
    assert not any(KERNEL_PIN in reason for reason in document["unqualified_reasons"])


def test_a_dirty_override_checkout_is_named(tmp_path: Path) -> None:
    checkout = _git_repo(tmp_path / "ns-cmsis-nn", dirty=True)
    document = _build(tmp_path, kernel=_local_kernel_module(checkout), kernel_source=KernelSource(path=checkout))

    [override] = document["overrides"]
    assert override["local_checkout"]["dirty"] is True
    assert any("uncommitted changes" in reason for reason in document["unqualified_reasons"])


def test_a_tree_git_cannot_describe_is_treated_as_dirty(tmp_path: Path) -> None:
    checkout = tmp_path / "loose-tree"
    checkout.mkdir()
    state = provenance.local_checkout_state(checkout)
    assert state["commit"] is None and state["dirty"] is True


def test_an_explicit_baseline_file_is_recorded_as_an_override(tmp_path: Path) -> None:
    path = tmp_path / "hpx-baseline.json"
    path.write_text(json.dumps(_BASELINE_DOC))
    document = _build(tmp_path, kernel=_git_kernel_module(), baseline=_baseline(path))

    [override] = document["overrides"]
    assert override == {
        "scope": "baseline",
        "name": "test-baseline-0001",
        "mode": "file",
        "requested": str(path),
        "content_hash": {"algorithm": "sha256", "value": _baseline(path).fingerprint},
    }
    # A different qualified baseline is still a qualified build -- it names pins,
    # it does not bypass them.
    assert document["qualification"] == "qualified"


def test_a_missing_lock_is_refused(tmp_path: Path) -> None:
    app_dir = tmp_path / "bd" / "nsx_app"
    app_dir.mkdir(parents=True)
    with pytest.raises(provenance.ProvenanceError, match="without an exact NSX lock"):
        provenance.build_provenance(
            _render(app_dir), repo_root=tmp_path, build_dir=tmp_path / "bd", lock_mode=provenance.LOCK_REUSED
        )


# --- build images ---------------------------------------------------------------------


def _compile_commands(path: Path, entries: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries))
    return path


def test_architecture_flags_count_kernel_and_server_units_only(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "app" / "modules" / "ns-cmsis-nn"
    server_dir = tmp_path / "repo" / "cmake" / "hardware"
    database = _compile_commands(
        tmp_path / "build" / "compile_commands.json",
        [
            {"file": str(kernel_dir / "Source" / "conv.c"), "command": "gcc -mcpu=cortex-m55 -mfloat-abi=hard -c conv.c"},
            {"file": str(kernel_dir / "Source" / "fc.c"), "arguments": ["gcc", "-mcpu=cortex-m55", "-mfloat-abi=hard", "-c", "fc.c"]},
            {"file": str(server_dir / "benchmark_server_main.c"), "command": "gcc -mcpu=cortex-m55 -mfloat-abi=hard -c main.c"},
            # An SDK unit built for a different core: real, and deliberately not
            # counted -- it is not code any measurement runs.
            {"file": str(tmp_path / "app" / "modules" / "nsx-ambiq-sdk" / "uart.c"), "command": "gcc -mcpu=cortex-m4 -c uart.c"},
        ],
    )

    flags, units = provenance.architecture_flags(database, kernel_dir=kernel_dir, server_dir=server_dir)
    assert flags == {"-mcpu=cortex-m55": 3, "-mfloat-abi=hard": 3}
    assert units == {"kernel": 2, "server": 1}


def test_a_mixed_isa_build_is_visible_as_two_counts(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernels"
    server_dir = tmp_path / "server"
    database = _compile_commands(
        tmp_path / "compile_commands.json",
        [
            {"file": str(kernel_dir / "a.c"), "command": "gcc -mcpu=cortex-m55 -c a.c"},
            {"file": str(kernel_dir / "b.c"), "command": "gcc -mcpu=cortex-m55+nomve -c b.c"},
            # A flag repeated on one command line must not outvote a unit.
            {"file": str(server_dir / "c.c"), "command": "gcc -mcpu=cortex-m55 -mcpu=cortex-m55 -c c.c"},
        ],
    )
    flags, _ = provenance.architecture_flags(database, kernel_dir=kernel_dir, server_dir=server_dir)
    assert flags == {"-mcpu=cortex-m55": 2, "-mcpu=cortex-m55+nomve": 1}


def test_build_image_records_the_binary_and_its_build_id(tmp_path: Path) -> None:
    binary = tmp_path / "hct_benchmark_server.elf"
    binary.write_bytes(b"elf-bytes")
    image = provenance.build_image_record(
        role="benchmark-server",
        target_name="hct_benchmark_server",
        binary=binary,
        build_id="hct-abc",
        compile_commands=tmp_path / "missing.json",
        kernel_dir=tmp_path / "kernels",
        server_dir=tmp_path / "server",
    )
    assert image["sha256"] and image["size_bytes"] == len(b"elf-bytes")
    assert image["build_id"] == "hct-abc" and image["binary_name"] == "hct_benchmark_server.elf"
    # A missing compile database costs the flags, not the record.
    assert image["architecture_flags"] == {} and image["translation_units"] == 0


def test_build_images_and_toolchain_are_carried(tmp_path: Path) -> None:
    binary = tmp_path / "fw.elf"
    binary.write_bytes(b"fw")
    document = _build(tmp_path, kernel=_git_kernel_module(), binary=binary, build_id="hct-xyz")
    assert provenance.recorded_build_id(document) == "hct-xyz"
    assert set(document["toolchain"]) == {"compiler", "compiler_version", "cmake_version", "neuralspotx_version"}
    assert document["toolchain"]["compiler"] == "arm-none-eabi-gcc"


# --- on-disk record and doctor --------------------------------------------------------


def test_the_record_round_trips_through_the_build_dir(tmp_path: Path) -> None:
    document = _build(tmp_path, kernel=_git_kernel_module())
    build_dir = tmp_path / "bd"
    path = provenance.write_provenance(build_dir, BOARD, document)

    # Beside the image it describes, not up in the app tree: the app's own
    # nsx.lock moves on the next time the app is re-locked.
    from helia_core_tester.hardware.firmware_build import output_dir

    assert path == output_dir(build_dir, BOARD) / "hct_provenance.json"
    assert provenance.read_provenance(build_dir, BOARD) == document
    assert provenance.read_provenance(tmp_path / "never-built", BOARD) is None

    path.write_text(json.dumps({"schema": "something-else"}))
    with pytest.raises(provenance.ProvenanceError, match="rebuild with `hardware build`"):
        provenance.read_provenance(build_dir, BOARD)


def test_doctor_describes_the_build_in_a_build_dir(tmp_path: Path) -> None:
    build_dir = tmp_path / "bd"
    assert "not built yet" in provenance.describe_build(build_dir, BOARD)

    provenance.write_provenance(build_dir, BOARD, _build(tmp_path, kernel=_git_kernel_module()))
    line = provenance.describe_build(build_dir, BOARD)
    assert line.startswith(f"{BOARD.id}: qualified against test-baseline-0001")
    assert KERNEL_PIN in line

    checkout = _git_repo(tmp_path / "kernels", dirty=True)
    provenance.write_provenance(
        build_dir, BOARD,
        _build(tmp_path, kernel=_local_kernel_module(checkout), kernel_source=KernelSource(path=checkout)),
    )
    line = provenance.describe_build(build_dir, BOARD)
    assert "development-overrides" in line and "uncommitted changes" in line


def test_summarize_kernels_is_one_log_line(tmp_path: Path) -> None:
    assert provenance.summarize_kernels(None) == "unknown"
    document = _build(tmp_path, kernel=_git_kernel_module())
    assert provenance.summarize_kernels(document) == f"ns-cmsis-nn@{KERNEL_PIN} (git, qualified)"

    checkout = _git_repo(tmp_path / "kernels")
    local = _build(tmp_path, kernel=_local_kernel_module(checkout), kernel_source=KernelSource(path=checkout))
    assert provenance.summarize_kernels(local).startswith(f"ns-cmsis-nn@content:{'f' * 16} (local, development-overrides")


# --- the bundle, and what the hpx dashboard reads out of it ---------------------------


def _dashboard_summary_dependency(summary: dict[str, Any], project: str) -> dict[str, str] | None:
    """`hpx_dashboard/dataset.py::_summary_dependency`, re-implemented here.

    Deliberately a copy and not an import: the dashboard is a separate repo with
    no dependency in either direction, and this test exists to fail if the
    tester's block stops being readable by the reader hpx bundles are read with.
    Only the current (non-legacy) branch is reproduced -- that is the one a
    tester bundle takes.
    """
    dependencies = summary.get("dependencies")
    modules = dependencies.get("modules") if isinstance(dependencies, dict) else None
    if not isinstance(modules, list):
        return None
    for module in modules:
        if not isinstance(module, dict) or module.get("project") != project:
            continue
        revision = {"project": project}
        ref = module.get("requested_ref")
        commit = module.get("peeled_commit")
        url = module.get("url")
        if isinstance(ref, str) and ref:
            revision["ref"] = ref
        if isinstance(commit, str) and commit:
            revision["commit"] = commit
        if isinstance(url, str) and url:
            revision["url"] = url
        return revision
    return None


def _bundle(tmp_path: Path, document: dict | None, *, provenance_files=()) -> Path:
    case = load_case_bundle(
        build_abs_s8_case_bundle(PROJECT_ROOT, output_root=tmp_path, case_id="abs_prov").manifest_path
    )
    result = HostSession(FakeTargetTransport()).run_many([case])
    return write_result_bundle(
        result, session_id="prov-session", output_root=tmp_path / "out",
        memory_report={}, kernel_catalog=[], dependencies=document, provenance_files=provenance_files,
    )


def test_the_bundle_carries_the_block_and_the_files_it_came_from(tmp_path: Path) -> None:
    build_dir = tmp_path / "bd"
    document = _build(tmp_path, kernel=_git_kernel_module())
    provenance_path = provenance.write_provenance(build_dir, BOARD, document)
    lock_snapshot = provenance_path.parent / "nsx.lock"
    lock_snapshot.write_text((build_dir / "nsx_app" / "nsx.lock").read_text())

    bundle = _bundle(tmp_path, document, provenance_files=(provenance_path, lock_snapshot))

    manifest = json.loads((bundle / "session_manifest.json").read_text())
    summary = json.loads((bundle / "session_summary.json").read_text())
    assert manifest["dependencies"] == document
    assert summary["dependencies"] == document
    # The build-side originals travel with the bundle, so it stays readable once
    # the build dir has been rebuilt or deleted.
    assert manifest["artifacts"]["provenance"] == ["hct_provenance.json", "nsx.lock"]
    assert json.loads((bundle / "hct_provenance.json").read_text()) == document
    assert (bundle / "nsx.lock").read_text() == lock_snapshot.read_text()


def test_a_bundle_with_no_provenance_omits_the_key_rather_than_nulling_it(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, None)
    assert "dependencies" not in json.loads((bundle / "session_manifest.json").read_text())
    assert "dependencies" not in json.loads((bundle / "session_summary.json").read_text())
    assert "provenance" not in json.loads((bundle / "session_manifest.json").read_text())["artifacts"]


def test_the_hpx_dashboard_lookup_reads_a_tester_bundle(tmp_path: Path) -> None:
    document = _build(tmp_path, kernel=_git_kernel_module())
    bundle = _bundle(tmp_path, document)
    summary = json.loads((bundle / "session_summary.json").read_text())

    assert _dashboard_summary_dependency(summary, "ns-cmsis-nn") == {
        "project": "ns-cmsis-nn",
        "ref": KERNEL_PIN,
        "commit": KERNEL_PIN,
        "url": "https://github.com/AmbiqAI/ns-cmsis-nn.git",
    }
    assert _dashboard_summary_dependency(summary, "nsx-ambiq-sdk")["commit"] == SDK_PIN
    # A project the build never resolved is absent, not empty.
    assert _dashboard_summary_dependency(summary, "arm-cmsis-nn") is None
    # A bundle with no block at all falls through to the dashboard's legacy branch.
    assert _dashboard_summary_dependency({}, "ns-cmsis-nn") is None
