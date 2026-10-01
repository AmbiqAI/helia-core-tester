"""Build provenance in the result bundle manifest."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from helia_core_tester.hardware import nsx_cli
from helia_core_tester.hardware import firmware_build, toolchain
from helia_core_tester.hardware.firmware_build import nsx_app_dir
from helia_core_tester.hardware.nsx_app import CMSIS_NN_REF, AppOptions, kernel_dir, save_options
from helia_core_tester.hardware.result_bundle import build_provenance, write_result_bundle
from helia_core_tester.hardware.session import SessionResult

COMMIT = "7c28064c047cde53cd87d8b9b61257ff43bdc549"
LOCK = f"""schema_version: 4
targets:
  apollo510_evb:
    generated_at: '2026-09-18T21:32:17+00:00'
    nsx_tool:
      version: 0.8.1
    manifest:
      path: nsx.yml
      hash: sha256:00
    target:
      board: apollo510_evb
      soc: apollo510
      toolchain: arm-none-eabi-gcc
    modules:
      nsx-cmsis-nn:
        project: ns-cmsis-nn
        kind: git
        constraint: {COMMIT}
        resolved:
          url: https://github.com/AmbiqAI/ns-cmsis-nn.git
          commit: {COMMIT}
          vendored_at: modules/ns-cmsis-nn
          content_hash: sha256:11
          acquired_at: '2026-09-18T20:11:46+00:00'
"""


def _fake_gcc(path: Path, version: str) -> Path:
    """A compiler that only reports a version."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\necho {version}\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _fake_build(build_dir: Path, options: AppOptions) -> str:
    """Records a finished build leaves behind."""
    compiler = _fake_gcc(build_dir.parent / "cached" / "arm-none-eabi-gcc", "14.2.1")
    probe = build_dir / "CMakeFiles" / "4.4.3" / "CMakeCCompiler.cmake"
    probe.parent.mkdir(parents=True)
    probe.write_text(f'set(CMAKE_C_COMPILER "{compiler}")\nset(CMAKE_C_COMPILER_ID "GNU")\n', encoding="utf-8")
    app_dir = nsx_app_dir(build_dir)
    module = kernel_dir(app_dir, options)
    (module / "Source").mkdir(parents=True)
    (module / "Source" / "k.c").write_text("int k;\n", encoding="utf-8")
    (app_dir / "nsx.lock").write_text(LOCK, encoding="utf-8")
    save_options(app_dir, options)
    firmware_build._record_built(build_dir, options)
    return nsx_cli.lock_digest(app_dir)


def _write_bundle(tmp_path: Path, build_dir: Path | None) -> Path:
    result = SessionResult(cases=(), protocol_trace=(), session_complete_cases=0, build_id="abc123")
    return write_result_bundle(
        result, session_id="prov", output_root=tmp_path, memory_report={}, kernel_catalog=[], build_dir=build_dir,
    )


def _manifest(bundle_root: Path) -> dict:
    return json.loads((bundle_root / "session_manifest.json").read_text(encoding="utf-8"))


def test_pinned_ref_build_is_stamped(tmp_path: Path) -> None:
    build_dir = tmp_path / "build"
    digest = _fake_build(build_dir, AppOptions(requantize_inline_asm=False))

    bundle_root = _write_bundle(tmp_path, build_dir)

    manifest = _manifest(bundle_root)
    assert manifest["firmware_build_id"] == "abc123"
    assert manifest["session_id"] == "prov"
    build = manifest["build"]
    tree = nsx_cli.tree_hash(kernel_dir(nsx_app_dir(build_dir), AppOptions()))
    assert build["kernels"] == {
        "ref": CMSIS_NN_REF, "commit": COMMIT, "root": None, "root_head": None, "root_dirty": None, "tree_hash": tree,
    }
    assert build["options"]["requantize_inline_asm"] is False
    assert build["options"]["cmsis_nn_root"] is None
    assert build["neuralspotx_version"] == nsx_cli.nsx_version()
    assert build["nsx_lock_sha256"] == digest
    assert manifest["artifacts"]["nsx_lock"] == "nsx.lock"
    assert (bundle_root / "nsx.lock").read_text(encoding="utf-8") == LOCK
    assert build["toolchain"] == {"name": "arm-none-eabi-gcc", "version": "14.2.1"}
    assert build["modules"] == [{
        "name": "nsx-cmsis-nn", "project": "ns-cmsis-nn", "kind": "git", "revision": COMMIT, "tag": None,
        "commit": COMMIT, "url": "https://github.com/AmbiqAI/ns-cmsis-nn.git",
    }]


def _git_checkout(root: Path) -> str:
    """One-commit checkout; returns HEAD."""
    (root / "Source").mkdir(parents=True)
    (root / "Source" / "k.c").write_text("int k;\n", encoding="utf-8")
    (root / "README").write_text("r\n", encoding="utf-8")
    git = ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-qm", "init"], check=True)
    return subprocess.run([*git, "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()


def test_toolchain_is_the_configured_compiler(tmp_path: Path, monkeypatch) -> None:
    # Downloaded GCC differs from the cached one.
    downloaded = _fake_gcc(tmp_path / "downloads" / "bin" / "arm-none-eabi-gcc", "15.1.0")
    monkeypatch.setattr(toolchain, "toolchain_bin_dir", lambda repo_root=None: downloaded.parent)
    monkeypatch.setenv("PATH", f"{downloaded.parent}:{os.environ.get('PATH', '')}")
    build_dir = tmp_path / "build"
    _fake_build(build_dir, AppOptions())

    assert build_provenance(build_dir)[0]["toolchain"] == {"name": "arm-none-eabi-gcc", "version": "14.2.1"}

    shutil.rmtree(build_dir / "CMakeFiles")
    firmware_build._record_built(build_dir, AppOptions())
    assert build_provenance(build_dir)[0]["toolchain"] is None


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
@pytest.mark.parametrize(("edit", "dirty"), [("README", False), ("Source/new.c", True)])
def test_local_root_records_build_checkout(tmp_path: Path, edit: str, dirty: bool) -> None:
    root = tmp_path / "ns-cmsis-nn"
    head = _git_checkout(root)
    (root / edit).write_text("edited\n", encoding="utf-8")
    build_dir = tmp_path / "build"
    _fake_build(build_dir, AppOptions(cmsis_nn_root=root))
    # Later edits must not leak in.
    (root / "Source" / "k.c").write_text("int later;\n", encoding="utf-8")

    kernels = build_provenance(build_dir)[0]["kernels"]

    assert kernels["ref"] is None
    assert kernels["root"] == str(root.resolve())
    assert kernels["root_head"] == head
    assert kernels["root_dirty"] is dirty


def test_relocked_app_drops_lock_fields(tmp_path: Path) -> None:
    build_dir = tmp_path / "build"
    _fake_build(build_dir, AppOptions())
    (nsx_app_dir(build_dir) / "nsx.lock").write_text(LOCK.replace("0.8.1", "0.9.0"), encoding="utf-8")

    provenance, lock_file = build_provenance(build_dir)

    assert lock_file is None
    assert provenance["kernels"]["commit"] is None
    assert provenance["modules"] is None
    assert provenance["kernels"]["ref"] == CMSIS_NN_REF
    assert provenance["neuralspotx_version"] == nsx_cli.nsx_version()


@pytest.mark.parametrize("with_dir", [False, True])
def test_missing_records_leave_nulls(tmp_path: Path, with_dir: bool) -> None:
    build_dir = tmp_path / "build" if with_dir else None
    if build_dir is not None:
        # Pre-NSX build: image, no app.
        (build_dir / "hardware").mkdir(parents=True)

    bundle_root = _write_bundle(tmp_path, build_dir)

    manifest = _manifest(bundle_root)
    assert manifest["build"] == {
        "options": None,
        "kernels": dict.fromkeys(("ref", "commit", "root", "root_head", "root_dirty", "tree_hash")),
        "neuralspotx_version": None,
        "nsx_lock_sha256": None,
        "modules": None,
        "toolchain": None,
    }
    assert "nsx_lock" not in manifest["artifacts"]
    assert not (bundle_root / "nsx.lock").exists()


def test_corrupt_record_fields_read_null(tmp_path: Path) -> None:
    build_dir = tmp_path / "build"
    root = tmp_path / "ns-cmsis-nn"
    root.mkdir()
    _fake_build(build_dir, AppOptions(cmsis_nn_root=root))
    app_dir = nsx_app_dir(build_dir)
    (app_dir / firmware_build.BUILT_LOCK).write_text(json.dumps({"lock": 1, "kernels": 123}), encoding="utf-8")
    bad = {"nsx_version": [], "root_head": {}, "root_dirty": "yes", "toolchain": {"version": 14}}
    (app_dir / firmware_build.BUILT_INFO).write_text(json.dumps(bad), encoding="utf-8")

    provenance, lock_file = build_provenance(build_dir)

    assert lock_file is None
    assert provenance["nsx_lock_sha256"] is None
    assert provenance["neuralspotx_version"] is None
    assert provenance["kernels"]["tree_hash"] is None
    assert provenance["kernels"]["root_head"] is None
    assert provenance["kernels"]["root_dirty"] is None
    assert provenance["toolchain"] == {"name": None, "version": None}


def test_rewrite_drops_stale_lock_copy(tmp_path: Path) -> None:
    build_dir = tmp_path / "build"
    _fake_build(build_dir, AppOptions())
    assert (_write_bundle(tmp_path, build_dir) / "nsx.lock").is_file()

    bundle_root = _write_bundle(tmp_path, None)

    assert not (bundle_root / "nsx.lock").exists()
    assert "nsx_lock" not in _manifest(bundle_root)["artifacts"]
