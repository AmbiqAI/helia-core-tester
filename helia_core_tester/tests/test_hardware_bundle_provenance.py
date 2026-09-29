"""Build provenance in the result bundle manifest."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from helia_core_tester.hardware import nsx_cli
from helia_core_tester.hardware.firmware_build import BUILT_LOCK, SYNC_STAMP, nsx_app_dir
from helia_core_tester.hardware.nsx_app import AppOptions, save_options
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


def _fake_build(build_dir: Path, options: AppOptions) -> str:
    """Records a finished build leaves behind."""
    app_dir = nsx_app_dir(build_dir)
    app_dir.mkdir(parents=True)
    (app_dir / "nsx.lock").write_text(LOCK, encoding="utf-8")
    digest = nsx_cli.lock_digest(app_dir)
    save_options(app_dir, options)
    (app_dir / BUILT_LOCK).write_text(json.dumps({"lock": digest, "kernels": "sha256:22"}), encoding="utf-8")
    (app_dir / SYNC_STAMP).write_text(f"{digest} 0.8.1", encoding="utf-8")
    return digest


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
    assert build["kernels"] == {
        "ref": "v7.35.1", "commit": COMMIT, "root": None, "root_head": None, "root_dirty": None, "tree_hash": "sha256:22",
    }
    assert build["options"]["requantize_inline_asm"] is False
    assert build["options"]["cmsis_nn_root"] is None
    assert build["neuralspotx_version"] == "0.8.1"
    assert build["nsx_lock_sha256"] == digest
    assert manifest["artifacts"]["nsx_lock"] == "nsx.lock"
    assert (bundle_root / "nsx.lock").read_text(encoding="utf-8") == LOCK


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_local_root_records_checkout(tmp_path: Path) -> None:
    root = tmp_path / "ns-cmsis-nn"
    root.mkdir()
    (root / "a.c").write_text("int a;\n", encoding="utf-8")
    git = ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "a.c"], check=True)
    subprocess.run([*git, "commit", "-qm", "init"], check=True)
    head = subprocess.run([*git, "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    (root / "a.c").write_text("int b;\n", encoding="utf-8")
    build_dir = tmp_path / "build"
    _fake_build(build_dir, AppOptions(cmsis_nn_root=root))

    kernels = build_provenance(build_dir)[0]["kernels"]

    assert kernels["ref"] is None
    assert kernels["root"] == str(root.resolve())
    assert kernels["root_head"] == head
    assert kernels["root_dirty"] is True


def test_relocked_app_drops_lock_fields(tmp_path: Path) -> None:
    build_dir = tmp_path / "build"
    _fake_build(build_dir, AppOptions())
    (nsx_app_dir(build_dir) / "nsx.lock").write_text(LOCK.replace("0.8.1", "0.9.0"), encoding="utf-8")

    provenance, lock_file = build_provenance(build_dir)

    assert lock_file is None
    assert provenance["kernels"]["commit"] is None
    assert provenance["neuralspotx_version"] is None
    assert provenance["kernels"]["ref"] == "v7.35.1"


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
    }
    assert "nsx_lock" not in manifest["artifacts"]
    assert not (bundle_root / "nsx.lock").exists()
