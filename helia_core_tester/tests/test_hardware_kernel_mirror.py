"""`mirror_kernels` copies what git would commit, nothing else."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from helia_core_tester.hardware.kernel_mirror import KernelMirrorError, mirror_kernels, nested_kernel_root


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    """Kernel checkout holding a nested tester."""
    root = tmp_path / "ns-cmsis-nn"
    _write(root / "Source" / "arm_add.c", "int add;\n")
    _write(root / "Source" / "arm_sub.c", "int sub;\n")
    _write(root / "Include" / "arm_nn.h", "#pragma once\n")
    _write(root / ".gitignore", "*.log\n")
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init")
    # Nested tester: its own repo, untracked here.
    tester = root / "Tests" / "helia-core-tester"
    _write(tester / "artifacts" / "big.bin", "x" * 1000)
    _git(tester, "init", "-q")
    return root


def _listing(mirror: Path) -> set[str]:
    return {path.relative_to(mirror).as_posix() for path in mirror.rglob("*") if path.is_file()}


def test_mirror_tracks_the_working_tree(checkout: Path, tmp_path: Path) -> None:
    tester = checkout / "Tests" / "helia-core-tester"
    build = tester / "build" / "hardware" / "apollo510_evb"
    _write(checkout / "build.log", "ignored\n")
    _write(checkout / "Source" / "arm_new.c", "int fresh;\n")

    first = mirror_kernels(checkout, build, exclude=[tester])
    mirror = build.resolve() / "kernel_src"
    assert first.path == mirror
    assert _listing(mirror) == {
        ".gitignore", "Include/arm_nn.h", "Source/arm_add.c", "Source/arm_sub.c", "Source/arm_new.c",
    }
    assert first.changed == first.files == 5
    source = checkout / "Source" / "arm_add.c"
    assert (mirror / "Source" / "arm_add.c").stat().st_mtime_ns == source.stat().st_mtime_ns

    again = mirror_kernels(checkout, build, exclude=[tester])
    assert again.changed == 0 and again.stamp == first.stamp

    # Modify one, delete one.
    source.write_text("int add; // edit\n", encoding="utf-8")
    (checkout / "Source" / "arm_sub.c").unlink()
    edited = mirror_kernels(checkout, build, exclude=[tester])
    assert edited.changed == 2 and edited.stamp != first.stamp
    assert (mirror / "Source" / "arm_add.c").read_text(encoding="utf-8") == "int add; // edit\n"
    assert not (mirror / "Source" / "arm_sub.c").exists()

    # Tester and build churn: no change.
    _write(tester / "artifacts" / "generated_tests" / "t.c", "int t;\n")
    assert mirror_kernels(checkout, build, exclude=[tester]).stamp == edited.stamp


def test_gitlink_tester_is_skipped(checkout: Path, tmp_path: Path) -> None:
    """A submodule entry is not a file."""
    tester = checkout / "Tests" / "helia-core-tester"
    sha = "0123456789abcdef0123456789abcdef01234567"
    _git(checkout, "update-index", "--add", "--cacheinfo", f"160000,{sha},Tests/helia-core-tester")
    result = mirror_kernels(checkout, tmp_path / "build")
    assert "Tests/helia-core-tester/artifacts/big.bin" not in _listing(result.path)
    assert result.files == 4
    # Explicit exclude gives the same set.
    assert mirror_kernels(checkout, tmp_path / "build", exclude=[tester]).changed == 0


def test_file_turned_dir_mirrors(checkout: Path, tmp_path: Path) -> None:
    mirror_kernels(checkout, tmp_path / "build")
    (checkout / "Include" / "arm_nn.h").unlink()
    _write(checkout / "Include" / "arm_nn.h" / "inner.h", "#pragma once\n")
    result = mirror_kernels(checkout, tmp_path / "build")
    assert "Include/arm_nn.h/inner.h" in _listing(result.path)


def test_root_must_be_git_and_outside_build(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(KernelMirrorError, match="not a git checkout"):
        mirror_kernels(plain, tmp_path / "build")
    with pytest.raises(KernelMirrorError, match="inside build dir"):
        mirror_kernels(tmp_path / "build" / "k", tmp_path / "build")


def test_nested_root_needs_every_marker(tmp_path: Path) -> None:
    root = tmp_path / "ns-cmsis-nn"
    tester = root / "Tests" / "helia-core-tester"
    tester.mkdir(parents=True)
    (root / "Include").mkdir()
    (root / "Source").mkdir()
    assert nested_kernel_root(tester) is None
    _write(root / "nsx" / "nsx-module.yaml", "")
    assert nested_kernel_root(tester) == root.resolve()
    assert nested_kernel_root(tmp_path / "standalone") is None
