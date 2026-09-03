"""Regression tests for issue #65.

CMSIS_NN_REPO_ROOT used to validate as this tool's own root against just
pyproject.toml + CMakeLists.txt, which the sibling ns-cmsis-nn kernel
checkout also has -- so pointing the override at a kernel checkout silently
"succeeded" and the run died several steps later with a misleading error
far from the real cause. These tests lock in the fix: validation now also
requires the tool's own helia_core_tester/ package directory, and the
failure messages are actionable.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from helia_core_tester.core.discovery import _is_repo_root, find_repo_root
from helia_core_tester.core.errors import RepoRootNotFoundError

TESTER_ROOT = Path(__file__).resolve().parents[2]


def _make_kernel_like_checkout(tmp_path: Path) -> Path:
    """A directory shaped like the ns-cmsis-nn kernel root: pyproject.toml +
    CMakeLists.txt present, but no helia_core_tester/ package -- the exact
    false-positive shape from issue #65."""
    kernel_root = tmp_path / "ns-cmsis-nn"
    kernel_root.mkdir()
    (kernel_root / "pyproject.toml").write_text('[project]\nname = "cmsis_nn"\n')
    (kernel_root / "CMakeLists.txt").write_text("# kernel cmake\n")
    return kernel_root


def test_real_tester_root_still_validates() -> None:
    assert _is_repo_root(TESTER_ROOT) is True


def test_kernel_shaped_checkout_no_longer_validates(tmp_path: Path) -> None:
    kernel_root = _make_kernel_like_checkout(tmp_path)
    assert _is_repo_root(kernel_root) is False


def test_env_override_pointed_at_kernel_checkout_fails_with_actionable_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kernel_root = _make_kernel_like_checkout(tmp_path)
    monkeypatch.setenv("CMSIS_NN_REPO_ROOT", str(kernel_root))

    with pytest.raises(RepoRootNotFoundError) as exc_info:
        find_repo_root()

    message = str(exc_info.value)
    # Names the actual missing marker...
    assert "helia_core_tester" in message
    # ...and points at the correct mechanism for a kernel checkout instead.
    assert "--cmsis-nn-root" in message or "CMSIS_NN_ROOT" in message


def test_env_override_pointed_at_real_tester_root_still_works(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CMSIS_NN_REPO_ROOT", str(TESTER_ROOT))
    assert find_repo_root() == TESTER_ROOT
