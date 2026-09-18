"""Where the hardware build's ns-cmsis-nn checkout comes from: --cmsis-nn-root, then
$CMSIS_NN_ROOT, then the nested <ns-cmsis-nn>/Tests/helia-core-tester layout."""

from __future__ import annotations

from pathlib import Path

import pytest

from helia_core_tester.hardware.dependency_sources import (
    CmsisNnSelection,
    CmsisNnSourceError,
    describe,
    resolve_cmsis_nn,
)


def _checkout(path: Path) -> Path:
    (path / "Include").mkdir(parents=True)
    (path / "Source").mkdir()
    return path.resolve()


def test_flag_beats_env_beats_nested_layout(tmp_path: Path) -> None:
    nested_root = _checkout(tmp_path / "ns-cmsis-nn")
    repo_root = nested_root / "Tests" / "helia-core-tester"
    repo_root.mkdir(parents=True)
    env_root = _checkout(tmp_path / "env")
    flag_root = _checkout(tmp_path / "flag")

    resolved = resolve_cmsis_nn(repo_root, env={})
    assert (resolved.root, resolved.selector, resolved.requested) == (nested_root, "layout.nested", None)

    resolved = resolve_cmsis_nn(repo_root, env={"CMSIS_NN_ROOT": str(env_root)})
    assert (resolved.root, resolved.selector, resolved.requested) == (env_root, "env.CMSIS_NN_ROOT", str(env_root))

    resolved = resolve_cmsis_nn(repo_root, CmsisNnSelection(root=flag_root), env={"CMSIS_NN_ROOT": str(env_root)})
    assert (resolved.root, resolved.selector) == (flag_root, "cli.--cmsis-nn-root")
    assert describe(resolved) == f"{flag_root} (cli.--cmsis-nn-root)"


def test_standalone_clone_without_flag_or_env_is_one_clear_error(tmp_path: Path) -> None:
    repo_root = tmp_path / "helia-core-tester"
    repo_root.mkdir()
    with pytest.raises(CmsisNnSourceError) as excinfo:
        resolve_cmsis_nn(repo_root, env={})
    message = str(excinfo.value)
    assert "--cmsis-nn-root PATH" in message and "CMSIS_NN_ROOT" in message
    assert str(tmp_path.resolve().parent) in message  # where the nested layout was looked for


def test_a_selected_path_must_look_like_a_checkout(tmp_path: Path) -> None:
    repo_root = tmp_path / "helia-core-tester"
    repo_root.mkdir()
    not_a_checkout = tmp_path / "somewhere"
    (not_a_checkout / "Include").mkdir(parents=True)  # no Source/
    with pytest.raises(CmsisNnSourceError, match=r"from --cmsis-nn-root is missing 'Source/'"):
        resolve_cmsis_nn(repo_root, CmsisNnSelection(root=not_a_checkout), env={})
    with pytest.raises(CmsisNnSourceError, match=r"from \$CMSIS_NN_ROOT does not exist"):
        resolve_cmsis_nn(repo_root, env={"CMSIS_NN_ROOT": str(tmp_path / "missing")})
    # An empty variable is "unset", not a path.
    with pytest.raises(CmsisNnSourceError, match="No ns-cmsis-nn checkout found"):
        resolve_cmsis_nn(repo_root, env={"CMSIS_NN_ROOT": ""})


def test_the_flag_path_is_expanded_and_resolved(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "helia-core-tester"
    repo_root.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    _checkout(tmp_path / "kernels")
    resolved = resolve_cmsis_nn(repo_root, CmsisNnSelection(root=Path("~/kernels")), env={})
    assert resolved.root == (tmp_path / "kernels").resolve() and resolved.requested == "~/kernels"


def test_ref_selection_is_rejected_until_the_baseline_lands(tmp_path: Path) -> None:
    with pytest.raises(CmsisNnSourceError, match="--cmsis-nn-ref is not supported yet"):
        resolve_cmsis_nn(tmp_path, CmsisNnSelection(ref="0" * 40), env={})
