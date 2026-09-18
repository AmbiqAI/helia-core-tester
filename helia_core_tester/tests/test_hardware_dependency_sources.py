"""Where the hardware build's ns-cmsis-nn checkout comes from: --cmsis-nn-root, then
$CMSIS_NN_ROOT, then the nested <ns-cmsis-nn>/Tests/helia-core-tester layout."""

from __future__ import annotations

import os
import subprocess
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


# --- provenance -------------------------------------------------------------------


def _git_repo(path: Path) -> str:
    """A one-commit git repository at `path` (Include/ + Source/); returns its HEAD."""
    _checkout(path)
    (path / "Include" / "arm_nnfunctions.h").write_text("/* header */\n")
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@x", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@x"}
    for args in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "init"]):
        subprocess.run(["git", "-C", str(path), *args], check=True, env={**os.environ, **env}, capture_output=True)
    return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()


def test_module_records_name_the_tree_state(tmp_path: Path) -> None:
    from helia_core_tester.generation.reuse import cmsis_nn_checkout_identity_for
    from helia_core_tester.hardware.dependency_sources import git_tree_identity, module_record

    repo_root = tmp_path / "helia-core-tester"
    repo_root.mkdir()
    clean = tmp_path / "clean"
    head = _git_repo(clean)

    record = module_record("nsx-cmsis-nn", "ns-cmsis-nn", clean, repo_root=repo_root, identity=cmsis_nn_checkout_identity_for(clean))
    assert record == {
        "name": "nsx-cmsis-nn", "project": "ns-cmsis-nn", "kind": "git", "requested_ref": None, "requested_tag": None,
        "peeled_commit": head, "content_hash": None, "url": "https://github.com/AmbiqAI/ns-cmsis-nn.git",
        "vendored_at": clean.resolve().as_posix(), "state": "git-clean",
    }

    (clean / "Include" / "arm_nnfunctions.h").write_text("/* edited */\n")
    dirty = module_record("nsx-cmsis-nn", "ns-cmsis-nn", clean, repo_root=repo_root, identity=cmsis_nn_checkout_identity_for(clean))
    assert dirty["kind"] == "git" and dirty["state"] == "git-dirty" and dirty["peeled_commit"] == head
    assert dirty["content_hash"]["algorithm"] == "sha256" and len(dirty["content_hash"]["value"]) == 64

    plain = _checkout(tmp_path / "plain")
    local = module_record("nsx-cmsis-nn", "ns-cmsis-nn", plain, repo_root=repo_root, identity=cmsis_nn_checkout_identity_for(plain))
    assert local["kind"] == "local" and local["state"] == "content" and local["peeled_commit"] is None
    assert local["content_hash"]["value"] != dirty["content_hash"]["value"]

    # Managed checkouts (SDK, neuralspotx, CMSIS_5): commit only, no content digest;
    # a path inside the repo is recorded relative to it.
    sdk = repo_root / "artifacts" / "downloads" / "nsx-ambiq-sdk"
    sdk_head = _git_repo(sdk)
    managed = module_record("nsx-ambiq-sdk", "nsx-ambiq-sdk", sdk, repo_root=repo_root, identity=git_tree_identity(sdk))
    assert managed["peeled_commit"] == sdk_head and managed["content_hash"] is None and managed["kind"] == "git"
    assert managed["vendored_at"] == "artifacts/downloads/nsx-ambiq-sdk"
    absent = module_record("CMSIS_5", "CMSIS_5", tmp_path / "nope", repo_root=repo_root, identity=git_tree_identity(tmp_path / "nope"))
    assert absent["kind"] == "absent" and absent["state"] == "absent" and absent["vendored_at"] is None
    # A git checkout nested inside another repository must not report the outer repo's commit.
    nested = _checkout(clean / "vendored-copy")
    assert git_tree_identity(nested) == {"state": "content"}


def test_dependencies_document_round_trips_and_names_the_override(tmp_path: Path) -> None:
    from helia_core_tester.hardware.dependency_sources import (
        ResolvedCmsisNn,
        build_dependencies_document,
        read_build_dependencies,
        summarize_kernels,
        write_build_dependencies,
    )

    repo_root = tmp_path / "helia-core-tester"
    repo_root.mkdir()
    kernels = tmp_path / "kernels"
    head = _git_repo(kernels)

    def _document(resolved):
        return build_dependencies_document(
            repo_root, resolved, cmake_defines={"CMSIS_NN_ROOT": str(kernels)}, kernel_target="cmsis-nn",
            build_profile="legacy-thin", kernel_compile_flags={"C_FLAGS": "-Ofast"}, toolchain={"arm_none_eabi_gcc": "gcc 14"},
        )

    document = _document(ResolvedCmsisNn(kernels, "env.CMSIS_NN_ROOT", str(kernels)))
    assert document["overrides"] == [
        {"scope": "module", "name": "nsx-cmsis-nn", "mode": "path", "requested": str(kernels), "selector": "env.CMSIS_NN_ROOT"}
    ]
    assert _document(ResolvedCmsisNn(kernels, "layout.nested", None))["overrides"] == []
    assert summarize_kernels(document) == f"{head} (git-clean)"
    assert summarize_kernels(None) == "unknown"

    build_dir = tmp_path / "bd"
    build_dir.mkdir()
    path = write_build_dependencies(build_dir, document)
    assert path.name == "hct_dependencies.json" and b"\r\n" not in path.read_bytes()
    assert read_build_dependencies(build_dir) == document
    assert read_build_dependencies(tmp_path / "never-configured") is None
    path.write_text('{"schema": "something-else"}')
    with pytest.raises(CmsisNnSourceError, match="not a hct.hardware.dependencies document"):
        read_build_dependencies(build_dir)


def test_dashboard_style_lookup_reads_the_kernel_commit_from_a_bundle_summary(tmp_path: Path) -> None:
    """The hpx dashboard finds the kernel-library commit by scanning
    summary["dependencies"]["modules"] for project == "ns-cmsis-nn" and reading
    requested_ref / peeled_commit / url. Mirror that lookup (no import: the tester has
    no dependency on hpx) so the record shape cannot drift away from it silently."""
    from helia_core_tester.hardware.dependency_sources import ResolvedCmsisNn, build_dependencies_document

    def dashboard_summary_dependency(summary: dict, project: str):
        for module in summary.get("dependencies", {}).get("modules", []):
            if module.get("project") != project:
                continue
            found = {"project": project}
            for key, source in (("ref", "requested_ref"), ("commit", "peeled_commit"), ("url", "url")):
                value = module.get(source)
                if isinstance(value, str) and value:
                    found[key] = value
            return found
        return None

    repo_root = tmp_path / "helia-core-tester"
    repo_root.mkdir()
    kernels = tmp_path / "kernels"
    head = _git_repo(kernels)
    document = build_dependencies_document(
        repo_root, ResolvedCmsisNn(kernels, "cli.--cmsis-nn-root", str(kernels)), cmake_defines={}, kernel_target="cmsis-nn",
        build_profile="legacy-thin", kernel_compile_flags=None, toolchain={},
    )
    summary = {"session_id": "s", "dependencies": document}
    assert dashboard_summary_dependency(summary, "ns-cmsis-nn") == {
        "project": "ns-cmsis-nn", "commit": head, "url": "https://github.com/AmbiqAI/ns-cmsis-nn.git",
    }
