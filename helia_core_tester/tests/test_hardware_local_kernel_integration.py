"""`--cmsis-nn-root` driven through the real neuralspotx lock and sync.

The unit tests elsewhere stub `neuralspotx.api`, which is right for asserting the
*order* of the build driver's calls but structurally cannot catch what this file
is for: whether NSX actually mirrors a local kernel checkout where the tester
looks for it, and whether a kernel edit re-locks instead of failing the frozen
sync. Those are properties of NSX's own resolution, so the real `lock_app` and
`sync_app` run here.

Network is stubbed out at `git_clone_at_commit`, the single entry point both the
lock's artifact hashing and the sync's vendoring use. The stub is deterministic
in (url, commit), so the hash the lock computes is the hash the sync reproduces
and the frozen verification is a real check rather than a bypassed one. The
module under test -- the local one -- is not stubbed at all.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from helia_core_tester.hardware import firmware_build, nsx_app
from helia_core_tester.hardware.boards import resolve_board
from helia_core_tester.hardware.dependency_baseline import resolve_baseline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BOARD = resolve_board("apollo510_evb")

_MODULE_YAML = """\
schema_version: 1
module:
  name: nsx-cmsis-nn
  type: algorithm
  version: "7.33.1"
support:
  ambiqsuite: true
  zephyr: false
summary: >-
  Stand-in for ns-cmsis-nn, carrying only what NSX validates.
build:
  cmake:
    package: nsx_cmsis_nn
    targets:
      - nsx::cmsis_nn
depends:
  required: []
  optional: []
compatibility:
  boards: ["*"]
  socs: ["*"]
  toolchains: [arm-none-eabi-gcc]
"""


def _stub_module_yaml(name: str) -> str:
    """The least NSX's validator and policy checks accept, with no dependencies.

    Two types cannot be flattened to `algorithm`, both because NSX's registry
    policy checks them (`module_registry/_policy.py`): a board module must
    depend on exactly one module of type `soc`, and on an enabled
    `sdk_provider`. The stub honours those rather than defeating real invariants
    of the registry it stands in for.
    """
    module_type = "algorithm"
    if name.endswith("-soc-hal"):
        module_type = "soc"
    elif name == "nsx-ambiqsuite":
        module_type = "sdk_provider"
    return (
        "schema_version: 1\n"
        "module:\n"
        f"  name: {name}\n"
        f"  type: {module_type}\n"
        '  version: "0.0.0"\n'
        "support:\n"
        "  ambiqsuite: true\n"
        "  zephyr: false\n"
        "build:\n"
        "  cmake:\n"
        f"    package: {name.replace('-', '_')}\n"
        "    targets:\n"
        f"      - nsx::{name.removeprefix('nsx-').replace('-', '_')}\n"
        "depends:\n"
        "  required: []\n"
        "  optional: []\n"
    )


def _project_module_metadata(url: str) -> dict[str, str]:
    """`{module name: metadata path}` for every module the registry maps to *url*."""
    from neuralspotx import api as nsx_api

    registry = nsx_api.load_registry()
    projects = {
        name: entry
        for name, entry in (registry.get("projects") or {}).items()
        if isinstance(entry, dict) and entry.get("url") == url
    }
    out: dict[str, str] = {}
    for name, entry in (registry.get("modules") or {}).items():
        if not isinstance(entry, dict) or entry.get("project") not in projects:
            continue
        out[str(name)] = str(entry.get("metadata") or "nsx-module.yaml")
    return out


def _fake_checkout(root: Path) -> Path:
    """A tree shaped enough like ns-cmsis-nn for NSX to accept it as a module."""
    (root / "Include").mkdir(parents=True)
    (root / "Source" / "BasicMathFunctions").mkdir(parents=True)
    (root / "nsx").mkdir(parents=True)
    (root / "Include" / "arm_nnfunctions.h").write_text("/* stub */\n", encoding="utf-8")
    (root / "Source" / "BasicMathFunctions" / "arm_abs_s8.c").write_text(
        "void arm_abs_s8(void) {}\n", encoding="utf-8"
    )
    (root / "nsx" / "nsx-module.yaml").write_text(_MODULE_YAML, encoding="utf-8")
    (root / "nsx" / "CMakeLists.txt").write_text(
        "add_library(nsx_cmsis_nn INTERFACE)\nadd_library(nsx::cmsis_nn ALIAS nsx_cmsis_nn)\n",
        encoding="utf-8",
    )
    return root


@pytest.fixture
def offline_git(monkeypatch):
    """Replace git cloning with a deterministic tree, keyed by (url, commit).

    Deterministic because the lock hashes what this produces and the sync has to
    reproduce the same hash; a stub that varied per call would make the frozen
    verification pass vacuously.
    """
    import neuralspotx.subprocess_utils as nsx_subprocess
    from neuralspotx.module_registry import _vendoring

    cloned: list[tuple[str, str]] = []

    def _fake_clone(url: str, dest: Path, commit: str) -> None:
        cloned.append((url, commit))
        dest = Path(dest)
        if dest.exists():
            import shutil

            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        (dest / "STUB").write_text(f"{url}\n{commit}\n", encoding="utf-8")
        # Closure resolution reads each module's nsx-module.yaml out of its
        # project clone, so a stub that produced only a marker file would fail
        # the lock before the local module was ever reached. Write a minimal
        # valid manifest at every metadata path the registry maps into this
        # project -- derived from the registry rather than hardcoded, so a
        # neuralspotx bump that moves a module cannot leave this silently stale.
        for name, meta in _project_module_metadata(url).items():
            # NSX looks for the metadata both at the registry path taken as
            # project-relative (how a monorepo like nsx-ambiq-sdk lays it out)
            # and at the clone root (how a single-module project does), so the
            # stub satisfies both rather than encoding a guess about which.
            for candidate in (dest / meta, dest / "nsx-module.yaml"):
                candidate.parent.mkdir(parents=True, exist_ok=True)
                candidate.write_text(_stub_module_yaml(name), encoding="utf-8")

    # Two bindings to cover: _vendoring imports the name at module level, while
    # the lock's artifact hashing imports it lazily from the package.
    monkeypatch.setattr(nsx_subprocess, "git_clone_at_commit", _fake_clone)
    monkeypatch.setattr(_vendoring, "git_clone_at_commit", _fake_clone, raising=False)
    return cloned


@pytest.fixture
def local_kernel_app(tmp_path: Path, offline_git, monkeypatch):
    """A rendered app whose kernels come from a local checkout, ready to lock."""
    monkeypatch.setattr(firmware_build, "ensure_host_tools", lambda repo_root, baseline=None: None)
    checkout = _fake_checkout(tmp_path / "ns-cmsis-nn")
    build_dir = tmp_path / "build"  # deliberately NOT under the checkout
    options = firmware_build.FirmwareOptions(cmsis_nn_root=checkout)
    render = nsx_app.render_app(
        BOARD,
        repo_root=PROJECT_ROOT,
        build_dir=build_dir,
        baseline=resolve_baseline(PROJECT_ROOT),
        cmsis_nn_root=checkout,
    )
    return checkout, build_dir, options, render


def test_local_kernel_checkout_is_mirrored_where_the_tester_looks(local_kernel_app) -> None:
    """The mirror must land at modules/ns-cmsis-nn and nsx_cmsis_nn must map to its
    nsx/ subdirectory -- NSX's defaults would put it under the *module* name and
    point CMake at the repository root, where `nsx::cmsis_nn` does not exist."""
    checkout, build_dir, options, render = local_kernel_app

    firmware_build.lock_and_sync(render, options)

    synced = nsx_app.synced_kernel_dir(render.app_dir)
    assert synced == render.app_dir / "modules" / "ns-cmsis-nn"
    assert (synced / "nsx" / "nsx-module.yaml").is_file()
    assert (synced / "Source" / "BasicMathFunctions" / "arm_abs_s8.c").is_file()

    modules_cmake = (render.app_dir / "cmake" / "nsx" / "modules.cmake").read_text()
    assert 'set(NSX_APP_MODULE_DIR_nsx_cmsis_nn "modules/ns-cmsis-nn/nsx")' in modules_cmake

    # Once the build records what it was made from, generation is pointed at the
    # mirror rather than the live checkout -- the tree the firmware compiled.
    nsx_app.commit_render_state(render)
    assert firmware_build.kernel_source_root(build_dir, options) == synced

    # And the manifest says local, with no stale git coordinates beside it.
    project = yaml.safe_load(render.nsx_yml)["module_registry"]["projects"]["ns-cmsis-nn"]
    assert project["local_path"] == str(checkout)
    assert "url" not in project and "revision" not in project

    lock = yaml.safe_load((render.app_dir / "nsx.lock").read_text())
    assert json.dumps(lock).count('"local"') or "local" in json.dumps(lock)


def test_a_kernel_edit_relocks_without_update_dependencies(local_kernel_app) -> None:
    """Editing kernels and rebuilding is the whole point of the flag.

    Nothing in the manifest changes when a source file is edited, so without the
    checkout's content hash in the render digest the stale lock would be reused
    and the frozen sync would fail with "drifted since lock" -- telling the user to
    pass --update-dependencies for something they never opted into.
    """
    checkout, build_dir, options, render = local_kernel_app

    firmware_build.lock_and_sync(render, options)
    nsx_app.commit_render_state(render)
    assert firmware_build.lock_reuse_reason(render) is None

    source = checkout / "Source" / "BasicMathFunctions" / "arm_abs_s8.c"
    source.write_text("void arm_abs_s8(void) { /* edited */ }\n", encoding="utf-8")

    edited = nsx_app.render_app(
        BOARD,
        repo_root=PROJECT_ROOT,
        build_dir=build_dir,
        baseline=resolve_baseline(PROJECT_ROOT),
        cmsis_nn_root=checkout,
    )
    assert edited.nsx_yml == render.nsx_yml, "the manifest is unchanged, which is the point"
    assert edited.digest != render.digest, "the checkout's contents must reach the digest"
    assert firmware_build.lock_reuse_reason(edited) == (
        "the rendered manifest or the dependency baseline changed"
    )

    # The rebuild succeeds with no --update-dependencies, and the mirror carries
    # the edit.
    firmware_build.lock_and_sync(edited, options)
    mirrored = nsx_app.synced_kernel_dir(edited.app_dir) / "Source" / "BasicMathFunctions" / "arm_abs_s8.c"
    assert "edited" in mirrored.read_text()


def test_an_edited_mirror_is_refused_for_a_local_source(local_kernel_app) -> None:
    """The real lock's content hash catches a mirror edited after the build.

    Frozen sync verified this mirror at build time; `--skip-flash` never syncs
    again, so nothing else would notice.
    """
    checkout, build_dir, options, render = local_kernel_app

    firmware_build.lock_and_sync(render, options)
    nsx_app.commit_render_state(render)
    synced = nsx_app.synced_kernel_dir(render.app_dir)
    assert firmware_build.kernel_source_root(build_dir, options) == synced

    target = synced / "Source" / "BasicMathFunctions" / "arm_abs_s8.c"
    target.write_text("void arm_abs_s8(void) { /* tampered */ }\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed since the build"):
        firmware_build.kernel_source_root(build_dir, options)


def test_an_app_dir_inside_the_checkout_is_refused(tmp_path: Path) -> None:
    """The nested layout puts the app under --cmsis-nn-root, where NSX's local
    vendoring silently does nothing and the content hash would cover the build
    output. Neither is fixable by care, so it is rejected."""
    checkout = _fake_checkout(tmp_path / "ns-cmsis-nn")
    nested = checkout / "Tests" / "helia-core-tester" / "build" / "hardware" / "apollo510_evb"

    with pytest.raises(nsx_app.AppRenderError, match="is inside --cmsis-nn-root"):
        nsx_app.plan_app(
            BOARD,
            repo_root=PROJECT_ROOT,
            build_dir=nested,
            baseline=resolve_baseline(PROJECT_ROOT),
            cmsis_nn_root=checkout,
        )

    # A symlink into the checkout is still inside it.
    link = tmp_path / "build-link"
    link.symlink_to(checkout / "Tests" / "helia-core-tester" / "build", target_is_directory=True)
    with pytest.raises(nsx_app.AppRenderError, match="is inside --cmsis-nn-root"):
        nsx_app.plan_app(
            BOARD,
            repo_root=PROJECT_ROOT,
            build_dir=link / "hardware" / "apollo510_evb",
            baseline=resolve_baseline(PROJECT_ROOT),
            cmsis_nn_root=checkout,
        )

    # ...and the same checkout with a build dir outside it renders fine.
    assert nsx_app.plan_app(
        BOARD,
        repo_root=PROJECT_ROOT,
        build_dir=tmp_path / "outside",
        baseline=resolve_baseline(PROJECT_ROOT),
        cmsis_nn_root=checkout,
    ).kernel_source.path == checkout
