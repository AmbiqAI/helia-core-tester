"""The NSX app render, the dependency baseline it pins from, and the lock decision.

The renderer's output is the manifest NSX resolves the firmware's dependencies
from, so these tests are mostly about the *text* -- what nsx.yml claims, and
whether it claims it consistently enough that `nsx lock` cannot resolve
something else. They run against the real pinned neuralspotx registry, so they
also fail if a neuralspotx bump changes this board's module set or ownership.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from helia_core_tester.hardware import nsx_app
from helia_core_tester.hardware.boards import resolve_board
from helia_core_tester.hardware.dependency_baseline import (
    CMSIS_NN_PROJECT,
    BaselineError,
    DependencyBaseline,
    default_baseline_path,
    load_baseline,
    parse_baseline,
    resolve_baseline,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BOARD = resolve_board("apollo510_evb")

_SHA_A = "a" * 40
_SHA_B = "b" * 40
_SHA_C = "c" * 40
_SHA_D = "d" * 40


def _minimal_document(**overrides) -> dict:
    document = {
        "schema": "hct.dependency-baseline",
        "schema_version": 1,
        "baseline_id": "test-baseline",
        "projects": {
            "ns-cmsis-nn": {"url": "https://example.invalid/ns-cmsis-nn.git", "ref": _SHA_A},
            "nsx-ambiq-sdk": {"url": "https://example.invalid/nsx-ambiq-sdk.git", "ref": _SHA_B},
            "neuralspotx": {"url": "https://example.invalid/neuralspotx.git", "ref": _SHA_C},
            "nsx-pmu-armv8m": {"url": "https://example.invalid/nsx-pmu-armv8m.git", "ref": _SHA_D},
        },
    }
    document.update(overrides)
    return document


def _baseline(**overrides) -> DependencyBaseline:
    return parse_baseline(_minimal_document(**overrides))


# --- the baseline file -------------------------------------------------------------


def test_repo_baseline_pins_every_project_the_app_resolves() -> None:
    """The shipped baseline must cover every project the rendered module list names,
    or `nsx lock` silently falls back to the packaged registry's mutable tags."""
    baseline = resolve_baseline(PROJECT_ROOT)
    modules = nsx_app.resolve_modules(BOARD.nsx_board, nsx_app.KernelSource(ref=_SHA_A))
    unpinned = sorted({m.project for m in modules if baseline.pin(m.project) is None})
    assert unpinned == [], f"projects with no baseline pin: {unpinned}"


def test_baseline_refs_are_immutable_commits() -> None:
    """A tag or branch would make "this build is qualified" depend on remote state."""
    baseline = resolve_baseline(PROJECT_ROOT)
    for name, project in baseline.projects.items():
        assert len(project.ref) == 40, f"{name} is pinned to {project.ref!r}, not a commit"

    with pytest.raises(BaselineError, match="40-character commit SHA"):
        _baseline(projects={"ns-cmsis-nn": {"url": "u", "ref": "v7.33.1"}})


def test_baseline_accepts_the_profilers_own_file() -> None:
    """`--baseline <hpx file>` builds against hpx's qualified pins, with no code
    dependency in either direction: only the shared subset is read."""
    hpx_shaped = _minimal_document(schema="hpx.compatibility-baseline")
    hpx_shaped["engines"] = {"something": "hpx-only"}  # hpx's extra sections are carried, not read
    baseline = parse_baseline(hpx_shaped)
    assert baseline.schema == "hpx.compatibility-baseline"
    assert baseline.project(CMSIS_NN_PROJECT).ref == _SHA_A
    assert baseline.to_dict()["engines"] == {"something": "hpx-only"}


def test_baseline_fingerprint_covers_the_whole_document() -> None:
    """The fingerprint identifies the file, not just the part this repo reads, so a
    bundle's baseline_fingerprint is comparable with a profiler run's."""
    first = _baseline()
    assert first.fingerprint == _baseline().fingerprint
    assert first.fingerprint != _baseline(baseline_id="other").fingerprint
    # A section this repo never reads still changes the identity.
    assert first.fingerprint != parse_baseline(_minimal_document(engines={"x": 1})).fingerprint


def test_baseline_rejects_a_missing_required_project() -> None:
    document = _minimal_document()
    del document["projects"]["nsx-pmu-armv8m"]
    with pytest.raises(BaselineError, match="missing required project"):
        parse_baseline(document)


def test_baseline_file_on_disk_loads(tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(_minimal_document()), encoding="utf-8")
    assert load_baseline(path).baseline_id == "test-baseline"
    assert default_baseline_path(PROJECT_ROOT) == PROJECT_ROOT / "assets" / "dependency_baseline.json"

    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(BaselineError, match="not valid JSON"):
        load_baseline(bad)


# --- the render --------------------------------------------------------------------


@pytest.fixture
def render(tmp_path: Path) -> nsx_app.AppRender:
    return nsx_app.render_app(
        BOARD,
        repo_root=PROJECT_ROOT,
        build_dir=tmp_path / "bd",
        baseline=resolve_baseline(PROJECT_ROOT),
    )


def test_nsx_yml_declares_the_profile_modules_plus_the_kernels(render) -> None:
    manifest = yaml.safe_load(render.nsx_yml)
    assert manifest["schema_version"] == 2
    assert manifest["project"]["name"] == "hct_benchmark_server"
    assert manifest["targets"] == {"default": "apollo510_evb", "supported": ["apollo510_evb"]}
    assert manifest["toolchain"] == "arm-none-eabi-gcc"

    names = [entry["name"] for entry in manifest["modules"]]
    profile = nsx_app.starter_profile(BOARD.nsx_board)
    assert names[:-1] == nsx_app.profile_module_names(profile), (
        "the starter profile is the source of truth for the board stack, in its order"
    )
    assert names[-1] == "nsx-cmsis-nn"


def test_every_module_is_pinned_to_the_baseline(render) -> None:
    manifest = yaml.safe_load(render.nsx_yml)
    baseline = resolve_baseline(PROJECT_ROOT)
    for entry in manifest["modules"]:
        assert entry["revision"] == baseline.pin(entry["project"]), entry["name"]


def test_module_registry_realigns_every_module_of_a_pinned_project(render) -> None:
    """NSX gives a module-level `revision` precedence over its project's, and the
    starter profile pins each migrated module to a tag. A project override alone
    would therefore be overruled and the app would build something other than the
    commit nsx.yml claims."""
    manifest = yaml.safe_load(render.nsx_yml)
    registry = manifest["module_registry"]
    baseline = resolve_baseline(PROJECT_ROOT)

    for project, entry in registry["projects"].items():
        expected = baseline.pin(project)
        if expected is not None:
            assert entry["revision"] == expected, project
    for name, entry in registry["modules"].items():
        expected = baseline.pin(entry["project"])
        if expected is not None:
            assert entry["revision"] == expected, name

    # Including the kernels, which the starter profile never mentions at all.
    assert registry["modules"]["nsx-cmsis-nn"]["revision"] == baseline.pin(CMSIS_NN_PROJECT)


def test_cmsis_nn_root_declares_a_local_module_source(tmp_path: Path) -> None:
    checkout = tmp_path / "ns-cmsis-nn"
    for sub in ("Include", "Source", "nsx"):
        (checkout / sub).mkdir(parents=True)
    (checkout / "nsx" / "nsx-module.yaml").write_text("module: {name: nsx-cmsis-nn}\n", encoding="utf-8")

    render = nsx_app.render_app(
        BOARD,
        repo_root=PROJECT_ROOT,
        build_dir=tmp_path / "bd",
        baseline=resolve_baseline(PROJECT_ROOT),
        cmsis_nn_root=checkout,
    )
    entry = next(e for e in yaml.safe_load(render.nsx_yml)["modules"] if e["name"] == "nsx-cmsis-nn")
    assert entry["source"] == {"path": str(checkout)}
    assert "project" not in entry and "revision" not in entry
    assert render.kernel_source.describe() == f"path:{checkout}"


def test_cmsis_nn_root_must_be_a_module_bearing_checkout(tmp_path: Path) -> None:
    bare = tmp_path / "not-a-checkout"
    bare.mkdir()
    with pytest.raises(nsx_app.AppRenderError, match="does not look like an ns-cmsis-nn checkout"):
        nsx_app.kernel_source_for(resolve_baseline(PROJECT_ROOT), bare)


def test_cmakelists_forces_the_kernel_switches_before_the_bootstrap(render) -> None:
    """An `option()` default cannot be overridden once its module has been added, so
    every kernel switch has to be set ahead of nsx_bootstrap_app()."""
    text = render.cmakelists
    bootstrap = text.index("nsx_bootstrap_app(")
    for name in ("NSX_CMSIS_NN_USE_REQUANTIZE_INLINE_ASM", "ARM_NN_ENABLE_F32", "ARM_NN_ENABLE_F16"):
        assert f'set({name} "ON" CACHE STRING' in text
        assert text.index(f"set({name}") < bootstrap, f"{name} is set too late to take effect"


def test_cmakelists_uses_nsxs_own_bootstrap_and_finalize(render) -> None:
    text = render.cmakelists
    assert "include(${CMAKE_CURRENT_LIST_DIR}/cmake/nsx/nsx_app_bootstrap.cmake)" in text
    assert "nsx_finalize_app(hct_benchmark_server)" in text
    # nsx::cmsis_nn, not a cmsis-nn subproject added by hand.
    assert "nsx::cmsis_nn" in text and "add_subdirectory" not in text


def test_cmakelists_compiles_the_server_sources_out_of_this_checkout(render) -> None:
    text = render.cmakelists
    assert f'set(HCT_HARDWARE_DIR "{PROJECT_ROOT / "cmake" / "hardware"}")' in text
    for name in ("benchmark_server_main.c", "benchmark_server_adapters.gen.c", "hct_build_id.c"):
        assert f'"${{HCT_HARDWARE_DIR}}/{name}"' in text
    assert '"${HCT_HARDWARE_DIR}/rtt/RTT/SEGGER_RTT.c"' in text
    assert "patch_build_id.py" in text, "the post-link build-id stamp must survive"
    assert f"HCT_SERVER_WORKSPACE_BYTES={BOARD.workspace_bytes}" in text


def test_symbol_ref_table_is_generated_from_the_synced_kernels(render, tmp_path: Path) -> None:
    """The header scan and the compiled archive must be the same tree, or the
    firmware's dispatch table references kernels the library never defined."""
    synced = nsx_app.synced_kernel_dir(nsx_app.app_dir_for(tmp_path / "bd"))
    assert f'set(HCT_KERNEL_SOURCE_DIR "{synced}")' in render.cmakelists
    assert '--cmsis-nn-root "${HCT_KERNEL_SOURCE_DIR}"' in render.cmakelists
    assert '--archive "$<TARGET_FILE:nsx_cmsis_nn>"' in render.cmakelists


def test_render_is_written_and_stable(render, tmp_path: Path) -> None:
    app_dir = tmp_path / "bd" / "nsx_app"
    assert (app_dir / "nsx.yml").read_text() == render.nsx_yml
    assert (app_dir / "CMakeLists.txt").read_text() == render.cmakelists

    # Rendering writes the app but makes no claim about it having been locked,
    # synced or built -- that is commit_render_state's job, after the fact.
    assert not (app_dir / nsx_app.RENDER_STATE).exists()
    nsx_app.commit_render_state(render)
    state = json.loads((app_dir / nsx_app.RENDER_STATE).read_text())
    assert state["render_digest"] == render.digest
    assert state["baseline_id"] == resolve_baseline(PROJECT_ROOT).baseline_id

    again = nsx_app.render_app(
        BOARD, repo_root=PROJECT_ROOT, build_dir=tmp_path / "bd",
        baseline=resolve_baseline(PROJECT_ROOT),
    )
    assert again.digest == render.digest, "the render must be deterministic"


def test_render_digest_moves_with_every_input_that_changes_the_firmware(tmp_path: Path) -> None:
    from helia_core_tester.hardware.firmware_build import FirmwareOptions

    baseline = resolve_baseline(PROJECT_ROOT)

    def _digest(**kwargs) -> str:
        return nsx_app.render_app(
            BOARD, repo_root=PROJECT_ROOT, build_dir=tmp_path / "bd", baseline=baseline, **kwargs
        ).digest

    base = _digest()
    assert base != _digest(
        kernel_options=FirmwareOptions(requantize_inline_asm=False).kernel_options()
    ), "the requantize switch must invalidate a reused lock and build dir"

    # A baseline edit that changes no pin this board uses still changes identity,
    # so a bundle cannot claim a baseline the lock never saw.
    other = parse_baseline({**baseline.to_dict(), "baseline_id": "something-else"})
    assert (
        nsx_app.render_app(
            BOARD, repo_root=PROJECT_ROOT, build_dir=tmp_path / "bd2", baseline=other
        ).digest
        != base
    )


def test_size_probe_render_swaps_the_target_and_the_float_switches(tmp_path: Path) -> None:
    """The probe is the same renderer with the variant's switches -- not a
    hand-mirrored -D list that can drift from what the firmware compiles."""
    from helia_core_tester.hardware.memory_report import SIZE_PROBE_VARIANTS, size_probe_build_dir

    variant = next(v for v in SIZE_PROBE_VARIANTS if v.name == "int")
    probe = nsx_app.render_app(
        BOARD,
        repo_root=PROJECT_ROOT,
        build_dir=size_probe_build_dir(tmp_path / "bd", variant),
        baseline=resolve_baseline(PROJECT_ROOT),
        kernel_options=variant.kernel_options(),
        build_size_probe=True,
    )
    assert "add_executable(hct_universal_size_probe" in probe.cmakelists
    assert "benchmark_server_main.c" not in probe.cmakelists
    assert 'set(ARM_NN_ENABLE_F32 "OFF" CACHE STRING' in probe.cmakelists
    assert 'set(ARM_NN_ENABLE_F16 "OFF" CACHE STRING' in probe.cmakelists
    assert "--enable-f32" not in probe.cmakelists
    assert probe.app_dir != nsx_app.app_dir_for(tmp_path / "bd"), "the probe needs its own app"


def test_write_app_keeps_nsxs_own_modules_cmake(tmp_path: Path) -> None:
    """`nsx lock`/`nsx sync` own cmake/nsx/modules.cmake -- it carries the resolved
    dependency order and the per-module directory mappings. Overwriting it with the
    bare placeholder would break the next configure."""
    build_dir = tmp_path / "bd"
    render = nsx_app.render_app(
        BOARD, repo_root=PROJECT_ROOT, build_dir=build_dir, baseline=resolve_baseline(PROJECT_ROOT)
    )
    modules_cmake = render.app_dir / "cmake" / "nsx" / "modules.cmake"
    assert modules_cmake.read_text() == render.modules_cmake

    resolved = "# Auto-generated by neuralspotx.\nset(NSX_APP_MODULES nsx-core)\n"
    modules_cmake.write_text(resolved, encoding="utf-8")
    nsx_app.render_app(
        BOARD, repo_root=PROJECT_ROOT, build_dir=build_dir, baseline=resolve_baseline(PROJECT_ROOT)
    )
    assert modules_cmake.read_text() == resolved


# --- review follow-ups -------------------------------------------------------------


def test_baseline_repository_url_reaches_the_module_registry(tmp_path: Path) -> None:
    """A pin is a commit *in a repository*; both halves have to reach the manifest.

    NSX resolves each project's URL from its packaged registry unless the app
    overrides it, and a starter profile's `project_overrides` entry carries only a
    revision. So a `--baseline` naming a fork would have had its refs fetched from
    the upstream URL -- failing, or worse succeeding on a SHA present in both.
    """
    document = _minimal_document()
    fork = "https://example.invalid/fork-of-ns-cmsis-nn.git"
    document["projects"]["ns-cmsis-nn"]["url"] = fork
    sdk_fork = "https://example.invalid/fork-of-nsx-ambiq-sdk.git"
    document["projects"]["nsx-ambiq-sdk"]["url"] = sdk_fork

    render = nsx_app.plan_app(
        BOARD, repo_root=PROJECT_ROOT, build_dir=tmp_path / "bd",
        baseline=parse_baseline(document),
    )
    projects = yaml.safe_load(render.nsx_yml)["module_registry"]["projects"]
    assert projects["ns-cmsis-nn"]["url"] == fork
    assert projects["ns-cmsis-nn"]["revision"] == _SHA_A
    # nsx-ambiq-sdk is the regression case: the starter profile overrides it with a
    # bare `revision`, so before this it reached the manifest with no url at all and
    # NSX silently used the packaged registry's repository.
    assert projects["nsx-ambiq-sdk"]["url"] == sdk_fork
    assert projects["nsx-ambiq-sdk"]["revision"] == _SHA_B

    # And a different baseline URL is a different render, so it cannot reuse a lock
    # resolved from the other repository.
    assert render.digest != nsx_app.plan_app(
        BOARD, repo_root=PROJECT_ROOT, build_dir=tmp_path / "bd",
        baseline=parse_baseline(_minimal_document()),
    ).digest


def test_plan_app_writes_nothing(tmp_path: Path) -> None:
    """`plan_app` answers "what would this render be" without touching the tree."""
    build_dir = tmp_path / "bd"
    planned = nsx_app.plan_app(
        BOARD, repo_root=PROJECT_ROOT, build_dir=build_dir, baseline=resolve_baseline(PROJECT_ROOT)
    )
    assert not nsx_app.app_dir_for(build_dir).exists()
    written = nsx_app.render_app(
        BOARD, repo_root=PROJECT_ROOT, build_dir=build_dir, baseline=resolve_baseline(PROJECT_ROOT)
    )
    assert planned.digest == written.digest
    assert (nsx_app.app_dir_for(build_dir) / "nsx.yml").read_text() == planned.nsx_yml


def test_render_state_is_not_a_side_effect_of_rendering(tmp_path: Path) -> None:
    """The state file claims a lock and a module tree exist for this render.

    Writing it at render time made the claim before it was true, which hid the one
    case `lock_reuse_reason` exists for: a baseline edit that leaves `nsx.yml`
    byte-identical (a new `baseline_id`, or a pin for a project this board does not
    resolve) keeps NSX's own manifest hash the same, so only the render digest can
    reject the stale lock -- and it cannot if it has already been overwritten.
    """
    build_dir = tmp_path / "bd"
    baseline = resolve_baseline(PROJECT_ROOT)
    first = nsx_app.render_app(
        BOARD, repo_root=PROJECT_ROOT, build_dir=build_dir, baseline=baseline
    )
    assert nsx_app.read_render_state(first.app_dir) is None
    nsx_app.commit_render_state(first)
    assert nsx_app.read_render_state(first.app_dir)["render_digest"] == first.digest

    renamed = parse_baseline({**baseline.to_dict(), "baseline_id": "renamed-baseline"})
    second = nsx_app.render_app(
        BOARD, repo_root=PROJECT_ROOT, build_dir=build_dir, baseline=renamed
    )
    assert second.nsx_yml == first.nsx_yml, "the manifest is unchanged, which is the point"
    assert second.digest != first.digest
    assert nsx_app.read_render_state(second.app_dir)["render_digest"] == first.digest, (
        "re-rendering must leave the previous build's state standing"
    )
