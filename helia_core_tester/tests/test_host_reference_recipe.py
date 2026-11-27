"""The host reference recipe from the library profile: parsing, path checks, and the build it drives."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from helia_core_tester.core import harness as harness_mod
from helia_core_tester.core import host_reference as hr
from helia_core_tester.core.library import LibraryProfileError, legacy_host_reference, load_profile, resolve_library
from helia_core_tester.generation.reference import host_kernels as hk
from helia_core_tester.tests.test_host_reference import _FAKE_CMAKELISTS, _case, _fake_tree, needs_cmake

_DSP_HARNESS = (
    "harness:\n  includes: [arm_math.h]\n  header_includes: [arm_math.h]\n"
    "  status:\n    type: arm_status\n    success: ARM_MATH_SUCCESS\n    argument_error: ARM_MATH_ARGUMENT_ERROR\n"
    "    codes: {ARM_MATH_SUCCESS: 0, ARM_MATH_ARGUMENT_ERROR: -1}\n"
)
_DSP_RECIPE = (
    "host_reference:\n  project: Testing/hct/host_reference\n  option: CMSISDSP_HOST_REFERENCE\n"
    "  manifest: cmsis_dsp_host_reference.json\n"
    "  build_inputs: [Include, Source, Testing/hct/host_reference]\n"
)


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(harness_mod.PROFILE_ENV, raising=False)
    monkeypatch.setattr(harness_mod, "_REGISTERED", {})
    harness_mod._library_of_root.cache_clear()
    harness_mod._profile_file.cache_clear()
    yield
    harness_mod._library_of_root.cache_clear()
    harness_mod._profile_file.cache_clear()


def _dsp_tree(root: Path) -> Path:
    for sub in ("Include", "Source", "Testing/hct/host_reference"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    (root / "Testing/hct/host_reference/CMakeLists.txt").write_text(
        'option(CMSISDSP_HOST_REFERENCE "" ON)\n')
    return root


def _profile(directory: Path, body: str) -> Path:
    path = directory / "hct-library.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def _dsp_profile(root: Path, recipe: str = _DSP_RECIPE) -> Path:
    return _profile(_dsp_tree(root), "schema: hct/library-profile/1\nlibrary: cmsis-dsp\n" + _DSP_HARNESS + recipe)


def test_the_built_in_recipe_is_the_one_ns_cmsis_nn_cmake_offers() -> None:
    nn = legacy_host_reference()
    assert (nn.library, nn.option, nn.manifest, nn.project) == (
        "ns-cmsis-nn", "NS_CMSIS_NN_HOST_REFERENCE", "ns_cmsis_nn_host_reference.json", ".")
    assert nn.build_inputs == ("CMakeLists.txt", "cmake", "Include", "Source")
    assert not nn.declared


def test_a_legacy_checkout_and_an_nn_profile_without_a_block_take_the_built_in_recipe(tmp_path) -> None:
    tree = _fake_tree(tmp_path)
    assert resolve_library(tree).host_reference == legacy_host_reference()
    profile = load_profile(_profile(tree, "schema: hct/library-profile/1\nlibrary: ns-cmsis-nn\n"))
    assert profile.host_reference == legacy_host_reference()


def test_another_library_without_a_block_has_none(tmp_path) -> None:
    profile = load_profile(_dsp_profile(tmp_path, recipe=""))
    assert profile.host_reference is None


def test_the_profile_names_a_project_below_the_library(tmp_path) -> None:
    recipe = load_profile(_dsp_profile(tmp_path)).host_reference
    assert recipe == hr.HostReferenceRecipe(
        library="cmsis-dsp", option="CMSISDSP_HOST_REFERENCE", manifest="cmsis_dsp_host_reference.json",
        build_inputs=("Include", "Source", "Testing/hct/host_reference"), project="Testing/hct/host_reference",
        declared=True)
    assert hr.supports(tmp_path, recipe)


def test_paths_are_relative_to_source_root_not_the_profile(tmp_path) -> None:
    pack = _dsp_tree(tmp_path) / "Testing" / "hct"
    profile = load_profile(_profile(
        pack, "schema: hct/library-profile/1\nlibrary: cmsis-dsp\nsource_root: ../..\n" + _DSP_HARNESS + _DSP_RECIPE))
    assert profile.source_root == tmp_path.resolve()
    assert profile.host_reference.project == "Testing/hct/host_reference"


def _recipe(**fields) -> str:
    base = {"project": "Testing/hct/host_reference", "option": "CMSISDSP_HOST_REFERENCE",
            "manifest": "cmsis_dsp_host_reference.json",
            "build_inputs": "[Include, Source, Testing/hct/host_reference]"}
    base.update(fields)
    return "host_reference:\n" + "".join(f"  {k}: {v}\n" for k, v in base.items() if v is not None)


@pytest.mark.parametrize("fields, said", [
    ({"option": None}, "'option' is a required property"),
    ({"option": "1BAD"}, "host_reference/option"),
    ({"manifest": "build/x.json"}, "host_reference/manifest"),
    ({"manifest": "x.txt"}, "host_reference/manifest"),
    ({"build_inputs": "[]"}, "host_reference/build_inputs"),
    ({"build_inputs": "[Include, Include, Testing/hct/host_reference]"}, "host_reference/build_inputs"),
    ({"project": "'a\\\\b'"}, "host_reference/project"),
    ({"extra": "1"}, "Additional properties are not allowed"),
])
def test_the_schema_refuses_a_bad_block(tmp_path, fields, said) -> None:
    with pytest.raises(LibraryProfileError, match=said):
        load_profile(_dsp_profile(tmp_path, _recipe(**fields)))


@pytest.mark.parametrize("fields, said", [
    ({"project": "/abs"}, "'project' '/abs' must be a relative POSIX path"),
    ({"project": "Testing/../Testing/hct/host_reference"}, "must be a relative POSIX path without '..'"),
    ({"project": "Missing"}, "'project' 'Missing' does not exist"),
    ({"project": "Include"}, "project 'Include' has no CMakeLists.txt"),
    ({"build_inputs": "[Include, Missing, Testing/hct/host_reference]"}, "'build_inputs' 'Missing' does not exist"),
    ({"build_inputs": "[Include, Source]"}, "no build input covers Testing/hct/host_reference/CMakeLists.txt"),
    ({"build_inputs": "[Testing/hct/host_reference/CMakeLists.txt]"}, None),
    ({"build_inputs": "[Testing]"}, None),
])
def test_paths_must_exist_inside_the_library_and_cover_the_project(tmp_path, fields, said) -> None:
    path = _dsp_profile(tmp_path, _recipe(**fields))
    if said is None:
        assert load_profile(path).host_reference.declared
        return
    with pytest.raises(LibraryProfileError, match=said.replace("(", r"\(").replace(")", r"\)")):
        load_profile(path)


def test_a_build_input_that_escapes_through_a_symlink_is_refused(tmp_path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    root = _dsp_tree(tmp_path / "lib")
    (root / "Escape").symlink_to(outside)
    with pytest.raises(LibraryProfileError, match="is outside the library root"):
        load_profile(_dsp_profile(root, _recipe(build_inputs="[Escape, Testing/hct/host_reference]")))


def test_parse_recipe_refuses_what_the_schema_would(tmp_path) -> None:
    good = {"option": "X", "manifest": "x.json", "build_inputs": ["CMakeLists.txt"]}
    for bad, said in [
        ([], "must be a mapping"),
        ({**good, "bogus": 1}, "unknown field"),
        ({**good, "option": "a-b"}, "'option' must be a CMake option name"),
        ({**good, "manifest": "../x.json"}, "'manifest' must be a bare .json file name"),
        ({**good, "build_inputs": "CMakeLists.txt"}, "'build_inputs' must be a list"),
        ({**good, "build_inputs": ["a", "a"]}, "'build_inputs' has duplicates"),
        ({**good, "build_inputs": [""]}, "'build_inputs' must be a non-empty path"),
        ({**good, "build_inputs": ["a\0b"]}, "'build_inputs' must be a non-empty path"),
        ({**good, "project": 3}, "'project' must be a non-empty path"),
    ]:
        with pytest.raises(hr.HostReferenceError, match=said):
            hr.parse_recipe("x", bad, "where")


def test_the_cache_key_follows_the_project(tmp_path) -> None:
    tree = _fake_tree(tmp_path)
    nn = legacy_host_reference()
    moved = hr.HostReferenceRecipe(**{**nn.__dict__, "project": "sub"})
    assert hr.host_reference_key(tree, nn, "cc", "cmake") != hr.host_reference_key(tree, moved, "cc", "cmake")


def test_another_library_without_a_recipe_is_not_scanned_as_ns_cmsis_nn(tmp_path) -> None:
    root = _dsp_tree(tmp_path)
    _dsp_profile(root, recipe="")
    with pytest.raises(hk.HostCheckError, match=r"\(cmsis-dsp\) has no host reference build"):
        hk.ensure_host_kernels(root, "m0", cache_root=tmp_path / "hk")


@pytest.mark.parametrize("recipe", ["", _DSP_RECIPE])
def test_dsp_mode_says_the_scan_is_ns_cmsis_nn_only(tmp_path, recipe) -> None:
    root = _dsp_tree(tmp_path)
    _dsp_profile(root, recipe=recipe)
    with pytest.raises(hk.HostCheckError, match=r"dsp host kernels come from hct's source scan, which reads ns-cmsis-nn only"):
        hk.ensure_host_kernels(root, "dsp", cache_root=tmp_path / "hk")


def test_a_project_reached_through_a_symlink_must_be_covered_where_it_really_is(tmp_path) -> None:
    root = _dsp_tree(tmp_path)
    real = root / "Other" / "hostref"
    real.mkdir(parents=True)
    (real / "CMakeLists.txt").write_text('option(CMSISDSP_HOST_REFERENCE "" ON)\n')
    (root / "Testing" / "link").symlink_to(real, target_is_directory=True)
    bad = _recipe(project="Testing/link", build_inputs="[Include, Testing]")
    with pytest.raises(LibraryProfileError, match="no build input covers Testing/link/CMakeLists.txt"):
        load_profile(_dsp_profile(root, bad))
    good = _recipe(project="Testing/link", build_inputs="[Include, Other]")
    assert load_profile(_dsp_profile(root, good)).host_reference.project == "Testing/link"


@pytest.mark.parametrize("mode", ["m0", "dsp"])
def test_a_broken_profile_is_a_host_check_error(tmp_path, mode) -> None:
    root = _dsp_tree(tmp_path)
    _dsp_profile(root, _recipe(build_inputs="[Include, Missing, Testing/hct/host_reference]"))
    with pytest.raises(hk.HostCheckError, match="'build_inputs' 'Missing' does not exist"):
        hk.ensure_host_kernels(root, mode, cache_root=tmp_path / "hk")


def test_a_missing_run_profile_is_a_host_check_error(tmp_path, monkeypatch) -> None:
    tree = _fake_tree(tmp_path)
    monkeypatch.setenv(harness_mod.PROFILE_ENV, str(tmp_path / "missing.yaml"))
    with pytest.raises(hk.HostCheckError, match="is not a file"):
        hk.ensure_host_kernels(tree, "dsp", cache_root=tmp_path / "hk")


def test_a_profile_whose_source_root_is_elsewhere_is_refused(tmp_path) -> None:
    lib = _dsp_tree(tmp_path / "lib")
    inner = lib / "inner"
    for sub in ("Include", "Source"):
        (inner / sub).mkdir(parents=True)
    _profile(inner, "schema: hct/library-profile/1\nlibrary: cmsis-dsp\nsource_root: ..\n" + _DSP_HARNESS + _DSP_RECIPE)
    with pytest.raises(hk.HostCheckError, match="has source_root .*, not .*inner"):
        hk.ensure_host_kernels(inner, "m0", cache_root=tmp_path / "hk")


def test_a_declared_recipe_does_not_need_the_ns_cmsis_nn_layout(tmp_path) -> None:
    root = tmp_path / "lib"
    (root / "Src").mkdir(parents=True)
    (root / "Src" / "CMakeLists.txt").write_text("project(x C)\n")
    _profile(root, "schema: hct/library-profile/1\nlibrary: cmsis-dsp\n" + _DSP_HARNESS
             + "host_reference:\n  project: Src\n  option: X_HOST_REFERENCE\n  manifest: x.json\n  build_inputs: [Src]\n")
    with pytest.raises(hk.HostCheckError, match="does not offer X_HOST_REFERENCE"):
        hk.ensure_host_kernels(root, "m0", cache_root=tmp_path / "hk")


_SUBPROJECT_RECIPE = (
    "host_reference:\n  project: Tests/hostref\n  option: NS_CMSIS_NN_HOST_REFERENCE\n"
    "  manifest: ns_cmsis_nn_host_reference.json\n  build_inputs: [Include, Source, Tests/hostref]\n"
)


def _subproject_tree(tmp_path: Path, **kwargs) -> Path:
    """A checkout whose host reference is a CMake project below it, named by its profile."""
    tree = _fake_tree(tmp_path, **kwargs)
    project = tree / "Tests" / "hostref"
    project.mkdir(parents=True)
    (project / "CMakeLists.txt").write_text(
        _FAKE_CMAKELISTS.replace("${CMAKE_CURRENT_SOURCE_DIR}", "${HCT_ROOT}")
        .replace("@CMAKE_CURRENT_SOURCE_DIR@", "@HCT_ROOT@")
        .replace("project(fake_kernels C)",
                 'project(fake_kernels C)\nget_filename_component(HCT_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)'))
    (tree / "CMakeLists.txt").write_text("project(no_host_reference C)\n")
    _profile(tree, "schema: hct/library-profile/1\nlibrary: ns-cmsis-nn\n" + _SUBPROJECT_RECIPE)
    return tree


def test_a_declared_recipe_whose_project_lacks_the_option_fails_rather_than_scanning(tmp_path) -> None:
    tree = _subproject_tree(tmp_path)
    (tree / "Tests" / "hostref" / "CMakeLists.txt").write_text("project(x C)\n")
    with pytest.raises(hk.HostCheckError, match="does not offer NS_CMSIS_NN_HOST_REFERENCE"):
        hk.ensure_host_kernels(tree, "m0", cache_root=tmp_path / "hk")


@needs_cmake
def test_the_profile_project_builds_the_reference_host_check_uses(tmp_path) -> None:
    tree = _subproject_tree(tmp_path)
    cases = tmp_path / "cases"
    _case(cases, "kernel_ok", "hct_kernel() == 42 && HCT_FAKE_REF == 1")
    report = hk.run_host_check([cases], tree, mode="m0", cache_root=tmp_path / "hk", jobs=2)
    assert report.provider == hk.PROVIDER_HOST_REFERENCE
    assert report.total == 1 and report.passed == 1


@needs_cmake
def test_the_profile_project_failure_is_a_host_check_error(tmp_path) -> None:
    tree = _subproject_tree(tmp_path, kernel="#error injected")
    with pytest.raises(hk.HostCheckError, match="host reference build failed"):
        hk.ensure_host_kernels(tree, "m0", cache_root=tmp_path / "hk")


def test_the_run_profile_wins_for_its_source_root(tmp_path, monkeypatch) -> None:
    root = _dsp_tree(tmp_path / "lib")
    pack = root / "Testing" / "hct"
    path = _profile(pack, "schema: hct/library-profile/1\nlibrary: cmsis-dsp\nsource_root: ../..\n"
                    + _DSP_HARNESS + _DSP_RECIPE)
    assert harness_mod.library_of_root(root).library == "ns-cmsis-nn"
    monkeypatch.setenv(harness_mod.PROFILE_ENV, str(path))
    assert harness_mod.library_of_root(root).host_reference.project == "Testing/hct/host_reference"
    monkeypatch.delenv(harness_mod.PROFILE_ENV)
    harness_mod.register_library(load_profile(path))
    assert harness_mod.library_of_root(root).library == "cmsis-dsp"


_REAL_DSP = os.environ.get("HCT_TEST_DSP_TREE", "").strip()


@needs_cmake
@pytest.mark.skipif(not _REAL_DSP, reason="set HCT_TEST_DSP_TREE to a CMSIS-DSP fork checkout with Testing/hct")
def test_real_cmsis_dsp_host_reference(tmp_path) -> None:
    root = Path(_REAL_DSP)
    recipe = hr.parse_recipe("cmsis-dsp", {
        "project": "Testing/hct/host_reference", "option": "CMSISDSP_HOST_REFERENCE",
        "manifest": "cmsis_dsp_host_reference.json",
        "build_inputs": ["Include", "PrivateInclude", "Source", "Testing/hct/host_reference"],
    }, "test", root=root)
    ref = hr.ensure_host_reference(root, recipe, cache_root=tmp_path / "cache")
    assert ref.library == "cmsis-dsp" and ref.archive.is_file()
    assert ref.float16_type in hr.FLOAT16_TYPES


def _src_only_library(root: Path) -> Path:
    """A library whose profile names a host reference and which has no Include/ or Source/."""
    (root / "Src").mkdir(parents=True)
    (root / "Src" / "CMakeLists.txt").write_text("project(x C)\n")
    _profile(root, "schema: hct/library-profile/1\nlibrary: cmsis-dsp\n" + _DSP_HARNESS
             + "host_reference:\n  project: Src\n  option: X_HOST_REFERENCE\n  manifest: x.json\n  build_inputs: [Src]\n")
    return root


def test_checkout_problem_names_what_is_wrong(tmp_path) -> None:
    assert hk.checkout_problem(None) == "no library checkout"
    assert "library checkout not found" in hk.checkout_problem(tmp_path / "missing")
    (tmp_path / "empty").mkdir()
    assert "not an ns-cmsis-nn checkout" in hk.checkout_problem(tmp_path / "empty")
    assert hk.checkout_problem(_fake_tree(tmp_path / "nn")) is None
    assert hk.checkout_problem(_src_only_library(tmp_path / "lib")) is None


def test_checkout_problem_reports_a_broken_profile(tmp_path) -> None:
    root = _dsp_tree(tmp_path)
    _dsp_profile(root, _recipe(build_inputs="[Include, Missing, Testing/hct/host_reference]"))
    assert "'build_inputs' 'Missing' does not exist" in hk.checkout_problem(root)


def test_a_library_without_a_recipe_still_needs_the_ns_cmsis_nn_layout(tmp_path) -> None:
    root = tmp_path / "lib"
    root.mkdir()
    _profile(root, "schema: hct/library-profile/1\nlibrary: cmsis-dsp\n" + _DSP_HARNESS)
    assert "not an ns-cmsis-nn checkout" in hk.checkout_problem(root)


def _step_config(tmp_path: Path, root: Path, **kw):
    from helia_core_tester.core.config import Config

    project = tmp_path / "proj"
    project.mkdir(exist_ok=True)
    return Config(project_root=project, generation_dir=hk.tester_root() / "helia_core_tester" / "generation",
                  generated_tests_root=tmp_path / "g", reports_root=tmp_path / "r", cmsis_nn_root=root, **kw)


def test_the_steps_take_a_library_whose_profile_names_its_layout(tmp_path) -> None:
    from helia_core_tester.core.steps.host_check import HostCheckStep
    from helia_core_tester.core.steps.record import RecordStep

    root = _src_only_library(tmp_path / "lib")
    assert HostCheckStep(_step_config(tmp_path, root)).validate() is None
    assert RecordStep(_step_config(tmp_path, root)).validate() is None


def test_the_steps_refuse_an_empty_checkout(tmp_path) -> None:
    from helia_core_tester.core.steps.generate import GenerateStep
    from helia_core_tester.core.steps.host_check import HostCheckStep
    from helia_core_tester.core.steps.record import RecordStep

    (tmp_path / "empty").mkdir()
    root = tmp_path / "empty"
    assert "host check: " in HostCheckStep(_step_config(tmp_path, root)).validate()
    assert "record: " in RecordStep(_step_config(tmp_path, root)).validate()
    message = GenerateStep(_step_config(tmp_path, root, golden="record")).validate()
    assert message is not None and "ns-cmsis-nn" in message


def test_checkout_problem_refuses_a_profile_whose_source_root_is_elsewhere(tmp_path) -> None:
    lib = _dsp_tree(tmp_path / "lib")
    inner = lib / "inner"
    inner.mkdir()
    _profile(inner, "schema: hct/library-profile/1\nlibrary: cmsis-dsp\nsource_root: ..\n" + _DSP_HARNESS + _DSP_RECIPE)
    assert "has source_root" in hk.checkout_problem(inner)
