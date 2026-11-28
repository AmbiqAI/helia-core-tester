"""The profile's build block, the build step that reads it, and the root CMake project it drives."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

from helia_core_tester.core.config import Config
from helia_core_tester.core.library import LibraryProfileError, load_profile, require_supported
from helia_core_tester.core.library_build import BuildSpec, BuildSpecError
from helia_core_tester.core.runtime_env import RuntimeEnvContext
from helia_core_tester.core.steps.build import BuildStep
from helia_core_tester.tests.test_pack import _library

ROOT = Path(__file__).resolve().parents[2]
_PACK = "pack:\n  api: \"1.0\"\n  descriptors: descriptors\n  modules: [basic]\n"
_BUILD = (
    "build:\n  target: dsplib\n"
    "  definitions: {CMSISCORE: \"{cmsis_path}/CMSIS/Core\", LOOPUNROLL: \"ON\"}\n"
    "  cpu_definitions:\n    m55: {MVEI: \"ON\"}\n    cortex-m4: {MVEI: \"OFF\"}\n"
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("HELIA_CORE_TESTER_LIBRARY", "HELIA_CORE_TESTER_CMSIS_NN_ROOT", "CMSIS_NN_ROOT",
                 "HCT_LIBRARY_PROFILE", "HELIA_CORE_TESTER_CONFIG"):
        monkeypatch.delenv(name, raising=False)


def _dsp(tmp_path: Path, build: str = _BUILD, cmake: bool = True) -> Path:
    root = tmp_path / "dsp"
    profile = _library(root, pack=_PACK + build)
    if cmake:
        (root / "CMakeLists.txt").write_text("add_library(dsplib STATIC probe.c)\n")
    return profile


def _load(tmp_path: Path, build: str, cmake: bool = True):
    root = tmp_path / "dsp"
    _library(root, pack=_PACK)
    if cmake:
        (root / "CMakeLists.txt").write_text("add_library(dsplib STATIC probe.c)\n")
    profile = root / "Testing" / "hct" / "hct-library.yaml"
    profile.write_text(profile.read_text() + build)
    return load_profile(profile)


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "tester"
    (root / "helia_core_tester" / "generation").mkdir(parents=True)
    (root / "helia_core_tester" / "fvp").mkdir(parents=True)
    (root / "helia_core_tester" / "fvp" / "build_and_run_fvp.py").write_text("")
    return root


def _config(tmp_path: Path, profile: Path, **overrides) -> Config:
    return Config(project_root=_repo(tmp_path), library=profile.parent,
                  _explicit_overrides={"project_root", "library", *overrides}, **overrides)


def _runtime_env(root: Path) -> RuntimeEnvContext:
    downloads = root / "dl"
    return RuntimeEnvContext(
        downloads_dir=downloads, ethos_path=downloads / "ethos", cmsis5_path=downloads / "CMSIS_5",
        toolchain_file=downloads / "toolchain.cmake", compiler_tag="gcc", fvp_exe=downloads / "fvp",
        child_env={"PATH": os.environ.get("PATH", "")})


def _defs(cmd: list[str]) -> list[str]:
    return [cmd[i + 1] for i, item in enumerate(cmd) if item == "--cmake-def"]


# --- the block ---------------------------------------------------------------------------------


def test_block_loads(tmp_path) -> None:
    build = load_profile(_dsp(tmp_path)).build
    assert build.project == (tmp_path / "dsp").resolve() and build.target == "dsplib"
    assert build.definitions == (("CMSISCORE", "{cmsis_path}/CMSIS/Core"), ("LOOPUNROLL", "ON"))
    assert dict(build.cpu_definitions) == {"cortex-m4": (("MVEI", "OFF"),), "cortex-m55": (("MVEI", "ON"),)}


def test_definitions_for_a_cpu(tmp_path) -> None:
    build = load_profile(_dsp(tmp_path)).build
    cmsis = tmp_path / "CMSIS_5"
    assert build.definitions_for("cortex-m55", cmsis) == [
        f"CMSISCORE={cmsis}/CMSIS/Core", "LOOPUNROLL=ON", "MVEI=ON"]
    assert build.definitions_for("m4", cmsis)[-1] == "MVEI=OFF"
    assert build.definitions_for("cortex-m0", cmsis) == [f"CMSISCORE={cmsis}/CMSIS/Core", "LOOPUNROLL=ON"]


def test_project_and_its_cmakelists(tmp_path) -> None:
    (tmp_path / "dsp" / "Lib").mkdir(parents=True)
    (tmp_path / "dsp" / "Lib" / "CMakeLists.txt").write_text("")
    assert _load(tmp_path, "build:\n  project: Lib\n  target: dsplib\n").build.project.name == "Lib"


@pytest.mark.parametrize("block, message", [
    ("build:\n  target: dsplib\n", "has no CMakeLists.txt"),
    ("build:\n  project: ../..\n  target: dsplib\n", "must not contain '..'"),
    ("build:\n  project: /tmp\n  target: dsplib\n", "must be a relative POSIX path"),
    ("build:\n  project: Missing\n  target: dsplib\n", "has no CMakeLists.txt"),
])
def test_project_problems(tmp_path, block, message) -> None:
    with pytest.raises(LibraryProfileError, match=message):
        _load(tmp_path, block, cmake=False)


@pytest.mark.parametrize("definitions, message", [
    ("{CMAKE_C_FLAGS: \"-O0\"}", "CMAKE_C_FLAGS is set by hct itself"),
    ("{HCT_LIBRARY_TARGET: x}", "is set by hct itself"),
    ("{CMSIS_PATH: x}", "is set by hct itself"),
    ("{ENABLE_COVERAGE: \"ON\"}", "is set by hct itself"),
    ("{ARM_NN_ENABLE_F32: \"ON\"}", "is set by hct itself"),
    ("{A: \"x;y\"}", "cannot carry"),
    ("{A: \"x\\\"y\"}", "cannot carry"),
    ("{A: \"${HOME}\"}", "cannot carry"),
    ("{A: \"x\\ny\"}", "cannot carry"),
    ("{A: \"{home}/x\"}", r"names \{home\}; the only placeholder is \{cmsis_path\}"),
    ("{A: \"{cmsis_path/x\"}", "unmatched brace"),
    ("{A: \"x}\"}", "unmatched brace"),
    ("{A: 1}", "is not of type 'string'"),
    ("{\"bad-name\": x}", "does not match"),
])
def test_definition_problems(tmp_path, definitions, message) -> None:
    with pytest.raises(LibraryProfileError, match=message):
        _load(tmp_path, f"build:\n  target: dsplib\n  definitions: {definitions}\n")


@pytest.mark.parametrize("cpus, message", [
    ("{cortex-m99: {A: x}}", "'cortex-m99' is not a CPU hct builds for"),
    ("{m55: {A: x}, cortex-m55: {A: y}}", "names cortex-m55 again"),
    ("{m55: {B: x}}", "B is also in definitions"),
    ("{m55: {CMAKE_X: x}}", "is set by hct itself"),
])
def test_cpu_definition_problems(tmp_path, cpus, message) -> None:
    with pytest.raises(LibraryProfileError, match=message):
        _load(tmp_path, f"build:\n  target: dsplib\n  definitions: {{B: y}}\n  cpu_definitions: {cpus}\n")


@pytest.mark.parametrize("block, message", [
    ("build:\n  project: .\n", "'target' is a required property"),
    ("build:\n  target: \"a b\"\n", "does not match"),
    ("build:\n  target: dsplib\n  extra: 1\n", "Additional properties"),
])
def test_block_shape(tmp_path, block, message) -> None:
    with pytest.raises(LibraryProfileError, match=message):
        _load(tmp_path, block)


def test_a_cmsis_path_that_cmake_cannot_carry(tmp_path) -> None:
    build = BuildSpec(project=tmp_path, target="t", definitions=(("A", "{cmsis_path}"),))
    with pytest.raises(BuildSpecError, match="cannot carry"):
        build.definitions_for("cortex-m55", tmp_path / "a;b")
    with pytest.raises(ValueError, match="Unsupported CPU"):
        build.definitions_for("cortex-m99", tmp_path)


def test_ns_cmsis_nn_names_no_build(tmp_path) -> None:
    root = tmp_path / "nn"
    (root / "Include").mkdir(parents=True)
    (root / "Include" / "arm_nnfunctions.h").write_text("")
    (root / "CMakeLists.txt").write_text("")
    (root / "hct-library.yaml").write_text(
        "schema: hct/library-profile/1\nlibrary: ns-cmsis-nn\nbuild:\n  target: cmsis-nn\n")
    with pytest.raises(LibraryProfileError, match="names build, but this hct builds ns-cmsis-nn from its own rules"):
        require_supported(load_profile(root / "hct-library.yaml"))


# --- the build step ----------------------------------------------------------------------------


def test_build_step_builds_each_cpu_from_the_block(tmp_path) -> None:
    profile = _dsp(tmp_path)
    config = _config(tmp_path, profile, cpu="cortex-m4,cortex-m55", suite="both")
    step = BuildStep(config, runtime_env=_runtime_env(tmp_path))
    assert step.validate() is None
    commands = step._build_commands()
    cpus = [cmd[cmd.index("--cpu") + 1] for cmd in commands]
    assert cpus == ["cortex-m4", "cortex-m55", "cortex-m4", "cortex-m55"]
    cmsis = tmp_path / "dl" / "CMSIS_5"
    for cmd, cpu in zip(commands, cpus):
        defs = _defs(cmd)
        assert defs == [
            f"CMSIS_OPTIMIZATION_LEVEL={config.optimization}",
            f"HCT_LIBRARY_DIR={(tmp_path / 'dsp').resolve()}", "HCT_LIBRARY_TARGET=dsplib",
            f"HCT_LIBRARY_COMPILE_OPTIONS={config.optimization}",
            f"CMSISCORE={cmsis}/CMSIS/Core", "LOOPUNROLL=ON", "MVEI=" + ("ON" if cpu == "cortex-m55" else "OFF"),
        ]
        assert not any(d.startswith(("ARM_NN_", "CMSIS_NN_ROOT")) for d in defs)


def test_build_step_without_a_runtime_env_takes_the_downloads_cmsis(tmp_path) -> None:
    config = _config(tmp_path, _dsp(tmp_path), cpu="cortex-m55")
    defs = _defs(BuildStep(config)._build_commands()[0])
    assert f"CMSISCORE={Path(config.downloads_dir).resolve() / 'CMSIS_5'}/CMSIS/Core" in defs


def test_build_step_refuses_a_library_without_a_build(tmp_path) -> None:
    config = _config(tmp_path, _dsp(tmp_path, build=""))
    assert "its profile names no build, so hct cannot build its FVP tests" in BuildStep(config).validate()


def test_build_step_refuses_coverage_for_another_library(tmp_path) -> None:
    config = _config(tmp_path, _dsp(tmp_path), coverage=True)
    assert BuildStep(config).validate() == "--coverage instruments ns-cmsis-nn only; cmsis-dsp builds without it"


def test_another_library_needs_no_root_cmakelists(tmp_path) -> None:
    config = _config(tmp_path, _dsp(tmp_path, build="", cmake=False))
    assert config.cmsis_nn_root_error() is None


def test_ns_cmsis_nn_commands_are_unchanged(tmp_path) -> None:
    root = tmp_path / "nn"
    (root / "Include").mkdir(parents=True)
    (root / "Include" / "arm_nnfunctions.h").write_text("")
    (root / "CMakeLists.txt").write_text("")
    config = Config(project_root=_repo(tmp_path), cmsis_nn_root=root, cpu="cortex-m4,cortex-m55", suite="int",
                    _explicit_overrides={"project_root", "cmsis_nn_root", "cpu", "suite"})
    step = BuildStep(config, runtime_env=_runtime_env(tmp_path))
    assert step.validate() is None
    [cmd] = step._build_commands()
    assert cmd[cmd.index("--cpu") + 1] == "cortex-m4,cortex-m55"
    assert _defs(cmd) == [f"CMSIS_OPTIMIZATION_LEVEL={config.optimization}", f"CMSIS_NN_ROOT={root.resolve()}",
                          "ARM_NN_ENABLE_F32=OFF", "ARM_NN_ENABLE_F16=OFF"]


# --- the root CMake project --------------------------------------------------------------------


@pytest.fixture
def cmake_project(tmp_path):
    for tool in ("cmake", "ninja"):
        if not shutil.which(tool):
            pytest.skip(f"requires {tool}")
    compiler = shutil.which("cc") or shutil.which("gcc")
    if compiler is None:
        pytest.skip("requires a C compiler")
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "probe.c").write_text("int probe(void) { return 0; }\n")
    (lib / "CMakeLists.txt").write_text(
        "add_library(dsplib STATIC probe.c)\n"
        'set_property(TARGET dsplib PROPERTY COMPILE_OPTIONS "")\n'
        "target_compile_definitions(dsplib PRIVATE \"PROBE_DEF=${PROBE_DEF}\")\n")
    cmsis = tmp_path / "cmsis"
    for name in ("CMSIS/Core/Include/cmsis_compiler.h", "Device/ARM/ARMCM55/Source/startup_ARMCM55.c",
                 "Device/ARM/ARMCM55/Source/system_ARMCM55.c"):
        (cmsis / name).parent.mkdir(parents=True, exist_ok=True)
        (cmsis / name).touch()

    def configure(*defs: str, check: bool = True):
        cmd = ["cmake", "-G", "Ninja", "-S", str(ROOT), "-B", str(tmp_path / "build"),
               f"-DCMSIS_PATH={cmsis}", f"-DCMAKE_C_COMPILER={compiler}", "-DTARGET_CPU=cortex-m55",
               "-DHELIA_BUILD_GENERATED_TESTS=OFF", f"-DCMSIS_NN_ROOT={tmp_path / 'no-nn'}",
               *[f"-D{d}" for d in defs]]
        shutil.rmtree(tmp_path / "build", ignore_errors=True)
        done = subprocess.run(cmd, text=True, capture_output=True)
        if check and done.returncode != 0:
            pytest.fail(done.stdout + done.stderr)
        return done

    return lib, configure, tmp_path / "build"


def test_cmake_builds_another_library(cmake_project) -> None:
    lib, configure, build = cmake_project
    configure(f"HCT_LIBRARY_DIR={lib}", "HCT_LIBRARY_TARGET=dsplib", "HCT_LIBRARY_COMPILE_OPTIONS=-O2 -g",
              "PROBE_DEF=7")
    commands = subprocess.check_output(["ninja", "-C", str(build), "-t", "commands"], text=True).splitlines()
    probe = next(shlex.split(line) for line in commands if " -c " in line and "/probe.c" in line)
    assert "-O2" in probe and "-g" in probe and "-DPROBE_DEF=7" in probe


@pytest.mark.parametrize("defs, message", [
    (("HCT_LIBRARY_TARGET=dsplib",), "HCT_LIBRARY_DIR and HCT_LIBRARY_TARGET are set together"),
    (("HCT_LIBRARY_DIR={lib}",), "HCT_LIBRARY_DIR and HCT_LIBRARY_TARGET are set together"),
    (("HCT_LIBRARY_DIR={lib}", "HCT_LIBRARY_TARGET=other"), "defines no target other"),
    (("HCT_LIBRARY_DIR={lib}", "HCT_LIBRARY_TARGET=cmsis-nn"), "ns-cmsis-nn builds from CMSIS_NN_ROOT"),
    (("HCT_LIBRARY_DIR={lib}/none", "HCT_LIBRARY_TARGET=dsplib"), "HCT_LIBRARY_DIR has no CMakeLists.txt"),
    (("HCT_LIBRARY_DIR={lib}", "HCT_LIBRARY_TARGET=dsplib", "ENABLE_COVERAGE=ON"), "instruments ns-cmsis-nn only"),
    (("HCT_LIBRARY_COMPILE_OPTIONS=-O2",), "HCT_LIBRARY_COMPILE_OPTIONS is for another library"),
])
def test_cmake_refuses(cmake_project, defs, message) -> None:
    lib, configure, _ = cmake_project
    done = configure(*(d.format(lib=lib) for d in defs), check=False)
    output = " ".join((done.stdout + done.stderr).split())
    assert done.returncode != 0
    if "ENABLE_COVERAGE=ON" in defs and "ENABLE_COVERAGE requires GCC" in output:
        pytest.skip("the host compiler is not GCC, which coverage is checked for first")
    assert message in output


def test_cmake_still_requires_ns_cmsis_nn_by_default(cmake_project) -> None:
    _, configure, _ = cmake_project
    done = configure(check=False)
    assert done.returncode != 0 and "cmsis-nn CMakeLists.txt not found" in done.stdout + done.stderr
