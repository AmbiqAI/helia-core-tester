"""Exercise the real root CMake policy with small host-buildable dependencies."""

import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
from tempfile import TemporaryDirectory

import pytest


ROOT = Path(__file__).resolve().parents[2]


def run(*args, **kwargs):
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT, **kwargs)


@pytest.fixture
def project(tmp_path):
    for tool in ("cmake", "ninja", "gcc"):
        if not shutil.which(tool):
            pytest.skip(f"requires {tool}")
    core = tmp_path / "core"
    core.mkdir()
    (core / "probe.c").write_text("int main(void) { return 0; }\n")
    (core / "CMakeLists.txt").write_text(
        "add_library(cmsis-nn STATIC probe.c)\n"
        # Remove inherited Arm ISA flags only in this host dependency fixture.
        # The production parent still sets coverage flags and launcher policy.
        'set_property(TARGET cmsis-nn PROPERTY COMPILE_OPTIONS "")\n'
    )
    cmsis = tmp_path / "cmsis"
    for name in (
        "CMSIS/Core/Include/cmsis_compiler.h",
        "Device/ARM/ARMCM55/Source/startup_ARMCM55.c",
        "Device/ARM/ARMCM55/Source/system_ARMCM55.c",
    ):
        path = cmsis / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    def configure(
        build, coverage, launcher, *, inherited=False, compiler="gcc", env=None
    ):
        child_env = dict(os.environ if env is None else env)
        child_env.pop("CMAKE_C_COMPILER_LAUNCHER", None)
        extra = []
        if inherited:
            child_env["CMAKE_C_COMPILER_LAUNCHER"] = launcher
        else:
            extra.append(f"-DCMAKE_C_COMPILER_LAUNCHER={launcher}")
        run(
            "cmake",
            "-G",
            "Ninja",
            "-S",
            str(ROOT),
            "-B",
            str(build),
            f"-DCMSIS_NN_ROOT={core}",
            f"-DCMSIS_PATH={cmsis}",
            f"-DCMAKE_C_COMPILER={shutil.which(compiler)}",
            "-DTARGET_CPU=cortex-m55",
            "-DHELIA_BUILD_GENERATED_TESTS=OFF",
            f"-DENABLE_COVERAGE={'ON' if coverage else 'OFF'}",
            *extra,
            env=child_env,
        )
        commands = run(
            "ninja", "-C", str(build), "-t", "commands", env=child_env
        ).splitlines()
        kernel = next(
            shlex.split(line)
            for line in commands
            if " -c " in line and "/probe.c" in line
        )
        harness = next(
            shlex.split(line)
            for line in commands
            if " -c " in line and "/retarget.c" in line
        )
        return kernel, harness

    return configure


@pytest.mark.parametrize("inherited", [False, True])
@pytest.mark.parametrize("initial_coverage", [False, True])
def test_coverage_launcher_reconfigure(project, tmp_path, inherited, initial_coverage):
    launcher = shutil.which("env")
    assert launcher is not None
    build = tmp_path / "build"
    for coverage in (initial_coverage, not initial_coverage, initial_coverage):
        kernel, harness = project(build, coverage, launcher, inherited=inherited)
        assert (launcher in kernel) is not coverage
        assert ("--coverage" in kernel) is coverage
        assert harness[0] == launcher
        assert "--coverage" not in harness


def test_coverage_profiles_and_ordinary_cache_hit(project, tmp_path):
    cache = shutil.which("sccache")
    # GCC12 supports the parent's freestanding coverage flag. This test examines
    # embedded profile paths; it does not execute Arm code or merge gcov streams.
    if not cache or not shutil.which("gcc-12"):
        pytest.skip("requires sccache and gcc-12 for real cache/profile checks")
    # A long pytest basetemp can exceed Unix socket path limits. Only the private
    # endpoint lives in this short, automatically removed directory, not builds.
    with TemporaryDirectory(prefix="hct-cache-", dir="/tmp") as socket_dir:
        env = dict(
            os.environ,
            SCCACHE_DIR=str(tmp_path / "cache"),
            SCCACHE_SERVER_UDS=str(Path(socket_dir) / "s"),
            SCCACHE_IDLE_TIMEOUT="60",
        )
        env.pop("SCCACHE_SERVER_PORT", None)
        run(cache, "--start-server", env=env)
        try:
            for coverage in (True, False):
                run(cache, "--zero-stats", env=env)
                for label in ("a", "b"):
                    build = tmp_path / f"{coverage}-{label}"
                    command, _ = project(
                        build, coverage, cache, compiler="gcc-12", env=env
                    )
                    output = build / command[command.index("-o") + 1]
                    output.parent.mkdir(parents=True, exist_ok=True)
                    run(*command, cwd=build, env=env)
                    data = output.read_bytes()
                    if coverage:
                        assert str(output.with_suffix(".gcda")).encode() in data
                        assert (
                            str(
                                tmp_path / f"True-{'b' if label == 'a' else 'a'}"
                            ).encode()
                            not in data
                        )
                    else:
                        assert b".gcda" not in data
                        executable = build / "probe"
                        run("gcc-12", str(output), "-o", str(executable), env=env)
                        run(str(executable), env=env)
                stats = run(cache, "--show-stats", env=env)
                # A real hit proves the positive control actually exercised caching.
                hits = int(
                    re.search(r"^Cache hits\s+(\d+)$", stats, re.MULTILINE).group(1)
                )
                assert hits == (0 if coverage else 1), stats
        finally:
            run(cache, "--stop-server", env=env)
