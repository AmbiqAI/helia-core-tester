"""The reference library's C unit tests, built with its sources under sanitizers."""

from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path

import pytest

from helia_core_tester.generation.reference.build import COMMON_DIR, INCLUDE_DIR, REFERENCE_ROOT, c_sources
from helia_core_tester.utils.host_compiler import find_host_cc

TEST_SOURCE = REFERENCE_ROOT / "tests" / "test_common.c"
BASE_FLAGS = ["-std=c11", "-O1", "-g", "-fno-fast-math", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror",
              "-fno-sanitize-recover=all"]


def _build_and_run(tmp_path: Path, sanitizers: str) -> subprocess.CompletedProcess:
    exe = tmp_path / f"test_common_{sanitizers.replace(',', '_')}"
    cmd = [find_host_cc(), *BASE_FLAGS, f"-fsanitize={sanitizers}", "-I", str(INCLUDE_DIR), "-I", str(COMMON_DIR),
           *[str(p) for p in c_sources()], str(TEST_SOURCE), "-lm", "-o", str(exe)]
    build = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert build.returncode == 0, build.stderr[-4000:]
    return subprocess.run([str(exe)], capture_output=True, text=True, timeout=300)


@lru_cache(maxsize=None)
def _asan_works(cc: str) -> bool:
    """ASan on some sandboxed macOS hosts hangs before main; probe a trivial program first."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        src, exe = Path(tmp) / "probe.c", Path(tmp) / "probe"
        src.write_text("int main(void){return 0;}\n")
        if subprocess.run([cc, "-fsanitize=address", str(src), "-o", str(exe)], capture_output=True).returncode:
            return False
        try:
            return subprocess.run([str(exe)], capture_output=True, timeout=20).returncode == 0
        except subprocess.TimeoutExpired:
            return False


def test_c_unit_tests_pass_under_ubsan(tmp_path: Path) -> None:
    run = _build_and_run(tmp_path, "undefined")
    assert run.returncode == 0, run.stdout[-4000:] + run.stderr[-4000:]
    assert run.stdout.strip().endswith("ok")


def test_c_unit_tests_pass_under_asan(tmp_path: Path) -> None:
    if not _asan_works(find_host_cc()):
        pytest.skip("AddressSanitizer does not run on this host (a trivial ASan program hangs or fails)")
    run = _build_and_run(tmp_path, "address,undefined")
    assert run.returncode == 0, run.stdout[-4000:] + run.stderr[-4000:]
    assert run.stdout.strip().endswith("ok")
