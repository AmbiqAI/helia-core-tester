"""gcc -E scan of a candidate."""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import tarfile
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from test_harness_lock import _git, _repo

from helia_core_tester.hardware import candidate_scan
from helia_core_tester.hardware.candidate_check import check_candidate, rule_counts
from helia_core_tester.hardware.toolchain import arm_tool

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
needs_gcc = pytest.mark.skipif(shutil.which(arm_tool("arm-none-eabi-gcc")) is None, reason="needs arm-none-eabi-gcc")
PASTE = "#define CAT(a, b) a##b\n"
CMSIS_NN_ROOT = os.environ.get("CMSIS_NN_ROOT", "")


@pytest.fixture
def kernels(tmp_path: Path) -> Path:
    return _repo(tmp_path / "nn", {
        "Source/Conv/a.c": '#include "k.h"\nKATTR int a(void) { return 0; }\n',
        "Source/Conv/b.c": '#include "arm_nn_types.h"\nint b;\n',
        "Include/k.h": "#define KATTR\n",
        "Include/arm_nn_types.h": '__attribute__((section(".x"))) int old;\n',
    })


def _hits(root: Path) -> set[tuple[str, str]]:
    report = check_candidate(root, _git(root, "rev-parse", "HEAD").strip())
    return {(f["rule"], f["path"]) for f in report["findings"]}


@pytest.mark.parametrize("text", [
    PASTE + 'CAT(_Pra, gma)("GCC optimize(\\"O3\\")")\nint a;\n',
    '%:define DCAT(a, b) a %:%: b\nDCAT(__attri, bute__)((optimize("O3"))) int a;\n',
    # Only -Ofast defines __OPTIMIZE__.
    PASTE + 'int a;\n#ifdef __OPTIMIZE__\nCAT(__attri, bute)( (section(".s")) ) int z;\n#endif\n',
    PASTE + 'static const char s[] = "//";\nCAT(_Pra, gma)("GCC optimize(\\"O3\\")")\n',
])
@needs_gcc
def test_nested_paste_found(kernels: Path, text: str) -> None:
    (kernels / "Source/Conv/a.c").write_text(text, encoding="utf-8")
    assert {rule for rule, path in _hits(kernels) if path == "Source/Conv/a.c"} & {"pragma", "attribute"}


@needs_gcc
def test_header_macro_reaches_unit(kernels: Path) -> None:
    # The unit itself is unchanged.
    text = PASTE + '#define KATTR CAT(__attri, bute__)((optimize("O3")))\n'
    (kernels / "Include/k.h").write_text(text, encoding="utf-8")
    assert ("attribute", "Source/Conv/a.c") in _hits(kernels)


@needs_gcc
def test_old_header_hits_pass(kernels: Path) -> None:
    (kernels / "Source/Conv/new.c").write_text('#include "arm_nn_types.h"\nint n;\n', encoding="utf-8")
    assert _hits(kernels) == set()


@needs_gcc
def test_failed_unit_fails(kernels: Path) -> None:
    (kernels / "Source/Conv/a.c").write_text("#error no\n", encoding="utf-8")
    assert ("scan_error", "Source/Conv/a.c") in _hits(kernels)


def test_missing_compiler_fails(kernels: Path, monkeypatch) -> None:
    monkeypatch.setattr(candidate_scan, "arm_tool", lambda name: "/nonexistent/gcc")
    found = candidate_scan.preprocess_findings(kernels, bytes(1024), rule_counts)
    assert [f["rule"] for f in found] == ["scan_error"]


def _fake_gcc(tmp_path: Path, body: str) -> str:
    script = tmp_path / "gcc"
    script.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    script.chmod(0o755)
    return str(script)


def test_slow_gcc_killed(tmp_path: Path) -> None:
    pid_file = tmp_path / "pid"
    gcc = _fake_gcc(tmp_path, f"sleep 60 &\necho $! > {pid_file}\nwait")
    start = time.monotonic()
    assert candidate_scan.run_capped([gcc], tmp_path, 1.0, 1 << 20) is None
    assert time.monotonic() - start < 10
    # The grandchild dies with the group.
    pid = int(pid_file.read_text())
    time.sleep(0.2)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_big_output_capped(tmp_path: Path) -> None:
    gcc = _fake_gcc(tmp_path, "yes")
    assert candidate_scan.run_capped([gcc], tmp_path, 30.0, 1 << 16) is None


def test_small_output_kept(tmp_path: Path) -> None:
    gcc = _fake_gcc(tmp_path, "echo ok")
    assert candidate_scan.run_capped([gcc], tmp_path, 30.0, 1 << 16) == b"ok\n"


def _tar(*members: tarfile.TarInfo) -> bytes:
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w") as tar:
        for member in members:
            tar.addfile(member, io.BytesIO(b"x" * member.size) if member.isfile() else None)
    return out.getvalue()


def _member(name: str, kind: bytes = tarfile.REGTYPE, link: str = "") -> tarfile.TarInfo:
    member = tarfile.TarInfo(name)
    member.type, member.linkname, member.size = kind, link, 1 if kind == tarfile.REGTYPE else 0
    return member


@pytest.mark.parametrize("member", [
    _member("../evil.c"),
    _member("/tmp/evil.c"),
    _member("Source/link", tarfile.SYMTYPE, "/etc/passwd"),
    _member("Source/hard", tarfile.LNKTYPE, "Source/a.c"),
    _member("Source/dev", tarfile.CHRTYPE),
])
def test_unsafe_tar_rejected(tmp_path: Path, member: tarfile.TarInfo) -> None:
    with pytest.raises(tarfile.TarError):
        candidate_scan.extract_tar(_tar(_member("Source/a.c"), member), tmp_path / "out")
    assert not (tmp_path / "evil.c").exists()


def test_unsafe_tar_fails_closed(kernels: Path, monkeypatch) -> None:
    # No gcc runs before the tar check.
    monkeypatch.setattr(candidate_scan, "arm_tool", lambda name: "/nonexistent/gcc")
    found = candidate_scan.preprocess_findings(kernels, _tar(_member("../evil.c")), rule_counts)
    assert [f["rule"] for f in found] == ["scan_error"]


def test_safe_tar_extracted(tmp_path: Path) -> None:
    candidate_scan.extract_tar(_tar(_member("Source/Conv/a.c")), tmp_path)
    assert (tmp_path / "Source/Conv/a.c").read_bytes() == b"x"


@needs_gcc
@pytest.mark.skipif(not CMSIS_NN_ROOT or not Path(CMSIS_NN_ROOT, ".git").exists(), reason="needs CMSIS_NN_ROOT")
def test_real_tree_clean(tmp_path: Path) -> None:
    root = tmp_path / "nn"
    subprocess.run(["git", "clone", "-q", CMSIS_NN_ROOT, str(root)], check=True)
    # A comment-only edit runs every scan.
    unit = root / "Source/ActivationFunctions/arm_relu_q7.c"
    unit.write_text(unit.read_text(encoding="utf-8") + "/* note */\n", encoding="utf-8")
    report = check_candidate(root, _git(root, "rev-parse", "HEAD").strip())
    assert report["ok"], report["findings"][:5]


def _many_units(tmp_path: Path, count: int) -> Path:
    root = tmp_path / "many"
    for index in range(count):
        (root / "Source" / f"u{index}.c").parent.mkdir(parents=True, exist_ok=True)
        (root / "Source" / f"u{index}.c").write_text("int u;\n", encoding="utf-8")
    return root


def test_scan_deadline_stops_all(tmp_path: Path, monkeypatch) -> None:
    gcc = _fake_gcc(tmp_path, "sleep 30")
    monkeypatch.setattr(candidate_scan, "arm_tool", lambda name: gcc)
    start = time.monotonic()
    found = candidate_scan.preprocess_findings(_many_units(tmp_path, 40), bytes(1024), rule_counts, deadline_s=1.0)
    assert time.monotonic() - start < 8
    assert found == [{"rule": "scan_error", "path": "", "message": "scan passed its 1 s deadline"}]


def test_too_many_units_refused(tmp_path: Path, monkeypatch) -> None:
    ran = tmp_path / "ran"
    gcc = _fake_gcc(tmp_path, f"touch {ran}")
    monkeypatch.setattr(candidate_scan, "arm_tool", lambda name: gcc)
    monkeypatch.setattr(candidate_scan, "MAX_UNITS", 2)
    found = candidate_scan.preprocess_findings(_many_units(tmp_path, 3), bytes(1024), rule_counts)
    assert [f["message"] for f in found] == ["too many units: 3 > 2"] and not ran.exists()


@needs_gcc
def test_first_failure_stops_scan(kernels: Path) -> None:
    for index in range(20):
        (kernels / f"Source/Conv/bad{index}.c").write_text("#error no\n", encoding="utf-8")
    found = candidate_scan.preprocess_findings(kernels, bytes(1024), rule_counts)
    assert len(found) == 1 and found[0]["rule"] == "scan_error"


def test_worker_returns_counts(tmp_path: Path, monkeypatch) -> None:
    # 4 MiB of kernel text, one hit.
    gcc = _fake_gcc(tmp_path, 'echo \'# 1 "Source/u.c"\'; echo "_Pragma(1)"; head -c 4194304 /dev/zero | tr "\\0" x; echo')
    found = candidate_scan._unit_counts(gcc, tmp_path, "Source/u.c", (), time.monotonic() + 30, "", rule_counts)
    assert set(found) == {"Source/u.c"}
    assert isinstance(found["Source/u.c"], Counter) and found["Source/u.c"]["pragma"] == 1


def test_jobs_in_flight_bounded(tmp_path: Path, monkeypatch) -> None:
    peak, live, lock = [0], [0], threading.Lock()

    class Pool(ThreadPoolExecutor):
        def submit(self, fn, *args):
            def run():
                try:
                    return fn(*args)
                finally:
                    with lock:
                        live[0] -= 1
            with lock:
                live[0] += 1
                peak[0] = max(peak[0], live[0])
            return super().submit(run)

    monkeypatch.setattr(candidate_scan, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(candidate_scan, "scan_workers", lambda: 2)
    gcc = _fake_gcc(tmp_path, "sleep 0.05")
    counts, failed = candidate_scan._tree_counts(gcc, _many_units(tmp_path, 12), rule_counts, {"c": ()},
                                                 time.monotonic() + 30, str(tmp_path / "abort"))
    assert counts == {} and not failed and peak[0] <= 2
