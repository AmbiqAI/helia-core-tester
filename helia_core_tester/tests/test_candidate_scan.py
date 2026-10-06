"""gcc -E scan of a candidate."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from test_harness_lock import _git, _repo

from helia_core_tester.hardware import candidate_scan
from helia_core_tester.hardware.candidate_check import check_candidate, rule_counts
from helia_core_tester.hardware.toolchain import arm_tool

pytestmark = [
    pytest.mark.skipif(shutil.which("git") is None, reason="needs git"),
    pytest.mark.skipif(shutil.which(arm_tool("arm-none-eabi-gcc")) is None, reason="needs arm-none-eabi-gcc"),
]
PASTE = "#define CAT(a, b) a##b\n"


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
    # Only -Ofast defines __OPTIMIZE__.
    PASTE + 'int a;\n#ifdef __OPTIMIZE__\nCAT(__attri, bute)( (section(".s")) ) int z;\n#endif\n',
    PASTE + 'static const char s[] = "//";\nCAT(_Pra, gma)("GCC optimize(\\"O3\\")")\n',
])
def test_nested_paste_found(kernels: Path, text: str) -> None:
    (kernels / "Source/Conv/a.c").write_text(text, encoding="utf-8")
    assert {rule for rule, path in _hits(kernels) if path == "Source/Conv/a.c"} & {"pragma", "attribute"}


def test_header_macro_reaches_unit(kernels: Path) -> None:
    # The unit itself is unchanged.
    text = PASTE + '#define KATTR CAT(__attri, bute__)((optimize("O3")))\n'
    (kernels / "Include/k.h").write_text(text, encoding="utf-8")
    assert ("attribute", "Source/Conv/a.c") in _hits(kernels)


def test_old_header_hits_pass(kernels: Path) -> None:
    (kernels / "Source/Conv/new.c").write_text('#include "arm_nn_types.h"\nint n;\n', encoding="utf-8")
    assert _hits(kernels) == set()


def test_failed_unit_fails(kernels: Path) -> None:
    (kernels / "Source/Conv/a.c").write_text("#error no\n", encoding="utf-8")
    assert ("scan_error", "Source/Conv/a.c") in _hits(kernels)


def test_missing_compiler_fails(kernels: Path, monkeypatch) -> None:
    monkeypatch.setattr(candidate_scan, "arm_tool", lambda name: "/nonexistent/gcc")
    found = candidate_scan.preprocess_findings(kernels, bytes(1024), rule_counts)
    assert [f["rule"] for f in found] == ["scan_error"]
