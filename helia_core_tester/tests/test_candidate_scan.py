"""gcc -E and object scans of a candidate."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from test_harness_lock import _git, _repo

from helia_core_tester.hardware import candidate_check, candidate_scan
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


def _build(tmp_path: Path, source: str, define: str = "-DBOARD_X") -> Path:
    """A build dir with one kernel object."""
    build = tmp_path / "build"
    src = build / "nsx_app/modules/nsx-cmsis-nn/Source/k.c"
    src.parent.mkdir(parents=True)
    src.write_text(source, encoding="utf-8")
    gcc = arm_tool("arm-none-eabi-gcc")
    subprocess.run([gcc, "-mcpu=cortex-m55", "-O2", "-c", str(src), "-o", str(build / "k.c.obj")], check=True)
    include = f"-I{build}/nsx_app/modules/nsx-cmsis-nn/Include"
    args = ["gcc", define, include, "-MD", "-MF", "k.d", "-o", "k.c.obj", "-c", str(src)]
    entry = {"directory": str(build), "file": str(src), "output": "k.c.obj", "arguments": args}
    (build / "compile_commands.json").write_text(json.dumps([entry]), encoding="utf-8")
    return build


def test_object_scan_flags_sections_and_scs(tmp_path: Path) -> None:
    source = (
        '__attribute__((section(".itcm_text"))) unsigned f(void)\n'
        "{ return *(volatile unsigned *)(0x70000000u * 2u + 0x1000u); }\n"
        '__attribute__((section(".data.fast"))) int g(int x) { return x + 1; }\n'
        "volatile unsigned *const p = (volatile unsigned *)(0x70000000u * 2u + 0x3000u);\n"
    )
    findings, summary, flags = candidate_scan.object_findings(_build(tmp_path, source))
    texts = {f["text"].split()[0] for f in findings if f["rule"] == "object_section"}
    assert texts == {".itcm_text", ".data.fast"} and summary["count"] == 1
    assert len([f for f in findings if f["rule"] == "object_address"]) >= 2
    assert flags["*"] == flags["Source/k.c"] == ("-DBOARD_X", "-IInclude")


def test_object_scan_clean(tmp_path: Path) -> None:
    source = "unsigned f(unsigned x) { return x > 0xE0000000u ? x : ~x; }\nconst int t[2] = {1, 2};\n"
    findings, summary, _ = candidate_scan.object_findings(_build(tmp_path, source))
    assert findings == [] and summary["count"] == 1


@pytest.mark.parametrize("source", [
    # Unaligned bytes: e0 10 e0 00.
    "const unsigned short h[3] = {0, 0xE010, 0xE000};\n",
    "extern char __Vectors[];\nchar *const v = __Vectors + 0xDFBFE010u;\n",
    "extern int harness_var;\nint *const w = &harness_var;\n",
    # Weak loses to a harness definition.
    "__attribute__((weak)) int harness_w;\nint *const w = &harness_w;\n",
])
def test_object_scan_hidden_addresses(tmp_path: Path, source: str) -> None:
    findings, _, _ = candidate_scan.object_findings(_build(tmp_path, source))
    assert [f["rule"] for f in findings] and {f["rule"] for f in findings} == {"object_address"}


def test_object_scan_needs_objects(tmp_path: Path) -> None:
    (tmp_path / "compile_commands.json").write_text("[]", encoding="utf-8")
    findings, _, _ = candidate_scan.object_findings(tmp_path)
    assert [f["rule"] for f in findings] == ["scan_error"]


def test_build_macros_reach_gcc_e(kernels: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(candidate_check, "kernels_match", lambda tree, module: True)
    text = PASTE + '#ifdef BOARD_X\nCAT(_Pra, gma)("GCC optimize(\\"O3\\")")\n#endif\nint a;\n'
    (kernels / "Source/Conv/a.c").write_text(text, encoding="utf-8")
    assert _hits(kernels) == set()
    report = check_candidate(kernels, _git(kernels, "rev-parse", "HEAD").strip(), build_dir=_build(tmp_path, "int k;\n"))
    assert {(f["rule"], f["path"]) for f in report["findings"]} == {("pragma", "Source/Conv/a.c")}
    assert report["objects"]["count"] == 1


def test_local_name_hides_no_global(tmp_path: Path) -> None:
    build = _build(tmp_path, "static int x;\nint *const p = &x;\n")
    other = build / "nsx_app/modules/nsx-cmsis-nn/Source/o.c"
    other.write_text("extern int x;\nint *const q = &x;\n", encoding="utf-8")
    gcc = arm_tool("arm-none-eabi-gcc")
    subprocess.run([gcc, "-mcpu=cortex-m55", "-O2", "-c", str(other), "-o", str(build / "o.c.obj")], check=True)
    entries = json.loads((build / "compile_commands.json").read_text(encoding="utf-8"))
    entries.append({**entries[0], "file": str(other), "output": "o.c.obj", "arguments": ["gcc", "-c", str(other)]})
    (build / "compile_commands.json").write_text(json.dumps(entries), encoding="utf-8")
    findings, _, flags = candidate_scan.object_findings(build)
    assert {f["path"] for f in findings} == {"Source/o.c"}
    assert flags["Source/o.c"] == () and flags["Source/k.c"][0] == "-DBOARD_X"


def test_per_unit_flags_reach_gcc_e(kernels: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(candidate_check, "kernels_match", lambda tree, module: True)
    text = PASTE + '#ifdef BOARD_B\nCAT(_Pra, gma)("GCC optimize(\\"O3\\")")\n#endif\nint b;\n'
    (kernels / "Source/Conv/b.c").write_text(text, encoding="utf-8")
    flags = {"*": ("-DBOARD_X",), "Source/Conv/a.c": ("-DBOARD_X",), "Source/Conv/b.c": ("-DBOARD_B",)}
    monkeypatch.setattr(candidate_check, "object_findings", lambda build: ([], {"count": 2}, flags))
    report = check_candidate(kernels, _git(kernels, "rev-parse", "HEAD").strip(), build_dir=tmp_path)
    assert ("pragma", "Source/Conv/b.c") in {(f["rule"], f["path"]) for f in report["findings"]}


def test_stale_build_fails(kernels: Path, tmp_path: Path) -> None:
    report = check_candidate(kernels, _git(kernels, "rev-parse", "HEAD").strip(), build_dir=_build(tmp_path, "int k;\n"))
    assert {f.get("message") for f in report["findings"]} == {"build dir holds other kernels"}
