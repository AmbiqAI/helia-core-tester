"""gcc -E and object scans of a candidate."""

from __future__ import annotations

import io
import json
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

from helia_core_tester.hardware import candidate_check, candidate_scan
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


def _build(tmp_path: Path, source: str, define: str = "-DBOARD_X", name: str = "k.c") -> Path:
    """A build dir with one kernel object."""
    return _build_units(tmp_path, {name: source}, define)


def _build_units(tmp_path: Path, sources: dict[str, str], define: str = "-DBOARD_X") -> Path:
    """A build dir with one kernel object per source."""
    build = tmp_path / "build"
    gcc = arm_tool("arm-none-eabi-gcc")
    include = f"-I{build}/nsx_app/modules/nsx-cmsis-nn/Include"
    entries = []
    for name, source in sources.items():
        src = build / "nsx_app/modules/nsx-cmsis-nn/Source" / name
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_text(source, encoding="utf-8")
        out = f"{Path(name).stem}.c.obj"
        subprocess.run([gcc, "-mcpu=cortex-m55", "-O2", "-c", str(src), "-o", str(build / out)], check=True)
        args = ["gcc", define, include, "-MD", "-MF", f"{Path(name).stem}.d", "-o", out, "-c", str(src)]
        entries.append({"directory": str(build), "file": str(src), "output": out, "arguments": args})
    (build / "compile_commands.json").write_text(json.dumps(entries), encoding="utf-8")
    return build


@needs_gcc
def test_object_scan_flags_sections_and_scs(tmp_path: Path) -> None:
    source = (
        '__attribute__((section(".itcm_text"))) unsigned f(void)\n'
        "{ return *(volatile unsigned *)(0x70000000u * 2u + 0x1000u); }\n"
        '__attribute__((section(".data.fast"))) int g(int x) { return x + 1; }\n'
        "volatile unsigned *const p = (volatile unsigned *)(0x70000000u * 2u + 0x3000u);\n"
    )
    findings, summary, flags = candidate_scan.object_findings(_build(tmp_path, source))
    texts = {f["text"].split()[0] for f in findings if f["rule"] == "object_section"}
    assert texts == {".itcm_text", ".data.fast", "code"} and summary["count"] == 1
    assert len([f for f in findings if f["rule"] == "object_address"]) >= 2
    assert flags["*"] == flags["Source/k.c"] == ("-DBOARD_X", "-IInclude")


RAM_ROUTINE = """.syntax unified
.thumb
.data
.global ramfn
.type ramfn, %function
.thumb_func
ramfn:
  bx lr
lab:
  bx lr
.text
.global caller
.type caller, %function
.thumb_func
caller:
  bl lab
  b ramfn
"""


@needs_gcc
def test_object_scan_flags_code_in_data(tmp_path: Path) -> None:
    findings, _, _ = candidate_scan.object_findings(_build(tmp_path, RAM_ROUTINE, name="k.S"))
    texts = {f["text"] for f in findings if f["rule"] == "object_section"}
    assert {"code ramfn in .data", "branch to .data in .data"} <= texts


@needs_gcc
@pytest.mark.parametrize("name", ["data hidden", "data\\thidden"])
def test_odd_section_name_flagged(tmp_path: Path, name: str) -> None:
    source = f'.section ".{name}","aw"\n.word 0xE0001004\n'
    findings, _, _ = candidate_scan.object_findings(_build(tmp_path, source, name="k.S"))
    rules = {f["rule"] for f in findings}
    assert any(f["rule"] == "object_section" and "hidden" in f["text"] for f in findings)
    assert "object_address" in rules


def _asm(*body: str) -> str:
    return ".syntax unified\n.thumb\n.text\n.global f\n.type f, %function\n.thumb_func\nf:\n" + "".join(
        f"  {line}\n" for line in body) + "  bx lr\n"


@needs_gcc
@pytest.mark.parametrize("insn", [
    ("cpsid i",), ("CPSIE f",), ("wfi",), ("wfe",), ("wfi.w",), ("it eq", "wfieq"),
    ("msr PRIMASK, r0",), ("msr basepri, r0",), ("msr BASEPRI_MAX, r0",), ("msr FAULTMASK, r0",),
    ("msr CONTROL, r0",), ("msr PRIMASK_NS, r0",), ("msr MSP, r0",), ("msr PSPLIM, r0",),
])
def test_object_scan_flags_forbidden_instructions(tmp_path: Path, insn: tuple[str, ...]) -> None:
    findings, _, _ = candidate_scan.object_findings(_build(tmp_path, _asm(*insn), name="k.S"))
    assert [f["rule"] for f in findings] == ["object_instruction"]
    assert findings[0]["path"] == "Source/k.S"
    assert insn[-1].split()[0].split(".")[0].lower() in findings[0]["text"].lower()


@needs_gcc
def test_object_scan_allows_lookalike_instructions(tmp_path: Path) -> None:
    source = _asm("msr APSR_nzcvq, r0", "msr APSR_g, r0", "mrs r0, PRIMASK", "mrs r0, CONTROL", "vmsr fpscr, r0", "sev")
    findings, summary, _ = candidate_scan.object_findings(_build(tmp_path, source, name="k.S"))
    assert findings == [] and summary["count"] == 1


@needs_gcc
def test_object_scan_sees_through_source_spelling(tmp_path: Path) -> None:
    # Pasted and stringized: no source rule sees "cpsid" or "wfi".
    source = (
        "#define CAT(a, b) a##b\n#define STR(x) #x\n#define XSTR(x) STR(x)\n"
        "void f(void) { __asm__ volatile (XSTR(CAT(cps, id)) \" i\\n\" XSTR(CAT(w, fi))); }\n"
    )
    findings, _, _ = candidate_scan.object_findings(_build(tmp_path, source))
    assert sorted(f["text"].split()[1] for f in findings if f["rule"] == "object_instruction") == ["cpsid", "wfi"]


@needs_gcc
@pytest.mark.parametrize(("source", "name"), [
    # cpsid i, wfe, cpsie i as data words that fall through from code.
    (_asm(".short 0xb672", ".short 0xbf20", ".short 0xb662"), "k.S"),
    # A const array called through a pointer.
    ("static const unsigned short blob[] = {0xb672, 0xbf30, 0xb662, 0x4770};\n"
     "void f(void) { ((void (*)(void))((unsigned)blob | 1u))(); }\n", "k.c"),
])
def test_object_scan_decodes_data_as_code(tmp_path: Path, source: str, name: str) -> None:
    findings, _, _ = candidate_scan.object_findings(_build(tmp_path, source, name=name))
    found = {f["text"].split()[1] for f in findings if f["rule"] == "object_instruction"}
    assert {"cpsid", "cpsie"} <= found and found & {"wfe", "wfi"}


# wfi as the second half of a 32-bit ldr.w, entered at `after - 2`.
HIDDEN = """.syntax unified
.thumb
.section .text.hide,"ax",%progbits
.global f
.type f, %function
.thumb_func
f:
  {local}
  .inst.w 0xf8dfbf30
after:
  bx lr
.section .text.call,"ax",%progbits
.global g
.type g, %function
.thumb_func
g:
  {remote}
  bx lr
.section .rodata.p,"a"
  .word 0
  {data}
"""


@needs_gcc
@pytest.mark.parametrize(("local", "remote", "data", "origin"), [
    (".thumb_set hidden, after - 2\n.global hidden", "", "", "instruction at .text.hide+0x0 is cut"),
    ("b.n after - 2", "", "", "branch at .text.hide+0x0"),
    ("", "bl after - 2", "", "R_ARM_THM_CALL"),
    ("", "b.w after - 2", "", "R_ARM_THM_JUMP24"),
    ("", "beq.w after - 2", "", "R_ARM_THM_JUMP19"),
    ("", "movw r0, #:lower16:(after - 1)", "", "R_ARM_THM_MOVW_ABS_NC"),
    ("", "movt r0, #:upper16:(after - 1)", "", "R_ARM_THM_MOVT_ABS"),
    ("", "", ".word after - 1", "R_ARM_ABS32"),
    # f carries the Thumb tag, so f + 3 enters at offset 2.
    ("", "", ".word f + 3", "R_ARM_ABS32"),
    ("", "movw r0, #:lower16:(f + 3)", "", "R_ARM_THM_MOVW_ABS_NC"),
])
def test_object_scan_flags_entry_inside_instruction(
    tmp_path: Path, local: str, remote: str, data: str, origin: str,
) -> None:
    source = HIDDEN.format(local=local, remote=remote, data=data)
    findings, _, _ = candidate_scan.object_findings(_build(tmp_path, source, name="k.S"))
    texts = [f["text"] for f in findings if f["rule"] == "object_instruction" and not f["text"].endswith("wfi")]
    assert len(texts) == 1 and texts[0].startswith(origin)
    assert ".text.hide+0x" in texts[0]


@needs_gcc
def test_object_scan_entry_unread_relocation_fails_closed(tmp_path: Path) -> None:
    source = HIDDEN.format(local="", remote="", data=".hword after")
    findings, _, _ = candidate_scan.object_findings(_build(tmp_path, source, name="k.S"))
    texts = [f["text"] for f in findings if f["rule"] == "object_instruction"]
    assert texts == ["R_ARM_ABS16 at .rodata.p+0x4 into .text.hide is unchecked"]


@needs_gcc
def test_object_scan_entry_at_boundaries_is_clean(tmp_path: Path) -> None:
    source = HIDDEN.format(local="b.n after", remote="bl after\n  b.w f\n  movw r0, #:lower16:after",
                           data=".word after + 1\n.word f\n.word f + 1")
    findings, summary, _ = candidate_scan.object_findings(_build(tmp_path, source, name="k.S"))
    assert findings == [] and summary["count"] == 1


def _raw(*body: str, tail: str = "") -> str:
    """Function f with exactly `body`, then `tail` outside any function."""
    return (".syntax unified\n.thumb\n.bss\nanchor:\n  .space 4\n.text\n.global f\n.type f, %function\n"
            ".thumb_func\nf:\n" + "".join(f"  {line}\n" for line in body) + tail)


def _instruction_texts(tmp_path: Path, source: str) -> list[str]:
    findings, _, _ = candidate_scan.object_findings(_build(tmp_path, source, name="k.S"))
    return [f["text"] for f in findings if f["rule"] == "object_instruction"]


@needs_gcc
@pytest.mark.parametrize(("body", "expected"), [
    # -dz lists data rows as starts, and -D reads both halfwords as one ldr.w.
    (["b.n .Lhidden", ".short 0xf8df", ".Lhidden:", ".short 0xbf30", "bx lr"],
     ["branch at .text+0x0 enters .text+0x4 in data"]),
    (["beq.n .Lhidden", "bx lr", ".short 0xf8df", ".Lhidden:", ".short 0xbf30"],
     ["branch at .text+0x0 enters .text+0x6 in data"]),
    (["cbz r0, .Lhidden", "bx lr", ".Lhidden:", ".short 0xbf30"],
     ["branch at .text+0x0 enters .text+0x4 in data"]),
    # A relocation that leaves the branch alone does not excuse it.
    (["b.n .Lafter - 2", ".reloc .-2, R_ARM_NONE, anchor", ".inst.w 0xf8dfbf30", ".Lafter:", "bx lr"],
     ["branch at .text+0x0 enters .text+0x4 inside an instruction"]),
    (["b.n .Lafter - 2", ".reloc .-2, R_ARM_ABS16, anchor", ".inst.w 0xf8dfbf30", ".Lafter:", "bx lr"],
     ["R_ARM_ABS16 at .text+0x0 rewrites code it does not fit",
      "branch at .text+0x0 enters .text+0x4 inside an instruction"]),
])
def test_object_scan_branches_follow_thumb_decoding(tmp_path: Path, body: list[str], expected: list[str]) -> None:
    assert [t for t in _instruction_texts(tmp_path, _raw(*body)) if not t.endswith("wfi")] == expected


@needs_gcc
@pytest.mark.parametrize(("body", "tail"), [
    # Literal pools stay data: loads, and symbols and words naming them.
    (["ldr r0, =0x12345678", "ldr r1, .Lpool", "ldr r2, pool", "bx lr", ".align 2", ".Lpool:", ".word 1",
      ".global pool", "pool:", ".word 2"], ".data\n.word pool\n.word .Lpool + 2\n"),
    # nop padding after a return, then a pool.
    (["ldr r0, =0x12345678", "bx lr", "nop", "nop.w", ".align 3", ".word 3"], ""),
    (["cmp r0, #0", "ite eq", "moveq r0, #1", "movne r0, #2", "pop {r4, pc}", ".word 0"], ""),
    (["b.w g", ".word 0"], ".thumb_func\ng:\n  udf #0\n"),
    (["tbb [pc, r0]", ".byte 0, 1", "bx lr"], ""),
    # The section end, reached without falling through.
    (["movs r0, #0", "bx lr"], ""),
])
def test_object_scan_code_and_pools_are_clean(tmp_path: Path, body: list[str], tail: str) -> None:
    assert _instruction_texts(tmp_path, _raw(*body, tail=tail)) == []


@needs_gcc
@pytest.mark.parametrize(("body", "tail", "expected"), [
    (["movs r0, #0", ".short 0x4770"], "", ["code at .text+0x0 can run into data at .text+0x2"]),
    # Inside an it block, a return may not run.
    (["cmp r0, #0", "it eq", "bxeq lr", ".word 0"], "", ["code at .text+0x0 can run into data at .text+0x6"]),
    (["movs r0, #0"], "", ["code at .text+0x0 can run into the section end"]),
    # A cut instruction at the section end.
    ([".short 0xf8df"], "", ["instruction at .text+0x0 is cut by a symbol or the section end"]),
    # A literal pool may be named, but not entered through bx or blx.
    (["bx lr", ".align 2", "pool:", ".word 0"], ".data\n.word pool + 1\n",
     ["R_ARM_ABS32 at .data+0x0 enters .text+0x5 in data"]),
    # In code only an instruction field may be relocated.
    ([".reloc ., R_ARM_ABS32, anchor", "nop.w", "bx lr"], "",
     ["R_ARM_ABS32 at .text+0x0 rewrites code it does not fit"]),
    (["bl g", ".reloc .-2, R_ARM_THM_CALL, g", "bx lr"], ".thumb_func\ng:\n  bx lr\n",
     ["R_ARM_THM_CALL at .text+0x2 rewrites code it does not fit",
      "R_ARM_THM_CALL at .text+0x2 enters .text+0xc01eea inside an instruction"]),
    ([".reloc ., R_ARM_THM_JUMP11, g", "nop.w", "bx lr"], ".thumb_func\ng:\n  bx lr\n",
     ["R_ARM_THM_JUMP11 at .text+0x0 rewrites code it does not fit",
      "R_ARM_THM_JUMP11 at .text+0x0 into .text is unchecked"]),
])
def test_object_scan_flags_code_layout(tmp_path: Path, body: list[str], tail: str, expected: list[str]) -> None:
    assert _instruction_texts(tmp_path, _raw(*body, tail=tail)) == expected


@needs_gcc
@pytest.mark.parametrize("body", [
    # Data-marked bytes a function symbol starts run as code.
    [".short 0xbf30", "bx lr"],
    [".short 0xf3af, 0x8003", "bx lr"],
    ["wfi", "bx lr"],
])
def test_object_scan_reports_each_sleep_once(tmp_path: Path, body: list[str]) -> None:
    assert [t.split()[-1].split(".")[0] for t in _instruction_texts(tmp_path, _raw(*body))] == ["wfi"]


def _layout(data: bytes, marks: dict[int, str] | None = None, labels: set[int] | None = None):
    return candidate_scan._thumb_layout(".text", memoryview(data), marks or {}, labels or set())


def test_thumb_layout_boundaries() -> None:
    assert _layout(b"") == ({}, (), [])
    # 0xf8df opens a 32-bit instruction; 0x4770 is bx lr.
    assert _layout(b"\xdf\xf8\x00\x00\x70\x47") == ({0: 4, 4: 2}, (), [])
    assert _layout(b"\x70\x47\xdf\xf8")[2] == ["instruction at .text+0x2 is cut by a symbol or the section end"]
    assert _layout(b"\x70\x47\xdf")[2] == ["instruction at .text+0x2 is cut by a symbol or the section end"]
    assert _layout(b"\xdf\xf8\x00\x00\x70\x47", labels={2})[2] == [
        "instruction at .text+0x0 is cut by a symbol or the section end"]
    # Data cutting an instruction, and code winning a mark clash.
    assert _layout(b"\xdf\xf8\x00\x00", marks={2: "d"})[2] == [
        "instruction at .text+0x0 is cut by a symbol or the section end"]
    assert _layout(b"\x70\x47\x00\x00", marks={0: "t", 2: "d"}) == ({0: 2}, ((2, 4),), [])
    assert _layout(b"\x00\x00\x70\x47", marks={0: "d", 2: "t"}) == ({2: 2}, ((0, 2),), [])


def test_landing_boundaries() -> None:
    layout = ({0: 4, 4: 2}, ((6, 10),), [])
    assert candidate_scan._landing(layout, 10, 0, True) is None
    assert candidate_scan._landing(layout, 10, 1, True) is None
    assert candidate_scan._landing(layout, 10, 5, True) is None
    assert candidate_scan._landing(layout, 10, 10, True) is None
    assert candidate_scan._landing(layout, 10, 2, False) == "inside an instruction"
    assert candidate_scan._landing(layout, 10, 3, True) == "inside an instruction"
    assert candidate_scan._landing(layout, 10, 6, False) is None
    assert candidate_scan._landing(layout, 10, 6, True) == "in data"
    assert candidate_scan._landing(layout, 10, 7, False) == "in data"
    assert candidate_scan._landing(layout, 10, 9, False) == "in data"


@needs_gcc
@pytest.mark.parametrize(("body", "mnemonic"), [
    # A 32-bit prefix swallows the next halfword in a linear decode.
    ('.section .rodata.x,"a"\n.short 0xf8df, 0xb672\n', "cpsid"),
    (_asm(".inst.w 0xf8dfb662"), "cpsie"),
    (_asm(".short 0xf8df, 0xf380, 0x8810"), "msr"),
    (_asm(".short 0xf8df, 0xf3af, 0x8610"), "cps"),
    # A byte-aligned section can land at an odd address.
    ('.section .rodata.x,"a"\n.byte 0, 0x72, 0xb6\n', "cpsid"),
])
def test_object_scan_finds_masks_at_any_halfword(tmp_path: Path, body: str, mnemonic: str) -> None:
    findings, _, _ = candidate_scan.object_findings(_build(tmp_path, ".syntax unified\n.thumb\n" + body, name="k.S"))
    assert [f["text"].split()[1] for f in findings if f["rule"] == "object_instruction"] == [mnemonic]


@needs_gcc
@pytest.mark.parametrize("body", [
    # vldrw.u32 q5, [sp, #128] is ed9d bf20: wfe inside an MVE load.
    _asm("vldrw.u32 q5, [sp, #128]"),
    # Odd offsets of a halfword-aligned section never start an instruction.
    '.section .rodata.x,"a"\n.p2align 1\n.byte 0, 0x72, 0xb6, 0\n',
    # msr to APSR (SYSm 0) inside another instruction.
    _asm(".short 0xf8df, 0xf380, 0x8800"),
])
def test_object_scan_ignores_masks_that_never_execute(tmp_path: Path, body: str) -> None:
    findings, summary, _ = candidate_scan.object_findings(_build(tmp_path, ".syntax unified\n.thumb\n" + body, name="k.S"))
    assert findings == [] and summary["count"] == 1


@needs_gcc
def test_object_scan_clean_calls_and_tables(tmp_path: Path) -> None:
    source = (
        "static int g(int x) { return x * 3; }\nint (*const tab[2])(int) = {g, 0};\nextern void ext(void);\n"
        "__attribute__((noinline)) static int h(int x) { return x + tab[0](x); }\n"
        "int f(int x) { ext(); switch (x) { case 0: return h(1); case 1: return 7; case 2: return 9;\n"
        "case 3: return 11; case 4: return 13; } return h(x); }\n"
    )
    findings, summary, _ = candidate_scan.object_findings(_build(tmp_path, source))
    assert findings == [] and summary["count"] == 1


# wfi as the second half of a 32-bit ldr.w in one object; `{call}` in another.
VICTIM = """.syntax unified
.thumb
.text
.global victim
.type victim, %function
.thumb_func
victim:
  .inst.w 0xf8dfbf30
  bx lr
.section .rodata.tab,"a"
.global tab
tab:
  .word 1, 2, 3
"""
CALLER = """.syntax unified
.thumb
.text
.global caller
.type caller, %function
.thumb_func
caller:
  {call}
  bx lr
.section .rodata.p,"a"
  {data}
"""


@needs_gcc
@pytest.mark.parametrize(("call", "data", "expected"), [
    ("bl victim + 2", "", "R_ARM_THM_CALL at .text+0x0 enters victim+0x2 (.text+0x2) inside an instruction"),
    ("b.w victim + 2", "", "R_ARM_THM_JUMP24 at .text+0x0 enters victim+0x2 (.text+0x2) inside an instruction"),
    ("movw r0, #:lower16:(victim + 3)", "",
     "R_ARM_THM_MOVW_ABS_NC at .text+0x0 enters victim+0x3 (.text+0x3) inside an instruction"),
    ("", ".word victim + 3", "R_ARM_ABS32 at .rodata.p+0x0 enters victim+0x3 (.text+0x3) inside an instruction"),
    ("", ".hword victim", "R_ARM_ABS16 at .rodata.p+0x0 into victim (.text) is unchecked"),
])
def test_object_scan_flags_entry_inside_another_objects_instruction(
    tmp_path: Path, call: str, data: str, expected: str,
) -> None:
    build = _build_units(tmp_path, {"v.S": VICTIM, "c.S": CALLER.format(call=call, data=data)})
    findings, summary, _ = candidate_scan.object_findings(build)
    hits = [f for f in findings if f["rule"] == "object_instruction" and not f["text"].endswith("wfi")]
    assert [(f["path"], f["text"]) for f in hits] == [("Source/c.S", expected)]
    assert summary["count"] == 2


@needs_gcc
def test_object_scan_cross_object_boundaries_are_clean(tmp_path: Path) -> None:
    call = "bl victim\n  b.w victim\n  movw r0, #:lower16:(tab + 6)\n  movt r0, #:upper16:(tab + 6)"
    build = _build_units(tmp_path, {"v.S": VICTIM, "c.S": CALLER.format(call=call, data=".word victim\n  .word tab + 5")})
    findings, summary, _ = candidate_scan.object_findings(build)
    assert [f for f in findings if not f["text"].endswith("wfi")] == [] and summary["count"] == 2


WEAK_VICTIM = VICTIM.replace(".global victim", ".weak victim")
# victim+2 starts an instruction here.
PLAIN_VICTIM = """.syntax unified
.thumb
.text
{bind}
.type victim, %function
.thumb_func
victim:
  nop
  nop
  bx lr
"""
OVERRIDE = "R_ARM_THM_CALL at .text+0x0 into weak victim+0x2 is unchecked: another definition can replace it"
ENTERS = "R_ARM_THM_CALL at .text+0x0 enters victim+0x2 (.text+0x2) inside an instruction"


def _entry_hits(build: Path) -> list[tuple[str, str]]:
    findings, summary, _ = candidate_scan.object_findings(build)
    assert summary["count"] >= 2
    return sorted((f["path"], f["text"]) for f in findings
                  if f["rule"] == "object_instruction" and not f["text"].endswith("wfi"))


@needs_gcc
@pytest.mark.parametrize(("victim", "call", "expected"), [
    # The weak definition is the one linked when nothing overrides it.
    (WEAK_VICTIM, "bl victim + 2", [ENTERS, OVERRIDE]),
    (WEAK_VICTIM, "b.w victim + 2", [ENTERS.replace("THM_CALL", "THM_JUMP24"),
                                     OVERRIDE.replace("THM_CALL", "THM_JUMP24")]),
    # A boundary of the weak body proves nothing about an override.
    (WEAK_VICTIM, "bl victim + 4", [OVERRIDE.replace("+0x2", "+0x4")]),
    (PLAIN_VICTIM.format(bind=".weak victim"), "bl victim + 2", [OVERRIDE]),
    (WEAK_VICTIM, "bl victim", []),
], ids=["bl_hidden", "bw_hidden", "bl_boundary", "bl_plain", "bl_entry"])
def test_object_scan_checks_calls_into_weak_code(tmp_path: Path, victim: str, call: str, expected: list[str]) -> None:
    build = _build_units(tmp_path, {"v.S": victim, "c.S": CALLER.format(call=call, data="")})
    assert _entry_hits(build) == [("Source/c.S", text) for text in sorted(expected)]


@needs_gcc
def test_object_scan_checks_own_weak_code(tmp_path: Path) -> None:
    own = WEAK_VICTIM + ".text\n.global caller\n.type caller, %function\n.thumb_func\ncaller:\n  bl victim + 2\n  bx lr\n"
    build = _build_units(tmp_path, {"v.S": own, "o.S": PLAIN_VICTIM.format(bind=".global other")})
    hits = _entry_hits(build)
    assert ("Source/v.S", OVERRIDE.replace(".text+0x0", ".text+0x6")) in hits
    assert any(path == "Source/v.S" and "victim+0x2 (.text+0x2) inside an instruction" in text for path, text in hits)


@needs_gcc
@pytest.mark.parametrize("order", ["strong_first", "weak_first"])
@pytest.mark.parametrize(("strong", "weak", "expected"), [
    # A strong definition always wins, so only it is checked.
    (VICTIM, PLAIN_VICTIM.format(bind=".weak victim"), [ENTERS]),
    (PLAIN_VICTIM.format(bind=".global victim"), WEAK_VICTIM, []),
], ids=["strong_hidden", "weak_hidden"])
def test_object_scan_strong_definition_beats_weak(
    tmp_path: Path, order: str, strong: str, weak: str, expected: list[str],
) -> None:
    units = {"s.S": strong, "w.S": weak}
    if order == "weak_first":
        units = dict(reversed(units.items()))
    build = _build_units(tmp_path, {**units, "c.S": CALLER.format(call="bl victim + 2", data="")})
    assert _entry_hits(build) == [("Source/c.S", text) for text in expected]


@needs_gcc
@pytest.mark.parametrize("order", ["local_first", "global_first"])
def test_object_scan_ignores_local_name_collisions(tmp_path: Path, order: str) -> None:
    # A local victim with a boundary at +2 must not stand in for the global one.
    units = {"l.S": PLAIN_VICTIM.format(bind=""), "g.S": VICTIM}
    if order == "global_first":
        units = dict(reversed(units.items()))
    build = _build_units(tmp_path, {**units, "c.S": CALLER.format(call="bl victim + 2", data="")})
    assert _entry_hits(build) == [("Source/c.S", ENTERS)]


@needs_gcc
@pytest.mark.parametrize("order", ["local_first", "global_first"])
def test_object_scan_local_name_collision_boundary_is_clean(tmp_path: Path, order: str) -> None:
    units = {"l.S": VICTIM.replace(".global victim\n", ""), "g.S": PLAIN_VICTIM.format(bind=".global victim")}
    if order == "global_first":
        units = dict(reversed(units.items()))
    build = _build_units(tmp_path, {**units, "c.S": CALLER.format(call="bl victim + 2", data="")})
    assert _entry_hits(build) == []


@needs_gcc
@pytest.mark.parametrize(("call", "expected"), [
    ("bl ext + 2", ["R_ARM_THM_CALL at .text+0x0 into ext+0x2 outside the kernels is unchecked"]),
    ("b.w ext + 4", ["R_ARM_THM_JUMP24 at .text+0x0 into ext+0x4 outside the kernels is unchecked"]),
    ("bl ext", []),
    ("b.w ext", []),
])
def test_object_scan_checks_calls_outside_the_kernels(tmp_path: Path, call: str, expected: list[str]) -> None:
    build = _build_units(tmp_path, {"o.S": PLAIN_VICTIM.format(bind=".global other"),
                                    "c.S": CALLER.format(call=call, data="")})
    assert _entry_hits(build) == [("Source/c.S", text) for text in expected]


def test_external_hits_weak_only_checks_every_weak_definition() -> None:
    clean = ((".text", 0, ({0: 2, 2: 2, 4: 2}, (), []), 6), True)
    hidden = ((".text", 0, ({0: 4, 4: 2}, (), []), 6), True)
    results = [
        ("Source/a.c", ([], {"f": [clean]}, [])),
        ("Source/b.c", ([], {"f": [hidden]}, [])),
        ("Source/c.c", ([], {}, [("f", 0, "R_ARM_THM_CALL at .text+0x0", True),
                                   ("f", None, "R_ARM_CALL at .text+0x4", True)])),
    ]
    assert [h["text"] for h in candidate_scan._external_hits(results)] == [
        "R_ARM_CALL at .text+0x4 into f (.text) is unchecked",
        "R_ARM_CALL at .text+0x4 into f (.text) is unchecked",
        "R_ARM_CALL at .text+0x4 into weak f is unchecked: another definition can replace it"]


def test_external_hits_ignores_symbols_without_code() -> None:
    results = [("Source/a.c", ([], {}, [("tab", 3, "R_ARM_ABS32 at .data+0x0", False)]))]
    assert candidate_scan._external_hits(results) == []


def test_external_hits_end_of_section_is_clean() -> None:
    results = [
        ("Source/a.c", ([], {"f": [((".text", 0, ({0: 4, 4: 2}, (), []), 6), False)]}, [])),
        ("Source/b.c", ([], {}, [("f", 6, "R_ARM_ABS32 at .data+0x0", False),
                                   ("f", 2, "R_ARM_ABS32 at .data+0x4", False)])),
    ]
    assert [h["text"] for h in candidate_scan._external_hits(results)] == [
        "R_ARM_ABS32 at .data+0x4 enters f+0x2 (.text+0x2) inside an instruction"]


def test_thumb_addend_short_data_is_unread() -> None:
    assert candidate_scan._thumb_addend("R_ARM_THM_CALL", memoryview(b"\x00\xf0"), 0) is None


def test_bad_elf_fails(tmp_path: Path) -> None:
    obj = tmp_path / "k.o"
    obj.write_bytes(b"\x7fELF\x01\x01" + bytes(10))
    with pytest.raises(ValueError):
        candidate_scan.elf_sections(obj)


@needs_gcc
def test_object_scan_clean(tmp_path: Path) -> None:
    source = "unsigned f(unsigned x) { return x > 0xE0000000u ? x : ~x; }\nconst int t[2] = {1, 2};\n"
    findings, summary, _ = candidate_scan.object_findings(_build(tmp_path, source))
    assert findings == [] and summary["count"] == 1


@needs_gcc
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


@needs_gcc
def test_build_macros_reach_gcc_e(kernels: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(candidate_check, "kernels_match", lambda tree, module: True)
    text = PASTE + '#ifdef BOARD_X\nCAT(_Pra, gma)("GCC optimize(\\"O3\\")")\n#endif\nint a;\n'
    (kernels / "Source/Conv/a.c").write_text(text, encoding="utf-8")
    # Source rules see only the paste.
    assert _hits(kernels) == {("build_probe", "Source/Conv/a.c")}
    report = check_candidate(kernels, _git(kernels, "rev-parse", "HEAD").strip(), build_dir=_build(tmp_path, "int k;\n"))
    found = {(f["rule"], f["path"]) for f in report["findings"]}
    assert found == {("pragma", "Source/Conv/a.c"), ("build_probe", "Source/Conv/a.c")}
    assert report["objects"]["count"] == 1


@needs_gcc
def test_board_header_probe_found(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(candidate_check, "kernels_match", lambda tree, module: True)
    header = PASTE + '#define B CAT(__has_, include)("board.h")\n#define A 1\n'
    root = _repo(tmp_path / "nn", {"Include/p.h": header,
                                   "Source/Conv/p.c": '#include "p.h"\n#if A\nint p;\n#endif\n'})
    (root / "Include/p.h").write_text(header.replace("A 1", "A B"), encoding="utf-8")
    (root / "Source/Conv/p.c").write_text(
        '#include "p.h"\n#if A\nCAT(_Pra, gma)("GCC optimize(\\"O3\\")")\n#endif\n', encoding="utf-8")
    build = _build(tmp_path, "int k;\n")
    # Only the real build sees board.h.
    (build / "bsp").mkdir()
    (build / "bsp/board.h").write_text("", encoding="utf-8")
    entries = json.loads((build / "compile_commands.json").read_text(encoding="utf-8"))
    entries[0]["arguments"].insert(1, f"-I{build}/bsp")
    (build / "compile_commands.json").write_text(json.dumps(entries), encoding="utf-8")
    sha = _git(root, "rev-parse", "HEAD").strip()
    assert ("pragma", "Source/Conv/p.c") not in _hits(root)
    found = {(f["rule"], f["path"]) for f in check_candidate(root, sha, build_dir=build)["findings"]}
    assert {("pragma", "Source/Conv/p.c"), ("build_probe", "Include/p.h")} <= found


@needs_gcc
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


@needs_gcc
def test_per_unit_flags_reach_gcc_e(kernels: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(candidate_check, "kernels_match", lambda tree, module: True)
    text = PASTE + '#ifdef BOARD_B\nCAT(_Pra, gma)("GCC optimize(\\"O3\\")")\n#endif\nint b;\n'
    (kernels / "Source/Conv/b.c").write_text(text, encoding="utf-8")
    flags = {"*": ("-DBOARD_X",), "Source/Conv/a.c": ("-DBOARD_X",), "Source/Conv/b.c": ("-DBOARD_B",)}
    monkeypatch.setattr(candidate_check, "object_findings", lambda build, deadline: ([], {"count": 2}, flags))
    report = check_candidate(kernels, _git(kernels, "rev-parse", "HEAD").strip(), build_dir=tmp_path)
    assert ("pragma", "Source/Conv/b.c") in {(f["rule"], f["path"]) for f in report["findings"]}


@needs_gcc
def test_stale_build_fails(kernels: Path, tmp_path: Path) -> None:
    report = check_candidate(kernels, _git(kernels, "rev-parse", "HEAD").strip(), build_dir=_build(tmp_path, "int k;\n"))
    assert {f.get("message") for f in report["findings"]} == {"build dir holds other kernels"}


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


def test_binutil_output_is_capped(tmp_path: Path, monkeypatch) -> None:
    flood = tmp_path / "flood"
    flood.write_text("#!/bin/sh\nexec yes x\n")
    flood.chmod(0o755)
    monkeypatch.setattr(candidate_scan, "arm_tool", lambda name: str(flood))
    monkeypatch.setattr(candidate_scan, "OUTPUT_CAP", 1 << 16)
    with pytest.raises(ValueError, match="hit limits"):
        candidate_scan.run_binutil("arm-none-eabi-objdump", ["-s", "x.o"])


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


@pytest.mark.parametrize(("cpus", "cap", "want"), [(64, 64 << 20, 4), (2, 64 << 20, 2), (None, 64 << 20, 1),
                                                   (64, 1 << 30, 1)])
def test_workers_fit_budget(monkeypatch, cpus, cap: int, want: int) -> None:
    monkeypatch.setattr(candidate_scan.os, "cpu_count", lambda: cpus)
    monkeypatch.setattr(candidate_scan, "OUTPUT_CAP", cap)
    assert candidate_scan.scan_workers() == want


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
    found, deps = candidate_scan._unit_counts(gcc, tmp_path, "Source/u.c", (), time.monotonic() + 30, "", rule_counts)
    assert set(found) == {"Source/u.c"} and deps is None
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


@needs_gcc
@pytest.mark.parametrize(("cap", "total", "message"), [
    (1024, 1 << 30, "object over 0 MiB: Source/k.c"),
    (1 << 30, 1024, "objects over 0 MiB in total"),
])
def test_big_objects_refused(tmp_path: Path, monkeypatch, cap: int, total: int, message: str) -> None:
    build = _build(tmp_path, "int k;\n")
    (build / "k.c.obj").write_bytes(bytes(4096))
    monkeypatch.setattr(candidate_scan, "OBJECT_CAP", cap)
    monkeypatch.setattr(candidate_scan, "OBJECTS_TOTAL_CAP", total)
    # Refused from stat alone: nothing read.
    monkeypatch.setattr(candidate_scan, "elf_sections", lambda obj: pytest.fail("read the object"))
    monkeypatch.setattr(candidate_scan, "run_binutil", lambda *args: pytest.fail("ran a binutil"))
    findings, _, _ = candidate_scan.object_findings(build)
    assert findings == [{"rule": "scan_error", "path": "", "message": message}]


@needs_gcc
def test_object_scan_deadline(tmp_path: Path, monkeypatch) -> None:
    build = _build(tmp_path, "int k;\n")
    slow = _fake_gcc(tmp_path, "sleep 30")
    monkeypatch.setattr(candidate_scan, "arm_tool", lambda name: slow)
    start = time.monotonic()
    findings, _, _ = candidate_scan.object_findings(build, deadline_s=1.0)
    assert time.monotonic() - start < 8
    assert [f["rule"] for f in findings] == ["scan_error"]


# --- scan cache ---------------------------------------------------------------------


@pytest.fixture
def scans(tmp_path: Path, monkeypatch):
    """Thread pool; counts gcc runs."""
    monkeypatch.setattr(candidate_scan, "_SCANS", {})
    monkeypatch.setattr(candidate_scan, "ProcessPoolExecutor", ThreadPoolExecutor)
    root = tmp_path / "tree"
    for rel, text in {"Source/u.c": "int u;\n", "Source/v.c": "int v;\n", "Include/h.h": "_Pragma(1)\n"}.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8")
    log = tmp_path / "ran"
    # Marker per file read, like gcc.
    gcc = _fake_gcc(tmp_path, f'for u; do :; done; echo "$u" >> {log}; echo "# 1 \\"$u\\""; cat "$u"; '
                              'echo \'# 1 "Include/h.h"\'; cat Include/h.h')

    def scan(configs=None, rules=rule_counts, **kw):
        log.unlink(missing_ok=True)
        counts, failed = candidate_scan._tree_counts(gcc, root, rules, configs or {"c": ("-DX",)},
                                                     time.monotonic() + 30, str(tmp_path / "abort"), **kw)
        ran = sorted(log.read_text().split()) if log.exists() else []
        return counts, ran

    return root, scan


def test_scan_memo_skips_gcc(scans) -> None:
    root, scan = scans
    first, ran = scan()
    assert ran == ["Source/u.c", "Source/v.c"] and first[("c", "Include/h.h")]["pragma"] == 1
    again, ran = scan()
    assert ran == [] and again == first


@pytest.mark.parametrize("change", ["unit", "header", "new_file", "flags", "flag_file", "env", "rules"])
def test_scan_key_misses_on_change(scans, monkeypatch, change) -> None:
    """Any gcc -E input change rescans."""
    root, scan = scans
    configs = {"c": ("-DX", "-include", "Source/f.h")}
    (root / "Source/f.h").write_text("\n")
    scan(configs)
    rules = rule_counts
    if change == "unit":
        (root / "Source/u.c").write_text("int u2;\n")
    elif change == "header":
        (root / "Include/h.h").write_text("_Pragma(2)\n")
    elif change == "new_file":
        (root / "Include/shadow.h").write_text("\n")
    elif change == "flags":
        configs = {"c": ("-DY", "-include", "Source/f.h")}
    elif change == "flag_file":
        # Not a marker: the key holds it.
        (root / "Source/f.h").write_text("#define Z\n")
    elif change == "env":
        monkeypatch.setenv("CPATH", "/elsewhere")
        candidate_scan._compiler_id.cache_clear()
    else:
        def rules(text):
            return rule_counts(text)
    _, ran = scan(configs, rules=rules)
    candidate_scan._compiler_id.cache_clear()
    assert ran == (["Source/u.c"] if change == "unit" else ["Source/u.c", "Source/v.c"])


def test_scan_cache_keeps_base_only(scans, tmp_path) -> None:
    root, scan = scans
    cache = tmp_path / "cache"
    scan(cache=cache)
    assert not cache.exists()
    candidate_scan._SCANS.clear()
    scan(cache=cache, store=True)
    stored = sorted(cache.glob("*.json"))
    assert len(stored) == 2
    # A fresh process reads them back.
    candidate_scan._SCANS.clear()
    _, ran = scan(cache=cache)
    assert ran == []


def test_scan_cache_drops_failed_units(scans, tmp_path) -> None:
    root, scan = scans
    cache = tmp_path / "cache"
    # cat fails: gcc exits nonzero.
    (root / "Include/h.h").unlink()
    scan(cache=cache, store=True)
    assert not cache.exists() and not candidate_scan._SCANS


@needs_gcc
def test_cached_base_still_finds_header_edit(kernels: Path, tmp_path: Path, monkeypatch) -> None:
    """Warm base cache; header edit still found."""
    monkeypatch.setattr(candidate_scan, "_SCANS", {})
    head = _git(kernels, "rev-parse", "HEAD").strip()
    cache = tmp_path / "cache"
    (kernels / "Source/Conv/b.c").write_text('#include "arm_nn_types.h"\nint b2;\n')
    assert check_candidate(kernels, head, scan_cache=cache)["ok"] and list(cache.glob("*.json"))
    (kernels / "Include/k.h").write_text(PASTE + '#define KATTR CAT(_Pra, gma)("GCC optimize(\\"O3\\")")\n')
    candidate_scan._SCANS.clear()
    report = check_candidate(kernels, head, scan_cache=cache)
    assert ("pragma", "Source/Conv/a.c") in {(f["rule"], f["path"]) for f in report["findings"]}


def test_marker_naming_fifo_does_not_hang(tmp_path: Path) -> None:
    """Non-regular deps: no read, no cache."""
    os.mkfifo(tmp_path / "pipe")
    (tmp_path / "Source").mkdir()
    (tmp_path / "Source/u.c").write_text("int u;\n")
    gcc = _fake_gcc(tmp_path, f'echo \'# 1 "Source/u.c"\'; echo \'# 1 "{tmp_path}/pipe" 1\'')
    found, deps = candidate_scan._unit_counts(gcc, tmp_path, "Source/u.c", (), time.monotonic() + 30, "", rule_counts)
    assert "Source/u.c" in found and deps is None


def test_cwd_marker_keeps_unit_cacheable(tmp_path: Path) -> None:
    """gcc -g names the cwd; skip it."""
    (tmp_path / "Source").mkdir()
    (tmp_path / "Source/u.c").write_text("int u;\n")
    gcc = _fake_gcc(tmp_path, f'echo \'# 0 "{tmp_path}//"\'; echo \'# 1 "Source/u.c"\'; echo "int u;"')
    found, deps = candidate_scan._unit_counts(gcc, tmp_path, "Source/u.c", (), time.monotonic() + 30, "", rule_counts)
    assert "Source/u.c" in found and deps is not None and list(deps) == ["Source/u.c"]


def test_directory_dep_reads_as_none(tmp_path: Path) -> None:
    """A directory dep has no digest."""
    assert candidate_scan._file_digest(tmp_path, "", {}) is None


def test_object_scan_drops_clang_flags(tmp_path: Path, monkeypatch) -> None:
    """gcc -E rejects clang flags; keep CONFIGS."""
    build = tmp_path / "build"
    src = build / "nsx_app/modules/nsx-cmsis-nn/Source/k.c"
    src.parent.mkdir(parents=True)
    src.write_text("int k;\n", encoding="utf-8")
    (build / "k.c.obj").write_bytes(b"")
    args = ["/opt/atfe/bin/clang", "--target=arm-none-eabi", "-o", "k.c.obj", "-c", str(src)]
    entry = {"directory": str(build), "file": str(src), "output": "k.c.obj", "arguments": args}
    (build / "compile_commands.json").write_text(json.dumps([entry]), encoding="utf-8")
    monkeypatch.setattr(candidate_scan, "_object_hits", lambda *a: ([], {}, []))
    monkeypatch.setattr(candidate_scan, "_defined_symbols", lambda *a, **k: frozenset())
    findings, summary, flags = candidate_scan.object_findings(build)
    assert findings == [] and summary["count"] == 1 and flags == {}


@pytest.mark.parametrize("name, ok", [
    (".ARM.exidx.text.arm_fn", True), (".rel.ARM.exidx.text.arm_fn", True), (".llvm_addrsig", True),
    (".note.GNU-stack", True), (".ARM.exidx", True), (".ARM.attributes.x", False), (".ARM.exidx.data.x", False),
    (".itcm_text", False),
])
def test_clang_sections_allowed(name: str, ok: bool) -> None:
    assert bool(candidate_scan._SECTION_OK.match(name) and candidate_scan._PLAIN_NAME.fullmatch(name)) == ok
