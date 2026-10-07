"""Harness digest and candidate diff check."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from helia_core_tester.cli import app
from helia_core_tester.hardware import boards, candidate_check, firmware_build, harness_lock
from helia_core_tester.hardware.candidate_check import check_candidate
from helia_core_tester.hardware.firmware_build import nsx_app_dir
from helia_core_tester.hardware.nsx_app import AppOptions, kernel_dir, save_options
from helia_core_tester.hardware.result_bundle import write_result_bundle
from helia_core_tester.hardware.session import SessionResult

runner = CliRunner()
pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


@pytest.fixture(autouse=True)
def _line_rules_only(monkeypatch) -> None:
    """test_candidate_scan covers gcc -E."""
    monkeypatch.setattr(candidate_check, "preprocess_findings", lambda *args: [])

LOCK = """schema_version: 4
targets:
  apollo510_evb:
    generated_at: '{stamp}'
    nsx_tool:
      version: 0.9.0
    manifest:
      path: nsx.yml
      hash: sha256:{manifest}
    target:
      board: apollo510_evb
    modules:
      nsx-ambiqsuite:
        project: nsx-ambiq-sdk
        kind: git
        resolved:
          commit: a9f4ec25
          content_hash: sha256:44
          acquired_at: '{stamp}'
      nsx-cmsis-nn:
        project: nsx-cmsis-nn
        kind: vendored
        resolved:
          content_hash: sha256:{kernel}
          acquired_at: '{stamp}'
"""
CACHE = "CMAKE_C_FLAGS:STRING={flags}\nCMAKE_C_FLAGS-ADVANCED:INTERNAL=1\nNSX_JLINK_SERIAL:UNINITIALIZED=1\n"


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout


def _repo(root: Path, files: dict[str, str]) -> Path:
    """A committed git repo with files."""
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8")
    _git(root.parent, "init", "-q", str(root))
    _git(root, "add", "-A")
    _git(root, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "base")
    return root


@pytest.fixture
def tester(tmp_path: Path, monkeypatch) -> Path:
    """A fake tester checkout as repo root."""
    root = _repo(tmp_path / "tester", {
        "cmake/hardware/main.c": "int main;\n",
        "assets/templates/hardware/nsx/CMakeLists.txt.j2": "x\n",
        "scripts/patch_build_id.py": "pass\n",
    })
    monkeypatch.setattr(firmware_build, "tester_repo_root", lambda: root)
    monkeypatch.setattr(boards, "repo_root", lambda: root)
    return root


def _build(build_dir: Path, kernel: str = "k", flags: str = "-O2", stamp: str = "t0", **opts) -> dict:
    """Fake one build; return the manifest's harness."""
    options = AppOptions(cmsis_nn_root=build_dir.parent / "kernels", **opts)
    app_dir = nsx_app_dir(build_dir)
    module = kernel_dir(app_dir, options)
    (module / "Source").mkdir(parents=True, exist_ok=True)
    (module / "Source" / "k.c").write_text(f"int {kernel};\n", encoding="utf-8")
    memory = app_dir / "boards" / "b" / "memory.cmake"
    if not memory.exists():
        memory.parent.mkdir(parents=True)
        memory.write_text("tcm\n", encoding="utf-8")
    lock = LOCK.format(stamp=stamp, manifest=stamp, kernel=kernel)
    (app_dir / "nsx.lock").write_text(lock, encoding="utf-8")
    (build_dir / "CMakeCache.txt").write_text(CACHE.format(flags=flags), encoding="utf-8")
    save_options(app_dir, options)
    firmware_build._record_built(build_dir, options)
    result = SessionResult(cases=(), protocol_trace=(), session_complete_cases=0, build_id="b")
    root = write_result_bundle(
        result, session_id="s", output_root=build_dir.parent, memory_report={}, kernel_catalog=[], build_dir=build_dir,
    )
    return json.loads((root / "session_manifest.json").read_text(encoding="utf-8"))


def test_digest_is_stable_across_reruns(tester: Path, tmp_path: Path) -> None:
    first = _build(tmp_path / "b", stamp="t0")
    second = _build(tmp_path / "b", stamp="t1")
    assert re.fullmatch(r"[0-9a-f]{64}", first["harness_digest"])
    assert harness_lock.same_harness(first, second)
    assert first["harness"]["tester_dirty"] is False


def test_kernel_change_keeps_digest(tester: Path, tmp_path: Path) -> None:
    base = _build(tmp_path / "b", kernel="a")
    candidate = _build(tmp_path / "b", kernel="b")
    assert harness_lock.same_harness(base, candidate)
    assert harness_lock.kernel_digest(base) != harness_lock.kernel_digest(candidate)


def test_module_source_edit_moves_digest(tester: Path, tmp_path: Path) -> None:
    sdk = nsx_app_dir(tmp_path / "b") / "modules" / "nsx-ambiq-sdk" / "hal.c"
    sdk.parent.mkdir(parents=True)
    sdk.write_text("int hal;\n", encoding="utf-8")
    first = _build(tmp_path / "b")
    sdk.write_text("int hal_edited;\n", encoding="utf-8")
    assert not harness_lock.same_harness(first, _build(tmp_path / "b"))
    assert "nsx-cmsis-nn" not in first["harness"]["inputs"]["firmware"]["module_trees"]


@pytest.mark.parametrize("change", ["source", "flags", "switch", "board"])
def test_harness_change_moves_digest(tester: Path, tmp_path: Path, change: str) -> None:
    base = _build(tmp_path / "b")
    kwargs: dict = {}
    if change == "source":
        (tester / "cmake/hardware/main.c").write_text("int main2;\n", encoding="utf-8")
        _git(tester, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qam", "edit")
    elif change == "flags":
        kwargs["flags"] = "-O3"
    elif change == "switch":
        kwargs["requantize_inline_asm"] = False
    else:
        memory = nsx_app_dir(tmp_path / "b") / "boards" / "b" / "memory.cmake"
        memory.write_text("mram\n", encoding="utf-8")
    assert not harness_lock.same_harness(base, _build(tmp_path / "b", **kwargs))


def test_dirty_tester_marks_bundle(tester: Path, tmp_path: Path) -> None:
    clean = _build(tmp_path / "b")
    (tester / "cmake/hardware/main.c").write_text("int dirty;\n", encoding="utf-8")
    dirty = _build(tmp_path / "b")
    assert dirty["harness"]["tester_dirty"] is True
    assert not harness_lock.same_harness(clean, dirty)


@pytest.mark.parametrize("flag", ["--skip-worktree", "--assume-unchanged"])
def test_hidden_edit_marks_tester_dirty(tester: Path, flag: str) -> None:
    assert harness_lock.tester_state(tester)["dirty"] is False
    _git(tester, "update-index", flag, "cmake/hardware/main.c")
    (tester / "cmake/hardware/main.c").write_text("int hidden;\n", encoding="utf-8")
    assert _git(tester, "status", "--porcelain") == ""
    state = harness_lock.tester_state(tester)
    assert state["dirty"] is True and state["diff"]


def test_no_build_means_no_digest(tester: Path) -> None:
    assert harness_lock.harness_record(None, tester)[0] is None
    assert not harness_lock.same_harness({}, {})


def test_run_refuses_dirty_tester(tmp_path: Path, monkeypatch) -> None:
    from helia_core_tester.hardware import hardware_pipeline

    def _pipeline(*args, **kwargs):
        called.append(kwargs)
        raise RuntimeError("stop here")

    called: list = []
    state = {"dirty": False}
    monkeypatch.setattr(harness_lock, "tester_state", lambda root: state)
    monkeypatch.setattr(hardware_pipeline, "run_hardware_pipeline", _pipeline)
    monkeypatch.setenv("HPX_JLINK_SERIAL", "1160003180")
    args = ["hardware", "run", "--skip-generate", "--build-dir", str(tmp_path / "b"), "--cmsis-nn-root", str(tmp_path)]
    assert runner.invoke(app, args).exit_code == 1 and called, "clean tester runs"
    called.clear()
    state["dirty"] = True
    result = runner.invoke(app, args)
    assert result.exit_code == 1 and "dirty" in result.output and not called
    result = runner.invoke(app, [*args, "--allow-dirty-tester"])
    assert called and "tester is dirty" in result.output


# --- candidate check ------------------------------------------------------------


@pytest.fixture
def kernels(tmp_path: Path) -> Path:
    root = _repo(tmp_path / "nn", {
        "Source/Conv/a.c": "int a;\n",
        "Include/arm_nnsupportfunctions.h": "int s;\n",
        "Include/arm_nnfunctions.h": "int f;\n",
        "Tests/t.c": "int t;\n",
        "nsx/CMakeLists.txt": "x\n",
        ".gitignore": "*.o\n",
    })
    _git(root, "tag", "base")
    return root


def _sha(root: Path) -> str:
    return _git(root, "rev-parse", "base").strip()


def _rules(root: Path) -> set[str]:
    return {finding["rule"] for finding in check_candidate(root, _sha(root))["findings"]}


def test_source_change_passes(kernels: Path) -> None:
    (kernels / "Source/Conv/a.c").write_text("int a; /* faster */\n", encoding="utf-8")
    (kernels / "Source/Conv/new.c").write_text("int b;\n", encoding="utf-8")
    report = check_candidate(kernels, _sha(kernels))
    assert report["ok"], report
    assert {f["path"] for f in report["files"]} == {"Source/Conv/a.c", "Source/Conv/new.c"}


@pytest.mark.parametrize(("rel", "text", "rule"), [
    ("Tests/t.c", "int t2;\n", "outside_allowlist"),
    ("nsx/CMakeLists.txt", "y\n", "outside_allowlist"),
    ("cmake/flags.cmake", "y\n", "outside_allowlist"),
    ("CMakeLists.txt", "y\n", "outside_allowlist"),
    ("Source/CMakeLists.txt", "y\n", "file_type"),
    ("Source/Conv/a.o", "y\n", "file_type"),
    ("Include/arm_nnfunctions.h", "int g;\n", "frozen_file"),
    ("Source/Conv/a.c", '__attribute__((section(".itcm"))) int a;\n', "attribute"),
    ("Source/Conv/a.c", '__attribute__((__section__(".itcm"))) int a;\n', "special_section"),
    ("Source/Conv/a.c", '[[gnu::section(".itcm")]] int a;\n', "attribute"),
    ("Source/Conv/a.c", "__attribute__((noinline,\n", "attribute"),
    ("Source/Conv/a.c", '__asm__(".pushsection .itcm");\n', "special_section"),
    ("Source/Conv/a.c", "#pragma GCC optimize(\"O3\")\n", "pragma"),
    ("Source/Conv/a.c", '_Pragma("GCC optimize(\\"O3\\")")\n', "pragma"),
    ("Source/Conv/a.c", "void g(void) { DWT->CYCCNT = 0; }\n", "measurement_access"),
    ("Source/Conv/a.c", "volatile int *c = (volatile int *)0xE0001004UL;\n", "measurement_access"),
    ("Source/Conv/a.c", "volatile int *c = (volatile int *)3758100484u;\n", "measurement_access"),
    ("Source/Conv/a.c", '#define P _Pra##gma("GCC optimize(\\"O3\\")")\n', "pragma"),
    ("Source/Conv/a.c", '#define A __attri ## bute__((optimize("O3")))\n', "attribute"),
    ("Source/Conv/a.c", '%:define P _Pra %:%: gma("GCC optimize(\\"O3\\")")\n', "pragma"),
    ("Source/Conv/a.c", '#define A __attri %:%: bute__((optimize("O3")))\n', "attribute"),
    ("Source/Conv/a.c", '#ifdef BOARD_X\n%:  pragma GCC optimize("O3")\n#endif\n', "pragma"),
    ("Source/Conv/a.c", '__attribute ( (section(".s")) ) int a;\n', "attribute"),
    ("Source/Conv/a.c", '#line 1 "/usr/include/x.h"\n', "build_probe"),
    ("Source/Conv/a.c", '# 1 "<x>"\n', "build_probe"),
    ("Source/Conv/a.c", "#ifdef __OPTIMIZE__\n", "build_probe"),
    ("Source/Conv/a.c", '#if __has_include("am_bsp.h")\n', "build_probe"),
    ("Source/Conv/a.c", "#include KERNEL_PATH\n", "include_escape"),
    ("Source/Conv/a.c", '%:include "../../Tests/t.c"\n', "include_escape"),
    ("Source/Conv/a.c", '%:pragma GCC optimize("O3")\n', "pragma"),
    ("Source/Conv/a.c", '__asm__("cpsid i");\n', "measurement_access"),
    ("Source/Conv/k.S", "    MSR PRIMASK, r0\n", "measurement_access"),
    ("Source/Conv/a.c", '__asm__("cps" "id i");\n', "measurement_access"),
    ("Source/Conv/a.c", '__asm__("cps"/*x*/"id i");\n', "measurement_access"),
    ("Source/Conv/a.c", '__asm__("cps\\151d i");\n', "measurement_access"),
    ("Source/Conv/a.c", '__asm__("cps\\x69\\x64 i");\n', "measurement_access"),
    ("Source/Conv/a.c", '_Pragma("GCC optimize(\\"O3\\")") /* #pragma once */\n', "pragma"),
    ("Source/Conv/a.c", "#pragma once\n_Pragma(\"GCC unroll 4\")\n", "pragma"),
    ("Source/Conv/a.c", '__asm__(".push"\n        "section .itcm");\n', "special_section"),
    ("Source/Conv/k.s", '.incbin "/etc/x"\n', "include_escape"),
    ("Source/Conv/a.c", "static const int golden[4];\n", "harness_reference"),
    ("Source/Conv/a.c", '#include "../../Tests/t.c"\n', "include_escape"),
])
def test_forbidden_change_fails(kernels: Path, rel: str, text: str, rule: str) -> None:
    (kernels / rel).parent.mkdir(parents=True, exist_ok=True)
    (kernels / rel).write_text(text, encoding="utf-8")
    assert rule in _rules(kernels)


@pytest.mark.parametrize(("value", "hit"), [
    ("3758096383u", False), ("3758096384u", True), ("3759144959UL", True), ("3759144960", False),
    ("0xDFFFFFFF", False), ("0xE0000000", True), ("0xe00fffffu", True), ("0xE0100000", False),
    ("3758000000", False), ("0xE000'1004", True), ("034000010004", True),
])
def test_scs_bounds(kernels: Path, value: str, hit: bool) -> None:
    (kernels / "Source/Conv/a.c").write_text(f"volatile int *c = (volatile int *){value};\n", encoding="utf-8")
    assert ("measurement_access" in _rules(kernels)) == hit


@pytest.mark.parametrize("text", [
    '#define CAT(a,b) a##b\n#if CAT(__has_, include)("board.h")\n_Pragma("GCC optimize(\\"O3\\")")\n#endif\n',
    # Unchanged-style #if reaching a new macro.
    '#define CAT(a,b) a##b\n#define B CAT(__has_, include)("board.h")\n#define A B\n#if A\nint z;\n#endif\n',
])
def test_macro_probe_fails(kernels: Path, text: str) -> None:
    (kernels / "Source/Conv/a.c").write_text(text, encoding="utf-8")
    assert "build_probe" in _rules(kernels)


def test_probe_after_comment_fails(kernels: Path) -> None:
    text = ('#define CAT(a,b) a##b\n/* note\n */ #if CAT(__has_, include)("missing.h")\n'
            'CAT(_Pra,gma)("GCC optimize(\\"O3\\")")\n#endif\n')
    (kernels / "Source/Conv/a.c").write_text(text, encoding="utf-8")
    assert "build_probe" in _rules(kernels)


def test_spliced_comment_hides_directive(kernels: Path) -> None:
    # gcc splices first: the pragma is comment.
    (kernels / "Source/Conv/a.c").write_text('// note \\\n#pragma GCC optimize("O3")\nint a;\n', encoding="utf-8")
    assert check_candidate(kernels, _sha(kernels))["ok"]


def test_comment_start_in_literal(kernels: Path) -> None:
    text = 'static const char s[] = "/*";\n_Pragma("GCC optimize(\\"O3\\")")\n// */\n'
    (kernels / "Source/Conv/a.c").write_text(text, encoding="utf-8")
    assert "pragma" in _rules(kernels)


def test_finding_lines_match_file(kernels: Path) -> None:
    text = 'int a;\n/* x\n y */\n#pragma GCC optimize("O3")\n#define D \\\n  _Pragma("x")\n'
    (kernels / "Source/Conv/a.c").write_text(text, encoding="utf-8")
    report = check_candidate(kernels, _sha(kernels))
    assert {f["line"] for f in report["findings"] if f["rule"] == "pragma"} == {4, 5}


def test_trigraph_fails(kernels: Path) -> None:
    (kernels / "Source/Conv/a.c").write_text('??=pragma GCC optimize("O3")\n', encoding="utf-8")
    assert "build_probe" in _rules(kernels)


PROBE_BASE = {
    "Include/p.h": '#define CAT(a, b) a##b\n#define B CAT(__has_, include)("board.h")\n#define A 1\n',
    "Source/Conv/p.c": '#include "p.h"\n#if A\nint p;\n#endif\nint q(int x) { return x; }\n',
}


@pytest.mark.parametrize(("rel", "text", "want"), [
    # Base probe routed into a base #if.
    ("Include/p.h", PROBE_BASE["Include/p.h"].replace("A 1", "A B"), "conditional reaches A"),
    ("Source/Conv/p.c", PROBE_BASE["Source/Conv/p.c"].replace(
        "int p;", 'CAT(_Pra,gma)("GCC optimize(\\"O3\\")")'), "uses pasting macro CAT"),
    ("Include/p.h", PROBE_BASE["Include/p.h"].replace("#define A 1\n", ""), "removed A, base line 3"),
    ("Include/p.h", PROBE_BASE["Include/p.h"] + "#undef A\n", "conditional reaches A"),
])
def test_conditional_macro_edit_fails(tmp_path: Path, rel: str, text: str, want: str) -> None:
    root = _repo(tmp_path / "nn", PROBE_BASE)
    (root / rel).write_text(text, encoding="utf-8")
    texts = {f.get("text") for f in check_candidate(root, _git(root, "rev-parse", "HEAD").strip())["findings"]
             if f["rule"] == "build_probe"}
    assert want in texts


def test_clean_edit_near_probe_passes(tmp_path: Path) -> None:
    root = _repo(tmp_path / "nn", PROBE_BASE)
    (root / "Source/Conv/p.c").write_text(
        PROBE_BASE["Source/Conv/p.c"].replace("return x;", "return x + 1;"), encoding="utf-8")
    (root / "Include/new.h").write_text("#ifndef NEW_H\n#define NEW_H\n#define TWICE(x) ((x) * 2)\n#endif\n",
                                        encoding="utf-8")
    report = check_candidate(root, _git(root, "rev-parse", "HEAD").strip())
    assert report["ok"], report["findings"]


def test_plain_conditional_passes(kernels: Path) -> None:
    text = "#if defined(ARM_MATH_MVEI) && __has_builtin(__builtin_expect)\nint z;\n#endif\n"
    (kernels / "Source/Conv/a.c").write_text(text, encoding="utf-8")
    assert check_candidate(kernels, _sha(kernels))["ok"]


def test_safe_attributes_pass(kernels: Path) -> None:
    text = "__attribute__((always_inline, aligned(4))) static int a;\n#pragma GCC unroll 4\n"
    (kernels / "Source/Conv/a.c").write_text(text, encoding="utf-8")
    assert check_candidate(kernels, _sha(kernels))["ok"]


def test_git_tricks_cannot_hide_changes(kernels: Path) -> None:
    (kernels / "Tests/t.c").write_text("int t2;\n", encoding="utf-8")
    _git(kernels, "update-index", "--skip-worktree", "Tests/t.c")
    (kernels / "Source/Conv/a.c").write_text('#pragma GCC optimize("O3")\n', encoding="utf-8")
    _git(kernels, "config", "diff.external", "true")
    (kernels / ".git/info/exclude").write_text("Source/Conv/hidden.c\n", encoding="utf-8")
    (kernels / "Source/Conv/hidden.c").write_text('__attribute__((section(".x"))) int h;\n', encoding="utf-8")
    report = check_candidate(kernels, _sha(kernels))
    rules = {(f["path"], f["rule"]) for f in report["findings"]}
    assert ("Tests/t.c", "outside_allowlist") in rules
    assert ("Source/Conv/a.c", "pragma") in rules
    assert ("Source/Conv/hidden.c", "attribute") in rules


def test_guard_edit_near_old_pragma_fails(tmp_path: Path) -> None:
    root = _repo(tmp_path / "nn", {"Source/Conv/a.c": '#if 0\n#pragma GCC optimize("O3")\n#endif\nint a;\n'})
    _git(root, "tag", "base")
    (root / "Source/Conv/a.c").write_text('#if 1\n#pragma GCC optimize("O3")\n#endif\nint a;\n', encoding="utf-8")
    assert _rules(root) == {"guard_change"}
    (root / "Source/Conv/a.c").write_text('#if 0\n#pragma GCC optimize("O3")\n#endif\nint a; /* faster */\n', encoding="utf-8")
    assert check_candidate(root, _sha(root))["ok"]
    (root / "Source/Conv/a.c").write_text('#pragma GCC optimize("O3")\nint a;\n', encoding="utf-8")
    assert _rules(root) == {"guard_change"}


def test_base_must_be_full_sha(kernels: Path) -> None:
    from helia_core_tester.hardware.candidate_check import CheckError

    with pytest.raises(CheckError, match="full commit SHA"):
        check_candidate(kernels, "base")


def test_token_inside_unchanged_attribute_fails(tmp_path: Path) -> None:
    root = _repo(tmp_path / "nn", {"Source/Conv/a.c": "__attribute__((\n    noinline))\nint a(void);\n"})
    _git(root, "tag", "base")
    text = '__attribute__((\n    noinline,\n    optimize("O3")))\nint a(void);\n'
    (root / "Source/Conv/a.c").write_text(text, encoding="utf-8")
    assert "attribute" in _rules(root)


def test_hidden_index_entry_fails(kernels: Path) -> None:
    _git(kernels, "update-index", "--skip-worktree", ".gitignore")
    report = check_candidate(kernels, _sha(kernels))
    assert [(f["rule"], f["path"]) for f in report["findings"]] == [("hidden_index_entry", ".gitignore")]


def test_symlink_fails(kernels: Path) -> None:
    (kernels / "Source/Conv/link.c").symlink_to(kernels / "Tests/t.c")
    assert "symlink" in _rules(kernels)


def test_symlinked_root_fails(kernels: Path, tmp_path: Path) -> None:
    import shutil

    shutil.copytree(kernels / "Source", tmp_path / "elsewhere")
    shutil.rmtree(kernels / "Source")
    (kernels / "Source").symlink_to(tmp_path / "elsewhere")
    assert "symlink" in _rules(kernels)


@pytest.mark.parametrize("text", [
    "void g(void) { DW\\\nT->CTRL = 0; }\n",
    "#prag\\\nma GCC optimize(\"O3\")\n",
    "__attri\\\nbute__((target(\"arch=armv8.1-m.main\"))) int a;\n",
])
def test_spliced_lines_cannot_hide_tokens(kernels: Path, text: str) -> None:
    (kernels / "Source/Conv/a.c").write_text(text, encoding="utf-8")
    report = check_candidate(kernels, _sha(kernels))
    assert not report["ok"] and report["findings"][0]["line"] == 1


def test_modules_cmake_keeps_unknown_kernel_location(tmp_path: Path) -> None:
    from helia_core_tester.hardware.harness_lock import modules_print

    known = 'set(NSX_APP_MODULE_DIR_nsx_cmsis_nn "modules/nsx-cmsis-nn")\n'
    assert "<kernels>" in modules_print(known)
    moved = 'set(NSX_APP_MODULE_DIR_nsx_cmsis_nn "/elsewhere/kernels")\n'
    assert modules_print(moved) == moved


def test_modules_cmake_hash_ignores_kernel_location(tmp_path: Path) -> None:
    from helia_core_tester.hardware.harness_lock import path_hash

    def tree(name: str, kernel_dir: str, project: str, extra: str = "") -> Path:
        root = tmp_path / name
        root.mkdir()
        (root / "modules.cmake").write_text(
            f'set(NSX_APP_MODULES\n    nsx-core\n    nsx-cmsis-nn\n{extra})\n'
            f'set(NSX_APP_MODULE_DIR_nsx_cmsis_nn "{kernel_dir}")\n'
            f"set(NSX_APP_PROJECT_DIRS\n{project}    modules/nsx-ambiq-sdk\n)\n",
            encoding="utf-8",
        )
        return root

    by_ref = tree("ref", "modules/ns-cmsis-nn/nsx", "    modules/ns-cmsis-nn\n")
    by_root = tree("root", "modules/nsx-cmsis-nn", "")
    assert path_hash(by_ref) == path_hash(by_root)
    assert path_hash(tree("more", "modules/nsx-cmsis-nn", "", "    nsx-segger-rtt\n")) != path_hash(by_root)


def test_cli_prints_json(kernels: Path) -> None:
    (kernels / "Tests/t.c").write_text("int t2;\n", encoding="utf-8")
    result = runner.invoke(app, ["candidate", "check", "--base", _sha(kernels), str(kernels)])
    assert result.exit_code == 1
    assert json.loads(result.output)["findings"][0]["rule"] == "outside_allowlist"
    bad = runner.invoke(app, ["candidate", "check", "--base", "nope", str(kernels)])
    assert bad.exit_code == 2 and json.loads(bad.output)["ok"] is False
