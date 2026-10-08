"""Kernel code digests and touched cases."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from helia_core_tester.hardware.code_graph import changed_nodes, code_graph, is_touched, object_nodes, read_graph
from helia_core_tester.hardware.nsx_app import CMSIS_NN_MODULE
from helia_core_tester.hardware.toolchain import arm_tool

GCC = arm_tool("arm-none-eabi-gcc")
pytestmark = pytest.mark.skipif(shutil.which(GCC) is None, reason="needs arm-none-eabi-gcc")
FLAGS = ["-mcpu=cortex-m55", "-mthumb", "-mfloat-abi=hard", "-O3", "-ffunction-sections", "-fdata-sections"]

HEADER = "static inline int helper(int x) { return x * HELPER_K; }\n"
KERNEL = """
#include "k.h"
const int table[4] = {1, 2, 3, TABLE_LAST};
NEIGHBOUR
DECL
__attribute__((noinline)) static int inner(int x) { return helper(x) + table[x & 3]; }
int kernel(int x) { return inner(x) BODY; }
OTHER
"""
OTHER = "int other(int x) { return x - 7; }\n"
WEAK_OTHER = "__attribute__((weak)) " + OTHER
CALLER = "int kernel(int);\nint caller(int x) { return kernel(x) + 1; }\nEXTRA"
NEIGHBOUR = """
const char *words[] = {"alpha", "beta", "gamma"};
int neighbour(int x) { int s = 0; for (int i = 0; i < x; i++) s += words[i % 3][0] * i; return s; }
"""


def _build(
    root: Path, *, body: str = "", k: int = 3, last: int = 4, neighbour: str = "", decl: str = "", other: str = OTHER, extra: str = "",
) -> Path:
    """Compile a fake kernel tree; return its build dir."""
    module = root / "modules" / CMSIS_NN_MODULE
    (module / "Source").mkdir(parents=True)
    (module / "Include").mkdir()
    (module / "Include/k.h").write_text(f"#define HELPER_K {k}\n#define TABLE_LAST {last}\n" + HEADER)
    kernel = KERNEL.replace("NEIGHBOUR", neighbour).replace("DECL", decl).replace("BODY", body).replace("OTHER", other)
    sources = {"Source/k.c": kernel, "Source/c.c": CALLER.replace("EXTRA", extra)}
    entries = []
    for rel, text in sources.items():
        (module / rel).write_text(text)
        obj = root / "obj" / f"{Path(rel).name}.obj"
        obj.parent.mkdir(exist_ok=True)
        args = [GCC, *FLAGS, "-I", str(module / "Include"), "-o", str(obj), "-c", str(module / rel)]
        subprocess.run(args, check=True)
        entries.append({"directory": str(root), "file": str(module / rel), "arguments": args, "output": str(obj)})
    (root / "compile_commands.json").write_text(json.dumps(entries))
    return root


def _graphs(tmp_path: Path, base: dict | None = None, **edit) -> tuple[dict, dict]:
    return code_graph(_build(tmp_path / "a", **(base or {})))["nodes"], code_graph(_build(tmp_path / "b", **edit))["nodes"]


def _touched(base: dict, cand: dict, root: str) -> bool:
    return is_touched(root, None, base, cand, changed_nodes(base, cand))


def test_same_source_same_digests(tmp_path) -> None:
    base, cand = _graphs(tmp_path)
    assert base == cand and not changed_nodes(base, cand)


def test_neighbour_code_leaves_kernel_alone(tmp_path) -> None:
    base, cand = _graphs(tmp_path, neighbour=NEIGHBOUR)
    assert "neighbour" in changed_nodes(base, cand)
    assert not _touched(base, cand, "kernel") and not _touched(base, cand, "caller")


def test_body_edit_touches_kernel_and_callers(tmp_path) -> None:
    base, cand = _graphs(tmp_path, body="+ 5")
    assert changed_nodes(base, cand) == {"kernel"}
    assert _touched(base, cand, "kernel") and _touched(base, cand, "caller")
    assert not _touched(base, cand, "other")


def test_inlined_helper_edit_touches_caller(tmp_path) -> None:
    base, cand = _graphs(tmp_path, k=5)
    assert changed_nodes(base, cand) == {"Source/k.c:inner"}
    assert _touched(base, cand, "kernel") and not _touched(base, cand, "other")


def test_table_edit_touches_its_readers(tmp_path) -> None:
    base, cand = _graphs(tmp_path, last=9)
    assert changed_nodes(base, cand) == {"table"}
    assert "table" in base["Source/k.c:inner"]["refs"]
    assert _touched(base, cand, "kernel") and not _touched(base, cand, "other")


def test_unknown_root_counts_as_touched(tmp_path) -> None:
    base, cand = _graphs(tmp_path)
    assert _touched(base, cand, "missing_kernel")


def test_timed_wrapper_edit_touches_case(tmp_path) -> None:
    base, cand = _graphs(tmp_path, body="+ 5")
    changed = changed_nodes(base, cand)
    # Inner route unchanged, wrapper changed.
    assert is_touched("kernel", "Source/k.c:inner", base, cand, changed)
    assert not is_touched("other", "Source/k.c:inner", base, cand, changed)


def test_stored_graph_round_trips(tmp_path) -> None:
    graph = code_graph(_build(tmp_path))
    assert read_graph(json.loads(json.dumps(graph))) == graph["nodes"]
    assert read_graph({"schema": "other"}) is None


def test_object_nodes_name_locals_by_unit(tmp_path) -> None:
    root = _build(tmp_path)
    nodes = object_nodes("Source/k.c", root / "obj" / "k.c.obj")
    assert {"kernel", "other", "table", "Source/k.c:inner"} <= nodes.keys()


def test_alignment_change_touches_kernel(tmp_path) -> None:
    base, cand = _graphs(tmp_path, decl="int other(int x) __attribute__((aligned(256)));")
    assert changed_nodes(base, cand) == {"other"}


def test_wrapper_helpers_count_sibling_kernels_do_not() -> None:
    def graph(transpose: str, sibling: str) -> dict:
        return {"wrap": {"digest": "w", "refs": ["kern", "sibling", "transpose"]}, "kern": {"digest": "k", "refs": []},
                "sibling": {"digest": sibling, "refs": []}, "transpose": {"digest": transpose, "refs": []}}

    routes = frozenset({"kern", "sibling"})
    base = graph("t", "s")
    for cand, touched in ((graph("t2", "s"), True), (graph("t", "s2"), False)):
        assert is_touched("wrap", "kern", base, cand, changed_nodes(base, cand), routes) is touched


def test_weak_strong_swap_touches_name(tmp_path) -> None:
    # Same bytes; the linker's pick moves.
    base, cand = _graphs(tmp_path, base={"extra": WEAK_OTHER}, other=WEAK_OTHER, extra=OTHER)
    assert "other" in changed_nodes(base, cand)


def test_duplicate_copy_edit_touches_name(tmp_path) -> None:
    base, cand = _graphs(tmp_path, base={"extra": WEAK_OTHER}, extra=WEAK_OTHER.replace("7", "8"))
    assert "other" in changed_nodes(base, cand) and _touched(base, cand, "other")
