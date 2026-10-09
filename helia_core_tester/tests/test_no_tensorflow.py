"""The tester runs without TensorFlow, Keras or LiteRT: generation never imports them."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_FORBIDDEN = ("tensorflow", "keras", "tf_keras", "ai_edge_litert", "tflite_runtime")


def _imported_roots(path: Path) -> set[str]:
    roots = set()
    for node in ast.walk(ast.parse(path.read_text(), str(path))):
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


def test_no_module_imports_tensorflow_or_litert() -> None:
    offenders = {
        str(path.relative_to(ROOT)): sorted(_imported_roots(path) & set(_FORBIDDEN))
        for path in (ROOT / "helia_core_tester").rglob("*.py")
    }
    assert {k: v for k, v in offenders.items() if v} == {}


def test_dependencies_name_no_tensorflow_or_litert() -> None:
    import tomllib

    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    declared = list(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        declared += extra
    names = {dep.split(";")[0].split("[")[0].split("<")[0].split(">")[0].split("=")[0].strip().lower().replace("_", "-")
             for dep in declared}
    assert names.isdisjoint({"tensorflow", "tensorflow-cpu", "keras", "tf-keras", "ai-edge-litert", "tflite-runtime"})


def test_lockfile_resolves_no_tensorflow_or_litert() -> None:
    lock = (ROOT / "uv.lock").read_text()
    for name in ("tensorflow", "keras", "ai-edge-litert", "tflite-runtime"):
        assert f'name = "{name}"' not in lock


@pytest.mark.parametrize("forbidden", _FORBIDDEN)
def test_generation_runs_with_the_frameworks_blocked(tmp_path: Path, forbidden: str) -> None:
    # A meta-path finder that refuses the framework: any import on the generation path fails loudly.
    script = f"""
import sys
class _Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] == {forbidden!r}:
            raise ImportError("blocked: " + name)
        return None
sys.meta_path.insert(0, _Block())
from helia_core_tester.generation.io.descriptors import load_all_descriptors
from helia_core_tester.generation.test_ops import generate_test
import helia_core_tester.generation.ops as ops
ops.get_op_map()
descs = {{d["name"]: d for d in load_all_descriptors({str(ROOT / "assets" / "descriptors")!r})}}
for name in ("convolve_float_default_f32", "fully_connected_float_default_f16", "batch_matmul_float_default_f32"):
    generate_test(descs[name], {str(tmp_path)!r}, seed=500)
print("ok")
"""
    proc = subprocess.run([sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    assert proc.stdout.strip().endswith("ok")
    assert not list(tmp_path.rglob("*.tflite"))
