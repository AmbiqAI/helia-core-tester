"""TensorFlow, Keras, LiteRT and flatbuffers are gone: every golden comes from the C reference.

Imports are read with `ast` (a comment citing TFLite semantics is fine; an import is not), and
importing the generation package is checked in a fresh interpreter, so a lazy import reached at
load time is caught too.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

FORBIDDEN_ROOTS = ("tensorflow", "keras", "ai_edge_litert", "flatbuffers", "tflite_runtime")

_PACKAGE = Path(__file__).resolve().parents[1]
_REPO = _PACKAGE.parent


def _modules() -> list[Path]:
    tests = _PACKAGE / "tests"
    return sorted(p for p in _PACKAGE.rglob("*.py") if tests not in p.parents)


def forbidden_imports(source: str) -> list[str]:
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module]
        else:
            continue
        found += [name for name in names if name.split(".")[0] in FORBIDDEN_ROOTS]
    return found


def test_the_scan_sees_every_import_form() -> None:
    source = ("import os\nimport tensorflow as tf\nfrom ai_edge_litert.interpreter import Interpreter\n"
              "def f():\n    import keras.layers\n    from flatbuffers import Builder\n"
              "try:\n    import tflite_runtime\nexcept ImportError:\n    pass\n"
              "# import tensorflow\nx = 'import keras'\nfrom . import tensorflow_free\n")
    assert forbidden_imports(source) == ["tensorflow", "ai_edge_litert.interpreter", "keras.layers",
                                         "flatbuffers", "tflite_runtime"]


def test_the_package_scan_covers_the_generator() -> None:
    names = {p.relative_to(_PACKAGE).as_posix() for p in _modules()}
    assert {"generation/test_ops.py", "generation/ops/_shared/base.py", "cli.py"} <= names
    assert not any(name.startswith("tests/") for name in names)


@pytest.mark.parametrize("path", _modules(), ids=lambda p: p.relative_to(_PACKAGE).as_posix())
def test_no_module_imports_a_model_framework(path: Path) -> None:
    assert forbidden_imports(path.read_text()) == []


def test_importing_the_generator_loads_no_model_framework() -> None:
    probe = (
        "import json, sys\n"
        "import helia_core_tester.cli\n"
        "import helia_core_tester.generation.test_ops\n"
        "from helia_core_tester.generation.ops import get_op_map\n"
        "ops = get_op_map()\n"
        "classes = [ops[name] for name in ops]\n"
        "import helia_core_tester.hardware.case_bundle, helia_core_tester.hardware.fake_target\n"
        f"roots = {FORBIDDEN_ROOTS!r}\n"
        "print(json.dumps(sorted(m for m in sys.modules if m.split('.')[0] in roots)))\n"
    )
    result = subprocess.run([sys.executable, "-c", probe], cwd=_REPO, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == []


def test_the_project_declares_no_model_framework() -> None:
    import tomllib

    project = tomllib.loads((_REPO / "pyproject.toml").read_text())
    declared = [*project["project"]["dependencies"],
                *(d for group in project.get("dependency-groups", {}).values() for d in group if isinstance(d, str))]
    names = {re.split(r"[<>=!~;\[ ]", d, maxsplit=1)[0].lower().replace("-", "_") for d in declared}
    assert names.isdisjoint({*FORBIDDEN_ROOTS, "ai_edge_litert", "tf_keras", "tensorflow_cpu"})
