"""Check each standalone case can be generated and gated on its public symbols."""

from pathlib import Path
import re

import pytest

from helia_core_tester.generation.io.descriptors import load_descriptor
from helia_core_tester.generation.test_ops import generate_test


DESCRIPTORS = (
    Path(__file__).resolve().parents[2] / "assets/descriptors/BasicMathFunctions"
)


@pytest.mark.parametrize("stem", ["tree_walk", "rbf_kernel", "exp"])
def test_required_symbols_match_the_rendered_calls(stem: str, tmp_path: Path) -> None:
    for desc in load_descriptor(str(DESCRIPTORS / f"{stem}.yaml")):
        generate_test(desc, str(tmp_path), seed=7, cpu="cortex-m55")
        case = next(tmp_path.rglob(desc["name"]))
        source = next(case.glob("*.c")).read_text()
        calls = set(re.findall(r"\b(arm_nn_[a-z0-9_]+)\(", source))
        assert calls == set(desc["required_kernel_symbols"])
        assert bool(list(case.glob("*.tflite"))) == (
            desc["name"] == "exp_edges_tail_f32"
        )
