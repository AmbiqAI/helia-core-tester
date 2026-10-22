"""Generated inputs reach the ranges kernels branch on."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from helia_core_tester.core.config import Config
from helia_core_tester.core.discovery import find_descriptors_dir
from helia_core_tester.generation.io.descriptors import load_all_descriptors


def _generate(tmp_path: Path, name: str) -> dict[str, np.ndarray]:
    """Generate one case; map array name to values."""
    from helia_core_tester.generation.test_ops import generate_test

    descs = {d["name"]: d for d in load_all_descriptors(str(find_descriptors_dir()))}
    generate_test(descs[name], str(tmp_path), seed=Config.seed)
    (case_dir,) = tmp_path.glob(f"*/{name}")
    arrays = {}
    for header in (case_dir / "includes").glob("*.h"):
        for array, body in re.findall(r"const\s+(?:int\w+|bool)\s+(\w+)\s*\[\s*\]\s*=\s*\{([^}]*)\}", header.read_text()):
            tokens = [t.strip() for t in body.split(",") if t.strip()]
            values = [int({"true": "1", "false": "0"}.get(t, t), 0) for t in tokens]
            arrays[array.removeprefix(name + "_")] = np.array(values)
    return arrays


@pytest.mark.parametrize(
    ("name", "reach"),
    [
        ("add_default_s8", 100),
        ("maximum_default_s8", 100),
        ("minimum_dual_s8", 64),
        ("mul_default_s8", 32),
        ("sub_dual_inputs_s8", 100),
    ],
)
def test_s8_inputs_span_range(tmp_path, name, reach):
    arrays = _generate(tmp_path, name)
    inputs = np.concatenate([arrays["input1"], arrays["input2"]])
    assert np.abs(inputs).max() >= reach


@pytest.mark.parametrize(
    ("name", "step"),
    [
        ("comparison_equal_nhwc_s16", 128),
        ("comparison_equal_s8", 1),
        ("comparison_less_batch_scalar_left_s8", 1),
        ("comparison_less_equal_batch_scalar_right_s8", 1),
        ("comparison_less_batch_scalar_left_s16", 128),
        ("comparison_less_equal_batch_scalar_right_s16", 128),
    ],
)
def test_comparison_has_near_pairs(tmp_path, name, step):
    arrays = _generate(tmp_path, name)
    desc = {d["name"]: d for d in load_all_descriptors(str(find_descriptors_dir()))}[name]
    shape_1, shape_2 = desc["input_1_shape"], desc["input_2_shape"]
    out_shape = np.broadcast_shapes(tuple(shape_1), tuple(shape_2))
    lhs = np.broadcast_to(arrays["input_1"].reshape(shape_1), out_shape).reshape(-1)
    rhs = np.broadcast_to(arrays["input_2"].reshape(shape_2), out_shape).reshape(-1)
    out = arrays["expected_output"].astype(bool)
    gap = np.abs(lhs - rhs)
    ties = out[gap == 0]
    assert ties.size
    # A near pair flips the tie result.
    assert np.any(out[gap == step] != ties[0])


@pytest.mark.parametrize("name", ["rsqrt_small_input_per_op_s16", "rsqrt_small_input_universal_s16"])
def test_rsqrt_small_inputs(tmp_path, name):
    arrays = _generate(tmp_path, name)
    assert arrays["input"].max() <= 1024
    assert arrays["input"].min() < 512
