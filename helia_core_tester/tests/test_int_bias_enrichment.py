"""Int-suite Convolve/FullyConnected cases must ship a bias a kernel can miss.

An all-zero bias tensor makes a dropped bias-add bit-identical to a correct
kernel, so these cases assert both that the bias is nonzero and that it is
large enough to move at least one output element by a full output
quantization step.  See AmbiqAI/helia-core-tester#77.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pytest

from helia_core_tester.generation.io.descriptors import load_all_descriptors
from helia_core_tester.generation.ops.ConvolutionFunctions.convolve import OpConvolve
from helia_core_tester.generation.ops.FullyConnectedFunctions.fully_connected import OpFullyConnected

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SEED = 1


def _descriptor(name: str) -> Dict:
    descriptors = load_all_descriptors(str(_PROJECT_ROOT / "assets" / "descriptors"))
    for desc in descriptors:
        if desc.get("name") == name:
            return desc
    raise AssertionError(f"descriptor {name} not found")


def _generate(name: str, out_dir: Path) -> str:
    """Generate one case and return its emitted header text."""
    desc = _descriptor(name)
    op_cls = OpConvolve if desc["operator"] == "Convolve" else OpFullyConnected
    op = op_cls(desc, seed=_SEED, target_cpu="cortex-m55")
    model = op.build_keras_model() if op.needs_keras_model() else None
    op.convert_to_tflite(model, str(out_dir / f"{name}.tflite"), _SEED)
    op.generate_c_files(out_dir)
    return (out_dir / "includes").glob(f"{name}_*.h").__next__().read_text()


def _int_array(header: str, suffix: str) -> Optional[List[int]]:
    match = re.search(rf"_{suffix}\[[0-9]*\]\s*=\s*\{{([^}}]*)\}}", header)
    if match is None:
        return None
    return [int(value) for value in re.findall(r"-?\d+", match.group(1))]


def _int_scalar(header: str, suffix: str) -> Optional[int]:
    match = re.search(rf"_{suffix}\s*=\s*(-?\d+);", header)
    return int(match.group(1)) if match else None


def _output_steps(header: str) -> float:
    """Largest output shift, in output quantization steps, a dropped bias causes."""
    biases = _int_array(header, "biases")
    # Per-tensor cases emit scalars where per-channel cases emit arrays.
    multipliers = _int_array(header, "multiplier") or [_int_scalar(header, "multiplier_val")]
    shifts = _int_array(header, "shift") or [_int_scalar(header, "shift_val")]
    assert biases and all(multipliers) and all(s is not None for s in shifts), (
        "case has no bias/requantization arrays"
    )

    channels = len(biases)
    if len(multipliers) == 1:
        multipliers = multipliers * channels
    if len(shifts) == 1:
        shifts = shifts * channels
    return max(
        abs(bias) * (multiplier / 2**31) * (2.0**shift)
        for bias, multiplier, shift in zip(biases, multipliers, shifts)
    )


# One per int kernel family that takes a bias array: per-channel s8, the
# s16 int64-bias path, and the s4 packed-weight path.
@pytest.mark.parametrize(
    "case_name",
    [
        "convolve_default_s8",
        "convolve_one_by_n_case_03_s8",
        "convolve_kernel_support_groups2_s8",
        "convolve_kernel_support_s16",
        "convolve_requantize_s64_s16",
        "convolve_generic_s4",
    ],
)
def test_int_convolve_bias_is_nonzero_and_detectable(case_name: str, tmp_path: Path) -> None:
    header = _generate(case_name, tmp_path)
    biases = _int_array(header, "biases")

    assert biases, f"{case_name} emits no bias array"
    assert any(bias != 0 for bias in biases), f"{case_name} ships an all-zero bias"
    assert _output_steps(header) >= 1.0, (
        f"{case_name} bias is below one output quantization step, so dropping "
        f"the bias-add reproduces the golden exactly"
    )


@pytest.mark.parametrize(
    "case_name",
    [
        "fully_connected_default_s16",
        "fully_connected_tail_rows5_cols13_bias_s4",
    ],
)
def test_int_fully_connected_bias_is_nonzero_and_detectable(case_name: str, tmp_path: Path) -> None:
    header = _generate(case_name, tmp_path)
    biases = _int_array(header, "biases")

    assert biases, f"{case_name} emits no bias array"
    assert any(bias != 0 for bias in biases), f"{case_name} ships an all-zero bias"
    assert _output_steps(header) >= 1.0, (
        f"{case_name} bias is below one output quantization step, so dropping "
        f"the bias-add reproduces the golden exactly"
    )


def test_s8_fully_connected_bias_reaches_the_kernel_via_the_weight_sum(tmp_path: Path) -> None:
    """The s8 FC kernel takes bias folded into the precomputed kernel sum.

    ``arm_vector_sum_s8`` accumulates the bias into the per-row sum and the
    kernel is then called with a NULL bias pointer, so a nonzero bias shows up
    in ``_weight_sum`` rather than in ``_biases``.
    """
    from helia_core_tester.generation.ops.ConvolutionFunctions.depthwise_conv import vector_sum_s8

    header = _generate("fully_connected_mve_case_01_s8", tmp_path)

    assert _int_array(header, "biases") is None
    emitted = _int_array(header, "weight_sum")
    assert emitted, "s8 FC case emits no precomputed weight sum"

    rows = len(emitted)
    weights = np.asarray(_int_array(header, "weights"), dtype=np.int8).reshape(rows, -1)
    bias_free = vector_sum_s8(
        vector_data=weights,
        vector_cols=weights.shape[1],
        vector_rows=rows,
        lhs_offset=int(re.search(r"\.input_offset\s*=\s*(-?\d+)", header).group(1)),
        rhs_offset=int(re.search(r"\.filter_offset\s*=\s*(-?\d+)", header).group(1)),
        bias_data=None,
    ).astype(np.int64)

    folded_bias = np.asarray(emitted, dtype=np.int64) - bias_free
    assert np.all(folded_bias != 0), "s8 FC weight sum carries no bias contribution"


@pytest.mark.parametrize("case_name", ["convolve_2x2_dilation_s8", "convolve_int16xint8_dilation_case_01_s16"])
def test_quantized_dilated_convolve_keeps_a_zero_bias(case_name: str, tmp_path: Path) -> None:
    """Known gap, deliberately not fixed here.

    A quantized dilated conv lowers to SpaceToBatchND -> Conv2D ->
    BatchToSpaceND -> Add.  The bias ends up in the trailing Add at output
    quantization scale rather than in the CONV_2D op at accumulator scale, so
    the generated case cannot carry it.  Guard the carve-out so it stays
    visible instead of quietly widening.
    """
    header = _generate(case_name, tmp_path)
    biases = _int_array(header, "biases")

    assert biases is not None
    assert all(bias == 0 for bias in biases)
