"""Int-suite Convolve/FullyConnected cases must ship a bias a kernel can miss.

An all-zero bias tensor makes a dropped bias-add bit-identical to a correct
kernel, so these cases assert both that the bias is nonzero and that it is
large enough to move at least one output element by a full output
quantization step.  See AmbiqAI/helia-core-tester#77.
"""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest

from helia_core_tester.generation.ops.ConvolutionFunctions import convolve as convolve_module
from helia_core_tester.generation.io.descriptors import load_all_descriptors
from helia_core_tester.generation.ops.ConvolutionFunctions.convolve import OpConvolve
from helia_core_tester.generation.ops.ConvolutionFunctions.depthwise_conv import OpDepthwiseConv
from helia_core_tester.generation.ops.FullyConnectedFunctions.fully_connected import OpFullyConnected

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SEED = 1
# An MVE target and a non-MVE one: the s8 FC bias is folded into the
# precomputed kernel sum only on the former.
_TARGETS = ("cortex-m55", "cortex-m4")


@lru_cache(maxsize=1)
def _all_descriptors() -> Tuple[Dict, ...]:
    """Every case below needs the whole descriptor tree, which cannot change
    mid-run; parsing it once keeps this module's runtime in the generation work
    rather than in YAML."""
    return tuple(load_all_descriptors(str(_PROJECT_ROOT / "assets" / "descriptors")))


def _pipeline_default_seed(name: str) -> int:
    """The seed the generation pipeline derives when none is passed on the
    command line (see generation/test_ops.py)."""
    return int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:4], "little")


def _descriptor(name: str) -> Dict:
    for desc in _all_descriptors():
        if desc.get("name") == name:
            return desc
    raise AssertionError(f"descriptor {name} not found")


def _generate(name: str, out_dir: Path, target_cpu: str, seed: int = _SEED) -> str:
    """Generate one case for one target and return its emitted header text."""
    out_dir.mkdir(parents=True, exist_ok=True)
    desc = _descriptor(name)
    op_cls = {
        "Convolve": OpConvolve,
        "DepthwiseConv": OpDepthwiseConv,
    }.get(desc["operator"], OpFullyConnected)
    op = op_cls(desc, seed=seed, target_cpu=target_cpu)
    model = op.build_keras_model() if op.needs_keras_model() else None
    op.convert_to_tflite(model, str(out_dir / f"{name}.tflite"), seed)
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


def _output_steps(header: str) -> List[float]:
    """Per-channel output shift, in output quantization steps, a dropped bias causes."""
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
    return [
        abs(bias) * (multiplier / 2**31) * (2.0**shift)
        for bias, multiplier, shift in zip(biases, multipliers, shifts)
    ]


def _bias_carrying_int_cases(*, weight_dtype_s4: bool) -> List[str]:
    """Names of the int Convolve/FullyConnected cases whose bias must be detectable.

    Derived from the descriptors rather than enumerated so a new case is held to
    the floor the moment it is added.  Float cases have no quantization step to
    clear and use_bias: false cases have no bias; quantized dilated convs are
    included, their bias being written into the lowered CONV_2D placeholder
    after conversion rather than coming from the Keras model.
    """
    names: List[str] = []
    for desc in _all_descriptors():
        if desc.get("operator") not in ("Convolve", "FullyConnected"):
            continue
        if str(desc.get("activation_dtype", "")).upper() in ("FP32", "FP16"):
            continue
        if not desc.get("use_bias", True):
            continue
        if (str(desc.get("weight_dtype", "S8")).upper() == "S4") is not weight_dtype_s4:
            continue
        names.append(desc["name"])
    assert names, "descriptor sweep found no bias-carrying int cases"
    return names


# The per-channel s8 and s16 int64-bias paths are built from a Keras model whose
# bias comes from SignedMagnitudeUniform, so every channel clears the floor.
@pytest.mark.parametrize("case_name", _bias_carrying_int_cases(weight_dtype_s4=False))
def test_int_bias_is_detectable_on_every_channel(case_name: str, tmp_path: Path) -> None:
    header = _generate(case_name, tmp_path, "cortex-m55")
    biases = _int_array(header, "biases")

    assert biases, f"{case_name} emits no bias array"
    assert all(bias != 0 for bias in biases), f"{case_name} ships a zero bias channel"
    assert min(_output_steps(header)) >= 1.0, (
        f"{case_name} has a channel whose bias is below one output quantization "
        f"step, so dropping the bias-add reproduces that channel exactly"
    )


# The s4 path builds its tensors with LiteRT directly and draws the bias from
# rng.integers(-128, 128), which can hand an individual channel a zero: only
# the case as a whole is guaranteed to detect a dropped bias-add.
@pytest.mark.parametrize("case_name", _bias_carrying_int_cases(weight_dtype_s4=True))
def test_s4_bias_is_detectable_on_at_least_one_channel(case_name: str, tmp_path: Path) -> None:
    header = _generate(case_name, tmp_path, "cortex-m55")
    biases = _int_array(header, "biases")

    assert biases, f"{case_name} emits no bias array"
    assert any(bias != 0 for bias in biases), f"{case_name} ships an all-zero bias"
    assert max(_output_steps(header)) >= 1.0, (
        f"{case_name} bias is below one output quantization step on every channel, "
        f"so dropping the bias-add reproduces the golden exactly"
    )


# The Convolve and FullyConnected cases with the narrowest measured margin over
# one output quantization step. The bias floor is a property of the case, not of
# the seed a run happens to use, so hold the tightest two at the seed the rest of
# this module pins and at the one the pipeline derives by default.
@pytest.mark.parametrize("case_name", ["convolve_one_by_n_case_07_s8", "fully_connected_default_s8"])
@pytest.mark.parametrize("seed", ["pinned", "pipeline_default"])
def test_tightest_int_bias_margins_survive_the_pipeline_default_seed(
    case_name: str, seed: str, tmp_path: Path
) -> None:
    seed_value = _SEED if seed == "pinned" else _pipeline_default_seed(case_name)
    header = _generate(case_name, tmp_path, "cortex-m55", seed=seed_value)
    biases = _int_array(header, "biases")

    assert biases, f"{case_name} emits no bias array at seed {seed_value}"
    assert all(bias != 0 for bias in biases), (
        f"{case_name} ships a zero bias channel at seed {seed_value}"
    )
    assert min(_output_steps(header)) >= 1.0, (
        f"{case_name} has a channel whose bias is below one output quantization "
        f"step at seed {seed_value}, so dropping the bias-add reproduces that "
        f"channel exactly"
    )


@pytest.mark.parametrize("target_cpu", _TARGETS)
@pytest.mark.parametrize("case_name", ["fully_connected_default_s16", "fully_connected_mve_case_01_s8"])
def test_int_fully_connected_bias_is_detectable_on_every_channel(
    case_name: str, target_cpu: str, tmp_path: Path
) -> None:
    header = _generate(case_name, tmp_path, target_cpu)
    biases = _int_array(header, "biases")

    assert biases, f"{case_name} emits no bias array on {target_cpu}"
    assert all(bias != 0 for bias in biases), f"{case_name} ships a zero bias channel"
    assert min(_output_steps(header)) >= 1.0, (
        f"{case_name} has a channel whose bias is below one output quantization "
        f"step, so dropping the bias-add reproduces that channel exactly"
    )


@pytest.mark.parametrize("target_cpu", _TARGETS)
def test_s8_fully_connected_weight_sum_is_gated_on_mve(target_cpu: str, tmp_path: Path) -> None:
    """Only an MVE build reads the precomputed kernel sum.

    ``arm_nn_vec_mat_mult_t_s8`` consults the sum in ``ctx->buf`` under
    ``ARM_MATH_MVEI`` and the bias pointer everywhere else, so a case that
    folds its bias into the sum on a non-MVE target loses the bias-add.
    """
    header = _generate("fully_connected_mve_case_01_s8", tmp_path, target_cpu)

    if target_cpu == "cortex-m55":
        assert _int_array(header, "weight_sum"), "MVE target emits no precomputed weight sum"
    else:
        assert _int_array(header, "weight_sum") is None, (
            "non-MVE target emits a precomputed weight sum the kernel never reads"
        )


def test_s8_fully_connected_bias_reaches_the_kernel_via_the_weight_sum(tmp_path: Path) -> None:
    """The s8 FC kernel takes bias folded into the precomputed kernel sum.

    ``arm_vector_sum_s8`` accumulates the bias into the per-row sum and the
    kernel is then called with a NULL bias pointer.  The header still declares
    the bias array, because the perf-stream bridge rebuilds the sum on device
    from it, so the two have to agree.
    """
    from helia_core_tester.generation.ops.ConvolutionFunctions.depthwise_conv import vector_sum_s8

    header = _generate("fully_connected_mve_case_01_s8", tmp_path, "cortex-m55")

    emitted = _int_array(header, "weight_sum")
    assert emitted, "s8 FC case emits no precomputed weight sum"
    declared_bias = _int_array(header, "biases")
    assert declared_bias, "s8 FC case declares no bias array for the bridge to read"

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
    assert np.array_equal(folded_bias, np.asarray(declared_bias, dtype=np.int64)), (
        "declared bias array disagrees with the bias folded into the weight sum"
    )


# The dilated lowering also covers DepthwiseConv, whose non-dilated cases draw
# their bias from a plain uniform and are not held to the per-channel floor.
@pytest.mark.parametrize(
    "case_name",
    ["depthwise_conv_dilation_s8", "depthwise_conv_dilation_s16"],
)
def test_dilated_depthwise_bias_is_detectable_on_every_channel(
    case_name: str, tmp_path: Path
) -> None:
    header = _generate(case_name, tmp_path, "cortex-m55")
    biases = _int_array(header, "biases")

    assert biases, f"{case_name} emits no bias array"
    assert all(bias != 0 for bias in biases), f"{case_name} ships a zero bias channel"
    assert min(_output_steps(header)) >= 1.0, (
        f"{case_name} has a channel whose bias is below one output quantization step"
    )


@pytest.mark.parametrize(
    "case_name",
    ["convolve_2x2_dilation_s8", "convolve_int16xint8_dilation_case_01_s16"],
)
def test_dilated_golden_moves_with_the_injected_bias(
    case_name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The golden has to be recomputed from the model the bias was written into.

    A quantized dilated conv lowers to SpaceToBatchND -> Conv2D ->
    BatchToSpaceND, and both the emitted bias tensor and the golden are read
    off that CONV_2D. If the injected bias reached the harness without the
    golden moving with it, every one of these cases would fail on hardware for
    a kernel that is behaving correctly.
    """
    with_bias = _generate(case_name, tmp_path / "with_bias", "cortex-m55")

    monkeypatch.setattr(convolve_module, "inject_hoisted_dilation_bias", lambda *_: False)
    without_bias = _generate(case_name, tmp_path / "without_bias", "cortex-m55")

    assert all(bias == 0 for bias in _int_array(without_bias, "biases") or [])
    assert any(bias != 0 for bias in _int_array(with_bias, "biases") or [])
    assert _int_array(with_bias, "expected_output") != _int_array(
        without_bias, "expected_output"
    ), f"{case_name} golden is unchanged by the injected bias"
