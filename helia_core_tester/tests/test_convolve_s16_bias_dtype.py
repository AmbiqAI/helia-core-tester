"""``bias_dtype`` pins the S16 Convolve bias width for both the golden and the kernel call.

The default S16 corpus passes an int64 bias, so the int32-only MVE paths in
arm_convolve_1x1_s16_ns_np_nd and arm_convolve_s16_group_ch_mult_1
(ns-cmsis-nn#540, #541) were never reached.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pytest

from helia_core_tester.generation.io.descriptors import load_all_descriptors
from helia_core_tester.generation.ops.ConvolutionFunctions.convolve import OpConvolve

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CPU = "cortex-m55"

_BASE_S16: Dict = {
    "operator": "Convolve",
    "name": "convolve_bias_dtype_probe_s16",
    "activation_dtype": "S16",
    "weight_dtype": "S8",
    "input_shape": [1, 2, 3, 8],
    "filter_shape": [1, 1, 8, 5],
    "strides": [1, 1],
    "padding": "VALID",
    "use_bias": True,
}


def _descriptor(name: str) -> Dict:
    for desc in load_all_descriptors(str(_PROJECT_ROOT / "assets" / "descriptors")):
        if desc.get("name") == name:
            return desc
    raise AssertionError(f"descriptor {name} not found")


def _generate(op: OpConvolve, out_dir: Path) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    name = op.desc["name"]
    op.convert_to_tflite(op.build_keras_model(), str(out_dir / f"{name}.tflite"), 1)
    op.generate_c_files(out_dir)
    return "\n".join(path.read_text() for path in sorted(out_dir.rglob(f"{name}*")) if path.suffix in {".c", ".h"})


def test_default_s16_bias_stays_int64() -> None:
    op = OpConvolve(dict(_BASE_S16), seed=1, target_cpu=_CPU)
    assert op._select_cmsis_convolve_kernel()["bias_c_type"] == "int64_t"


@pytest.mark.parametrize("value, expected", [("S32", "int32_t"), ("s32", "int32_t"), ("S64", "int64_t")])
def test_bias_dtype_pins_kernel_bias_type(value: str, expected: str) -> None:
    op = OpConvolve({**_BASE_S16, "bias_dtype": value}, seed=1, target_cpu=_CPU)
    assert op._select_cmsis_convolve_kernel()["bias_c_type"] == expected


@pytest.mark.parametrize("value", ["S16", "S8", "int32", "", 32])
def test_bias_dtype_rejects_unknown_width(value) -> None:
    op = OpConvolve({**_BASE_S16, "bias_dtype": value}, seed=1, target_cpu=_CPU)
    with pytest.raises(ValueError, match="unsupported bias_dtype"):
        op._select_cmsis_convolve_kernel()


@pytest.mark.parametrize(
    "activation_dtype, weight_dtype",
    [("S8", "S8"), ("S8", "S4"), ("FP32", "FP32"), ("FP16", "FP16")],
)
def test_bias_dtype_rejected_outside_s16(activation_dtype: str, weight_dtype: str) -> None:
    desc = {
        **_BASE_S16,
        "name": "convolve_bias_dtype_probe_s8",
        "activation_dtype": activation_dtype,
        "weight_dtype": weight_dtype,
        "bias_dtype": "S32",
    }
    op = OpConvolve(desc, seed=1, target_cpu=_CPU)
    with pytest.raises(ValueError, match="only supported for S16 x S8"):
        op._select_cmsis_convolve_kernel()
    with pytest.raises(ValueError, match="only supported for S16 x S8"):
        op.convert_to_tflite(None, "/nonexistent/never-written.tflite", 1)


@pytest.mark.parametrize(
    "name",
    [
        "convolve_int32_bias_1x1_in8_out7_s16",
        "convolve_int32_bias_group_ch_mult1_k3x3_w12_s16",
    ],
)
def test_int32_bias_case_emits_int32_bias_end_to_end(name: str, tmp_path: Path) -> None:
    text = _generate(OpConvolve(_descriptor(name), seed=1, target_cpu=_CPU), tmp_path)
    assert f"static const int32_t {name}_biases[]" in text
    assert ".is_int32_bias = true" in text
    assert "int64_t" not in text


def test_bias_width_mismatch_between_golden_and_kernel_fails(tmp_path: Path) -> None:
    """Fault injection: a model converted with the default int64 bias must not be
    silently narrowed into an int32 kernel call."""
    desc = _descriptor("convolve_int32_bias_1x1_in8_out7_s16")
    name = desc["name"]
    unpinned = OpConvolve({k: v for k, v in desc.items() if k != "bias_dtype"}, seed=1, target_cpu=_CPU)
    unpinned.convert_to_tflite(unpinned.build_keras_model(), str(tmp_path / f"{name}.tflite"), 1)

    pinned = OpConvolve(desc, seed=1, target_cpu=_CPU)
    with pytest.raises(ValueError, match="bias_dtype pins int32_t but the converted model carries a int64 bias"):
        pinned.generate_c_files(tmp_path)
