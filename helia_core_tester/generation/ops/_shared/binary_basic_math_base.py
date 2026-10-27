"""Shared helpers for binary basic-math operators."""

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from helia_core_tester.generation.entry import check_entry_fault, resolve_entry
from helia_core_tester.generation.ops._shared.base import OperationBase
from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

TensorQuant = Tuple[Sequence[float], Sequence[int]]
QuantPresets = Mapping[str, Mapping[str, Mapping[str, TensorQuant]]]

# Every tensor takes the builder's default quantization.
SHARED_QUANT_PRESET = "shared"
_PRESET_TENSORS = ("input_1_quant", "input_2_quant", "output_quant")


def resolve_quant_preset(
    operator: str,
    presets: QuantPresets,
    quant_preset: Any,
    activation_dtype: str,
) -> Dict[str, TensorQuant]:
    """Per-tensor quantization for `quant_preset`; empty for the shared default.

    An unknown preset, or a preset that has no quantization for this activation
    dtype (float cases included), is a descriptor error.
    """
    if quant_preset is None or quant_preset == SHARED_QUANT_PRESET:
        return {}
    if not isinstance(quant_preset, str) or quant_preset not in presets:
        known = sorted({SHARED_QUANT_PRESET, *presets})
        raise ValueError(f"{operator} has no quant_preset {quant_preset!r}; known: {known}")
    by_dtype = presets[quant_preset]
    if activation_dtype not in by_dtype:
        raise ValueError(
            f"{operator} quant_preset {quant_preset!r} has no {activation_dtype} quantization; "
            f"it applies to {sorted(by_dtype)}"
        )
    tensors = by_dtype[activation_dtype]
    if set(tensors) != set(_PRESET_TENSORS):
        raise ValueError(f"{operator} quant_preset {quant_preset!r} must define {list(_PRESET_TENSORS)}")
    return dict(tensors)


def operand_quant(preset: Mapping[str, TensorQuant], default: Tuple[float, int]) -> Dict[str, Tuple[float, int]]:
    """Per-tensor (scale, zero_point) of input1, input2 and output: the preset's, else `default` for all three."""
    if not preset:
        return {role: default for role in ("input1", "input2", "output")}
    quant: Dict[str, Tuple[float, int]] = {}
    for role, key in (("input1", "input_1_quant"), ("input2", "input_2_quant"), ("output", "output_quant")):
        scales, zero_points = preset[key]
        if len(scales) != 1 or len(zero_points) != 1:
            raise ValueError(f"{key} must be per-tensor (one scale and one zero point), got {preset[key]!r}")
        quant[role] = (float(scales[0]), int(zero_points[0]))
    return quant


_CMSIS_DIMS_RANK = 4


def broadcast_cmsis_dims(
    operator: str,
    input1_shape: Sequence[int],
    input2_shape: Sequence[int],
    output_shape: Sequence[int],
) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
    """cmsis_nn_dims for a broadcasting binary op, each shape right-aligned to NHWC.

    The kernels broadcast dim by dim, so a lower-rank operand must be padded with
    leading ones (NumPy/TFLite rules); nhwc_to_cmsis_dims alone reads rank 2 as HW.
    Shapes outside rank 1..4, non-positive extents, or an output that is not the
    broadcast of the inputs are descriptor errors.
    """
    shapes = {"input_1": input1_shape, "input_2": input2_shape, "output": output_shape}
    padded = {}
    for role, shape in shapes.items():
        if shape is None:
            raise ValueError(f"{operator} {role} has no shape")
        dims = tuple(shape)
        if not 1 <= len(dims) <= _CMSIS_DIMS_RANK or any(
            isinstance(d, bool) or not isinstance(d, (int, np.integer)) or d < 1 for d in dims
        ):
            raise ValueError(f"{operator} {role} shape {list(dims)} must be positive extents of rank 1 through 4")
        padded[role] = (1,) * (_CMSIS_DIMS_RANK - len(dims)) + tuple(int(d) for d in dims)
    try:
        expected = np.broadcast_shapes(tuple(input1_shape), tuple(input2_shape))
    except ValueError as exc:
        raise ValueError(
            f"{operator} inputs {list(input1_shape)} and {list(input2_shape)} do not broadcast"
        ) from exc
    if tuple(expected) != tuple(int(d) for d in output_shape):
        raise ValueError(
            f"{operator} output shape {list(output_shape)} is not the broadcast {list(expected)} of its inputs"
        )
    to_dims = TemplateContextBuilder.nhwc_to_cmsis_dims
    return to_dims(padded["input_1"]), to_dims(padded["input_2"]), to_dims(padded["output"])


class BinaryBasicMathBase(OperationBase):
    """Shared helpers for Add, Sub, Mul, Maximum, and Minimum."""

    # `hint: call_style: broadcast` selects the dims-taking float entry point
    # (arm_elementwise_{sub,add,mul}_broadcast_{f32,f16}, ns-cmsis-nn#415).
    FLOAT_BROADCAST_CALL_STYLE = "broadcast"

    # s8 draw reach, in quantized units.
    S8_REACH = 128

    def _widen_s8(self, unit: np.ndarray, scale: float, c_type: str) -> np.ndarray:
        """Stretch [-1, 1] draws over s8."""
        if c_type != "int8_t":
            return unit
        return unit * np.float32(self.S8_REACH * float(scale))

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Fail at load for an entry this operator does not call, rather than gating the case on it.
        self._direct_entry_kernel()

    def _float_broadcast_call(self, *, auto_on_shape_mismatch: bool) -> bool:
        """Return True when the float path must call the broadcast entry point.

        The hint is the explicit switch. `auto_on_shape_mismatch` lets an operator
        whose flat float path cannot represent two shapes at all (sub) take the
        broadcast kernel from the shapes alone; add and mul keep their pre-#415
        materialised-broadcast flat call for un-hinted mismatched shapes, since
        descriptors already pin that behaviour.
        """
        call_style = str(self.desc.get("hint", {}).get("call_style", "")).strip().lower()
        if call_style == self.FLOAT_BROADCAST_CALL_STYLE:
            return True
        if not auto_on_shape_mismatch:
            return False
        shape_1 = self.desc.get("input_1_shape")
        shape_2 = self.desc.get("input_2_shape")
        if shape_1 is None or shape_2 is None:
            return False
        return tuple(shape_1) != tuple(shape_2)

    def _direct_entry_kernel(self) -> Optional[Dict[str, Any]]:
        """Kernel info for an `entry:` case: a public kernel of this operator, bound from the kernel
        contract with the router's values (the s8 row-broadcast entries take the router's arguments)."""
        entry = self.desc.get("entry")
        if not entry:
            return None
        dtype = self.tensor_dtype("input")
        resolved = resolve_entry(
            str(self.desc.get("operator")),
            str(entry),
            activation_dtype=dtype,
            weight_dtype=dtype,
            cpu=self.target_cpu,
            desc=self.desc,
            extra_roles={"input_1": dtype, "input_2": dtype},
        )
        check_entry_fault(self.desc, resolved)
        c_type = self.tensor_c_type("input")
        return {
            "kernel_fn": resolved["kernel_fn"],
            "input_c_type": c_type,
            "output_c_type": c_type,
            "float_kernel": dtype in ("FP32", "FP16"),
        }
