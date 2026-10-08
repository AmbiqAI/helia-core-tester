"""Shared helpers for binary basic-math operators."""

from typing import Any, Dict, Optional

import numpy as np

from helia_core_tester.generation.entry import check_entry_fault, resolve_entry
from helia_core_tester.generation.ops._shared.base import OperationBase


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

    @staticmethod
    def _requantize_np(values: np.ndarray, multiplier: int, shift: int) -> np.ndarray:
        left_shift = shift if shift > 0 else 0
        right_shift = -shift if shift < 0 else 0
        prod = values.astype(np.int64) * (1 << left_shift)
        mult = (1 << 30) + (prod * int(multiplier))
        res = (mult >> 31).astype(np.int64)
        if right_shift == 0:
            return res.astype(np.int32)
        remainder_mask = (1 << right_shift) - 1
        remainder = res & remainder_mask
        result = res >> right_shift
        threshold = remainder_mask >> 1
        threshold = threshold + (result < 0)
        result = result + (remainder > threshold)
        return result.astype(np.int32)

