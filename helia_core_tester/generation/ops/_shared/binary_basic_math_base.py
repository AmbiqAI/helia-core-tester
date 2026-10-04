"""Shared helpers for binary basic-math operators."""

from typing import Any, Dict, Optional

import numpy as np

from helia_core_tester.generation.kernel_dispatch import check_entry_fault, resolve_direct_entry
from helia_core_tester.generation.ops._shared.base import OperationBase


class BinaryBasicMathBase(OperationBase):
    """Shared helpers for Add, Sub, Mul, Maximum, and Minimum."""

    # `hint: call_style: broadcast` selects the dims-taking float entry point
    # (arm_elementwise_{sub,add,mul}_broadcast_{f32,f16}, ns-cmsis-nn#415).
    FLOAT_BROADCAST_CALL_STYLE = "broadcast"

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
        """Kernel info for an `entry:` case, which calls an s8 entry with its router's arguments."""
        entry = self.desc.get("entry")
        if not entry:
            return None
        resolved = resolve_direct_entry(
            str(self.desc.get("operator")),
            str(entry),
            self.tensor_dtype("input"),
            self.desc.get("weight_dtype", "S8"),
        )
        check_entry_fault(self.desc, resolved)
        return {
            "kernel_fn": resolved["kernel_fn"],
            "input_c_type": "int8_t",
            "output_c_type": "int8_t",
            "float_kernel": False,
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

