"""Shared helpers for binary basic-math operators."""

import numpy as np

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

