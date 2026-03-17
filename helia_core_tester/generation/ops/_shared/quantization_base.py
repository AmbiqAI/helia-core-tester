"""Shared helpers for quantization-family operators."""

import numpy as np

from helia_core_tester.generation.ops._shared.base import OperationBase


class QuantizationFamilyBase(OperationBase):
    """Shared descriptor helpers for quantize/dequantize/requantize ops."""

    def has_relu_activation(self) -> bool:
        return self.activation_name() in {"RELU", "RELU6"}

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
