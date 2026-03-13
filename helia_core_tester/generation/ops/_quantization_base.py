"""Shared helpers for quantization-family operators."""

from helia_core_tester.generation.ops.base import OperationBase


class QuantizationFamilyBase(OperationBase):
    """Shared descriptor helpers for quantize/dequantize/requantize ops."""

    def has_relu_activation(self) -> bool:
        return self.activation_name() in {"RELU", "RELU6"}

