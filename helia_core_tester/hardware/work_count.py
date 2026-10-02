"""Per-case work counts (MACs, ops) from the case bundle.

Counts follow the usual TFLite/TFLM conventions so cycles/MAC compares
across tools: a MAC kernel counts one MAC per multiply-accumulate of the
dense computation (padding taps included) and two ops per MAC; a pooling
kernel counts one op per window tap; an elementwise, activation,
comparison, quantization, softmax or reduction kernel counts one op per
element it reads. Data-movement kernels (pad, transpose, gather, ...)
have no standard op count and report none.
"""

from __future__ import annotations

from math import prod
from typing import Any

from .case_bundle import CaseBundle

# Output elements times the reduction length.
_MAC_OPERATORS = frozenset({"Convolve", "DepthwiseConv", "FullyConnected", "TransposeConv", "BatchMatMul"})
_POOL_OPERATORS = frozenset({"AvgPool", "MaxPool"})
# One op per input element.
_REDUCE_OPERATORS = frozenset({"ArgMax", "ArgMin", "Mean", "ReduceMax", "ReduceMin", "ReduceSum"})
# One op per output element.
_ELEMENT_OPERATORS = frozenset({
    "Abs", "Add", "Sub", "Mul", "Maximum", "Minimum", "Rsqrt", "RsqrtUniversal", "Sqrt", "SquaredDifference",
    "Clamp", "HardSwishCompat", "HardSwishPrecise", "LeakyRelu", "Logistic", "NNActivationFloat", "PReLU",
    "PReLUScalar", "Relu", "Relu6", "Tanh",
    "Equal", "NotEqual", "Greater", "GreaterEqual", "Less", "LessEqual",
    "Quantize", "Dequantize", "Requantize", "BatchNorm", "Softmax", "SoftmaxS8S16", "SelectV2", "Where",
})


def _dims(bundle: CaseBundle, role: str) -> tuple[int, ...] | None:
    return next((blob.dimensions for blob in bundle.blobs if blob.role == role), None)


def _case_macs(bundle: CaseBundle, operator: str, output: tuple[int, ...] | None, scalars: dict[str, Any]) -> int | None:
    """Dense MACs, or None if shapes are missing."""
    if operator == "BatchMatMul":
        lhs = _dims(bundle, "input_0")
        if output is None or lhs is None:
            return None
        return prod(output) * (lhs[-2] if scalars.get("adj_x") else lhs[-1])
    weights = _dims(bundle, "weights")
    if output is None or weights is None:
        return None
    if operator == "TransposeConv":
        # Each input element scatters a kernel.
        source = _dims(bundle, "input_0")
        return None if source is None else prod(source) * prod(weights) // source[-1]
    # Conv, depthwise, FC: weights per output channel.
    return prod(output) * prod(weights) // output[-1]


def case_work(bundle: CaseBundle) -> dict[str, int | None]:
    """`macs` and `ops` for one case; None where undefined."""
    operator = str(bundle.manifest.get("operator", ""))
    scalars: dict[str, Any] = bundle.manifest.get("serialized_scalar_parameters", {})
    output = _dims(bundle, "expected_output")
    macs = ops = None
    if operator in _MAC_OPERATORS:
        macs = _case_macs(bundle, operator, output, scalars)
        ops = None if macs is None else 2 * macs
    elif operator in _POOL_OPERATORS and output is not None and "pool_h" in scalars:
        ops = prod(output) * int(scalars["pool_h"]) * int(scalars["pool_w"])
    elif operator in _REDUCE_OPERATORS and (source := _dims(bundle, "input_0")) is not None:
        ops = prod(source)
    elif operator in _ELEMENT_OPERATORS and output is not None:
        ops = prod(output)
    return {"macs": macs, "ops": ops}


def per_unit(cycles: float | None, units: int | None) -> float | None:
    """`cycles / units` to 4 places, or None."""
    if cycles is None or not units:
        return None
    return round(cycles / units, 4)
