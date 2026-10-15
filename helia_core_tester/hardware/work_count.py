"""Per-case work counts (MACs, ops) from the case bundle.

Counts follow the usual TFLite/TFLM conventions so cycles/MAC compares
across tools: a MAC kernel counts one MAC per multiply-accumulate of the
dense computation (padding taps included) and two ops per MAC (LSTM and
SVDF count their gate and feature/time matrix products only); a pooling
kernel counts one op per window tap; an elementwise, activation,
comparison, quantization, softmax or reduction kernel counts one op per
element it reads. Data-movement kernels (pad, transpose, gather, ...)
have no standard op count and report none.
"""

from __future__ import annotations

from math import isqrt, prod
from typing import Any

from .case_bundle import CaseBundle

# Output elements times the reduction length.
_MAC_OPERATORS = frozenset({
    "Convolve", "DepthwiseConv", "FullyConnected", "TransposeConv", "BatchMatMul", "LSTMUnidirectional", "SVDF",
})
_POOL_OPERATORS = frozenset({"AvgPool", "MaxPool"})
# One op per input element.
_REDUCE_OPERATORS = frozenset({"ArgMax", "ArgMin", "Mean", "ReduceMax", "ReduceMin", "ReduceSum"})
# One op per output element.
_ELEMENT_OPERATORS = frozenset({
    "Abs", "Add", "Sub", "Mul", "Maximum", "Minimum", "Rsqrt", "RsqrtUniversal", "Sqrt", "SquaredDifference",
    "Clamp", "HardSwishCompat", "HardSwishPrecise", "LeakyRelu", "Logistic", "NNActivationFloat", "PReLU",
    "PReLUScalar", "Relu", "Relu6", "Tanh",
    "Equal", "NotEqual", "Greater", "GreaterEqual", "Less", "LessEqual",
    "Quantize", "Dequantize", "Requantize", "BatchNorm", "Softmax", "SoftmaxS8S16", "SelectV2",
})


def _dims(bundle: CaseBundle, role: str) -> tuple[int, ...] | None:
    return next((blob.dimensions for blob in bundle.blobs if blob.role == role), None)


def _lstm_macs(bundle: CaseBundle, output: tuple[int, ...] | None) -> int | None:
    """Four gates, each an input and a recurrent product, per batch and step.

    The bridge sends flat blobs: input B*T*I, weights 4*H*(I+H), output B*T*H,
    so H**2 = weights*output / (4*(input+output)). Inconsistent sizes give None.
    """
    source = _dims(bundle, "input_0")
    weights = _dims(bundle, "weights")
    if source is None or weights is None or output is None:
        return None
    inputs, outputs, packed = prod(source), prod(output), prod(weights)
    if min(inputs, outputs, packed) <= 0:
        return None
    hidden_sq, remainder = divmod(packed * outputs, 4 * (inputs + outputs))
    hidden = isqrt(hidden_sq)
    if remainder or hidden * hidden != hidden_sq or outputs % hidden or (inputs * hidden) % outputs:
        return None
    if 4 * hidden * (inputs * hidden // outputs + hidden) != packed:
        return None
    return packed * (outputs // hidden)


def _svdf_macs(bundle: CaseBundle) -> int | None:
    """Feature (F x I) and time (F x M) products per batch.

    Shapes: input (B, I, ...), weights_feature (F, I, ...), weights_time in
    input_2 (F, M, ...). The rank sum is additions only.
    """
    source = _dims(bundle, "input_0")
    feature = _dims(bundle, "weights")
    time = _dims(bundle, "input_2")
    if source is None or feature is None or time is None or min(len(source), len(feature), len(time)) < 2:
        return None
    if feature[1] != source[1] or time[0] != feature[0] or min(source[0], prod(feature), prod(time)) <= 0:
        return None
    return source[0] * (prod(feature) + prod(time))


def _case_macs(bundle: CaseBundle, operator: str, output: tuple[int, ...] | None, scalars: dict[str, Any]) -> int | None:
    """Dense MACs, or None if shapes are missing."""
    if operator == "LSTMUnidirectional":
        return _lstm_macs(bundle, output)
    if operator == "SVDF":
        return _svdf_macs(bundle)
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
