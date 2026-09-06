"""Bias enrichment for operators whose Keras model feeds the quantized pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import keras
import numpy as np

# Output quantization steps a hoisted-bias case's injected bias is worth per
# channel. The floor clears the Convolve family's 1 LSB comparison tolerance
# with margin so a dropped bias-add cannot hide inside it, and the ceiling
# keeps the bias a few percent of the int8 output range, which the conv output
# tensor's scale was calibrated without.
_DILATION_BIAS_MIN_STEPS = 3.0
_DILATION_BIAS_MAX_STEPS = 8.0


class SignedMagnitudeUniform(keras.initializers.Initializer):
    """Seed-derived values with ``|value|`` uniform in ``[minval, maxval]``.

    A plain uniform over ``[-limit, limit]`` can draw arbitrarily close to
    zero, and a bias element below one output quantization step is
    indistinguishable from no bias at all in the golden. Sampling the
    magnitude and the sign separately keeps every channel above the
    detection floor, which matters most for single-output-channel cases
    where there is no other channel to carry the signal.

    Args:
        minval: Smallest absolute value produced. Must be > 0.
        maxval: Largest absolute value produced. Must be >= ``minval``.
        seed: Seed for the generator; identical seeds give identical tensors.
    """

    def __init__(self, minval: float, maxval: float, seed: Optional[int] = None):
        if minval <= 0:
            raise ValueError(f"minval must be positive, got {minval}")
        if maxval < minval:
            raise ValueError(f"maxval ({maxval}) must be >= minval ({minval})")
        self.minval = float(minval)
        self.maxval = float(maxval)
        self.seed = seed

    def __call__(self, shape, dtype=None):
        rng = np.random.default_rng(self.seed)
        shape = tuple(int(dim) for dim in shape)
        magnitude = rng.uniform(self.minval, self.maxval, size=shape)
        signs = rng.choice((-1.0, 1.0), size=shape)
        return keras.ops.convert_to_tensor(
            (magnitude * signs).astype(np.float32), dtype=dtype or "float32"
        )

    def get_config(self) -> Dict[str, Any]:
        return {"minval": self.minval, "maxval": self.maxval, "seed": self.seed}


def inject_hoisted_dilation_bias(tflite_path: str | Path, seed: Optional[int]) -> bool:
    """Give a lowered quantized dilated conv the accumulator-scale bias it lost.

    TF lowers a quantized dilated Conv2D/DepthwiseConv2D to SpaceToBatchND ->
    conv -> BatchToSpaceND -> Add: the conv op keeps a zero placeholder bias
    and the Keras bias is applied by the trailing Add in the output
    quantization domain, which is not a bias a CMSIS-NN kernel can be handed.
    Both the emitted bias tensor and the golden come from the conv op's own
    output, so writing a real int32/int64 bias into that placeholder is what
    puts the bias-add back under test -- the interpreter run that produces the
    golden then sees the same bias the kernel is called with.

    The magnitude is drawn in output quantization steps rather than converted
    from the Keras bias: the conv output tensor's scale was calibrated on the
    bias-free convolution, so a bias sized for the (wider) final output range
    would saturate every element.

    Args:
        tflite_path: Converted model, rewritten in place when the pattern matches.
        seed: Case seed; identical seeds give identical bias tensors.

    Returns:
        True if a bias was injected, False if the model is not a lowered
        quantized dilated conv (float graphs and undilated graphs included).
    """
    from ai_edge_litert import schema_py_generated as litert
    import flatbuffers

    from helia_core_tester.generation.utils.litert_utils import (
        get_tensor_data_from_litert,
        load_litert_model,
    )

    model, subgraph = load_litert_model(str(tflite_path))

    def builtin_code(operator: Any) -> int:
        return model.operatorCodes[operator.opcodeIndex].builtinCode

    if not any(
        builtin_code(op) == litert.BuiltinOperator.SPACE_TO_BATCH_ND for op in subgraph.operators
    ):
        return False

    conv_op = None
    for op in subgraph.operators:
        if builtin_code(op) in (
            litert.BuiltinOperator.CONV_2D,
            litert.BuiltinOperator.DEPTHWISE_CONV_2D,
        ):
            conv_op = op
            break
    if conv_op is None or conv_op.inputs is None or len(conv_op.inputs) < 3:
        return False

    bias_tensor = subgraph.tensors[int(conv_op.inputs[2])]
    bias_np_dtype = {
        litert.TensorType.INT32: np.int32,
        litert.TensorType.INT64: np.int64,
    }.get(bias_tensor.type)
    if bias_np_dtype is None:
        return False

    existing = get_tensor_data_from_litert(bias_tensor, model)
    if existing is not None and np.any(existing != 0):
        return False

    input_tensor = subgraph.tensors[int(conv_op.inputs[0])]
    weight_tensor = subgraph.tensors[int(conv_op.inputs[1])]
    output_tensor = subgraph.tensors[int(conv_op.outputs[0])]
    for tensor in (input_tensor, weight_tensor, output_tensor):
        if tensor.quantization is None or tensor.quantization.scale is None:
            return False
        if len(tensor.quantization.scale) == 0:
            return False

    channels = int(bias_tensor.shape[0]) if bias_tensor.shape is not None else 0
    if channels <= 0:
        return False

    weight_scales = np.asarray(weight_tensor.quantization.scale, dtype=np.float64)
    if weight_scales.size == 1:
        weight_scales = np.repeat(weight_scales, channels)
    if weight_scales.size != channels:
        return False

    input_scale = float(input_tensor.quantization.scale[0])
    output_scale = float(output_tensor.quantization.scale[0])
    accumulator_scales = input_scale * weight_scales
    if not np.all(accumulator_scales > 0):
        return False

    steps = np.asarray(
        SignedMagnitudeUniform(
            _DILATION_BIAS_MIN_STEPS, _DILATION_BIAS_MAX_STEPS, seed
        )((channels,)),
        dtype=np.float64,
    )
    bias = np.round(steps * output_scale / accumulator_scales).astype(bias_np_dtype)

    # The placeholder never gets written in place: the exporter deduplicates
    # identical constant buffers, so a zero bias can share storage with the
    # lowering's own zero paddings/crops (same length whenever the channel
    # count lines up), and overwriting it corrupts SpaceToBatchND. Buffer 0 is
    # likewise the shared empty-data sentinel. Give the bias one of its own.
    buffer = litert.BufferT()
    buffer.data = np.frombuffer(bias.tobytes(), dtype=np.uint8)
    model.buffers.append(buffer)
    bias_tensor.buffer = len(model.buffers) - 1

    builder = flatbuffers.Builder(1024)
    model_offset = model.Pack(builder)
    file_identifier = getattr(litert.Model, "FileIdentifier", lambda: b"TFL3")()
    builder.Finish(model_offset, file_identifier)
    Path(tflite_path).write_bytes(bytes(builder.Output()))
    return True
