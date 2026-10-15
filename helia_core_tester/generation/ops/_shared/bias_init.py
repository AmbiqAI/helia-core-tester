"""Bias enrichment for operators whose Keras model feeds the quantized pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import keras
import numpy as np

# Output quantization steps a hoisted-bias case's injected bias is worth per
# channel. The floor clears the Convolve family's 1 LSB comparison tolerance
# with margin so a dropped bias-add cannot hide inside it, and the ceiling
# keeps the bias a few percent of the output tensor's own range, which its
# scale was calibrated without.
_DILATION_BIAS_MIN_STEPS = 3.0
_DILATION_BIAS_MAX_STEPS = 8.0


class HoistedBiasInjectionError(ValueError):
    """A model handed to :func:`inject_hoisted_dilation_bias` did not match.

    Subclasses ``ValueError`` so the argument-validation contract of this
    module is unchanged for callers that do not care which precondition
    failed.
    """


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


def bias_is_hoisted_by_lowering(dilation: Any, *, is_float: bool) -> bool:
    """True when the converter moves this case's bias out of the conv op.

    A quantized dilated conv lowers to SpaceToBatchND -> conv ->
    BatchToSpaceND -> Add, which strands the bias in the trailing Add; a float
    graph keeps its bias fused into the conv whatever the dilation is.

    Args:
        dilation: Descriptor dilation, an int or a pair.
        is_float: True for FP32/FP16 cases.
    """
    if is_float:
        return False
    rates = (dilation,) if isinstance(dilation, (int, float)) else tuple(dilation)
    return any(int(rate) != 1 for rate in rates)


def inject_hoisted_dilation_bias(tflite_path: str | Path, seed: Optional[int]) -> None:
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

    Raises:
        HoistedBiasInjectionError: The model is not a lowered quantized dilated
            conv with an injectable placeholder bias. The message names the
            precondition that failed, which is what tells a caller whether the
            graph was never lowered, or was lowered into a shape this does not
            know how to bias.
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
        raise HoistedBiasInjectionError(
            "the graph has no SpaceToBatchND, so the converter never hoisted a bias"
        )

    conv_op = None
    for op in subgraph.operators:
        if builtin_code(op) in (
            litert.BuiltinOperator.CONV_2D,
            litert.BuiltinOperator.DEPTHWISE_CONV_2D,
        ):
            conv_op = op
            break
    if conv_op is None:
        raise HoistedBiasInjectionError(
            "the lowered graph has no CONV_2D or DEPTHWISE_CONV_2D operator"
        )
    if conv_op.inputs is None or len(conv_op.inputs) < 3:
        raise HoistedBiasInjectionError(
            "the lowered conv op has no bias input "
            f"({0 if conv_op.inputs is None else len(conv_op.inputs)} inputs)"
        )

    # TFLite marks an absent optional input with -1, which would index the
    # tensor list from the end.
    if int(conv_op.inputs[2]) < 0:
        raise HoistedBiasInjectionError(
            "the lowered conv op's bias input is the -1 absent-optional sentinel"
        )

    bias_tensor = subgraph.tensors[int(conv_op.inputs[2])]
    bias_np_dtype = {
        litert.TensorType.INT32: np.int32,
        litert.TensorType.INT64: np.int64,
    }.get(bias_tensor.type)
    if bias_np_dtype is None:
        raise HoistedBiasInjectionError(
            f"the placeholder bias is tensor type {bias_tensor.type}, not INT32 or INT64"
        )

    existing = get_tensor_data_from_litert(bias_tensor, model)
    if existing is not None and np.any(existing != 0):
        raise HoistedBiasInjectionError(
            "the lowered conv op already carries a non-zero bias, so the Keras bias "
            "was not hoisted into the trailing Add"
        )

    input_tensor = subgraph.tensors[int(conv_op.inputs[0])]
    weight_tensor = subgraph.tensors[int(conv_op.inputs[1])]
    output_tensor = subgraph.tensors[int(conv_op.outputs[0])]
    for role, tensor in (
        ("input", input_tensor),
        ("weight", weight_tensor),
        ("output", output_tensor),
    ):
        if (
            tensor.quantization is None
            or tensor.quantization.scale is None
            or len(tensor.quantization.scale) == 0
        ):
            raise HoistedBiasInjectionError(
                f"the lowered conv op's {role} tensor carries no quantization scale"
            )

    channels = int(bias_tensor.shape[0]) if bias_tensor.shape is not None else 0
    if channels <= 0:
        raise HoistedBiasInjectionError(
            f"the placeholder bias has no channels (shape {bias_tensor.shape})"
        )

    weight_scales = np.asarray(weight_tensor.quantization.scale, dtype=np.float64)
    if weight_scales.size == 1:
        weight_scales = np.repeat(weight_scales, channels)
    if weight_scales.size != channels:
        raise HoistedBiasInjectionError(
            f"the weight tensor has {weight_scales.size} scales for {channels} bias channels"
        )

    input_scale = float(input_tensor.quantization.scale[0])
    output_scale = float(output_tensor.quantization.scale[0])
    accumulator_scales = input_scale * weight_scales
    if not np.all(accumulator_scales > 0):
        raise HoistedBiasInjectionError(
            "the input x weight accumulator scale is not positive on every channel"
        )

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


def patch_requant(tflite_path: str, desc: Dict[str, Any], seed: int) -> None:
    """Push requant to its edges, in place.

    `wide_bias: [lo, hi]` rewrites each channel's int64 bias to a
    random-sign magnitude in [lo, hi] accumulator steps, which the
    converter never emits, and widens the output scale so no output
    can saturate. `min_effective_scale` then shrinks the output scale
    until the smallest in * weight / out scale equals it, so one
    accumulator step moves an output step.
    """
    import flatbuffers
    from ai_edge_litert import schema_py_generated as litert
    from helia_core_tester.generation.utils.litert_utils import get_tensor_data_from_litert, load_litert_model

    model, subgraph = load_litert_model(str(tflite_path))
    ops = [op for op in subgraph.operators
           if model.operatorCodes[op.opcodeIndex].builtinCode == litert.BuiltinOperator.TRANSPOSE_CONV]
    if len(ops) != 1:
        raise ValueError(f"expected one TRANSPOSE_CONV, found {len(ops)}")
    op = ops[0]
    # Inputs: output shape, weights, data, bias.
    weights, data = subgraph.tensors[int(op.inputs[1])], subgraph.tensors[int(op.inputs[2])]
    out = subgraph.tensors[int(op.outputs[0])]
    in_scale = float(data.quantization.scale[0])
    weight_scales = np.asarray(weights.quantization.scale, dtype=np.float64)
    if desc.get('wide_bias'):
        if len(op.inputs) < 4 or int(op.inputs[3]) < 0:
            raise ValueError("wide_bias needs a bias input")
        bias = subgraph.tensors[int(op.inputs[3])]
        lo, hi = (float(v) for v in desc['wide_bias'])
        values = np.round(np.asarray(SignedMagnitudeUniform(lo, hi, seed)((weight_scales.size,)),
                                     dtype=np.float64)).astype(np.int64)
        # Own buffer: constants may share one.
        buffer = litert.BufferT()
        buffer.data = np.frombuffer(values.tobytes(), dtype=np.uint8)
        model.buffers.append(buffer)
        bias.buffer = len(model.buffers) - 1
        # Largest accumulator any output can reach.
        filters = np.abs(get_tensor_data_from_litert(weights, model).astype(np.int64))
        reach = 32768 * filters.reshape(filters.shape[0], -1).sum(axis=1) + np.abs(values)
        out.quantization.scale = [float(np.max(reach * in_scale * weight_scales)) / 32767.0]
    if desc.get('min_effective_scale'):
        out.quantization.scale = [in_scale * float(np.min(weight_scales)) / float(desc['min_effective_scale'])]
    builder = flatbuffers.Builder(1024)
    builder.Finish(model.Pack(builder), getattr(litert.Model, "FileIdentifier", lambda: b"TFL3")())
    Path(tflite_path).write_bytes(bytes(builder.Output()))
