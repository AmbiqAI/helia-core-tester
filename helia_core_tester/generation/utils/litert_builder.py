"""
LiteRT single-op model builder utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

try:
    from ai_edge_litert import schema_py_generated as litert
    import flatbuffers
    LITERT_AVAILABLE = True
except ImportError:  # pragma: no cover - runtime dependency
    litert = None
    flatbuffers = None
    LITERT_AVAILABLE = False


_INT4_TENSOR_TYPE = getattr(litert.TensorType, "INT4", None) if LITERT_AVAILABLE else None

_DTYPE_MAP = {
    "fp32": litert.TensorType.FLOAT32 if LITERT_AVAILABLE else None,
    "float32": litert.TensorType.FLOAT32 if LITERT_AVAILABLE else None,
    "fp16": litert.TensorType.FLOAT16 if LITERT_AVAILABLE else None,
    "float16": litert.TensorType.FLOAT16 if LITERT_AVAILABLE else None,
    "s8": litert.TensorType.INT8 if LITERT_AVAILABLE else None,
    "int8": litert.TensorType.INT8 if LITERT_AVAILABLE else None,
    "s16": litert.TensorType.INT16 if LITERT_AVAILABLE else None,
    "int16": litert.TensorType.INT16 if LITERT_AVAILABLE else None,
    "s32": litert.TensorType.INT32 if LITERT_AVAILABLE else None,
    "bool": litert.TensorType.BOOL if LITERT_AVAILABLE else None,
    "bool8": litert.TensorType.BOOL if LITERT_AVAILABLE else None,
    "int32": litert.TensorType.INT32 if LITERT_AVAILABLE else None,
    "s4": _INT4_TENSOR_TYPE,
    "int4": _INT4_TENSOR_TYPE,
}


def _default_quant(tensor_type: int) -> Optional[Tuple[Sequence[float], Sequence[int]]]:
    if not LITERT_AVAILABLE:
        return None
    if tensor_type == litert.TensorType.INT8:
        return ([0.125], [0])
    if tensor_type == litert.TensorType.INT16:
        return ([1.0 / 32768.0], [0])
    return None


_TENSOR_TYPE_TO_NP = (
    {
        litert.TensorType.INT8: np.int8,
        litert.TensorType.INT16: np.int16,
        litert.TensorType.INT32: np.int32,
        litert.TensorType.UINT8: np.uint8,
        litert.TensorType.INT64: np.int64,
        litert.TensorType.FLOAT32: np.float32,
        litert.TensorType.FLOAT16: np.float16,
        **({litert.TensorType.INT4: np.int8} if _INT4_TENSOR_TYPE is not None else {}),
    }
    if LITERT_AVAILABLE
    else {}
)


def pack_s4(values: np.ndarray) -> np.ndarray:
    """Pack int4 values into int8 bytes (low nibble first), matching op_conv.py."""
    vals = np.asarray(values, dtype=np.int8).flatten()
    if np.any(vals < -8) or np.any(vals > 7):
        raise ValueError("int4 values must be in [-8, 7]")
    # op_conv.py packs raw 4-bit nibbles into uint8, low nibble first.
    # Use two's complement by masking with 0x0F and packing pairs.
    vals_u4 = (vals.astype(np.int16) & 0x0F).astype(np.uint8)
    if vals_u4.size % 2 != 0:
        vals_u4 = np.append(vals_u4, np.uint8(0))
    temp = np.reshape(vals_u4, (vals_u4.size // 2, 2)).astype(np.uint8)
    packed = 0xFF & ((0xF0 & (temp[:, 1] << 4)) | (temp[:, 0] & 0x0F))
    return packed.astype(np.uint8).view(np.int8)


def unpack_s4(packed: np.ndarray, num_elems: int) -> np.ndarray:
    """Unpack int4 values from int8 bytes (low nibble first, two's complement)."""
    p = np.asarray(packed, dtype=np.int8).view(np.uint8).flatten()
    low = p & 0x0F
    high = (p >> 4) & 0x0F
    vals = np.empty(p.size * 2, dtype=np.int8)
    vals[0::2] = low.astype(np.int8)
    vals[1::2] = high.astype(np.int8)
    # Convert from unsigned 4-bit to signed int4
    vals = ((vals + 8) & 0x0F) - 8
    return vals[:num_elems]


@dataclass
class TensorSpec:
    name: str
    shape: Iterable[int]
    tensor_type: int
    is_input: bool = False
    is_output: bool = False
    quantization: Optional[Tuple[Sequence[float], Sequence[int]]] = None
    quantized_dimension: Optional[int] = None
    data: Optional[Sequence[int] | np.ndarray] = None


class LiteRtSingleOpBuilder:
    def __init__(self, op_name: str):
        if not LITERT_AVAILABLE:
            raise ImportError("ai_edge_litert is not available. Install it with: pip install ai-edge-litert")
        self.op_name = op_name
        self._tensors: list[TensorSpec] = []
        self._op_inputs: list[int] = []
        self._op_outputs: list[int] = []
        self._op_options = None
        self._op_options_type = None

    def add_tensor(self, spec: TensorSpec) -> int:
        idx = len(self._tensors)
        self._tensors.append(spec)
        return idx

    def add_operator(
        self,
        op_name: str,
        *,
        inputs: Sequence[int],
        outputs: Sequence[int],
        options,
        options_type,
    ) -> None:
        self.op_name = op_name
        self._op_inputs = list(inputs)
        self._op_outputs = list(outputs)
        self._op_options = options
        self._op_options_type = options_type

    def build(self) -> bytes:
        model = litert.ModelT()
        model.version = 3

        # Buffers: buffer[0] is empty per TFLite convention
        buffers: list[litert.BufferT] = []
        buffers.append(litert.BufferT())

        tensors: list[litert.TensorT] = []
        subgraph_inputs: list[int] = []
        subgraph_outputs: list[int] = []

        for idx, spec in enumerate(self._tensors):
            tensor = litert.TensorT()
            tensor.name = spec.name
            tensor.shape = list(int(v) for v in spec.shape)
            tensor.type = spec.tensor_type
            tensor.isVariable = False

            if spec.quantization is not None:
                scales, zero_points = spec.quantization
                q = litert.QuantizationParametersT()
                q.scale = list(float(v) for v in scales)
                q.zeroPoint = list(int(v) for v in zero_points)
                if spec.quantized_dimension is not None:
                    q.quantizedDimension = int(spec.quantized_dimension)
                else:
                    q.quantizedDimension = 0
                tensor.quantization = q

            if spec.data is not None:
                buf = litert.BufferT()
                np_dtype = _TENSOR_TYPE_TO_NP.get(spec.tensor_type, np.int8)
                data = np.array(spec.data, dtype=np_dtype).tobytes()
                buf.data = data
                buffers.append(buf)
                tensor.buffer = len(buffers) - 1
            else:
                tensor.buffer = 0

            tensors.append(tensor)

            if spec.is_input:
                subgraph_inputs.append(idx)
            if spec.is_output:
                subgraph_outputs.append(idx)

        opcode = litert.OperatorCodeT()
        builtin = getattr(litert.BuiltinOperator, self.op_name, None)
        if builtin is None:
            raise ValueError(f"Unsupported builtin op '{self.op_name}'")
        opcode.builtinCode = builtin

        op = litert.OperatorT()
        op.opcodeIndex = 0
        op.inputs = list(self._op_inputs)
        op.outputs = list(self._op_outputs)
        op.builtinOptionsType = self._op_options_type
        op.builtinOptions = self._op_options

        subgraph = litert.SubGraphT()
        subgraph.tensors = tensors
        subgraph.inputs = subgraph_inputs
        subgraph.outputs = subgraph_outputs
        subgraph.operators = [op]
        subgraph.name = "main"

        model.operatorCodes = [opcode]
        model.subgraphs = [subgraph]
        model.buffers = buffers

        builder = flatbuffers.Builder(1024)
        model_offset = model.Pack(builder)
        file_identifier = getattr(litert.Model, "FileIdentifier", lambda: b"TFL3")()
        builder.Finish(model_offset, file_identifier)
        return bytes(builder.Output())


def _broadcast_shape(shape_a: Sequence[int], shape_b: Sequence[int]) -> Tuple[int, ...]:
    a = list(int(v) for v in shape_a)
    b = list(int(v) for v in shape_b)
    max_len = max(len(a), len(b))
    a = [1] * (max_len - len(a)) + a
    b = [1] * (max_len - len(b)) + b
    out = []
    for dim_a, dim_b in zip(a, b):
        if dim_a == dim_b or dim_a == 1 or dim_b == 1:
            out.append(max(dim_a, dim_b))
        else:
            raise ValueError(f"Shapes not broadcastable: {shape_a} vs {shape_b}")
    return tuple(out)


def _require_litert() -> None:
    if not LITERT_AVAILABLE:
        raise ImportError("ai_edge_litert is not available. Install it with: pip install ai-edge-litert")


def _resolve_tensor_type(dtype: str) -> int:
    tensor_type = _DTYPE_MAP.get(dtype.lower())
    if tensor_type is None:
        raise ValueError(f"Unsupported dtype '{dtype}'.")
    return tensor_type


def _normalize_shape(input_shape: Iterable[int], *, op_label: str) -> Tuple[int, ...]:
    shape = tuple(int(dim) for dim in input_shape)
    if len(shape) < 1:
        raise ValueError(f"{op_label} expects a non-empty input shape.")
    return shape


def build_unary_same_shape_op(
    *,
    op_name: str,
    input_shape: Iterable[int],
    dtype: str = "int8",
    output_dtype: Optional[str] = None,
    options=None,
    options_type=None,
) -> bytes:
    _require_litert()

    input_tensor_type = _resolve_tensor_type(dtype)
    output_tensor_type = _resolve_tensor_type(output_dtype or dtype)
    input_shape = _normalize_shape(input_shape, op_label=op_name)

    builder = LiteRtSingleOpBuilder(op_name=op_name)

    input_tensor_idx = builder.add_tensor(
        TensorSpec(
            name="input",
            shape=input_shape,
            tensor_type=input_tensor_type,
            is_input=True,
            quantization=_default_quant(input_tensor_type),
        )
    )

    output_tensor_idx = builder.add_tensor(
        TensorSpec(
            name="output",
            shape=input_shape,
            tensor_type=output_tensor_type,
            is_output=True,
            quantization=_default_quant(output_tensor_type),
        )
    )

    builder.add_operator(
        op_name,
        inputs=[input_tensor_idx],
        outputs=[output_tensor_idx],
        options=options,
        options_type=litert.BuiltinOptions.NONE if options_type is None else options_type,
    )

    return builder.build()


def build_binary_broadcast_op(
    *,
    op_name: str,
    input_1_shape: Iterable[int],
    input_2_shape: Iterable[int],
    dtype: str = "int8",
    input_1_quant: Optional[Tuple[Sequence[float], Sequence[int]]] = None,
    input_2_quant: Optional[Tuple[Sequence[float], Sequence[int]]] = None,
    output_quant: Optional[Tuple[Sequence[float], Sequence[int]]] = None,
    options=None,
    options_type=None,
) -> bytes:
    _require_litert()

    tensor_type = _resolve_tensor_type(dtype)
    input_1_shape = _normalize_shape(input_1_shape, op_label=op_name)
    input_2_shape = _normalize_shape(input_2_shape, op_label=op_name)
    output_shape = _broadcast_shape(input_1_shape, input_2_shape)

    builder = LiteRtSingleOpBuilder(op_name=op_name)

    input1_tensor_idx = builder.add_tensor(
        TensorSpec(
            name="input1",
            shape=input_1_shape,
            tensor_type=tensor_type,
            is_input=True,
            quantization=input_1_quant if input_1_quant is not None else _default_quant(tensor_type),
        )
    )

    input2_tensor_idx = builder.add_tensor(
        TensorSpec(
            name="input2",
            shape=input_2_shape,
            tensor_type=tensor_type,
            is_input=True,
            quantization=input_2_quant if input_2_quant is not None else _default_quant(tensor_type),
        )
    )

    output_tensor_idx = builder.add_tensor(
        TensorSpec(
            name="output",
            shape=output_shape,
            tensor_type=tensor_type,
            is_output=True,
            quantization=output_quant if output_quant is not None else _default_quant(tensor_type),
        )
    )

    builder.add_operator(
        op_name,
        inputs=[input1_tensor_idx, input2_tensor_idx],
        outputs=[output_tensor_idx],
        options=options,
        options_type=litert.BuiltinOptions.NONE if options_type is None else options_type,
    )

    return builder.build()


def build_shape_transform_op(
    *,
    op_name: str,
    input_shape: Iterable[int],
    output_shape: Iterable[int],
    dtype: str = "int8",
    extra_input_tensors: Optional[Sequence[TensorSpec]] = None,
    options=None,
    options_type=None,
) -> bytes:
    _require_litert()

    tensor_type = _resolve_tensor_type(dtype)
    input_shape = _normalize_shape(input_shape, op_label=op_name)
    output_shape = _normalize_shape(output_shape, op_label=op_name)

    builder = LiteRtSingleOpBuilder(op_name=op_name)

    op_inputs = [
        builder.add_tensor(
            TensorSpec(
                name="input",
                shape=input_shape,
                tensor_type=tensor_type,
                is_input=True,
                quantization=_default_quant(tensor_type),
            )
        )
    ]

    for tensor in extra_input_tensors or ():
        op_inputs.append(builder.add_tensor(tensor))

    output_tensor_idx = builder.add_tensor(
        TensorSpec(
            name="output",
            shape=output_shape,
            tensor_type=tensor_type,
            is_output=True,
            quantization=_default_quant(tensor_type),
        )
    )

    builder.add_operator(
        op_name,
        inputs=op_inputs,
        outputs=[output_tensor_idx],
        options=options,
        options_type=litert.BuiltinOptions.NONE if options_type is None else options_type,
    )

    return builder.build()


def build_transpose_op(
    *,
    input_shape: Iterable[int],
    permutation: Iterable[int],
    dtype: str = "int8",
) -> bytes:
    _require_litert()

    tensor_type = _resolve_tensor_type(dtype)
    input_shape = _normalize_shape(input_shape, op_label="TRANSPOSE")
    permutation = tuple(int(dim) for dim in permutation)
    rank = len(input_shape)
    if len(permutation) != rank or sorted(permutation) != list(range(rank)):
        raise ValueError(
            f"TRANSPOSE permutation must be a permutation of rank {rank}, got {permutation}."
        )
    output_shape = tuple(input_shape[dim] for dim in permutation)

    builder = LiteRtSingleOpBuilder(op_name="TRANSPOSE")

    input_tensor_idx = builder.add_tensor(
        TensorSpec(
            name="input",
            shape=input_shape,
            tensor_type=tensor_type,
            is_input=True,
            quantization=_default_quant(tensor_type),
        )
    )

    permutation_tensor_idx = builder.add_tensor(
        TensorSpec(
            name="perm",
            shape=(rank,),
            tensor_type=litert.TensorType.INT32,
            data=np.array(permutation, dtype=np.int32),
        )
    )

    output_tensor_idx = builder.add_tensor(
        TensorSpec(
            name="output",
            shape=output_shape,
            tensor_type=tensor_type,
            is_output=True,
            quantization=_default_quant(tensor_type),
        )
    )

    builder.add_operator(
        "TRANSPOSE",
        inputs=[input_tensor_idx, permutation_tensor_idx],
        outputs=[output_tensor_idx],
        options=None,
        options_type=litert.BuiltinOptions.NONE,
    )

    return builder.build()


def build_arg_reduction_op(
    *,
    op_name: str,
    input_shape: Iterable[int] = (1, 4, 4, 3),
    axis: int = -1,
    dtype: str = "int8",
) -> bytes:
    _require_litert()

    if op_name not in {"ARG_MAX", "ARG_MIN"}:
        raise ValueError(f"Unsupported arg reduction '{op_name}'.")

    tensor_type = _resolve_tensor_type(dtype)
    input_shape = _normalize_shape(input_shape, op_label=op_name)
    rank = len(input_shape)

    axis_norm = axis
    if axis_norm < 0:
        axis_norm += rank
    if axis_norm < 0 or axis_norm >= rank:
        raise ValueError(f"Axis {axis} out of range for rank {rank}.")

    output_shape = list(input_shape)
    del output_shape[axis_norm]
    if not output_shape:
        output_shape = [1]

    builder = LiteRtSingleOpBuilder(op_name=op_name)

    input_tensor_idx = builder.add_tensor(
        TensorSpec(
            name="input",
            shape=input_shape,
            tensor_type=tensor_type,
            is_input=True,
            quantization=_default_quant(tensor_type),
        )
    )

    axis_tensor_idx = builder.add_tensor(
        TensorSpec(
            name="axis",
            shape=(1,),
            tensor_type=litert.TensorType.INT32,
            data=[axis_norm],
        )
    )

    output_tensor_idx = builder.add_tensor(
        TensorSpec(
            name="output",
            shape=tuple(output_shape),
            tensor_type=litert.TensorType.INT32,
            is_output=True,
        )
    )

    if op_name == "ARG_MAX":
        options = litert.ArgMaxOptionsT()
        options.outputType = litert.TensorType.INT32
        options_type = litert.BuiltinOptions.ArgMaxOptions
    else:
        options = litert.ArgMinOptionsT()
        options.outputType = litert.TensorType.INT32
        options_type = litert.BuiltinOptions.ArgMinOptions

    builder.add_operator(
        op_name,
        inputs=[input_tensor_idx, axis_tensor_idx],
        outputs=[output_tensor_idx],
        options=options,
        options_type=options_type,
    )

    return builder.build()


def build_space_to_depth_op(
    *,
    input_shape: Iterable[int] = (1, 4, 4, 3),
    block_size: int = 2,
    dtype: str = "int8",
) -> bytes:
    if not LITERT_AVAILABLE:
        raise ImportError("ai_edge_litert is not available. Install it with: pip install ai-edge-litert")

    tensor_type = _DTYPE_MAP.get(dtype.lower())
    if tensor_type is None:
        raise ValueError(f"Unsupported dtype '{dtype}'.")

    input_shape = tuple(int(dim) for dim in input_shape)
    if len(input_shape) != 4:
        raise ValueError("SpaceToDepth expects a 4D NHWC input shape.")

    if block_size <= 0:
        raise ValueError("block_size must be > 0.")

    n, h, w, c = input_shape
    if h % block_size != 0 or w % block_size != 0:
        raise ValueError("Input height and width must be divisible by block_size.")

    output_shape = (n, h // block_size, w // block_size, c * block_size * block_size)

    builder = LiteRtSingleOpBuilder(op_name="SPACE_TO_DEPTH")

    input_tensor_idx = builder.add_tensor(
        TensorSpec(
            name="input",
            shape=input_shape,
            tensor_type=tensor_type,
            is_input=True,
            quantization=_default_quant(tensor_type),
        )
    )

    output_tensor_idx = builder.add_tensor(
        TensorSpec(
            name="output",
            shape=output_shape,
            tensor_type=tensor_type,
            is_output=True,
            quantization=_default_quant(tensor_type),
        )
    )

    options = litert.SpaceToDepthOptionsT()
    options.blockSize = int(block_size)

    builder.add_operator(
        "SPACE_TO_DEPTH",
        inputs=[input_tensor_idx],
        outputs=[output_tensor_idx],
        options=options,
        options_type=litert.BuiltinOptions.SpaceToDepthOptions,
    )

    return builder.build()


def build_depth_to_space_op(
    *,
    input_shape: Iterable[int],
    block_size: int,
    dtype: str = "int8",
) -> bytes:
    if not LITERT_AVAILABLE:
        raise ImportError("ai_edge_litert is not available. Install it with: pip install ai-edge-litert")

    tensor_type = _DTYPE_MAP.get(dtype.lower())
    if tensor_type is None:
        raise ValueError(f"Unsupported dtype '{dtype}'.")

    input_shape = tuple(int(dim) for dim in input_shape)
    if len(input_shape) != 4:
        raise ValueError("DepthToSpace expects a 4D NHWC input shape.")

    if block_size <= 0:
        raise ValueError("block_size must be > 0.")

    n, h, w, c = input_shape
    if c % (block_size * block_size) != 0:
        raise ValueError("Input channels must be divisible by block_size^2.")

    output_shape = (n, h * block_size, w * block_size, c // (block_size * block_size))

    builder = LiteRtSingleOpBuilder(op_name="DEPTH_TO_SPACE")

    input_tensor_idx = builder.add_tensor(
        TensorSpec(
            name="input",
            shape=input_shape,
            tensor_type=tensor_type,
            is_input=True,
            quantization=_default_quant(tensor_type),
        )
    )

    output_tensor_idx = builder.add_tensor(
        TensorSpec(
            name="output",
            shape=output_shape,
            tensor_type=tensor_type,
            is_output=True,
            quantization=_default_quant(tensor_type),
        )
    )

    options = litert.DepthToSpaceOptionsT()
    options.blockSize = int(block_size)

    builder.add_operator(
        "DEPTH_TO_SPACE",
        inputs=[input_tensor_idx],
        outputs=[output_tensor_idx],
        options=options,
        options_type=litert.BuiltinOptions.DepthToSpaceOptions,
    )

    return builder.build()


def build_reshape_op(
    *,
    input_shape: Iterable[int],
    target_shape: Iterable[int],
    dtype: str = "int8",
) -> bytes:
    if not LITERT_AVAILABLE:
        raise ImportError("ai_edge_litert is not available. Install it with: pip install ai-edge-litert")

    tensor_type = _DTYPE_MAP.get(dtype.lower())
    if tensor_type is None:
        raise ValueError(f"Unsupported dtype '{dtype}'.")

    input_shape = tuple(int(dim) for dim in input_shape)
    target_shape = tuple(int(dim) for dim in target_shape)

    if int(np.prod(input_shape)) != int(np.prod(target_shape)):
        raise ValueError("Reshape requires input and target shapes with the same number of elements.")

    builder = LiteRtSingleOpBuilder(op_name="RESHAPE")

    input_tensor_idx = builder.add_tensor(
        TensorSpec(
            name="input",
            shape=input_shape,
            tensor_type=tensor_type,
            is_input=True,
            quantization=_default_quant(tensor_type),
        )
    )

    shape_tensor_idx = builder.add_tensor(
        TensorSpec(
            name="new_shape",
            shape=(len(target_shape),),
            tensor_type=litert.TensorType.INT32,
            data=list(target_shape),
        )
    )

    output_tensor_idx = builder.add_tensor(
        TensorSpec(
            name="output",
            shape=target_shape,
            tensor_type=tensor_type,
            is_output=True,
            quantization=_default_quant(tensor_type),
        )
    )

    options = litert.ReshapeOptionsT()
    options.newShape = list(int(v) for v in target_shape)

    builder.add_operator(
        "RESHAPE",
        inputs=[input_tensor_idx, shape_tensor_idx],
        outputs=[output_tensor_idx],
        options=options,
        options_type=litert.BuiltinOptions.ReshapeOptions,
    )

    return builder.build()


def build_space_to_batch_nd_op(
    *,
    input_shape: Iterable[int],
    block_shape: Iterable[int],
    paddings: Sequence[Sequence[int]],
    dtype: str = "int8",
) -> bytes:
    if not LITERT_AVAILABLE:
        raise ImportError("ai_edge_litert is not available. Install it with: pip install ai-edge-litert")

    tensor_type = _DTYPE_MAP.get(dtype.lower())
    if tensor_type is None:
        raise ValueError(f"Unsupported dtype '{dtype}'.")

    input_shape = tuple(int(dim) for dim in input_shape)
    if len(input_shape) != 4:
        raise ValueError("SpaceToBatchND expects a 4D NHWC input shape.")

    block_shape = [int(v) for v in block_shape]
    if len(block_shape) != 2:
        raise ValueError("SpaceToBatchND supports 2D block_shape.")

    pad_h = [int(paddings[0][0]), int(paddings[0][1])]
    pad_w = [int(paddings[1][0]), int(paddings[1][1])]
    pad_h_total = pad_h[0] + pad_h[1]
    pad_w_total = pad_w[0] + pad_w[1]

    n, h, w, c = input_shape
    bh, bw = block_shape
    out_h = (h + pad_h_total) // bh
    out_w = (w + pad_w_total) // bw
    out_n = n * bh * bw

    output_shape = (out_n, out_h, out_w, c)

    builder = LiteRtSingleOpBuilder(op_name="SPACE_TO_BATCH_ND")

    input_tensor_idx = builder.add_tensor(
        TensorSpec(
            name="input",
            shape=input_shape,
            tensor_type=tensor_type,
            is_input=True,
            quantization=_default_quant(tensor_type),
        )
    )

    block_tensor_idx = builder.add_tensor(
        TensorSpec(
            name="block_shape",
            shape=(len(block_shape),),
            tensor_type=litert.TensorType.INT32,
            data=block_shape,
        )
    )

    paddings_flat = [pad_h, pad_w]
    paddings_tensor_idx = builder.add_tensor(
        TensorSpec(
            name="paddings",
            shape=(2, 2),
            tensor_type=litert.TensorType.INT32,
            data=np.array(paddings_flat, dtype=np.int32).flatten().tolist(),
        )
    )

    output_tensor_idx = builder.add_tensor(
        TensorSpec(
            name="output",
            shape=output_shape,
            tensor_type=tensor_type,
            is_output=True,
            quantization=_default_quant(tensor_type),
        )
    )

    options = litert.SpaceToBatchNDOptionsT()

    builder.add_operator(
        "SPACE_TO_BATCH_ND",
        inputs=[input_tensor_idx, block_tensor_idx, paddings_tensor_idx],
        outputs=[output_tensor_idx],
        options=options,
        options_type=litert.BuiltinOptions.SpaceToBatchNDOptions,
    )

    return builder.build()


def build_batch_to_space_nd_op(
    *,
    input_shape: Iterable[int],
    block_shape: Iterable[int],
    crops: Sequence[Sequence[int]],
    dtype: str = "int8",
) -> bytes:
    if not LITERT_AVAILABLE:
        raise ImportError("ai_edge_litert is not available. Install it with: pip install ai-edge-litert")

    tensor_type = _DTYPE_MAP.get(dtype.lower())
    if tensor_type is None:
        raise ValueError(f"Unsupported dtype '{dtype}'.")

    input_shape = tuple(int(dim) for dim in input_shape)
    if len(input_shape) != 4:
        raise ValueError("BatchToSpaceND expects a 4D NHWC input shape.")

    block_shape = [int(v) for v in block_shape]
    if len(block_shape) != 2:
        raise ValueError("BatchToSpaceND supports 2D block_shape.")

    crop_h = [int(crops[0][0]), int(crops[0][1])]
    crop_w = [int(crops[1][0]), int(crops[1][1])]

    n, h, w, c = input_shape
    bh, bw = block_shape
    out_n = n // (bh * bw)
    out_h = h * bh - crop_h[0] - crop_h[1]
    out_w = w * bw - crop_w[0] - crop_w[1]

    output_shape = (out_n, out_h, out_w, c)

    builder = LiteRtSingleOpBuilder(op_name="BATCH_TO_SPACE_ND")

    input_tensor_idx = builder.add_tensor(
        TensorSpec(
            name="input",
            shape=input_shape,
            tensor_type=tensor_type,
            is_input=True,
            quantization=_default_quant(tensor_type),
        )
    )

    block_tensor_idx = builder.add_tensor(
        TensorSpec(
            name="block_shape",
            shape=(len(block_shape),),
            tensor_type=litert.TensorType.INT32,
            data=block_shape,
        )
    )

    crops_flat = [crop_h, crop_w]
    crops_tensor_idx = builder.add_tensor(
        TensorSpec(
            name="crops",
            shape=(2, 2),
            tensor_type=litert.TensorType.INT32,
            data=np.array(crops_flat, dtype=np.int32).flatten().tolist(),
        )
    )

    output_tensor_idx = builder.add_tensor(
        TensorSpec(
            name="output",
            shape=output_shape,
            tensor_type=tensor_type,
            is_output=True,
            quantization=_default_quant(tensor_type),
        )
    )

    options = litert.BatchToSpaceNDOptionsT()

    builder.add_operator(
        "BATCH_TO_SPACE_ND",
        inputs=[input_tensor_idx, block_tensor_idx, crops_tensor_idx],
        outputs=[output_tensor_idx],
        options=options,
        options_type=litert.BuiltinOptions.BatchToSpaceNDOptions,
    )

    return builder.build()


def build_resize_nearest_neighbor_op(
    *,
    input_shape: Iterable[int],
    new_size: Sequence[int],
    align_corners: bool = False,
    half_pixel_centers: bool = False,
    dtype: str = "int8",
) -> bytes:
    if not LITERT_AVAILABLE:
        raise ImportError("ai_edge_litert is not available. Install it with: pip install ai-edge-litert")

    tensor_type = _DTYPE_MAP.get(dtype.lower())
    if tensor_type is None:
        raise ValueError(f"Unsupported dtype '{dtype}'.")

    input_shape = tuple(int(dim) for dim in input_shape)
    if len(input_shape) != 4:
        raise ValueError("ResizeNearestNeighbor expects a 4D NHWC input shape.")

    new_h, new_w = int(new_size[0]), int(new_size[1])
    n, _, _, c = input_shape
    output_shape = (n, new_h, new_w, c)

    builder = LiteRtSingleOpBuilder(op_name="RESIZE_NEAREST_NEIGHBOR")

    input_tensor_idx = builder.add_tensor(
        TensorSpec(
            name="input",
            shape=input_shape,
            tensor_type=tensor_type,
            is_input=True,
            quantization=_default_quant(tensor_type),
        )
    )

    size_tensor_idx = builder.add_tensor(
        TensorSpec(
            name="size",
            shape=(2,),
            tensor_type=litert.TensorType.INT32,
            data=[new_h, new_w],
        )
    )

    output_tensor_idx = builder.add_tensor(
        TensorSpec(
            name="output",
            shape=output_shape,
            tensor_type=tensor_type,
            is_output=True,
            quantization=_default_quant(tensor_type),
        )
    )

    options = litert.ResizeNearestNeighborOptionsT()
    options.alignCorners = bool(align_corners)
    options.halfPixelCenters = bool(half_pixel_centers)

    builder.add_operator(
        "RESIZE_NEAREST_NEIGHBOR",
        inputs=[input_tensor_idx, size_tensor_idx],
        outputs=[output_tensor_idx],
        options=options,
        options_type=litert.BuiltinOptions.ResizeNearestNeighborOptions,
    )

    return builder.build()


def build_pad_op(
    *,
    input_shape: Iterable[int],
    paddings: Sequence[Sequence[int]],
    dtype: str = "int8",
) -> bytes:
    if not LITERT_AVAILABLE:
        raise ImportError("ai_edge_litert is not available. Install it with: pip install ai-edge-litert")

    tensor_type = _DTYPE_MAP.get(dtype.lower())
    if tensor_type is None:
        raise ValueError(f"Unsupported dtype '{dtype}'.")

    input_shape = tuple(int(dim) for dim in input_shape)
    if len(input_shape) != 4:
        raise ValueError("Pad expects a 4D NHWC input shape.")

    if len(paddings) != 4:
        raise ValueError("Pad expects paddings for 4 dimensions.")

    pad_list = [[int(p[0]), int(p[1])] for p in paddings]
    output_shape = tuple(
        int(input_shape[i] + pad_list[i][0] + pad_list[i][1]) for i in range(4)
    )

    builder = LiteRtSingleOpBuilder(op_name="PAD")

    input_tensor_idx = builder.add_tensor(
        TensorSpec(
            name="input",
            shape=input_shape,
            tensor_type=tensor_type,
            is_input=True,
            quantization=_default_quant(tensor_type),
        )
    )

    paddings_tensor_idx = builder.add_tensor(
        TensorSpec(
            name="paddings",
            shape=(4, 2),
            tensor_type=litert.TensorType.INT32,
            data=np.array(pad_list, dtype=np.int32).flatten().tolist(),
        )
    )

    output_tensor_idx = builder.add_tensor(
        TensorSpec(
            name="output",
            shape=output_shape,
            tensor_type=tensor_type,
            is_output=True,
            quantization=_default_quant(tensor_type),
        )
    )

    builder.add_operator(
        "PAD",
        inputs=[input_tensor_idx, paddings_tensor_idx],
        outputs=[output_tensor_idx],
        options=None,
        options_type=litert.BuiltinOptions.NONE,
    )

    return builder.build()


def build_sub_op(
    *,
    input_1_shape: Iterable[int],
    input_2_shape: Iterable[int],
    dtype: str = "int8",
) -> bytes:
    if not LITERT_AVAILABLE:
        raise ImportError("ai_edge_litert is not available. Install it with: pip install ai-edge-litert")

    tensor_type = _DTYPE_MAP.get(dtype.lower())
    if tensor_type is None:
        raise ValueError(f"Unsupported dtype '{dtype}'.")

    input_1_shape = tuple(int(dim) for dim in input_1_shape)
    input_2_shape = tuple(int(dim) for dim in input_2_shape)
    if len(input_1_shape) < 1 or len(input_2_shape) < 1:
        raise ValueError("Sub expects non-empty input shapes.")

    output_shape = _broadcast_shape(input_1_shape, input_2_shape)

    builder = LiteRtSingleOpBuilder(op_name="SUB")

    input1_tensor_idx = builder.add_tensor(
        TensorSpec(
            name="input1",
            shape=input_1_shape,
            tensor_type=tensor_type,
            is_input=True,
            quantization=_default_quant(tensor_type),
        )
    )

    input2_tensor_idx = builder.add_tensor(
        TensorSpec(
            name="input2",
            shape=input_2_shape,
            tensor_type=tensor_type,
            is_input=True,
            quantization=_default_quant(tensor_type),
        )
    )

    output_tensor_idx = builder.add_tensor(
        TensorSpec(
            name="output",
            shape=output_shape,
            tensor_type=tensor_type,
            is_output=True,
            quantization=_default_quant(tensor_type),
        )
    )

    options = litert.SubOptionsT()
    options.fusedActivationFunction = litert.ActivationFunctionType.NONE

    builder.add_operator(
        "SUB",
        inputs=[input1_tensor_idx, input2_tensor_idx],
        outputs=[output_tensor_idx],
        options=options,
        options_type=litert.BuiltinOptions.SubOptions,
    )

    return builder.build()


def build_mul_op(
    *,
    input_1_shape: Iterable[int],
    input_2_shape: Iterable[int],
    dtype: str = "int8",
) -> bytes:
    if not LITERT_AVAILABLE:
        raise ImportError("ai_edge_litert is not available. Install it with: pip install ai-edge-litert")

    tensor_type = _DTYPE_MAP.get(dtype.lower())
    if tensor_type is None:
        raise ValueError(f"Unsupported dtype '{dtype}'.")

    input_1_shape = tuple(int(dim) for dim in input_1_shape)
    input_2_shape = tuple(int(dim) for dim in input_2_shape)
    if len(input_1_shape) < 1 or len(input_2_shape) < 1:
        raise ValueError("Mul expects non-empty input shapes.")

    output_shape = _broadcast_shape(input_1_shape, input_2_shape)

    builder = LiteRtSingleOpBuilder(op_name="MUL")

    input1_tensor_idx = builder.add_tensor(
        TensorSpec(
            name="input1",
            shape=input_1_shape,
            tensor_type=tensor_type,
            is_input=True,
            quantization=_default_quant(tensor_type),
        )
    )

    input2_tensor_idx = builder.add_tensor(
        TensorSpec(
            name="input2",
            shape=input_2_shape,
            tensor_type=tensor_type,
            is_input=True,
            quantization=_default_quant(tensor_type),
        )
    )

    output_tensor_idx = builder.add_tensor(
        TensorSpec(
            name="output",
            shape=output_shape,
            tensor_type=tensor_type,
            is_output=True,
            quantization=_default_quant(tensor_type),
        )
    )

    options = litert.MulOptionsT()
    options.fusedActivationFunction = litert.ActivationFunctionType.NONE

    builder.add_operator(
        "MUL",
        inputs=[input1_tensor_idx, input2_tensor_idx],
        outputs=[output_tensor_idx],
        options=options,
        options_type=litert.BuiltinOptions.MulOptions,
    )

    return builder.build()


def build_comparison_op(
    *,
    input_1_shape: Iterable[int],
    input_2_shape: Iterable[int],
    op_name: str,
    dtype: str = "int8",
) -> bytes:
    if not LITERT_AVAILABLE:
        raise ImportError("ai_edge_litert is not available. Install it with: pip install ai-edge-litert")

    tensor_type = _DTYPE_MAP.get(dtype.lower())
    if tensor_type is None:
        raise ValueError(f"Unsupported dtype '{dtype}'.")

    input_1_shape = tuple(int(dim) for dim in input_1_shape)
    input_2_shape = tuple(int(dim) for dim in input_2_shape)
    if len(input_1_shape) < 1 or len(input_2_shape) < 1:
        raise ValueError("Comparison expects non-empty input shapes.")

    output_shape = _broadcast_shape(input_1_shape, input_2_shape)

    builder = LiteRtSingleOpBuilder(op_name=op_name)

    input1_tensor_idx = builder.add_tensor(
        TensorSpec(
            name="input1",
            shape=input_1_shape,
            tensor_type=tensor_type,
            is_input=True,
            quantization=_default_quant(tensor_type),
        )
    )

    input2_tensor_idx = builder.add_tensor(
        TensorSpec(
            name="input2",
            shape=input_2_shape,
            tensor_type=tensor_type,
            is_input=True,
            quantization=_default_quant(tensor_type),
        )
    )

    output_tensor_idx = builder.add_tensor(
        TensorSpec(
            name="output",
            shape=output_shape,
            tensor_type=litert.TensorType.BOOL,
            is_output=True,
            quantization=None,
        )
    )

    builder.add_operator(
        op_name,
        inputs=[input1_tensor_idx, input2_tensor_idx],
        outputs=[output_tensor_idx],
        options=None,
        options_type=litert.BuiltinOptions.NONE,
    )

    return builder.build()


def build_concat_op(
    *,
    input_shapes: Sequence[Sequence[int]],
    axis: int,
    dtype: str = "int8",
) -> bytes:
    if not LITERT_AVAILABLE:
        raise ImportError("ai_edge_litert is not available. Install it with: pip install ai-edge-litert")

    tensor_type = _DTYPE_MAP.get(dtype.lower())
    if tensor_type is None:
        raise ValueError(f"Unsupported dtype '{dtype}'.")

    shapes = [tuple(int(dim) for dim in shape) for shape in input_shapes]
    if not shapes:
        raise ValueError("Concat expects at least one input shape.")

    rank = len(shapes[0])
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank:
        raise ValueError(f"Axis {axis} out of range for rank {rank}.")

    output_shape = list(shapes[0])
    output_shape[axis] = sum(int(shape[axis]) for shape in shapes)
    output_shape = tuple(output_shape)

    builder = LiteRtSingleOpBuilder(op_name="CONCATENATION")

    input_indices = []
    for i, shape in enumerate(shapes):
        quant = _default_quant(tensor_type)
        input_idx = builder.add_tensor(
            TensorSpec(
                name=f"input{i+1}",
                shape=shape,
                tensor_type=tensor_type,
                is_input=True,
                quantization=quant,
            )
        )
        input_indices.append(input_idx)

    output_quant = _default_quant(tensor_type)
    output_idx = builder.add_tensor(
        TensorSpec(
            name="output",
            shape=output_shape,
            tensor_type=tensor_type,
            is_output=True,
            quantization=output_quant,
        )
    )

    options = litert.ConcatenationOptionsT()
    options.axis = int(axis)
    options.fusedActivationFunction = litert.ActivationFunctionType.NONE

    builder.add_operator(
        "CONCATENATION",
        inputs=input_indices,
        outputs=[output_idx],
        options=options,
        options_type=litert.BuiltinOptions.ConcatenationOptions,
    )

    return builder.build()


def build_gather_op(
    *,
    input_shape: Iterable[int],
    indices_shape: Iterable[int],
    axis: int = 0,
    batch_dims: int = 0,
    dtype: str = "int8",
) -> bytes:
    if not LITERT_AVAILABLE:
        raise ImportError("ai_edge_litert is not available. Install it with: pip install ai-edge-litert")

    tensor_type = _DTYPE_MAP.get(dtype.lower())
    if tensor_type is None:
        raise ValueError(f"Unsupported dtype '{dtype}'.")

    input_shape = tuple(int(dim) for dim in input_shape)
    indices_shape = tuple(int(dim) for dim in indices_shape)
    rank = len(input_shape)
    if rank == 0:
        raise ValueError("Gather expects input shape with at least one dimension.")

    axis_norm = axis
    if axis_norm < 0:
        axis_norm += rank
    if axis_norm < 0 or axis_norm >= rank:
        raise ValueError(f"Axis {axis} out of range for rank {rank}.")

    if batch_dims < 0 or batch_dims > axis_norm:
        raise ValueError("batch_dims must be between 0 and axis.")

    output_shape = input_shape[:axis_norm] + indices_shape[batch_dims:] + input_shape[axis_norm + 1 :]

    builder = LiteRtSingleOpBuilder(op_name="GATHER")

    input_tensor_idx = builder.add_tensor(
        TensorSpec(
            name="input",
            shape=input_shape,
            tensor_type=tensor_type,
            is_input=True,
            quantization=_default_quant(tensor_type),
        )
    )

    indices_tensor_idx = builder.add_tensor(
        TensorSpec(
            name="indices",
            shape=indices_shape,
            tensor_type=litert.TensorType.INT32,
            is_input=True,
        )
    )

    output_tensor_idx = builder.add_tensor(
        TensorSpec(
            name="output",
            shape=output_shape,
            tensor_type=tensor_type,
            is_output=True,
            quantization=_default_quant(tensor_type),
        )
    )

    options = litert.GatherOptionsT()
    options.axis = int(axis_norm)
    options.batchDims = int(batch_dims)

    builder.add_operator(
        "GATHER",
        inputs=[input_tensor_idx, indices_tensor_idx],
        outputs=[output_tensor_idx],
        options=options,
        options_type=litert.BuiltinOptions.GatherOptions,
    )

    return builder.build()


def build_gather_nd_op(
    *,
    params_shape: Iterable[int],
    indices_shape: Iterable[int],
    dtype: str = "int8",
) -> bytes:
    if not LITERT_AVAILABLE:
        raise ImportError("ai_edge_litert is not available. Install it with: pip install ai-edge-litert")

    tensor_type = _DTYPE_MAP.get(dtype.lower())
    if tensor_type is None:
        raise ValueError(f"Unsupported dtype '{dtype}'.")

    params_shape = tuple(int(dim) for dim in params_shape)
    indices_shape = tuple(int(dim) for dim in indices_shape)
    if len(indices_shape) < 1:
        raise ValueError("indices_shape must have at least one dimension.")

    indices_nd = int(indices_shape[-1])
    if indices_nd <= 0 or indices_nd > len(params_shape):
        raise ValueError("indices_shape last dimension must be in [1, params_rank].")

    output_shape = indices_shape[:-1] + params_shape[indices_nd:]

    builder = LiteRtSingleOpBuilder(op_name="GATHER_ND")

    params_tensor_idx = builder.add_tensor(
        TensorSpec(
            name="params",
            shape=params_shape,
            tensor_type=tensor_type,
            is_input=True,
            quantization=_default_quant(tensor_type),
        )
    )

    indices_tensor_idx = builder.add_tensor(
        TensorSpec(
            name="indices",
            shape=indices_shape,
            tensor_type=litert.TensorType.INT32,
            is_input=True,
        )
    )

    output_tensor_idx = builder.add_tensor(
        TensorSpec(
            name="output",
            shape=output_shape,
            tensor_type=tensor_type,
            is_output=True,
            quantization=_default_quant(tensor_type),
        )
    )

    options = litert.GatherNdOptionsT()

    builder.add_operator(
        "GATHER_ND",
        inputs=[params_tensor_idx, indices_tensor_idx],
        outputs=[output_tensor_idx],
        options=options,
        options_type=litert.BuiltinOptions.GatherNdOptions,
    )

    return builder.build()


def build_split_op(
    *,
    input_shape: Iterable[int],
    axis: int,
    num_splits: Optional[int] = None,
    size_splits: Optional[Sequence[int]] = None,
    dtype: str = "int8",
) -> bytes:
    if not LITERT_AVAILABLE:
        raise ImportError("ai_edge_litert is not available. Install it with: pip install ai-edge-litert")

    tensor_type = _DTYPE_MAP.get(dtype.lower())
    if tensor_type is None:
        raise ValueError(f"Unsupported dtype '{dtype}'.")

    input_shape = tuple(int(dim) for dim in input_shape)
    rank = len(input_shape)
    if rank == 0:
        raise ValueError("Split expects input shape with at least one dimension.")

    axis_norm = axis
    if axis_norm < 0:
        axis_norm += rank
    if axis_norm < 0 or axis_norm >= rank:
        raise ValueError(f"Axis {axis} out of range for rank {rank}.")

    if size_splits is not None:
        split_dims = [int(v) for v in size_splits]
        if not split_dims:
            raise ValueError("size_splits must not be empty.")
        num_splits_val = len(split_dims)
    elif num_splits is not None:
        if num_splits <= 0:
            raise ValueError("num_splits must be > 0.")
        axis_size = int(input_shape[axis_norm])
        if axis_size % num_splits != 0:
            raise ValueError("num_splits must evenly divide the axis dimension.")
        split_size = axis_size // num_splits
        split_dims = [split_size] * int(num_splits)
        num_splits_val = int(num_splits)
    else:
        raise ValueError("Split requires either num_splits or size_splits.")

    output_shapes = []
    for dim in split_dims:
        out_shape = list(input_shape)
        out_shape[axis_norm] = int(dim)
        output_shapes.append(tuple(out_shape))

    builder = LiteRtSingleOpBuilder(op_name="SPLIT_V" if size_splits is not None else "SPLIT")

    input_tensor_idx = builder.add_tensor(
        TensorSpec(
            name="input",
            shape=input_shape,
            tensor_type=tensor_type,
            is_input=True,
            quantization=_default_quant(tensor_type),
        )
    )

    axis_tensor_idx = builder.add_tensor(
        TensorSpec(
            name="axis",
            shape=(1,),
            tensor_type=litert.TensorType.INT32,
            data=[axis_norm],
        )
    )

    input_indices = []
    if size_splits is not None:
        size_tensor_idx = builder.add_tensor(
            TensorSpec(
                name="size_splits",
                shape=(len(split_dims),),
                tensor_type=litert.TensorType.INT32,
                data=split_dims,
            )
        )
        input_indices = [input_tensor_idx, size_tensor_idx, axis_tensor_idx]
        options = litert.SplitVOptionsT()
        options.numSplits = int(num_splits_val)
        options_type = litert.BuiltinOptions.SplitVOptions
        op_name = "SPLIT_V"
    else:
        input_indices = [axis_tensor_idx, input_tensor_idx]
        options = litert.SplitOptionsT()
        options.numSplits = int(num_splits_val)
        options_type = litert.BuiltinOptions.SplitOptions
        op_name = "SPLIT"

    output_indices = []
    for idx, shape in enumerate(output_shapes):
        output_idx = builder.add_tensor(
            TensorSpec(
                name=f"output{idx}",
                shape=shape,
                tensor_type=tensor_type,
                is_output=True,
                quantization=_default_quant(tensor_type),
            )
        )
        output_indices.append(output_idx)

    builder.add_operator(
        op_name,
        inputs=input_indices,
        outputs=output_indices,
        options=options,
        options_type=options_type,
    )

    return builder.build()


def build_strided_slice_op(
    *,
    input_shape: Iterable[int],
    begin: Sequence[int],
    end: Sequence[int],
    strides: Sequence[int],
    shrink_axis_mask: int = 0,
    dtype: str = "int8",
) -> bytes:
    """Build a single-op STRIDED_SLICE model, mainly to unlock float (FP16)
    dtypes not reachable via the Keras/TFLiteConverter path used for the
    quantized variants. Output shape is derived directly from Python slicing
    semantics rather than TF's op resolver, so masks other than
    ``shrink_axis_mask`` are intentionally unsupported.
    """
    if not LITERT_AVAILABLE:
        raise ImportError("ai_edge_litert is not available. Install it with: pip install ai-edge-litert")

    tensor_type = _DTYPE_MAP.get(dtype.lower())
    if tensor_type is None:
        raise ValueError(f"Unsupported dtype '{dtype}'.")

    input_shape = tuple(int(dim) for dim in input_shape)
    rank = len(input_shape)
    begin = [int(v) for v in begin]
    end = [int(v) for v in end]
    strides = [int(v) for v in strides]
    if not (len(begin) == len(end) == len(strides) == rank):
        raise ValueError("begin/end/strides must match input rank.")

    output_shape = []
    for i in range(rank):
        dim = input_shape[i]
        if shrink_axis_mask & (1 << i):
            output_shape.append(1)
        else:
            output_shape.append(len(range(dim)[begin[i]:end[i]:strides[i]]))

    kept_axes = [i for i in range(rank) if not (shrink_axis_mask & (1 << i))]
    squeezed_output_shape = tuple(output_shape[i] for i in kept_axes) or (1,)

    builder = LiteRtSingleOpBuilder(op_name="STRIDED_SLICE")

    input_tensor_idx = builder.add_tensor(
        TensorSpec(
            name="input",
            shape=input_shape,
            tensor_type=tensor_type,
            is_input=True,
            quantization=_default_quant(tensor_type),
        )
    )

    begin_tensor_idx = builder.add_tensor(
        TensorSpec(name="begin", shape=(rank,), tensor_type=litert.TensorType.INT32, data=begin)
    )
    end_tensor_idx = builder.add_tensor(
        TensorSpec(name="end", shape=(rank,), tensor_type=litert.TensorType.INT32, data=end)
    )
    strides_tensor_idx = builder.add_tensor(
        TensorSpec(name="strides", shape=(rank,), tensor_type=litert.TensorType.INT32, data=strides)
    )

    output_tensor_idx = builder.add_tensor(
        TensorSpec(
            name="output",
            shape=squeezed_output_shape,
            tensor_type=tensor_type,
            is_output=True,
            quantization=_default_quant(tensor_type),
        )
    )

    options = litert.StridedSliceOptionsT()
    options.beginMask = 0
    options.endMask = 0
    options.ellipsisMask = 0
    options.newAxisMask = 0
    options.shrinkAxisMask = int(shrink_axis_mask)
    options.offset = False

    builder.add_operator(
        "STRIDED_SLICE",
        inputs=[input_tensor_idx, begin_tensor_idx, end_tensor_idx, strides_tensor_idx],
        outputs=[output_tensor_idx],
        options=options,
        options_type=litert.BuiltinOptions.StridedSliceOptions,
    )

    return builder.build()


def build_prelu_op(
    *,
    input_shape: Iterable[int],
    alpha_shape: Optional[Iterable[int]] = None,
    alpha_values: Optional[Sequence[float]] = None,
    dtype: str = "int8",
) -> bytes:
    if not LITERT_AVAILABLE:
        raise ImportError("ai_edge_litert is not available. Install it with: pip install ai-edge-litert")

    tensor_type = _DTYPE_MAP.get(dtype.lower())
    if tensor_type is None:
        raise ValueError(f"Unsupported dtype '{dtype}'.")

    input_shape = tuple(int(dim) for dim in input_shape)
    if len(input_shape) < 2:
        raise ValueError("PReLU expects an input shape with batch dimension.")

    if alpha_shape is None:
        alpha_shape = input_shape[1:]
    alpha_shape = tuple(int(dim) for dim in alpha_shape)
    if len(alpha_shape) == 0:
        raise ValueError("alpha_shape must include at least one dimension.")

    num_alpha = int(np.prod(alpha_shape))
    if num_alpha <= 0:
        raise ValueError("alpha_shape must have positive dimensions.")

    if alpha_values is None:
        alpha_values = np.linspace(0.05, 0.25, num=num_alpha, dtype=np.float32).tolist()
    else:
        alpha_values = list(alpha_values)
        if len(alpha_values) == 1:
            alpha_values = [float(alpha_values[0])] * num_alpha
        elif len(alpha_values) != num_alpha:
            raise ValueError(
                f"alpha_values has {len(alpha_values)} entries, expected {num_alpha} to match alpha_shape."
            )

    quant = _default_quant(tensor_type)
    if quant is None:
        raise ValueError("Missing default quantization for tensor type.")
    scale = float(quant[0][0])
    zero_point = int(quant[1][0])

    alpha_float = np.array(alpha_values, dtype=np.float32).reshape(alpha_shape)
    alpha_q = np.round(alpha_float / scale + zero_point).astype(np.int32)
    if tensor_type == litert.TensorType.INT8:
        alpha_q = np.clip(alpha_q, -128, 127).astype(np.int8)
    elif tensor_type == litert.TensorType.INT16:
        alpha_q = np.clip(alpha_q, -32768, 32767).astype(np.int16)
    else:
        raise ValueError(f"Unsupported tensor type for alpha: {tensor_type}")

    builder = LiteRtSingleOpBuilder(op_name="PRELU")

    input_tensor_idx = builder.add_tensor(
        TensorSpec(
            name="input",
            shape=input_shape,
            tensor_type=tensor_type,
            is_input=True,
            quantization=quant,
        )
    )

    alpha_tensor_idx = builder.add_tensor(
        TensorSpec(
            name="alpha",
            shape=alpha_shape,
            tensor_type=tensor_type,
            is_input=False,
            quantization=quant,
            data=alpha_q,
        )
    )

    output_tensor_idx = builder.add_tensor(
        TensorSpec(
            name="output",
            shape=input_shape,
            tensor_type=tensor_type,
            is_output=True,
            quantization=quant,
        )
    )

    builder.add_operator(
        "PRELU",
        inputs=[input_tensor_idx, alpha_tensor_idx],
        outputs=[output_tensor_idx],
        options=None,
        options_type=litert.BuiltinOptions.NONE,
    )

    return builder.build()


def _require_int4() -> int:
    if _INT4_TENSOR_TYPE is None:
        raise RuntimeError("LiteRT INT4 tensor type is not available in this environment.")
    return _INT4_TENSOR_TYPE


def build_conv2d_s4_op(
    *,
    input_shape: Iterable[int],
    filter_shape: Iterable[int],
    strides: Iterable[int],
    padding: str,
    dilation: Iterable[int],
    use_bias: bool,
    input_quant: Tuple[Sequence[float], Sequence[int]],
    weight_quant: Tuple[Sequence[float], Sequence[int]],
    output_quant: Tuple[Sequence[float], Sequence[int]],
    weights_int4: np.ndarray,
    biases: Optional[np.ndarray],
) -> bytes:
    if not LITERT_AVAILABLE:
        raise ImportError("ai_edge_litert is not available. Install it with: pip install ai-edge-litert")

    int4_type = _require_int4()
    input_shape = tuple(int(v) for v in input_shape)
    filter_shape = tuple(int(v) for v in filter_shape)  # OHWI
    strides = tuple(int(v) for v in strides)
    dilation = tuple(int(v) for v in dilation)

    builder = LiteRtSingleOpBuilder(op_name="CONV_2D")

    input_tensor_idx = builder.add_tensor(
        TensorSpec(
            name="input",
            shape=input_shape,
            tensor_type=litert.TensorType.INT8,
            is_input=True,
            quantization=input_quant,
        )
    )

    packed_weights = pack_s4(weights_int4)
    weight_tensor_idx = builder.add_tensor(
        TensorSpec(
            name="weights",
            shape=filter_shape,
            tensor_type=int4_type,
            quantization=weight_quant,
            quantized_dimension=0,  # output channel
            data=packed_weights,
        )
    )

    bias_tensor_idx = -1
    if use_bias and biases is not None:
        bias_tensor_idx = builder.add_tensor(
            TensorSpec(
                name="bias",
                shape=(filter_shape[0],),
                tensor_type=litert.TensorType.INT32,
                data=biases.astype(np.int32),
            )
        )

    in_h, in_w = input_shape[1], input_shape[2]
    kh, kw = filter_shape[1], filter_shape[2]
    eff_kh = (kh - 1) * dilation[0] + 1
    eff_kw = (kw - 1) * dilation[1] + 1
    if str(padding).lower() == "same":
        out_h = int(np.ceil(in_h / strides[0]))
        out_w = int(np.ceil(in_w / strides[1]))
    else:
        out_h = int(np.floor((in_h - eff_kh) / strides[0]) + 1)
        out_w = int(np.floor((in_w - eff_kw) / strides[1]) + 1)
    output_shape = (input_shape[0], out_h, out_w, filter_shape[0])
    output_tensor_idx = builder.add_tensor(
        TensorSpec(
            name="output",
            shape=output_shape,
            tensor_type=litert.TensorType.INT8,
            is_output=True,
            quantization=output_quant,
        )
    )

    options = litert.Conv2DOptionsT()
    options.padding = litert.Padding.SAME if str(padding).lower() == "same" else litert.Padding.VALID
    options.strideH = int(strides[0])
    options.strideW = int(strides[1])
    options.dilationHFactor = int(dilation[0])
    options.dilationWFactor = int(dilation[1])
    options.fusedActivationFunction = litert.ActivationFunctionType.NONE

    op_inputs = [input_tensor_idx, weight_tensor_idx]
    if use_bias and bias_tensor_idx >= 0:
        op_inputs.append(bias_tensor_idx)

    builder.add_operator(
        "CONV_2D",
        inputs=op_inputs,
        outputs=[output_tensor_idx],
        options=options,
        options_type=litert.BuiltinOptions.Conv2DOptions,
    )

    return builder.build()


def build_depthwise_conv2d_s4_op(
    *,
    input_shape: Iterable[int],
    filter_shape: Iterable[int],
    strides: Iterable[int],
    padding: str,
    dilation: Iterable[int],
    depth_multiplier: int,
    use_bias: bool,
    input_quant: Tuple[Sequence[float], Sequence[int]],
    weight_quant: Tuple[Sequence[float], Sequence[int]],
    output_quant: Tuple[Sequence[float], Sequence[int]],
    weights_int4: np.ndarray,
    biases: Optional[np.ndarray],
) -> bytes:
    if not LITERT_AVAILABLE:
        raise ImportError("ai_edge_litert is not available. Install it with: pip install ai-edge-litert")

    int4_type = _require_int4()
    input_shape = tuple(int(v) for v in input_shape)
    filter_shape = tuple(int(v) for v in filter_shape)  # TFLite depthwise: [1, H, W, C_out]
    strides = tuple(int(v) for v in strides)
    dilation = tuple(int(v) for v in dilation)

    builder = LiteRtSingleOpBuilder(op_name="DEPTHWISE_CONV_2D")

    input_tensor_idx = builder.add_tensor(
        TensorSpec(
            name="input",
            shape=input_shape,
            tensor_type=litert.TensorType.INT8,
            is_input=True,
            quantization=input_quant,
        )
    )

    packed_weights = pack_s4(weights_int4)
    weight_tensor_idx = builder.add_tensor(
        TensorSpec(
            name="weights",
            shape=filter_shape,
            tensor_type=int4_type,
            quantization=weight_quant,
            quantized_dimension=3,
            data=packed_weights,
        )
    )

    bias_tensor_idx = -1
    if use_bias and biases is not None:
        out_ch = filter_shape[3]
        bias_tensor_idx = builder.add_tensor(
            TensorSpec(
                name="bias",
                shape=(out_ch,),
                tensor_type=litert.TensorType.INT32,
                data=biases.astype(np.int32),
            )
        )

    in_h, in_w = input_shape[1], input_shape[2]
    kh, kw = filter_shape[1], filter_shape[2]
    eff_kh = (kh - 1) * dilation[0] + 1
    eff_kw = (kw - 1) * dilation[1] + 1
    if str(padding).lower() == "same":
        out_h = int(np.ceil(in_h / strides[0]))
        out_w = int(np.ceil(in_w / strides[1]))
    else:
        out_h = int(np.floor((in_h - eff_kh) / strides[0]) + 1)
        out_w = int(np.floor((in_w - eff_kw) / strides[1]) + 1)
    out_ch = filter_shape[3]
    output_shape = (input_shape[0], out_h, out_w, out_ch)
    output_tensor_idx = builder.add_tensor(
        TensorSpec(
            name="output",
            shape=output_shape,
            tensor_type=litert.TensorType.INT8,
            is_output=True,
            quantization=output_quant,
        )
    )

    options = litert.DepthwiseConv2DOptionsT()
    options.padding = litert.Padding.SAME if str(padding).lower() == "same" else litert.Padding.VALID
    options.strideH = int(strides[0])
    options.strideW = int(strides[1])
    options.dilationHFactor = int(dilation[0])
    options.dilationWFactor = int(dilation[1])
    options.depthMultiplier = int(depth_multiplier)
    options.fusedActivationFunction = litert.ActivationFunctionType.NONE

    op_inputs = [input_tensor_idx, weight_tensor_idx]
    if use_bias and bias_tensor_idx >= 0:
        op_inputs.append(bias_tensor_idx)

    builder.add_operator(
        "DEPTHWISE_CONV_2D",
        inputs=op_inputs,
        outputs=[output_tensor_idx],
        options=options,
        options_type=litert.BuiltinOptions.DepthwiseConv2DOptions,
    )

    return builder.build()


def build_fully_connected_s4_op(
    *,
    input_shape: Iterable[int],
    filter_shape: Iterable[int],
    use_bias: bool,
    input_quant: Tuple[Sequence[float], Sequence[int]],
    weight_quant: Tuple[Sequence[float], Sequence[int]],
    output_quant: Tuple[Sequence[float], Sequence[int]],
    weights_int4: np.ndarray,
    biases: Optional[np.ndarray],
) -> bytes:
    if not LITERT_AVAILABLE:
        raise ImportError("ai_edge_litert is not available. Install it with: pip install ai-edge-litert")

    int4_type = _require_int4()
    input_shape = tuple(int(v) for v in input_shape)
    filter_shape = tuple(int(v) for v in filter_shape)  # O, I

    builder = LiteRtSingleOpBuilder(op_name="FULLY_CONNECTED")

    input_tensor_idx = builder.add_tensor(
        TensorSpec(
            name="input",
            shape=input_shape,
            tensor_type=litert.TensorType.INT8,
            is_input=True,
            quantization=input_quant,
        )
    )

    packed_weights = pack_s4(weights_int4)
    weight_tensor_idx = builder.add_tensor(
        TensorSpec(
            name="weights",
            shape=filter_shape,
            tensor_type=int4_type,
            quantization=weight_quant,
            quantized_dimension=0,
            data=packed_weights,
        )
    )

    bias_tensor_idx = -1
    if use_bias and biases is not None:
        bias_tensor_idx = builder.add_tensor(
            TensorSpec(
                name="bias",
                shape=(filter_shape[0],),
                tensor_type=litert.TensorType.INT32,
                data=biases.astype(np.int32),
            )
        )

    output_shape = (input_shape[0], filter_shape[0])
    output_tensor_idx = builder.add_tensor(
        TensorSpec(
            name="output",
            shape=output_shape,
            tensor_type=litert.TensorType.INT8,
            is_output=True,
            quantization=output_quant,
        )
    )

    options = litert.FullyConnectedOptionsT()
    options.fusedActivationFunction = litert.ActivationFunctionType.NONE
    options.weightsFormat = litert.FullyConnectedOptionsWeightsFormat.DEFAULT

    op_inputs = [input_tensor_idx, weight_tensor_idx]
    if use_bias and bias_tensor_idx >= 0:
        op_inputs.append(bias_tensor_idx)

    builder.add_operator(
        "FULLY_CONNECTED",
        inputs=op_inputs,
        outputs=[output_tensor_idx],
        options=options,
        options_type=litert.BuiltinOptions.FullyConnectedOptions,
    )

    return builder.build()
