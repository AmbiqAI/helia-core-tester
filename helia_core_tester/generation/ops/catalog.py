"""Canonical operator catalog for grouped CMSIS-NN parity layout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Literal, Optional


ParityKind = Literal["cmsis", "extension"]


@dataclass(frozen=True)
class OperatorSpec:
    operator: str
    family: str
    parity_kind: ParityKind
    module_path: str
    class_name: str
    descriptor_relpaths: tuple[str, ...]
    template_relpath: Optional[str]
    artifact_family_dir: str
    rationale: Optional[str] = None

    @property
    def descriptor_relpath(self) -> Optional[str]:
        if not self.descriptor_relpaths:
            return None
        return self.descriptor_relpaths[0]

    @property
    def descriptor_stem(self) -> Optional[str]:
        if self.descriptor_relpath is None:
            return None
        return Path(self.descriptor_relpath).stem


def _spec(
    operator: str,
    family: str,
    module_basename: str,
    class_name: str,
    descriptor_relpath: Optional[str] = None,
    template_relpath: Optional[str] = None,
    *,
    descriptor_relpaths: Optional[tuple[str, ...]] = None,
    parity_kind: ParityKind = "cmsis",
    rationale: Optional[str] = None,
) -> OperatorSpec:
    return OperatorSpec(
        operator=operator,
        family=family,
        parity_kind=parity_kind,
        module_path=f"helia_core_tester.generation.ops.{family}.{module_basename}",
        class_name=class_name,
        descriptor_relpaths=descriptor_relpaths or ((descriptor_relpath,) if descriptor_relpath else ()),
        template_relpath=template_relpath,
        artifact_family_dir=family,
        rationale=rationale,
    )


OPERATOR_SPECS: Dict[str, OperatorSpec] = {

    "HardSwishCompat": _spec(
        "HardSwishCompat",
        "ActivationFunctions",
        "hard_swish_compat",
        "OpHardSwishCompat",
        "ActivationFunctions/hard_swish_compat.yaml",
        "ActivationFunctions/hard_swish",
    ),
    "PReLU": _spec(
        "PReLU",
        "ActivationFunctions",
        "prelu",
        "OpPReLU",
        descriptor_relpaths=("ActivationFunctions/prelu.yaml", "ActivationFunctions/prelu_float.yaml"),
        template_relpath="ActivationFunctions/prelu",
    ),
    "PReLUScalar": _spec(
        "PReLUScalar",
        "ActivationFunctions",
        "prelu_scalar",
        "OpPReLUScalar",
        "ActivationFunctions/prelu_scalar.yaml",
        "ActivationFunctions/prelu_scalar",
    ),
    "Clamp": _spec("Clamp", "ActivationFunctions", "clamp", "OpClamp", "ActivationFunctions/clamp.yaml", "ActivationFunctions/clamp"),
    "NNActivationS16": _spec(
        "NNActivationS16",
        "ActivationFunctions",
        "nn_activation_s16",
        "OpNNActivationS16",
        None,
        "ActivationFunctions/nn_activation",
    ),
    "Relu": _spec("Relu", "ActivationFunctions", "relu", "OpRelu", "ActivationFunctions/relu.yaml", "ActivationFunctions/relu"),
    "Relu6": _spec("Relu6", "ActivationFunctions", "relu6", "OpRelu6", "ActivationFunctions/relu6.yaml", "ActivationFunctions/relu6"),
    "LeakyRelu": _spec("LeakyRelu", "ActivationFunctions", "leaky_relu", "OpLeakyRelu", "ActivationFunctions/leaky_relu.yaml", "ActivationFunctions/leaky_relu"),
    "Tanh": _spec("Tanh", "ActivationFunctions", "tanh", "OpTanh", "ActivationFunctions/tanh.yaml", "ActivationFunctions/tanh"),
    "Logistic": _spec("Logistic", "ActivationFunctions", "logistic", "OpLogistic", "ActivationFunctions/logistic.yaml", "ActivationFunctions/logistic"),
    "HardSwishPrecise": _spec(
        "HardSwishPrecise",
        "ActivationFunctions",
        "hard_swish_precise",
        "OpHardSwishPrecise",
        "ActivationFunctions/hard_swish_precise.yaml",
        "ActivationFunctions/hard_swish",
    ),
    "Abs": _spec(
        "Abs",
        "BasicMathFunctions",
        "abs",
        "OpAbs",
        descriptor_relpaths=("BasicMathFunctions/abs.yaml", "BasicMathFunctions/abs_float.yaml"),
        template_relpath="BasicMathFunctions/abs",
    ),
    "Add": _spec(
        "Add",
        "BasicMathFunctions",
        "add",
        "OpAdd",
        descriptor_relpaths=("BasicMathFunctions/add.yaml", "BasicMathFunctions/add_float.yaml"),
        template_relpath="BasicMathFunctions/add",
    ),
    "Sub": _spec(
        "Sub",
        "BasicMathFunctions",
        "sub",
        "OpSub",
        descriptor_relpaths=("BasicMathFunctions/sub.yaml", "BasicMathFunctions/sub_float.yaml"),
        template_relpath="BasicMathFunctions/sub",
    ),
    "Mul": _spec(
        "Mul",
        "BasicMathFunctions",
        "mul",
        "OpMul",
        descriptor_relpaths=("BasicMathFunctions/mul.yaml", "BasicMathFunctions/mul_float.yaml"),
        template_relpath="BasicMathFunctions/mul",
    ),
    "Maximum": _spec(
        "Maximum",
        "BasicMathFunctions",
        "minmax",
        "OpMinMax",
        descriptor_relpaths=("BasicMathFunctions/maximum.yaml", "BasicMathFunctions/maximum_float.yaml"),
        template_relpath="BasicMathFunctions/minmax",
    ),
    "Minimum": _spec(
        "Minimum",
        "BasicMathFunctions",
        "minmax",
        "OpMinMax",
        descriptor_relpaths=("BasicMathFunctions/minimum.yaml", "BasicMathFunctions/minimum_float.yaml"),
        template_relpath="BasicMathFunctions/minmax",
    ),
    "Mean": _spec("Mean", "BasicMathFunctions", "mean", "OpMean", "BasicMathFunctions/mean.yaml", "BasicMathFunctions/mean"),
    "ReduceMax": _spec("ReduceMax", "BasicMathFunctions", "reduce_max", "OpReduceMax", "BasicMathFunctions/reduce_max.yaml", "BasicMathFunctions/reduce_max"),
    "ReduceSum": _spec(
        "ReduceSum",
        "BasicMathFunctions",
        "reduce_sum",
        "OpReduceSum",
        "BasicMathFunctions/reduce_sum_float.yaml",
        "BasicMathFunctions/reduce_sum",
    ),
    "ReduceMin": _spec("ReduceMin", "BasicMathFunctions", "reduce_min", "OpReduceMin", "BasicMathFunctions/reduce_min.yaml", "BasicMathFunctions/reduce_min"),
    "ArgMax": _spec("ArgMax", "BasicMathFunctions", "argmax", "OpArgMax", "BasicMathFunctions/argmax.yaml", "BasicMathFunctions/argmax"),
    "ArgMin": _spec("ArgMin", "BasicMathFunctions", "argmin", "OpArgMin", "BasicMathFunctions/argmin.yaml", "BasicMathFunctions/argmin"),
    "Sqrt": _spec("Sqrt", "BasicMathFunctions", "sqrt", "OpSqrt", "BasicMathFunctions/sqrt.yaml", "BasicMathFunctions/sqrt"),
    "Rsqrt": _spec("Rsqrt", "BasicMathFunctions", "rsqrt", "OpRsqrt", "BasicMathFunctions/rsqrt.yaml", "BasicMathFunctions/rsqrt"),
    "Comparison": _spec("Comparison", "ComparisonFunctions", "comparison", "OpComparison", "ComparisonFunctions/comparison.yaml", "ComparisonFunctions/comparison"),
    "Concatenation": _spec(
        "Concatenation",
        "ConcatenationFunctions",
        "concatenation",
        "OpConcatenation",
        descriptor_relpaths=("ConcatenationFunctions/concatenation.yaml", "ConcatenationFunctions/concatenation_float.yaml"),
        template_relpath="ConcatenationFunctions/concatenation",
    ),
    "Split": _spec(
        "Split",
        "ConcatenationFunctions",
        "split",
        "OpSplit",
        descriptor_relpaths=("ConcatenationFunctions/split.yaml", "ConcatenationFunctions/split_float.yaml"),
        template_relpath="ConcatenationFunctions/split",
    ),
    "Convolve": _spec(
        "Convolve",
        "ConvolutionFunctions",
        "convolve",
        "OpConvolve",
        descriptor_relpaths=("ConvolutionFunctions/convolve.yaml", "ConvolutionFunctions/convolve_float.yaml"),
        template_relpath="ConvolutionFunctions/convolve",
    ),
    "DepthwiseConv": _spec(
        "DepthwiseConv",
        "ConvolutionFunctions",
        "depthwise_conv",
        "OpDepthwiseConv",
        descriptor_relpaths=("ConvolutionFunctions/depthwise_conv.yaml", "ConvolutionFunctions/depthwise_conv_float.yaml"),
        template_relpath="ConvolutionFunctions/depthwise_conv",
    ),
    "TransposeConv": _spec(
        "TransposeConv",
        "ConvolutionFunctions",
        "transpose_conv",
        "OpTransposeConv",
        descriptor_relpaths=("ConvolutionFunctions/transpose_conv.yaml", "ConvolutionFunctions/transpose_conv_float.yaml"),
        template_relpath="ConvolutionFunctions/transpose_conv",
    ),
    "FullyConnected": _spec(
        "FullyConnected",
        "FullyConnectedFunctions",
        "fully_connected",
        "OpFullyConnected",
        descriptor_relpaths=("FullyConnectedFunctions/fully_connected.yaml", "FullyConnectedFunctions/fully_connected_float.yaml"),
        template_relpath="FullyConnectedFunctions/fully_connected",
    ),
    "BatchMatMul": _spec(
        "BatchMatMul",
        "FullyConnectedFunctions",
        "batch_matmul",
        "OpBatchMatMul",
        descriptor_relpaths=("FullyConnectedFunctions/batch_matmul.yaml", "FullyConnectedFunctions/batch_matmul_float.yaml"),
        template_relpath="FullyConnectedFunctions/batch_matmul",
    ),
    "Gather": _spec("Gather", "GatherFunctions", "gather", "OpGather", "GatherFunctions/gather.yaml", "GatherFunctions/gather"),
    "GatherND": _spec("GatherND", "GatherFunctions", "gather_nd", "OpGatherND", "GatherFunctions/gather_nd.yaml", "GatherFunctions/gather_nd"),
    "LSTMUnidirectional": _spec(
        "LSTMUnidirectional",
        "LSTMFunctions",
        "lstm_unidirectional",
        "OpLSTMUnidirectional",
        descriptor_relpaths=("LSTMFunctions/lstm_unidirectional.yaml", "LSTMFunctions/lstm_unidirectional_float.yaml"),
        template_relpath="LSTMFunctions/lstm_unidirectional",
    ),
    "GRUUnidirectional": _spec(
        "GRUUnidirectional",
        "LSTMFunctions",
        "gru_unidirectional",
        "OpGRUUnidirectional",
        descriptor_relpaths=("LSTMFunctions/gru_unidirectional_float.yaml",),
        template_relpath="LSTMFunctions/gru_unidirectional",
    ),
    "Requantize": _spec("Requantize", "NNSupportFunctions", "requantize", "OpRequantize", "NNSupportFunctions/requantize.yaml", "NNSupportFunctions/requantize"),
    "BatchNorm": _spec(
        "BatchNorm",
        "NNSupportFunctions",
        "batch_norm",
        "OpBatchNorm",
        "NNSupportFunctions/batch_norm_float.yaml",
        "NNSupportFunctions/batch_norm",
    ),
    "Pad": _spec(
        "Pad",
        "PadFunctions",
        "pad",
        "OpPad",
        descriptor_relpaths=("PadFunctions/pad.yaml", "PadFunctions/pad_float.yaml"),
        template_relpath="PadFunctions/pad",
    ),
    "AvgPool": _spec(
        "AvgPool",
        "PoolingFunctions",
        "avg_pool",
        "OpAvgPool",
        descriptor_relpaths=("PoolingFunctions/avg_pool.yaml", "PoolingFunctions/avg_pool_float.yaml"),
        template_relpath="PoolingFunctions/avg_pool",
    ),
    "MaxPool": _spec(
        "MaxPool",
        "PoolingFunctions",
        "max_pool",
        "OpMaxPool",
        descriptor_relpaths=("PoolingFunctions/max_pool.yaml", "PoolingFunctions/max_pool_float.yaml"),
        template_relpath="PoolingFunctions/max_pool",
    ),
    "Quantize": _spec(
        "Quantize",
        "QuantizationFunctions",
        "quantize",
        "OpQuantize",
        descriptor_relpaths=("QuantizationFunctions/quantize.yaml", "QuantizationFunctions/quantize_float.yaml"),
        template_relpath="QuantizationFunctions/quantize",
    ),
    "Dequantize": _spec(
        "Dequantize",
        "QuantizationFunctions",
        "dequantize",
        "OpDequantize",
        descriptor_relpaths=("QuantizationFunctions/dequantize.yaml", "QuantizationFunctions/dequantize_float.yaml"),
        template_relpath="QuantizationFunctions/dequantize",
    ),
    "Reshape": _spec(
        "Reshape",
        "ReshapeFunctions",
        "reshape",
        "OpReshape",
        descriptor_relpaths=("ReshapeFunctions/reshape.yaml", "ReshapeFunctions/reshape_float.yaml"),
        template_relpath="ReshapeFunctions/reshape",
    ),
    "ResizeNearestNeighbor": _spec(
        "ResizeNearestNeighbor",
        "ReshapeFunctions",
        "resize_nearest_neighbor",
        "OpResizeNearestNeighbor",
        "ReshapeFunctions/resize_nearest_neighbor.yaml",
        "ReshapeFunctions/resize_nearest_neighbor",
    ),
    "SquaredDifference": _spec("SquaredDifference", "BasicMathFunctions", "squared_difference", "OpSquaredDifference", "BasicMathFunctions/squared_difference.yaml", "BasicMathFunctions/squared_difference"),
    "SpaceToDepth": _spec("SpaceToDepth", "ReshapeFunctions", "space_to_depth", "OpSpaceToDepth", "ReshapeFunctions/space_to_depth.yaml", "ReshapeFunctions/space_to_depth"),
    "DepthToSpace": _spec("DepthToSpace", "ReshapeFunctions", "depth_to_space", "OpDepthToSpace", "ReshapeFunctions/depth_to_space.yaml", "ReshapeFunctions/depth_to_space"),
    "SpaceToBatchND": _spec("SpaceToBatchND", "ReshapeFunctions", "space_to_batch_nd", "OpSpaceToBatchND", "ReshapeFunctions/space_to_batch_nd.yaml", "ReshapeFunctions/space_to_batch_nd"),
    "BatchToSpaceND": _spec("BatchToSpaceND", "ReshapeFunctions", "batch_to_space_nd", "OpBatchToSpaceND", "ReshapeFunctions/batch_to_space_nd.yaml", "ReshapeFunctions/batch_to_space_nd"),
    "SVDF": _spec(
        "SVDF",
        "SVDFunctions",
        "svdf",
        "OpSVDF",
        descriptor_relpaths=("SVDFunctions/svdf.yaml", "SVDFunctions/svdf_float.yaml"),
        template_relpath="SVDFunctions/svdf",
    ),
    "Softmax": _spec(
        "Softmax",
        "SoftmaxFunctions",
        "softmax",
        "OpSoftmax",
        descriptor_relpaths=("SoftmaxFunctions/softmax.yaml", "SoftmaxFunctions/softmax_float.yaml"),
        template_relpath="SoftmaxFunctions/softmax",
    ),
    "StridedSlice": _spec(
        "StridedSlice",
        "StridedSliceFunctions",
        "strided_slice",
        "OpStridedSlice",
        descriptor_relpaths=("StridedSliceFunctions/strided_slice.yaml", "StridedSliceFunctions/strided_slice_float.yaml"),
        template_relpath="StridedSliceFunctions/strided_slice",
    ),
    "Transpose": _spec(
        "Transpose",
        "TransposeFunctions",
        "transpose",
        "OpTranspose",
        descriptor_relpaths=("TransposeFunctions/transpose.yaml", "TransposeFunctions/transpose_float.yaml"),
        template_relpath="TransposeFunctions/transpose",
    ),
    "NNActivationFloat": _spec(
        "NNActivationFloat",
        "ActivationFunctions",
        "nn_activation_float",
        "OpNNActivationFloat",
        "ActivationFunctions/nn_activation_float.yaml",
        "ActivationFunctions/nn_activation_float",
    ),
    "Fill": _spec(
        "Fill",
        "TesterExtensions",
        "fill",
        "OpFill",
        None,
        None,
        parity_kind="extension",
        rationale="Tester-only utility op with no CMSIS-NN family mapping or generated templates.",
    ),
    "Squeeze": _spec(
        "Squeeze",
        "TesterExtensions",
        "squeeze",
        "OpSqueeze",
        "TesterExtensions/squeeze.yaml",
        "TesterExtensions/squeeze",
        parity_kind="extension",
        rationale="Tester-only op retained for existing standalone generation coverage.",
    ),
    "VariableUpdate": _spec(
        "VariableUpdate",
        "TesterExtensions",
        "variable_update",
        "OpVariableUpdate",
        None,
        None,
        parity_kind="extension",
        rationale="Tester-only stateful op kept public but isolated from CMSIS parity families.",
    ),
    "Tile": _spec("Tile", "TileFunctions", "tile", "OpTile", "TileFunctions/tile.yaml", "TileFunctions/tile"),
    "BroadcastTo": _spec("BroadcastTo", "BroadcastFunctions", "broadcast_to", "OpBroadcastTo", "BroadcastFunctions/broadcast_to.yaml", "BroadcastFunctions/broadcast_to"),
    "ScatterNd": _spec("ScatterNd", "ScatterFunctions", "scatter_nd", "OpScatterNd", "ScatterFunctions/scatter_nd.yaml", "ScatterFunctions/scatter_nd"),
    "MirrorPad": _spec("MirrorPad", "PadFunctions", "mirror_pad", "OpMirrorPad", "PadFunctions/mirror_pad.yaml", "PadFunctions/mirror_pad"),
    "SelectV2": _spec("SelectV2", "SelectFunctions", "select_v2", "OpSelectV2", "SelectFunctions/select_v2.yaml", "SelectFunctions/select_v2"),
    "Where": _spec("Where", "SelectFunctions", "where", "OpWhere", "SelectFunctions/where.yaml", "SelectFunctions/where"),
    "ReverseSequence": _spec("ReverseSequence", "ReverseSequenceFunctions", "reverse_sequence", "OpReverseSequence", "ReverseSequenceFunctions/reverse_sequence.yaml", "ReverseSequenceFunctions/reverse_sequence"),
    "DynamicUpdateSlice": _spec("DynamicUpdateSlice", "DynamicUpdateSliceFunctions", "dynamic_update_slice", "OpDynamicUpdateSlice", "DynamicUpdateSliceFunctions/dynamic_update_slice.yaml", "DynamicUpdateSliceFunctions/dynamic_update_slice"),
}


def get_operator_spec(operator: str) -> OperatorSpec:
    try:
        return OPERATOR_SPECS[operator]
    except KeyError as exc:
        raise KeyError(f"Unknown operator: {operator}") from exc


def iter_operator_specs() -> Iterable[OperatorSpec]:
    return OPERATOR_SPECS.values()


def template_candidates(operator: str, template_path: str) -> Iterable[str]:
    spec = get_operator_spec(operator)
    seen: set[str] = set()
    if template_path not in seen:
        seen.add(template_path)
        yield template_path
    if template_path.startswith("common/"):
        return
    if spec.template_relpath:
        filename = Path(template_path).name
        candidate = f"{spec.template_relpath}/{filename}"
        if candidate not in seen:
            yield candidate


def validate_catalog_paths(repo_root: Path) -> list[str]:
    errors: list[str] = []
    repo_root = Path(repo_root)
    ops_root = repo_root / "helia_core_tester" / "generation" / "ops"
    descriptors_root = repo_root / "assets" / "descriptors"
    templates_root = repo_root / "assets" / "templates"

    for spec in iter_operator_specs():
        module_relpath = Path(*spec.module_path.split(".")[-2:]).with_suffix(".py")
        if not (ops_root / module_relpath).exists():
            errors.append(f"Missing module for {spec.operator}: {module_relpath}")
        for descriptor_relpath in spec.descriptor_relpaths:
            if not (descriptors_root / descriptor_relpath).exists():
                errors.append(f"Missing descriptor for {spec.operator}: {descriptor_relpath}")
        if spec.template_relpath is not None and not (templates_root / spec.template_relpath).exists():
            errors.append(f"Missing template directory for {spec.operator}: {spec.template_relpath}")

    return errors


__all__ = [
    "OPERATOR_SPECS",
    "OperatorSpec",
    "ParityKind",
    "get_operator_spec",
    "iter_operator_specs",
    "template_candidates",
    "validate_catalog_paths",
]
