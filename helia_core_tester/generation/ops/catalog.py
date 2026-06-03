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
    descriptor_relpath: Optional[str]
    template_relpath: Optional[str]
    artifact_family_dir: str
    rationale: Optional[str] = None

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
    descriptor_relpath: Optional[str],
    template_relpath: Optional[str],
    *,
    parity_kind: ParityKind = "cmsis",
    rationale: Optional[str] = None,
) -> OperatorSpec:
    return OperatorSpec(
        operator=operator,
        family=family,
        parity_kind=parity_kind,
        module_path=f"helia_core_tester.generation.ops.{family}.{module_basename}",
        class_name=class_name,
        descriptor_relpath=descriptor_relpath,
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
    "PReLU": _spec("PReLU", "ActivationFunctions", "prelu", "OpPReLU", "ActivationFunctions/prelu.yaml", "ActivationFunctions/prelu"),
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
    "Abs": _spec("Abs", "BasicMathFunctions", "abs", "OpAbs", "BasicMathFunctions/abs.yaml", "BasicMathFunctions/abs"),
    "Add": _spec("Add", "BasicMathFunctions", "add", "OpAdd", "BasicMathFunctions/add.yaml", "BasicMathFunctions/add"),
    "Sub": _spec("Sub", "BasicMathFunctions", "sub", "OpSub", "BasicMathFunctions/sub.yaml", "BasicMathFunctions/sub"),
    "Mul": _spec("Mul", "BasicMathFunctions", "mul", "OpMul", "BasicMathFunctions/mul.yaml", "BasicMathFunctions/mul"),
    "Maximum": _spec("Maximum", "BasicMathFunctions", "minmax", "OpMinMax", "BasicMathFunctions/maximum.yaml", "BasicMathFunctions/minmax"),
    "Minimum": _spec("Minimum", "BasicMathFunctions", "minmax", "OpMinMax", "BasicMathFunctions/minimum.yaml", "BasicMathFunctions/minmax"),
    "Mean": _spec("Mean", "BasicMathFunctions", "mean", "OpMean", "BasicMathFunctions/mean.yaml", "BasicMathFunctions/mean"),
    "ReduceMax": _spec("ReduceMax", "BasicMathFunctions", "reduce_max", "OpReduceMax", "BasicMathFunctions/reduce_max.yaml", "BasicMathFunctions/reduce_max"),
    "ReduceMin": _spec("ReduceMin", "BasicMathFunctions", "reduce_min", "OpReduceMin", "BasicMathFunctions/reduce_min.yaml", "BasicMathFunctions/reduce_min"),
    "ArgMax": _spec("ArgMax", "BasicMathFunctions", "argmax", "OpArgMax", "BasicMathFunctions/argmax.yaml", "BasicMathFunctions/argmax"),
    "ArgMin": _spec("ArgMin", "BasicMathFunctions", "argmin", "OpArgMin", "BasicMathFunctions/argmin.yaml", "BasicMathFunctions/argmin"),
    "Sqrt": _spec("Sqrt", "BasicMathFunctions", "sqrt", "OpSqrt", "BasicMathFunctions/sqrt.yaml", "BasicMathFunctions/sqrt"),
    "Rsqrt": _spec("Rsqrt", "BasicMathFunctions", "rsqrt", "OpRsqrt", "BasicMathFunctions/rsqrt.yaml", "BasicMathFunctions/rsqrt"),
    "Comparison": _spec("Comparison", "ComparisonFunctions", "comparison", "OpComparison", "ComparisonFunctions/comparison.yaml", "ComparisonFunctions/comparison"),
    "Concatenation": _spec("Concatenation", "ConcatenationFunctions", "concatenation", "OpConcatenation", "ConcatenationFunctions/concatenation.yaml", "ConcatenationFunctions/concatenation"),
    "Split": _spec("Split", "ConcatenationFunctions", "split", "OpSplit", "ConcatenationFunctions/split.yaml", "ConcatenationFunctions/split"),
    "Convolve": _spec("Convolve", "ConvolutionFunctions", "convolve", "OpConvolve", "ConvolutionFunctions/convolve.yaml", "ConvolutionFunctions/convolve"),
    "DepthwiseConv": _spec("DepthwiseConv", "ConvolutionFunctions", "depthwise_conv", "OpDepthwiseConv", "ConvolutionFunctions/depthwise_conv.yaml", "ConvolutionFunctions/depthwise_conv"),
    "TransposeConv": _spec("TransposeConv", "ConvolutionFunctions", "transpose_conv", "OpTransposeConv", "ConvolutionFunctions/transpose_conv.yaml", "ConvolutionFunctions/transpose_conv"),
    "FullyConnected": _spec("FullyConnected", "FullyConnectedFunctions", "fully_connected", "OpFullyConnected", "FullyConnectedFunctions/fully_connected.yaml", "FullyConnectedFunctions/fully_connected"),
    "BatchMatMul": _spec("BatchMatMul", "FullyConnectedFunctions", "batch_matmul", "OpBatchMatMul", "FullyConnectedFunctions/batch_matmul.yaml", "FullyConnectedFunctions/batch_matmul"),
    "Gather": _spec("Gather", "GatherFunctions", "gather", "OpGather", "GatherFunctions/gather.yaml", "GatherFunctions/gather"),
    "GatherND": _spec("GatherND", "GatherFunctions", "gather_nd", "OpGatherND", "GatherFunctions/gather_nd.yaml", "GatherFunctions/gather_nd"),
    "LSTMUnidirectional": _spec("LSTMUnidirectional", "LSTMFunctions", "lstm_unidirectional", "OpLSTMUnidirectional", "LSTMFunctions/lstm_unidirectional.yaml", "LSTMFunctions/lstm_unidirectional"),
    "Requantize": _spec("Requantize", "NNSupportFunctions", "requantize", "OpRequantize", "NNSupportFunctions/requantize.yaml", "NNSupportFunctions/requantize"),
    "Pad": _spec("Pad", "PadFunctions", "pad", "OpPad", "PadFunctions/pad.yaml", "PadFunctions/pad"),
    "AvgPool": _spec("AvgPool", "PoolingFunctions", "avg_pool", "OpAvgPool", "PoolingFunctions/avg_pool.yaml", "PoolingFunctions/avg_pool"),
    "MaxPool": _spec("MaxPool", "PoolingFunctions", "max_pool", "OpMaxPool", "PoolingFunctions/max_pool.yaml", "PoolingFunctions/max_pool"),
    "Quantize": _spec("Quantize", "QuantizationFunctions", "quantize", "OpQuantize", "QuantizationFunctions/quantize.yaml", "QuantizationFunctions/quantize"),
    "Dequantize": _spec("Dequantize", "QuantizationFunctions", "dequantize", "OpDequantize", "QuantizationFunctions/dequantize.yaml", "QuantizationFunctions/dequantize"),
    "Reshape": _spec("Reshape", "ReshapeFunctions", "reshape", "OpReshape", "ReshapeFunctions/reshape.yaml", "ReshapeFunctions/reshape"),
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
    "SVDF": _spec("SVDF", "SVDFunctions", "svdf", "OpSVDF", "SVDFunctions/svdf.yaml", "SVDFunctions/svdf"),
    "Softmax": _spec("Softmax", "SoftmaxFunctions", "softmax", "OpSoftmax", "SoftmaxFunctions/softmax.yaml", "SoftmaxFunctions/softmax"),
    "StridedSlice": _spec("StridedSlice", "StridedSliceFunctions", "strided_slice", "OpStridedSlice", "StridedSliceFunctions/strided_slice.yaml", "StridedSliceFunctions/strided_slice"),
    "Transpose": _spec("Transpose", "TransposeFunctions", "transpose", "OpTranspose", "TransposeFunctions/transpose.yaml", "TransposeFunctions/transpose"),
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
        if spec.descriptor_relpath is not None and not (descriptors_root / spec.descriptor_relpath).exists():
            errors.append(f"Missing descriptor for {spec.operator}: {spec.descriptor_relpath}")
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
