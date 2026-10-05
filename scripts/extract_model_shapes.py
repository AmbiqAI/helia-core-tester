#!/usr/bin/env python3
"""Write s8 kernel descriptors from real model layers.

Reads the CONV_2D, DEPTHWISE_CONV_2D, FULLY_CONNECTED and AVERAGE_POOL_2D
operators of int8 `.tflite` models and emits one s8 descriptor per unique layer
shape and fused activation. Weights stay random; only shapes, strides, padding
and the fused activation carry over. Each descriptor is named after the first
layer with that key (`<stem>_mlperf_<model>_l<op index>_s8`); a comment lists
the other layers.
The cases replace a marked block at the end of each operator's descriptor file.

Usage, with M the helia-profiler checkout's tests/fixtures/mlperf_tiny
(the MLPerf Tiny models the hpx nightly profiles):
    uv run python scripts/extract_model_shapes.py \
        ad=$M/ad/ad01_int8.tflite ic=$M/ic/ic_resnet_int8.tflite \
        kws=$M/kws/kws_ref_model.tflite vww=$M/vww/vww_96_int8.tflite
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from ai_edge_litert import schema_py_generated as fb

from helia_core_tester.generation.ops.catalog import get_operator_spec
from helia_core_tester.generation.utils.litert_utils import load_litert_model

_DESCRIPTOR_DIR = Path(__file__).resolve().parent.parent / "assets" / "descriptors"
_OUTPUTS = {
    "CONV_2D": "Convolve",
    "DEPTHWISE_CONV_2D": "DepthwiseConv",
    "FULLY_CONNECTED": "FullyConnected",
    "AVERAGE_POOL_2D": "AvgPool",
}
_OP_NAMES = {v: k for k, v in vars(fb.BuiltinOperator).items() if not k.startswith("_")}
_ACT_NAMES = {fb.ActivationFunctionType.NONE: "NONE", fb.ActivationFunctionType.RELU: "RELU",
              fb.ActivationFunctionType.RELU6: "RELU6"}
_PAD_NAMES = {fb.Padding.SAME: "SAME", fb.Padding.VALID: "VALID"}


def layer_fields(op_name: str, opts, shapes: list[list[int]]) -> dict:
    """Map one tflite operator to descriptor fields."""
    act = _ACT_NAMES[opts.fusedActivationFunction]
    if op_name == "FULLY_CONNECTED":
        return {"activation": act, "input_shape": shapes[0], "filter_shape": shapes[1],
                "use_bias": len(shapes) > 2}
    fields = {"activation": act, "input_shape": shapes[0]}
    if op_name == "AVERAGE_POOL_2D":
        fields["pool_size"] = [opts.filterHeight, opts.filterWidth]
    elif op_name == "CONV_2D":
        out_ch, kh, kw, in_ch = shapes[1]
        fields["filter_shape"] = [kh, kw, in_ch, out_ch]
    else:
        _, kh, kw, _ = shapes[1]
        fields["filter_shape"] = [kh, kw, shapes[0][3], opts.depthMultiplier]
        fields["depth_multiplier"] = opts.depthMultiplier
    fields["strides"] = [opts.strideH, opts.strideW]
    fields["padding"] = _PAD_NAMES[opts.padding]
    if op_name != "AVERAGE_POOL_2D":
        dilation = [opts.dilationHFactor, opts.dilationWFactor]
        if dilation != [1, 1]:
            fields["dilation"] = dilation
        fields["use_bias"] = len(shapes) > 2
    return fields


def model_layers(tag: str, path: Path):
    """Yield (op name, layer label, fields) per kernel layer."""
    model, graph = load_litert_model(str(path))
    for index, op in enumerate(graph.operators):
        code = model.operatorCodes[op.opcodeIndex]
        op_name = _OP_NAMES[max(code.builtinCode, code.deprecatedBuiltinCode)]
        if op_name not in _OUTPUTS:
            continue
        tensors = [graph.tensors[i] for i in op.inputs if i >= 0]
        if tensors[0].type != fb.TensorType.INT8:
            sys.exit(f"{path}: op {index} is not int8")
        shapes = [[int(d) for d in t.shape] for t in tensors]
        yield op_name, f"{tag}_l{index}", layer_fields(op_name, op.builtinOptions, shapes)


def layer_key(fields: dict) -> tuple:
    """Dedup key: every descriptor field."""
    return tuple((k, str(v)) for k, v in fields.items())


_BEGIN = "# BEGIN mlperf model shapes (scripts/extract_model_shapes.py)\n"
_END = "# END mlperf model shapes\n"


def render(stem: str, operator: str, layers: list[tuple[str, dict, list[str]]]) -> str:
    """Render the marked descriptor block."""
    docs = []
    for label, fields, also in layers:
        desc = {"operator": operator, "name": f"{stem}_mlperf_{label}_s8",
                "activation_dtype": "S8", "weight_dtype": "S8",
                "hint": {"call_style": "per_tensor"}, **fields}
        note = f"# Same layer: {', '.join(also)}\n" if also else ""
        docs.append(note + yaml.safe_dump(desc, sort_keys=False, default_flow_style=None))
    return "---\n" + _BEGIN + "---\n".join(docs) + _END


def replace_block(path: Path, block: str) -> None:
    """Swap the marked block in `path`."""
    text = path.read_text()
    start = text.find("---\n" + _BEGIN)
    if start >= 0:
        end = text.index(_END, start) + len(_END)
        text = text[:start] + text[end:]
    path.write_text(text + block)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("models", nargs="+", help="tag=path/to/model.tflite")
    args = parser.parse_args()
    groups: dict[str, dict[tuple, tuple[str, dict, list[str]]]] = {k: {} for k in _OUTPUTS}
    for spec in args.models:
        tag, _, path = spec.partition("=")
        for op_name, label, fields in model_layers(tag, Path(path)):
            key = layer_key(fields)
            if key in groups[op_name]:
                groups[op_name][key][2].append(label)
            else:
                groups[op_name][key] = (label, fields, [])
    for op_name, operator in _OUTPUTS.items():
        spec = get_operator_spec(operator)
        out = _DESCRIPTOR_DIR / spec.descriptor_relpath
        replace_block(out, render(spec.descriptor_stem, operator, list(groups[op_name].values())))
        print(f"{out.relative_to(_DESCRIPTOR_DIR)}: {len(groups[op_name])} cases")


if __name__ == "__main__":
    main()
