"""The tester's copy of the depthwise routing rule matches ns-cmsis-nn v7.36.0.

v7.36.0 routes dilated 1D depthwise layers (no vertical extent, unit stride, no vertical
padding) to arm_depthwise_conv_s8_opt and arm_depthwise_conv_fast_s16. Fault cases are
admitted only where the selected route actually diagnoses the fault, so the rule decides
which fault descriptors can exist.
"""

from __future__ import annotations

import pytest

from helia_core_tester.generation.ops.ConvolutionFunctions.depthwise_conv import (
    OpDepthwiseConv,
    _opt_dilation_supported,
)


def _dims(n: int = 1, h: int = 1, w: int = 240, c: int = 24) -> dict:
    return {"n": n, "h": h, "w": w, "c": c}


def _params(*, dilation=(1, 2), stride=(1, 1), pad=(0, 3), ch_mult: int = 1) -> dict:
    return {
        "ch_mult": ch_mult,
        "dilation_h": dilation[0],
        "dilation_w": dilation[1],
        "stride_h": stride[0],
        "stride_w": stride[1],
        "pad_h": pad[0],
        "pad_w": pad[1],
    }


@pytest.mark.parametrize(
    ("params", "input_dims", "filter_dims", "output_dims", "expected"),
    [
        (_params(dilation=(1, 1)), _dims(h=5), _dims(h=3, w=3), _dims(h=5), True),
        (_params(dilation=(1, 2)), _dims(), _dims(w=7), _dims(), True),
        (_params(dilation=(1, 16)), _dims(), _dims(w=7), _dims(), True),
        (_params(dilation=(2, 1)), _dims(), _dims(w=7), _dims(), False),
        (_params(dilation=(2, 2)), _dims(h=5), _dims(h=3, w=3), _dims(h=5), False),
        (_params(dilation=(1, 2)), _dims(h=2), _dims(w=7), _dims(), False),
        (_params(dilation=(1, 2)), _dims(), _dims(h=2, w=7), _dims(), False),
        (_params(dilation=(1, 2)), _dims(), _dims(w=7), _dims(h=2), False),
        (_params(dilation=(1, 2), stride=(1, 2)), _dims(), _dims(w=7), _dims(), False),
        (_params(dilation=(1, 2), stride=(2, 1)), _dims(), _dims(w=7), _dims(), False),
        (_params(dilation=(1, 2), pad=(1, 3)), _dims(), _dims(w=7), _dims(), False),
    ],
    ids=[
        "unit-dilation-2d",
        "dilated-1d-d2",
        "dilated-1d-d16",
        "vertical-dilation",
        "dilated-2d",
        "input-height-2",
        "filter-height-2",
        "output-height-2",
        "stride-w-2",
        "stride-h-2",
        "vertical-padding",
    ],
)
def test_opt_dilation_predicate(params, input_dims, filter_dims, output_dims, expected) -> None:
    assert _opt_dilation_supported(params, input_dims, filter_dims, output_dims) is expected


def _op(required_capabilities=("dsp",)) -> OpDepthwiseConv:
    op = OpDepthwiseConv.__new__(OpDepthwiseConv)
    op.desc = {"name": "probe", "operator": "DepthwiseConv", "required_capabilities": list(required_capabilities)}
    return op


def _context(kernel_fn: str, *, dilation=(1, 2), n: int = 1, c: int = 24, filter_w: int = 7) -> dict:
    return {
        "kernel_fn": kernel_fn,
        "float_kernel": False,
        "dw_conv_params": _params(dilation=dilation),
        "input_dims": _dims(n=n, c=c),
        "filter_dims": _dims(w=filter_w, c=c),
        "output_dims": _dims(c=c),
    }


@pytest.mark.parametrize(
    ("kernel_fn", "kind"),
    [
        ("arm_depthwise_conv_wrapper_s8", "null_weight_sum_ctx"),
        ("arm_depthwise_conv_wrapper_s8", "null_ctx_buf"),
        ("arm_depthwise_conv_wrapper_s16", "null_ctx_buf"),
    ],
)
def test_dilated_1d_layer_reaches_the_optimized_route(kernel_fn: str, kind: str) -> None:
    _op()._check_fault_reachable(kind, _context(kernel_fn))


@pytest.mark.parametrize(
    ("kernel_fn", "context"),
    [
        ("arm_depthwise_conv_wrapper_s8", _context("arm_depthwise_conv_wrapper_s8", dilation=(2, 2))),
        ("arm_depthwise_conv_wrapper_s8", _context("arm_depthwise_conv_wrapper_s8", n=2)),
        ("arm_depthwise_conv_wrapper_s16", _context("arm_depthwise_conv_wrapper_s16", dilation=(2, 1))),
        ("arm_depthwise_conv_wrapper_s16", _context("arm_depthwise_conv_wrapper_s16", filter_w=512)),
        ("arm_depthwise_conv_wrapper_s4", _context("arm_depthwise_conv_wrapper_s4")),
    ],
    ids=["s8-dilated-2d", "s8-batch-2", "s16-vertical-dilation", "s16-large-filter", "s4-dilated-1d-unchanged"],
)
def test_layers_off_the_optimized_route_are_rejected(kernel_fn: str, context: dict) -> None:
    with pytest.raises(ValueError, match="only checks it on the optimized route"):
        _op()._check_fault_reachable("null_ctx_buf", context)
