"""Test-only kernel contract for rendering contract-bound templates without an ns-cmsis-nn checkout.

It holds every function of the operators whose templates bind their calls from the contract
(BOUND_PREFIXES), kernels and scratch-size queries alike, since the templates bind both.

Those templates render their kernel calls from the ns-cmsis-nn kernel contract and refuse
to render without one. The pure-Python pytest job has no checkout, so unit tests that render a
contract-bound case fall back to the committed fixture, but only when nothing better exists: a
checkout that ships the export always wins, and a test that points CMSIS_NN_ROOT at a real
directory keeps that directory's behaviour, including refusing to render.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

from helia_core_tester.contract.ir import load_contract_set

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "contract" / "bound_operators"
BOUND_PREFIXES = ("arm_convolve_", "arm_depthwise_", "arm_fully_connected_", "arm_batch_matmul_",
                  "arm_avgpool_", "arm_avg_pool_", "arm_max_pool_", "arm_relu", "arm_clamp_",
                  "arm_hard_swish_", "arm_leaky_relu_", "arm_logistic_", "arm_tanh_", "arm_nn_activation_",
                  "arm_prelu_", "arm_abs_", "arm_nn_abs_", "arm_mean_", "arm_nn_mean_", "arm_reduce_",
                  "arm_add_", "arm_sub_", "arm_mul_", "arm_elementwise_", "arm_squared_difference_",
                  "arm_maximum_", "arm_minimum_", "arm_argmax_", "arm_argmin_", "arm_nn_fill_", "arm_sqrt_",
                  "arm_equal_", "arm_not_equal_", "arm_greater_", "arm_less_", "arm_comparison_",
                  "arm_broadcast_to_", "arm_batch_to_space_", "arm_space_to_batch_", "arm_depth_to_space_",
                  "arm_space_to_depth_", "arm_strided_slice_", "arm_pad_", "arm_transpose_", "arm_gather_",
                  "arm_resize_nearest_neighbor_", "arm_pack_", "arm_mirror_pad_", "arm_tile_",
                  "arm_reverse_sequence_", "arm_select_v2_", "arm_scatter_nd_", "arm_dynamic_update_slice_",
                  "arm_where_", "arm_requantize_", "arm_batch_norm_", "arm_softmax_",
                  "arm_split_", "arm_unpack_", "arm_quantize_", "arm_dequantize_",
                  "arm_concatenation_", "arm_rsqrt_", "arm_reshape_", "arm_nn_sqrt_")


def fallback_resolver(resolve: Callable[[], Optional[Path]]) -> Callable[[], Optional[Path]]:
    def resolver() -> Optional[Path]:
        root = resolve()
        if root is not None and (os.environ.get("CMSIS_NN_ROOT") or load_contract_set(root).present):
            return root
        return FIXTURE_ROOT

    return resolver


def subprocess_contract_env() -> dict:
    """The environment for a generation subprocess: it gets no session fixture, so point
    CMSIS_NN_ROOT at the fallback contract when no checkout with an export is configured."""
    from helia_core_tester.contract import render

    env = dict(os.environ)
    root = fallback_resolver(render.resolve_cmsis_nn_root)()
    if root == FIXTURE_ROOT:
        env["CMSIS_NN_ROOT"] = str(FIXTURE_ROOT)
    return env
