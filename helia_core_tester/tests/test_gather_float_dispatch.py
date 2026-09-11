"""Float GATHER/GATHER_ND dispatch and its adjacent consumer (helia-core-tester#149).

Two things the descriptors alone do not pin. That a copy operator refuses to
convert element types, and that a dtype the hardware kernel registry does not
carry leaves the bundle bridge able to skip the case rather than aborting the
whole run on it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from helia_core_tester.generation.ops.GatherFunctions.gather import OpGather
from helia_core_tester.generation.ops.GatherFunctions.gather_nd import OpGatherND
from helia_core_tester.perf_stream.generated_test_bridge import (
    UnsupportedGeneratedTestError,
    _kernel_id,
)
from helia_core_tester.perf_stream.kernel_registry import lookup_kernel_id


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "op_class,operator",
    [(OpGather, "Gather"), (OpGatherND, "GatherND")],
)
@pytest.mark.parametrize(
    "dtype,gather_fn,gather_nd_fn",
    [
        ("S8", "arm_gather_s8", "arm_gather_nd_s8"),
        ("S16", "arm_gather_s16", "arm_gather_nd_s16"),
        ("FP32", "arm_gather_f32", "arm_gather_nd_f32"),
        ("FP16", "arm_gather_f16", "arm_gather_nd_f16"),
    ],
)
def test_matching_element_dtypes_resolve_to_their_own_kernel(
    op_class, operator, dtype, gather_fn, gather_nd_fn
):
    """Pin the kernel each dtype dispatches to, not just that the dtype round-trips.

    Asserting only that _element_dtype() returns what the descriptor asked for would pass
    with two entries of the kernel table transposed, which is the mistake the table exists
    to prevent. The C type is pinned too, since the emitted arrays are declared with it.
    """
    desc = {
        "operator": operator,
        "name": f"probe_{dtype.lower()}",
        "tensor_dtypes": {"input": dtype, "output": dtype},
        "input_shape": [2, 3, 4],
        "indices_shape": [2, 2],
    }
    op = op_class(desc, seed=0, target_cpu="cortex-m55")
    assert op._element_dtype() == dtype

    expected_fn = gather_fn if operator == "Gather" else gather_nd_fn
    selector = (
        op._select_cmsis_gather_kernel
        if operator == "Gather"
        else op._select_cmsis_gather_nd_kernel
    )
    kernel_info = selector()
    assert kernel_info["kernel_fn"] == expected_fn
    expected_c_type = {
        "S8": "int8_t",
        "S16": "int16_t",
        "FP32": "float",
        "FP16": "float16_t",
    }[dtype]
    assert kernel_info["input_c_type"] == expected_c_type
    assert kernel_info["output_c_type"] == expected_c_type


@pytest.mark.parametrize(
    "case_name,axis_size,index_count",
    [
        ("gather_float_axis_inner_f32", 4, 3),
        ("gather_float_axis_outer_f16", 3, 2),
        ("gather_float_indices_rank2_f32", 4, 4),
        ("gather_float_axis_middle_f32", 3, 2),
    ],
)
def test_a_case_drawing_no_more_indices_than_its_axis_has_slices_gets_distinct_ones(
    case_name, axis_size, index_count
):
    """The property that lets an index case fail at all, over many seeds rather than one.

    Uniform draws once produced {0, 0} for the outer-axis case, so a kernel that ignored
    the index array and always read slice zero passed the only case covering that copy
    shape. This calls the generator's own draw rather than reimplementing it, across a
    hundred seeds, so it pins the property instead of the luck of one seed.
    """
    desc = {
        "operator": "Gather",
        "name": case_name,
        "tensor_dtypes": {"input": "FP32", "output": "FP32"},
        "input_shape": [2, 3, 4],
        "indices_shape": [index_count],
    }
    for seed in range(100):
        op = OpGather(desc, seed, target_cpu="cortex-m55")
        drawn = op._draw_indices((index_count,), axis_size)
        assert drawn.size == index_count
        assert len(set(drawn.tolist())) == index_count, (
            f"seed {seed} drew {drawn.tolist()}, which cannot distinguish a kernel that "
            f"ignores the index array from one that honours it"
        )
        assert drawn.min() >= 0 and drawn.max() < axis_size


def test_the_repeated_index_case_still_repeats():
    """Its repeats are forced by pigeonhole, so the distinctness rule must not remove them.

    Five indices drawn from an axis of extent two: every seed must repeat, or the case
    stops catching a kernel that consumes a source slice.
    """
    desc = {
        "operator": "Gather",
        "name": "gather_float_repeated_index_f16",
        "tensor_dtypes": {"input": "FP16", "output": "FP16"},
        "input_shape": [2, 3, 4],
        "indices_shape": [5],
    }
    for seed in range(100):
        op = OpGather(desc, seed, target_cpu="cortex-m55")
        drawn = op._draw_indices((5,), 2)
        assert len(set(drawn.tolist())) < drawn.size
        assert drawn.min() >= 0 and drawn.max() < 2
