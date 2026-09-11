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
@pytest.mark.parametrize("dtype", ["S8", "S16", "FP32", "FP16"])
def test_matching_element_dtypes_resolve_to_their_own_kernel(op_class, operator, dtype):
    desc = {
        "operator": operator,
        "name": f"probe_{dtype.lower()}",
        "tensor_dtypes": {"input": dtype, "output": dtype},
        "input_shape": [2, 3, 4],
        "indices_shape": [2, 2],
    }
    op = op_class(desc, seed=0, target_cpu="cortex-m55")
    assert op._element_dtype() == dtype


@pytest.mark.parametrize(
    "op_class,operator",
    [(OpGather, "Gather"), (OpGatherND, "GatherND")],
)
def test_a_copy_operator_refuses_to_convert_element_types(op_class, operator):
    # Accepting this would gather at the input type and compare against a golden built at
    # the same type, so the case would pass while proving nothing about the conversion it
    # asked for.
    desc = {
        "operator": operator,
        "name": "probe_mismatched",
        "tensor_dtypes": {"input": "FP32", "output": "FP16"},
        "input_shape": [2, 3, 4],
        "indices_shape": [2, 2],
    }
    op = op_class(desc, seed=0, target_cpu="cortex-m55")
    with pytest.raises(ValueError, match="cannot convert"):
        op._element_dtype()


@pytest.mark.parametrize("operator", ["Gather", "GatherND"])
def test_an_unregistered_dtype_is_skippable_rather_than_fatal(operator):
    """The float gather entry points have no hardware kernel registry entry.

    lookup_kernel_id's own contract says callers should treat that as an
    UnsupportedGeneratedTestError; before #149 no caller did, so introducing a float
    case into a family the registry only carries in integer form would abort the whole
    bundle run instead of skipping that one case.
    """
    root = _repo_root()
    assert _kernel_id(root, family="GatherFunctions", operator=operator, dtype="S8") > 0
    for dtype in ("FP32", "FP16"):
        with pytest.raises(UnsupportedGeneratedTestError):
            _kernel_id(root, family="GatherFunctions", operator=operator, dtype=dtype)


def test_the_conversion_does_not_hide_a_registered_lookup():
    """The wrapper must stay a pass-through for everything the registry does carry."""
    root = _repo_root()
    assert _kernel_id(root, family="GatherFunctions", operator="Gather", dtype="S8") == (
        lookup_kernel_id(root, family="GatherFunctions", operator="Gather", dtype="S8")
    )
