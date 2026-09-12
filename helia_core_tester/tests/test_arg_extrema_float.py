"""Read generated inputs/indices; assert the selected contract independently."""

from pathlib import Path
import re

import numpy as np
import pytest
import yaml

from helia_core_tester.generation.test_ops import generate_test
from helia_core_tester.generation.ops.BasicMathFunctions.argmin import OpArgMin
from helia_core_tester.generation.ops.BasicMathFunctions.argmax import OpArgMax
from helia_core_tester.generation.ops._shared.arg_extrema_reference import (
    arg_extrema_reference,
)
from helia_core_tester.generation.ops._shared.arg_extrema_float import float_arg_kernel

ROOT = Path(__file__).resolve().parents[2]
CASES = [
    desc
    for kind in ("min", "max")
    for desc in yaml.safe_load_all(
        (
            ROOT / f"assets/descriptors/BasicMathFunctions/arg{kind}_float.yaml"
        ).read_text()
    )
]


@pytest.mark.parametrize("desc", CASES, ids=lambda desc: desc["name"])
def test_emitted_arg_indices(tmp_path, desc):
    generate_test(desc, str(tmp_path), seed=500)
    case = tmp_path / "BasicMathFunctions" / desc["name"]
    header = next((case / "includes").glob("*.h")).read_text()
    source = next(case.glob("*.c")).read_text()
    half = desc["tensor_dtypes"]["input"] == "FP16"
    word, dtype = (np.uint16, np.float16) if half else (np.uint32, np.float32)
    shape, axis = tuple(desc["input_shape"]), desc["axis"]
    kind = "min" if desc["operator"] == "ArgMin" else "max"

    def array(suffix):
        raw = re.search(rf"_{suffix}\[\] = \{{([^}}]+)", header).group(1)
        return [int(x.strip(), 0) for x in raw.split(",") if x.strip()]

    bits = np.array(array("input_bits"), dtype=word).reshape(shape)
    expected = np.array(array("expected_output"), dtype=np.int32)
    assert len(expected) == bits.size // shape[axis]
    assert "int32_t" in header and "EXACT_INT" in source
    assert (
        "memcpy(" in source and f"arm_arg{kind}_{'f16' if half else 'f32'}(" in source
    )
    if "special" in desc["name"]:
        # Finite first tie, both zero orders, first/later/all NaN, infinity,
        # positive/negative subnormals. Literal indices, not the golden oracle.
        np.testing.assert_array_equal(
            expected,
            [
                0,
                0,
                0,
                0,
                2,
                0,
                0 if kind == "min" else 1,
                0 if kind == "min" else 1,
                1 if kind == "min" else 0,
            ],
        )
    else:
        values = bits.view(dtype)
        fn = np.argmin if kind == "min" else np.argmax
        np.testing.assert_array_equal(expected, fn(values, axis=axis).flatten())
        assert {0, shape[axis] // 2, shape[axis] - 1} <= set(expected)
        if shape[0] > 1 and axis != 0:
            batches = expected.reshape(shape[0], -1)
            assert np.all(batches[0] != batches[1])
        # Constant-zero and last-index implementations must not pass.
        assert np.any(expected != 0) and np.any(expected != shape[axis] - 1)


@pytest.mark.parametrize("dtype", [np.uint16, np.uint32])
@pytest.mark.parametrize("axis", range(4))
@pytest.mark.parametrize("kind", ["min", "max"])
def test_reference_finite_axis_relative_indices(dtype, axis, kind):
    float_dtype = np.float16 if dtype == np.uint16 else np.float32
    values = np.arange(120, dtype=float_dtype).reshape(2, 3, 4, 5)
    fn = np.argmin if kind == "min" else np.argmax
    np.testing.assert_array_equal(
        arg_extrema_reference(values.view(dtype), axis, kind), fn(values, axis=axis)
    )


@pytest.mark.parametrize(
    "bits,axis,kind,message",
    [
        (np.zeros((1, 1, 1, 1), dtype=np.int32), 3, "min", "uint16 or uint32"),
        (np.zeros((2,), dtype=np.uint16), 0, "min", "canonical 4D"),
        (np.zeros((1, 1, 1, 1), dtype=np.uint16), -1, "min", "canonical 4D"),
        (np.zeros((1, 1, 1, 1), dtype=np.uint16), 4, "min", "canonical 4D"),
        (np.zeros((1, 1, 1, 1), dtype=np.uint16), 3, "sum", "min or max"),
        (np.zeros((0, 0, 1, 1), dtype=np.uint16), 1, "min", "Empty reduction"),
    ],
)
def test_reference_rejects_invalid_contract(bits, axis, kind, message):
    with pytest.raises(ValueError, match=message):
        arg_extrema_reference(bits, axis, kind)


def test_reference_empty_output_with_positive_axis():
    assert arg_extrema_reference(
        np.empty((0, 3, 2, 4), dtype=np.uint16), 1, "min"
    ).shape == (0, 2, 4)


@pytest.mark.parametrize("cls", [OpArgMin, OpArgMax])
@pytest.mark.parametrize("output", ["FP16", "FP32", "S16"])
def test_float_arg_requires_s32_output(tmp_path, cls, output):
    desc = dict(
        name="bad",
        operator=cls.__name__[2:],
        input_shape=[1, 1, 1, 4],
        axis=3,
        tensor_dtypes={"input": "FP32", "output": output},
    )
    with pytest.raises(ValueError, match="S32 output"):
        cls(desc, seed=0, target_cpu="cortex-m55").generate_c_files(tmp_path)


@pytest.mark.parametrize(
    "change,message",
    [
        ({"input_shape": [1, 1, 4]}, "rank4"),
        ({"input_shape": [1, 1, 0, 4]}, "rank4"),
        ({"axis": -1}, "canonical axis"),
        ({"axis": 1.5}, "canonical axis"),
        ({"hint": {"extras": {"input_bits": [-1]}}}, "unsigned integers"),
        ({"hint": {"extras": {"input_bits": [1 << 32]}}}, "unsigned integers"),
        ({"hint": {"extras": {"input_bits": [0.5]}}}, "unsigned integers"),
        ({"hint": {"extras": {"input_bits": [0]}}}, "reshape"),
    ],
)
def test_fixture_contract_rejections(tmp_path, change, message):
    desc = dict(
        name="bad",
        operator="ArgMin",
        input_shape=[1, 1, 1, 4],
        axis=3,
        tensor_dtypes={"input": "FP32", "output": "S32"},
    )
    desc.update(change)
    with pytest.raises(ValueError, match=message):
        OpArgMin(desc, seed=0, target_cpu="cortex-m55").generate_c_files(tmp_path)


def test_float_dispatch_rejects_integer_input():
    desc = dict(
        name="bad",
        operator="ArgMin",
        input_shape=[1, 1, 1, 4],
        axis=3,
        tensor_dtypes={"input": "S8", "output": "S32"},
    )
    with pytest.raises(ValueError, match="FP16 or FP32"):
        float_arg_kernel(OpArgMin(desc, seed=0, target_cpu="cortex-m55"), "min")


@pytest.mark.parametrize("kind", ["min", "max"])
@pytest.mark.parametrize(
    "legacy,effective", [(None, "S16"), ("S8", "S16"), ("S16", "S8")]
)
def test_integer_arg_emitted_dtype_precedence(tmp_path, kind, legacy, effective):
    from tensorflow.lite.python import schema_py_generated as schema

    name = f"arg{kind}_dtype"
    desc = dict(
        name=name,
        operator="ArgMin" if kind == "min" else "ArgMax",
        input_shape=[1, 2, 3, 4],
        axis=3,
        tensor_dtypes={"input": effective, "output": "S32"},
    )
    if legacy is not None:
        desc["activation_dtype"] = legacy
    generate_test(desc, str(tmp_path), seed=500)
    case = tmp_path / "BasicMathFunctions" / name
    model = schema.Model.GetRootAsModel((case / f"{name}.tflite").read_bytes(), 0)
    graph = model.Subgraphs(0)
    width = 16 if effective == "S16" else 8
    expected_type = schema.TensorType.INT16 if width == 16 else schema.TensorType.INT8
    assert graph.Tensors(graph.Inputs(0)).Type() == expected_type
    assert graph.Tensors(graph.Outputs(0)).Type() == schema.TensorType.INT32
    source = (case / f"{name}_arg{kind}.c").read_text()
    header = (case / "includes" / f"{name}_arg{kind}.h").read_text()
    assert f"arm_arg{kind}_s{width}(" in source
    assert f"const int{width}_t* __restrict input" in source
    assert f"const int{width}_t {name}_input[]" in header
