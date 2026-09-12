"""Inspect emitted integer fixtures, including their real LiteRT goldens."""

from pathlib import Path
import re

import numpy as np
import pytest
import yaml

from helia_core_tester.generation.test_ops import generate_test
from helia_core_tester.generation.ops._shared.reduce_extrema_integer import (
    boundary_inputs, resize_integer_interpreter,
)


ROOT = Path(__file__).resolve().parents[2]
DESCRIPTORS = [
    desc
    for kind in ("min", "max")
    for desc in yaml.safe_load_all(
        (ROOT / f"assets/descriptors/BasicMathFunctions/reduce_{kind}.yaml").read_text()
    )
]


@pytest.mark.parametrize("defect,match", [
    ("shape", "input shape"), ("dtype", "tensor dtypes"),
    ("scales", "matching quantization"), ("zero_points", "matching quantization"),
    ("per_axis", "per-tensor"), ("empty_scales", "per-tensor"),
    ("empty_zero_points", "per-tensor"), ("scalar_dimension", None),
])
def test_integer_metadata_rejections(defect, match):
    from copy import deepcopy

    shape = (2, 4, 5, 8)
    input_detail = dict(index=0, shape=shape, dtype=np.int8,
                        quantization_parameters=dict(scales=np.array([0.5]),
                                                     zero_points=np.array([0]),
                                                     quantized_dimension=0))
    output_detail = deepcopy(input_detail)
    if defect == "shape":
        input_detail["shape"] = (1, 4, 5, 8)
    elif defect == "dtype":
        output_detail["dtype"] = np.int16
    elif defect in ("scales", "zero_points"):
        output_detail["quantization_parameters"][defect] += 1
    else:
        for detail in (input_detail, output_detail):
            qp = detail["quantization_parameters"]
            if defect == "per_axis":
                qp.update(scales=np.array([0.5, 0.25]), zero_points=np.array([0, 0]))
            elif defect != "scalar_dimension":
                qp[defect.removeprefix("empty_")] = np.array([])
        output_detail["quantization_parameters"]["quantized_dimension"] = 1

    class Interpreter:
        def get_input_details(self): return [input_detail]
        def get_output_details(self): return [output_detail]
        def resize_tensor_input(self, index, new_shape, strict): pass
        def allocate_tensors(self): pass

    if match is None:
        # Dimension metadata has no effect when each tensor has one scale.
        resize_integer_interpreter(Interpreter(), shape)
    else:
        with pytest.raises(ValueError, match=match):
            resize_integer_interpreter(Interpreter(), shape)


@pytest.mark.parametrize("dtype,kind,match", [
    (np.float32, "min", "int8 or int16"), (np.int8, "sum", "min or max"),
])
def test_boundary_input_rejections(dtype, kind, match):
    with pytest.raises(ValueError, match=match):
        boundary_inputs((1, 2, 2, 1), [1, 2], dtype, kind)


@pytest.mark.parametrize("desc", DESCRIPTORS, ids=lambda desc: desc["name"])
def test_emitted_integer_boundary_and_batch_sensitivity(tmp_path, desc):
    generate_test(desc, str(tmp_path), seed=500)
    case = tmp_path / "BasicMathFunctions" / desc["name"]
    header = next((case / "includes").glob("*.h")).read_text()
    dims = re.search(r"_input_dims = \{([^}]+)", header).group(1)
    shape = tuple(int(x) for x in re.findall(r"= (\d+)", dims))
    assert shape == tuple(desc["input_shape"])

    def array(suffix):
        text = re.search(rf"_{suffix}\[\] = \{{([^}}]+)", header).group(1)
        return np.array([int(x.strip()) for x in text.split(",") if x.strip()])

    values = array("input").reshape(shape)
    expected = array("expected_output")
    axes = desc["axes"]
    retained = [i for i in range(4) if i not in axes]
    domains = values.transpose(retained + axes).reshape(
        -1, int(np.prod([shape[i] for i in axes]))
    )
    reduce = np.min if desc["operator"] == "ReduceMin" else np.max
    # Diagnostic integer selection independently checks the emitted LiteRT golden.
    np.testing.assert_array_equal(reduce(domains, axis=1), expected)
    assert domains.shape[1] > 1
    assert np.any(np.abs(reduce(domains[:, 1:], axis=1) - expected) > 1)
    assert np.any(np.abs(reduce(domains[:, :-1], axis=1) - expected) > 1)
    if "axes13" in desc["name"] or "_hw_" in desc["name"]:
        assert shape[0] == 2
        batches = expected.reshape(shape[0], -1)
        assert np.all(np.abs(batches[1] - batches[0]) > 1)
