"""Inspect emitted integer fixtures, including their real LiteRT goldens."""

from pathlib import Path
import re

import numpy as np
import pytest
import yaml

from helia_core_tester.generation.test_ops import generate_test


ROOT = Path(__file__).resolve().parents[2]
DESCRIPTORS = [
    desc
    for kind in ("min", "max")
    for desc in yaml.safe_load_all(
        (ROOT / f"assets/descriptors/BasicMathFunctions/reduce_{kind}.yaml").read_text()
    )
]


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
