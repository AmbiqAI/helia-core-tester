"""Check emitted fixtures and element-type dispatch for float reduce extrema."""

from pathlib import Path
import re

import pytest
import yaml

from helia_core_tester.generation.ops.BasicMathFunctions.reduce_max import OpReduceMax
from helia_core_tester.generation.ops.BasicMathFunctions.reduce_min import OpReduceMin


@pytest.mark.parametrize("kind,cls", [("max", OpReduceMax), ("min", OpReduceMin)])
@pytest.mark.parametrize(
    "dtype,suffix,ctype",
    [
        ("S8", "s8", "int8_t"),
        ("S16", "s16", "int16_t"),
        ("FP32", "f32", "float"),
        ("FP16", "f16", "float16_t"),
    ],
)
def test_dispatch_and_reject_conversion(kind, cls, dtype, suffix, ctype):
    desc = {
        "operator": "Reduce" + kind.title(),
        "name": "probe",
        "input_shape": [2, 2, 3, 2],
        "tensor_dtypes": {"input": dtype, "output": dtype},
    }
    op = cls(desc, seed=0, target_cpu="cortex-m55")
    selected = getattr(op, f"_select_cmsis_reduce_{kind}_kernel")()
    assert selected == {
        "kernel_fn": f"arm_reduce_{kind}_{suffix}",
        "input_c_type": ctype,
        "output_c_type": ctype,
    }
    desc["tensor_dtypes"]["output"] = "FP16" if dtype != "FP16" else "FP32"
    with pytest.raises(ValueError, match="matching"):
        cls(desc, seed=0, target_cpu="cortex-m55")._element_dtype()


@pytest.mark.parametrize("kind,cls", [("max", OpReduceMax), ("min", OpReduceMin)])
@pytest.mark.parametrize(
    "suffix,sign,inf,quiet",
    [
        ("f32", 0x80000000, 0x7F800000, 0x400000),
        ("f16", 0x8000, 0x7C00, 0x200),
    ],
)
def test_emitted_special_bits_distinguish_ties_and_nan_payloads(
    tmp_path, kind, cls, suffix, sign, inf, quiet
):
    root = Path(__file__).resolve().parents[2]
    path = root / f"assets/descriptors/BasicMathFunctions/reduce_{kind}_float.yaml"
    descriptors = list(yaml.safe_load_all(path.read_text()))
    desc = next(
        d for d in descriptors if d["name"] == f"reduce_{kind}_special_{suffix}"
    )
    cls(desc, seed=0, target_cpu="cortex-m55").generate_c_files(tmp_path)
    header = next((tmp_path / "includes").glob("*.h")).read_text()
    arrays = re.findall(r"_((?:input|expected)_bits)\[\] = \{([^}]+)", header)
    raw = {name: [int(x.strip(), 16) for x in text.split(",")] for name, text in arrays}
    assert raw["input_bits"][:6] == [sign, 0, 0, 0, sign, sign]
    expected = raw["expected_bits"]
    assert expected[:2] == [sign, 0]
    assert expected[2:4] == ([2, sign] if kind == "max" else [0, sign + 2])
    assert expected[6:9] == [inf | quiet] * 3
    assert expected[9] == (1 if kind == "max" else sign + 1)
    source = next(tmp_path.glob("*.c")).read_text()
    assert "HELIA_VALIDATE_OUTPUTS(" not in source
    assert re.search(r"HELIA_VALIDATE_FLOAT_BITS\(actual, .*?, .*?, 0, i,", source)
