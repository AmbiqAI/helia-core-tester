"""INT32 predicate references and generated API contracts. Refs #478."""

import re
from pathlib import Path

import numpy as np
import pytest
import yaml

from helia_core_tester.generation.ops.ComparisonFunctions.comparison import OpComparison
from helia_core_tester.generation.ops.SelectFunctions.select_v2 import OpSelectV2
from helia_core_tester.generation.utils.temp_sizer_probe import missing_header_symbols

ROOT = Path(__file__).resolve().parents[2]


def descriptors():
    result = []
    for group, name in [
        ("ComparisonFunctions", "comparison"),
        ("SelectFunctions", "select_v2"),
    ]:
        path = ROOT / "assets" / "descriptors" / group / (name + ".yaml")
        result.extend(
            d
            for d in yaml.safe_load_all(path.read_text())
            if d["activation_dtype"] == "S32"
        )
    return result


@pytest.mark.parametrize("desc", descriptors(), ids=lambda d: d["name"])
def test_int32_reference_and_call(desc, tmp_path):
    cls = OpComparison if desc["operator"] == "Comparison" else OpSelectV2
    op = cls(desc, seed=7, target_cpu="cortex-m55")
    op.generate_c_files(tmp_path)
    call = op.reference
    assert call is not None
    result = call.output()
    if cls is OpComparison:
        assert call.entry == "comparison_s32"
        values = [call.inputs["input1"], call.inputs["input2"]]
        expected = getattr(np, desc["operation"])(*values)
        assert all(v.dtype == np.int32 for v in values)
        assert set(np.asarray(result, dtype=bool).flat) == {False, True}
        suffix = "comparison"
    else:
        assert call.entry == "select_v2"
        values = [call.inputs["condition"].astype(np.bool_), call.inputs["x"], call.inputs["y"]]
        expected = np.where(*values)
        assert values[1].dtype == values[2].dtype == np.int32
        suffix = "select_v2"
    np.testing.assert_array_equal(np.asarray(result).astype(expected.dtype), expected)
    assert not list(tmp_path.rglob("*.tflite"))
    source = (tmp_path / (desc["name"] + "_" + suffix + ".c")).read_text()
    (symbol,) = desc["required_kernel_symbols"]
    assert symbol + "(" in source
    if cls is OpComparison:
        code = re.sub(r"/\*.*?\*/|//[^\n]*", " ", source, flags=re.S)
        args = code.split(symbol + "(", 1)[1].split(");", 1)[0]
        assert len(args.split(",")) == 6  # No context or requantization parameters.
    else:
        assert "HELIA_VALIDATE_OUTPUTS" in source
        assert "EXACT_INT" in source


def test_required_symbols_skip_older_core(tmp_path):
    include = tmp_path / "Include"
    include.mkdir()
    (include / "arm_nnfunctions.h").write_text(
        "/* no INT32 predicate declarations */\n"
    )
    for desc in descriptors():
        assert (
            missing_header_symbols(desc["required_kernel_symbols"], tmp_path)
            == desc["required_kernel_symbols"]
        )


@pytest.mark.parametrize(
    "cls,name",
    [(OpComparison, "comparison_equal_tail_s32"), (OpSelectV2, "select_v2_tail_s32")],
)
def test_int32_reference_failure_is_not_hidden(cls, name, tmp_path, monkeypatch):
    desc = next(d for d in descriptors() if d["name"] == name)
    op = cls(desc, seed=7, target_cpu="cortex-m55")

    def unavailable(_):
        raise RuntimeError("reference unavailable")

    monkeypatch.setattr(op, "reference_golden", unavailable)
    with pytest.raises(RuntimeError, match="reference unavailable"):
        op.generate_c_files(tmp_path)
