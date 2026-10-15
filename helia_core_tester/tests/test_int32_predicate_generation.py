"""INT32 predicate references and generated API contracts. Refs #478."""

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


class RecordingInterpreter:
    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.inputs = {}
        self.result = None

    def __getattr__(self, name):
        return getattr(self.interpreter, name)

    def set_tensor(self, index, value):
        self.inputs[index] = value.copy()
        self.interpreter.set_tensor(index, value)

    def get_tensor(self, index):
        self.result = self.interpreter.get_tensor(index)
        return self.result


@pytest.mark.parametrize("desc", descriptors(), ids=lambda d: d["name"])
def test_int32_reference_and_call(desc, tmp_path, monkeypatch):
    cls = OpComparison if desc["operator"] == "Comparison" else OpSelectV2
    op = cls(desc, seed=7, target_cpu="cortex-m55")
    model = tmp_path / (desc["name"] + ".tflite")
    op.convert_to_tflite(None, str(model), 7)
    interpreter = RecordingInterpreter(op.load_litert_interpreter(str(model)))
    monkeypatch.setattr(op, "load_litert_interpreter", lambda _: interpreter)
    op.generate_c_files(tmp_path)
    values = [interpreter.inputs[d["index"]] for d in interpreter.get_input_details()]
    if cls is OpComparison:
        expected = getattr(np, desc["operation"])(*values)
        assert all(v.dtype == np.int32 for v in values)
        assert set(interpreter.result.flat) == {False, True}
        suffix = "comparison"
    else:
        expected = np.where(*values)
        assert values[0].dtype == np.bool_
        assert values[1].dtype == values[2].dtype == np.int32
        suffix = "select_v2"
    np.testing.assert_array_equal(interpreter.result, expected)
    source = (tmp_path / (desc["name"] + "_" + suffix + ".c")).read_text()
    (symbol,) = desc["required_kernel_symbols"]
    assert symbol + "(" in source
    if cls is OpComparison:
        call = source.split(symbol + "(", 1)[1].split(");", 1)[0]
        assert len(call.split(",")) == 6  # No context or requantization parameters.
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
    model = tmp_path / (name + ".tflite")
    op.convert_to_tflite(None, str(model), 7)

    def unavailable(_):
        raise RuntimeError("reference unavailable")

    monkeypatch.setattr(op, "load_litert_interpreter", unavailable)
    with pytest.raises(RuntimeError, match="reference unavailable"):
        op.generate_c_files(tmp_path)
