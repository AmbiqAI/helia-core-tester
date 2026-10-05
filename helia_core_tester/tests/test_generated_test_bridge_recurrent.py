"""SVDF and LSTM bridge from tiny synthetic headers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from helia_core_tester.hardware.adapter_specs import generated_test_bridge_scalar_fields
from helia_core_tester.hardware.case_bundle import load_case_bundle
from helia_core_tester.hardware.generated_test_bridge import (
    GeneratedTestCase,
    UnsupportedGeneratedTestError,
    build_case_bundle_from_generated_test,
)
from helia_core_tester.hardware.kernel_registry import lookup_kernel_id

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _case(tmp_path: Path, family: str, descriptor: dict, header: str) -> GeneratedTestCase:
    directory = tmp_path / "generated" / descriptor["name"]
    (directory / "includes").mkdir(parents=True)
    (directory / "includes" / f"{descriptor['name']}.h").write_text(header, encoding="utf-8")
    descriptor = {**descriptor, "resolved_comparison": {"mode": "exact_int"}}
    (directory / "descriptor.yaml").write_text(yaml.safe_dump(descriptor), encoding="utf-8")
    return GeneratedTestCase(name=descriptor["name"], cpu="cortex-m55", family=family, directory=directory, descriptor=descriptor)


def _array(c_type: str, name: str, values) -> str:
    return f"static const {c_type} {name}[] = {{ {', '.join(str(int(v)) for v in values)} }};\n"


def _dims(name: str, n: int, h: int, w: int = 1, c: int = 1) -> str:
    return f"static const cmsis_nn_dims {name} = {{ .n = {n}, .h = {h}, .w = {w}, .c = {c} }};\n"


def _svdf_header(name: str, *, state_type: str, bias: bool) -> str:
    # 2 batches, 4 inputs, 4 filters, 3 time steps, rank 2.
    macro = name.upper()
    defines = {
        "RANK": 2, "INPUT_OFFSET": 3, "OUTPUT_OFFSET": -4, "INPUT_MULTIPLIER": 11, "INPUT_SHIFT": -1,
        "OUTPUT_MULTIPLIER": 22, "OUTPUT_SHIFT": -2, "INPUT_ACTIVATION_MIN": -32768,
        "INPUT_ACTIVATION_MAX": 32767, "OUTPUT_ACTIVATION_MIN": -128, "OUTPUT_ACTIVATION_MAX": 127,
    }
    text = "".join(f"#define {macro}_{key} {value}\n" for key, value in defines.items())
    text += _dims(f"{name}_input_dims", 2, 4) + _dims(f"{name}_weights_feature_dims", 4, 4)
    text += _dims(f"{name}_weights_time_dims", 1, 3) + _dims(f"{name}_state_dims", 2, 4, 3)
    text += _dims(f"{name}_output_dims", 2, 2) + _dims(f"{name}_bias_dims", 1, 1, 1, 2)
    text += _array("int8_t", f"{name}_input_data", range(8)) + _array("int8_t", f"{name}_weights_feature", range(16))
    text += _array(state_type, f"{name}_weights_time", range(12)) + _array(state_type, f"{name}_state_init", range(24))
    text += _array("int8_t", f"{name}_output_ref", range(4))
    return text + (_array("int32_t", f"{name}_bias", (7, 8)) if bias else "")


def _blob(bundle, role: str):
    return next(blob for blob in bundle.blobs if blob.role == role)


def test_svdf_sends_every_time_filter(tmp_path: Path) -> None:
    name = "svdf_synthetic_s8"
    case = _case(tmp_path, "SVDFunctions", {"operator": "SVDF", "name": name, "activation_dtype": "S8"},
                 _svdf_header(name, state_type="int8_t", bias=True))
    bundle = load_case_bundle(build_case_bundle_from_generated_test(
        PROJECT_ROOT, case, output_root=tmp_path / "out", require_fvp_pass=False).manifest_path)

    assert bundle.kernel_id == lookup_kernel_id(PROJECT_ROOT, family="SVDFunctions", operator="SVDF")
    # Header dims say n=1; all 4 filters ship.
    weights_time = _blob(bundle, "input_2")
    assert weights_time.dimensions[:2] == (4, 3)
    assert np.fromfile(weights_time.path, dtype=np.int8).tolist() == list(range(12))
    assert _blob(bundle, "input_1").mutable_data
    assert np.fromfile(_blob(bundle, "meta_0").path, dtype=np.int32).tolist() == [2, 11, -1, 22, -2, -32768, 32767]
    scalars = bundle.manifest["serialized_scalar_parameters"]
    assert set(scalars) <= set(generated_test_bridge_scalar_fields("run_svdf_once"))
    assert (scalars["input_offset"], scalars["output_offset"]) == (3, -4)


def test_svdf_s16_state_without_bias(tmp_path: Path) -> None:
    name = "svdf_state_s16_synthetic_s8"
    case = _case(tmp_path, "SVDFunctions", {"operator": "SVDF", "name": name, "activation_dtype": "S8"},
                 _svdf_header(name, state_type="int16_t", bias=False))
    bundle = build_case_bundle_from_generated_test(PROJECT_ROOT, case, output_root=tmp_path / "out", require_fvp_pass=False)

    assert bundle.kernel_id == lookup_kernel_id(PROJECT_ROOT, family="SVDFunctions", operator="SVDFStateS16")
    assert {blob.role: blob.dtype for blob in bundle.blobs}["input_1"] == "S16"
    assert "bias" not in {blob.role for blob in bundle.blobs}


def _lstm_header(dataset: str, *, batch: int, steps: int, inputs: int, hidden: int) -> str:
    macro = dataset.upper()
    params = {"TIME_MAJOR": "false", "BATCH_SIZE": batch, "TIME_STEPS": steps, "INPUT_SIZE": inputs, "HIDDEN_SIZE": hidden}
    params.update({key: index for index, key in enumerate((
        "INPUT_ZERO_POINT", "FORGET_TO_CELL_MULTIPLIER", "FORGET_TO_CELL_SHIFT", "INPUT_TO_CELL_MULTIPLIER",
        "INPUT_TO_CELL_SHIFT", "CELL_CLIP", "CELL_SCALE_POWER", "OUTPUT_MULTIPLIER", "OUTPUT_SHIFT",
        "OUTPUT_ZERO_POINT"), start=100)})
    for offset, gate in enumerate(("INPUT", "FORGET", "CELL", "OUTPUT")):
        for source_offset, source in enumerate(("INPUT", "HIDDEN")):
            params[f"{gate}_GATE_{source}_MULTIPLIER"] = 1000 + 10 * offset + 2 * source_offset
            params[f"{gate}_GATE_{source}_SHIFT"] = -(1000 + 10 * offset + 2 * source_offset + 1)
    text = "".join(f"#define {macro}_{key} {value}\n" for key, value in params.items())
    text += _array("int8_t", f"{dataset}_input_tensor", range(batch * steps * inputs))
    for index, gate in enumerate(("input", "forget", "cell", "output")):
        text += _array("int8_t", f"{dataset}_{gate}_gate_input_weights", [index] * hidden * inputs)
        text += _array("int8_t", f"{dataset}_{gate}_gate_hidden_weights", [10 + index] * hidden * hidden)
        text += _array("int32_t", f"{dataset}_{gate}_gate_bias", [20 + index] * hidden)
    return text + _array("int8_t", f"{dataset}_output", range(batch * steps * hidden))


def test_lstm_packs_gates_in_input_forget_cell_output_order(tmp_path: Path) -> None:
    descriptor = {"operator": "LSTMUnidirectional", "name": "lstm_synthetic_s8", "activation_dtype": "S8", "dataset": "lstm_x"}
    case = _case(tmp_path, "LSTMFunctions", descriptor, _lstm_header("lstm_x", batch=1, steps=2, inputs=3, hidden=2))
    bundle = build_case_bundle_from_generated_test(PROJECT_ROOT, case, output_root=tmp_path / "out", require_fvp_pass=False)

    weights = np.fromfile(_blob(bundle, "weights").path, dtype=np.int8).tolist()
    assert weights == [0] * 6 + [1] * 6 + [2] * 6 + [3] * 6 + [10] * 4 + [11] * 4 + [12] * 4 + [13] * 4
    assert np.fromfile(_blob(bundle, "bias").path, dtype=np.int32).tolist() == [20, 20, 21, 21, 22, 22, 23, 23]
    meta = np.fromfile(_blob(bundle, "meta_0").path, dtype=np.int32).tolist()
    assert meta[:15] == [0, 1, 2, 3, 2, *range(100, 110)]
    # Per gate: input mult/shift, hidden mult/shift.
    assert meta[15:19] == [1000, -1001, 1002, -1003]
    assert meta[27:31] == [1030, -1031, 1032, -1033]


def test_lstm_s16_is_skipped(tmp_path: Path) -> None:
    descriptor = {"operator": "LSTMUnidirectional", "name": "lstm_synthetic_s16", "activation_dtype": "S16", "dataset": "lstm_x"}
    case = _case(tmp_path, "LSTMFunctions", descriptor, "")
    with pytest.raises(UnsupportedGeneratedTestError, match="S16"):
        build_case_bundle_from_generated_test(PROJECT_ROOT, case, output_root=tmp_path / "out", require_fvp_pass=False)
