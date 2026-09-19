"""Float policy at the generated-data, host session and persisted-report boundaries."""

import csv
import json
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pytest
import yaml

from helia_core_tester.generation.io.descriptors import load_descriptor
from helia_core_tester.generation.ops.BasicMathFunctions.abs import OpAbs
from helia_core_tester.hardware.case_bundle import blob_numpy, load_case_bundle
from helia_core_tester.hardware.comparison import compare_output
from helia_core_tester.hardware.fake_target import (
    FakeKernelAdapter,
    FakeTargetTransport,
)
from helia_core_tester.hardware.generated_test_bridge import (
    GeneratedTestCase,
    build_case_bundle_from_generated_test,
)
from helia_core_tester.hardware.hctp import MessageType
from helia_core_tester.hardware.result_bundle import write_result_bundle
from helia_core_tester.hardware.session import HostSession
from helia_core_tester.hardware.wire import CatalogEntry, decode_correctness_ack

ROOT = Path(__file__).resolve().parents[2]
FLOAT = {"mode": "float", "atol": 0.001, "rtol": 0.001}


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("actual", [1.0, np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("expected", [1.0, np.nan, np.inf, -np.inf])
def test_classifications(dtype, actual, expected):
    result = compare_output(
        np.array([actual], dtype=dtype), np.array([expected], dtype=dtype), FLOAT
    )
    matches = (np.isnan(actual) and np.isnan(expected)) or actual == expected
    assert result.passed == matches
    assert result.mismatch_count == (0 if matches else 1)
    assert result.max_abs_diff == (0 if matches else np.inf)


def test_finite_tolerance_and_other_modes():
    for mode, actual, expected, passed in [
        ({"mode": "float", "atol": 0.125, "rtol": 0}, [1.125], [1], True),
        ({"mode": "float", "atol": 0.125, "rtol": 0}, [1.25], [1], False),
        ({"mode": "float", "atol": 0, "rtol": 0.125}, [2.25], [2], True),
        (FLOAT, [-0.0], [0.0], True),
        (FLOAT, [], [], True),
        ({"mode": "exact_int"}, [2], [1], False),
        ({"mode": "tolerant_int", "tolerance": 1}, [2], [1], True),
        ({"mode": "bool"}, [0], [1], False),
        ({"mode": "none"}, [np.nan], [1], True),
    ]:
        assert (
            compare_output(np.array(actual), np.array(expected), mode).passed == passed
        )


def test_mask_precedes_classification_and_metrics():
    assert compare_output(
        np.array([]), np.array([]), {**FLOAT, "nonfinite_mask": []}
    ).passed
    result = compare_output(
        np.array([[np.nan, 123, 1.25, np.inf]]),
        np.array([[0, 0, 1, np.inf]]),
        {**FLOAT, "nonfinite_mask": [1, 1, 0, 0]},
    )
    assert not result.passed
    assert result.mismatch_count == 1
    assert result.max_abs_diff == 0.25
    assert compare_output(
        np.array([np.nan]), np.array([0]), {**FLOAT, "nonfinite_mask": [1]}
    ).passed


@pytest.mark.parametrize("mask", [[1, 0], [2], [0.5], ["0"], [[0]], None])
def test_invalid_mask_is_not_silently_coerced(mask):
    with pytest.raises(ValueError, match="mask"):
        compare_output(
            np.array([1.0]), np.array([1.0]), {**FLOAT, "nonfinite_mask": mask}
        )


@pytest.fixture(params=["f32", "f16"])
def generated_abs(request, tmp_path):
    name = f"abs_float_nonfinite_{request.param}"
    desc = next(
        d
        for d in load_descriptor(
            str(ROOT / "assets/descriptors/BasicMathFunctions/abs_float.yaml")
        )
        if d["name"] == name
    )
    directory = tmp_path / name
    directory.mkdir()
    op = OpAbs(desc, seed=500, target_cpu="cortex-m55")
    op.convert_to_tflite(None, str(directory / f"{name}.tflite"), 500)
    op.generate_c_files(directory)
    (directory / "descriptor.yaml").write_text(yaml.safe_dump(desc))
    return GeneratedTestCase(
        name, "cortex-m55", "BasicMathFunctions", directory, desc, suite="float"
    )


def _bridge(case, tmp_path):
    bundle = build_case_bundle_from_generated_test(
        ROOT, case, output_root=tmp_path, fvp_gate="off"
    )
    loaded = load_case_bundle(bundle.manifest_path)
    assert loaded.comparison == bundle.comparison
    return loaded


class OutputAdapter(FakeKernelAdapter):
    def __init__(self, bundle, output):
        self.entry = CatalogEntry(
            bundle.kernel_id,
            "test_float_output",
            "BasicMathFunctions",
            1,
            bundle.expected_output.dtype,
            1,
            True,
            True,
            False,
            0,
        )
        self.output = output

    def invoke(self, blobs, scalar_parameters):
        return self.output.copy()


class AckTransport(FakeTargetTransport):
    def __init__(self, adapter):
        self.adapter = adapter
        self.acks = []
        super().__init__()

    def _emit_target_info(self):
        self._adapters = {self.adapter.entry.kernel_id: self.adapter}
        self._catalog = (self.adapter.entry,)
        super()._emit_target_info()

    def _handle_frame(self, frame):
        if frame.header.message_type == MessageType.CORRECTNESS_ACK:
            self.acks.append(decode_correctness_ack(frame.payload).passed)
        super()._handle_frame(frame)

    def _case_blobs(self):
        # The shared fake's built-in adapters only decode integer inputs.
        return {
            spec.role: np.frombuffer(
                self._accumulators[blob_id].finish(), dtype=self.adapter.output.dtype
            ).reshape(spec.dimensions)
            for blob_id, spec in self._blob_specs.items()
        }


@pytest.mark.parametrize("corrupt", [False, True])
def test_generated_mask_survives_reload_session_ack_and_reports(
    generated_abs, tmp_path, corrupt
):
    bundle = _bridge(generated_abs, tmp_path)
    assert bundle.comparison["nonfinite_mask"] == [1] * 3 + [0] * 125
    expected = blob_numpy(bundle.expected_output)
    actual = np.abs(blob_numpy(bundle.input_blob))
    np.testing.assert_array_equal(actual.reshape(-1)[3:], expected.reshape(-1)[3:])
    # Both finite and nonfinite arbitrary values are valid only at the exact masked lanes.
    actual.reshape(-1)[:3] = [123, np.nan, -np.inf]
    if corrupt:
        actual.reshape(-1)[3] = np.nan
    _check_session_reports(bundle, actual, tmp_path, not corrupt)


@pytest.mark.parametrize(
    "actual_value,expected_value,passed",
    [
        (np.nan, 1, False),
        (1, np.nan, False),
        (np.inf, -np.inf, False),
        (np.nan, np.nan, True),
        (np.inf, np.inf, True),
    ],
)
def test_strict_bundle_session_reports(
    generated_abs, tmp_path, actual_value, expected_value, passed
):
    bundle = _bridge(generated_abs, tmp_path)
    # A persisted strict bundle has no mask; its supplied golden remains authoritative.
    bundle.manifest["correctness_comparison"].pop("nonfinite_mask")
    bundle.manifest_path.write_text(json.dumps(bundle.manifest))
    expected = blob_numpy(bundle.expected_output).copy()
    expected.reshape(-1)[0] = expected_value
    bundle.expected_output.path.write_bytes(expected.tobytes())
    bundle = load_case_bundle(bundle.manifest_path)
    actual = expected.copy()
    actual.reshape(-1)[0] = actual_value
    _check_session_reports(bundle, actual, tmp_path, passed)


def _check_session_reports(bundle, actual, tmp_path, passed):
    transport = AckTransport(OutputAdapter(bundle, actual))
    result = HostSession(transport).run(bundle)
    assert transport.acks == [passed]
    assert result.comparison.passed == passed
    assert result.comparison.mismatch_count == int(not passed)
    assert result.session_complete_cases == 1
    reports = write_result_bundle(
        result,
        session_id="float-check",
        output_root=tmp_path,
        memory_report={},
        kernel_catalog=[],
    )
    correctness = json.loads(
        (reports / "correctness" / f"{bundle.case_id}.json").read_text()
    )
    assert correctness["passed"] == passed
    assert correctness["mismatch_count"] == int(not passed)
    assert correctness["comparison"] == bundle.comparison
    assert (
        json.loads((reports / "cases.json").read_text())[0]["comparison_passed"]
        == passed
    )
    with (reports / "case_summary.csv").open() as stream:
        assert next(csv.DictReader(stream))["comparison_passed"] == str(passed).lower()
    assert len(ElementTree.parse(reports / "junit.xml").findall(".//failure")) == int(
        not passed
    )


@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "short",
        "nonbinary",
        "all",
        "wrong_output",
        "wrong_count",
        "ambiguous",
    ],
)
def test_invalid_generated_mask_fails_admission(generated_abs, tmp_path, damage):
    source = next(generated_abs.directory.glob("*.c"))
    text = source.read_text()
    if damage == "missing":
        text = text.replace("HELIA_VALIDATE_FLOATS_MASKED", "MISSING_VALIDATION")
    elif damage in ("short", "nonbinary", "all"):
        start = text.index("{", text.index(f"{generated_abs.name}_expected_mask[]")) + 1
        end = text.index("}", start)
        values = {"short": "1, 0", "nonbinary": "2," + "0," * 127, "all": "1," * 128}
        text = text[:start] + values[damage] + text[end:]
    elif damage == "wrong_output":
        text = text.replace(
            f"{generated_abs.name}_expected_output,", "unrelated_expected_output,"
        )
    elif damage == "wrong_count":
        text = text.replace(f"{generated_abs.name.upper()}_OUTPUT_SIZE,", "1,")
    else:
        start = text.index("HELIA_VALIDATE_FLOATS_MASKED(")
        end = text.index(");", start) + 2
        text += "\n" + text[start:end]
    source.write_text(text)
    with pytest.raises(ValueError, match="mask"):
        _bridge(generated_abs, tmp_path)
