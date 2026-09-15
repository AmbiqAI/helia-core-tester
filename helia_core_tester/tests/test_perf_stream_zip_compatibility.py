"""Exercise the reporting/bridge zip contracts without external artifacts."""

import csv
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from helia_core_tester.generation.io.dtypes import resolve_comparison
from helia_core_tester.generation.test_ops import generate_test
from helia_core_tester.perf_stream import generated_test_bridge, result_bundle
from helia_core_tester.perf_stream.case_bundle import (
    build_abs_s8_case_bundle,
    load_case_bundle,
)
from helia_core_tester.perf_stream.fake_target import FakeTargetTransport
from helia_core_tester.perf_stream.session import HostSession

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def session(tmp_path):
    bundle = load_case_bundle(
        build_abs_s8_case_bundle(ROOT, output_root=tmp_path).manifest_path
    )
    return HostSession(FakeTargetTransport()).run_many([bundle])


@pytest.mark.parametrize("empty", [False, True])
def test_result_writer_accepts_legacy_zip(tmp_path, monkeypatch, session, empty):
    if empty:
        session = replace(
            session,
            cases=(replace(session.cases[0], samples=(), normalized_samples=()),),
        )
    monkeypatch.setattr(result_bundle, "zip", lambda *args: zip(*args), raising=False)
    root = result_bundle.write_result_bundle(
        session,
        session_id="legacy",
        output_root=tmp_path,
        memory_report={},
        kernel_catalog=[],
    )
    with (root / "raw_samples.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    case = session.cases[0]
    expected = [
        (sample.sample_index, normalized.cycles_per_invocation, counter["name"])
        for sample, normalized in zip(case.samples, case.normalized_samples)
        for counter in sample.counters
    ]
    assert bool(expected) is not empty
    assert [
        (
            int(row["sample_index"]),
            float(row["cycles_per_invocation"]),
            row["counter_name"],
        )
        for row in rows
    ] == expected


@pytest.mark.parametrize("delta", [-1, 1])
def test_result_writer_rejects_unpaired_samples(tmp_path, session, delta):
    case = session.cases[0]
    assert case.normalized_samples
    normalized = (
        case.normalized_samples[:-1]
        if delta < 0
        else case.normalized_samples + (case.normalized_samples[-1],)
    )
    malformed = replace(session, cases=(replace(case, normalized_samples=normalized),))
    with pytest.raises(ValueError):
        result_bundle.write_result_bundle(
            malformed,
            session_id="mismatch",
            output_root=tmp_path,
            memory_report={},
            kernel_catalog=[],
        )


def test_generated_concatenation_accepts_legacy_zip(tmp_path, monkeypatch):
    descriptor = next(
        yaml.safe_load_all(
            (
                ROOT / "assets/descriptors/ConcatenationFunctions/concatenation.yaml"
            ).read_text()
        )
    )
    generate_test(descriptor, str(tmp_path), seed=500)
    case_dir = tmp_path / "ConcatenationFunctions" / descriptor["name"]
    descriptor["resolved_comparison"] = resolve_comparison(descriptor)
    case = generated_test_bridge.GeneratedTestCase(
        descriptor["name"], "cortex-m55", "ConcatenationFunctions", case_dir, descriptor
    )
    monkeypatch.setattr(
        generated_test_bridge, "zip", lambda *args: zip(*args), raising=False
    )
    bundle = generated_test_bridge.build_case_bundle_from_generated_test(
        ROOT, case, output_root=tmp_path / "bridge", require_fvp_pass=False
    )
    inputs = [blob for blob in bundle.blobs if blob.blob_id in (1, 2)]
    assert len(inputs) == 2
    assert [blob.dimensions for blob in inputs] == [(1, 4, 4, 3), (1, 4, 4, 1)]
    arrays = [
        np.frombuffer(blob.path.read_bytes(), dtype=np.int8).reshape(blob.dimensions)
        for blob in inputs
    ]
    expected = np.concatenate(arrays, axis=3)
    assert bundle.expected_output.dimensions == expected.shape
    assert bundle.expected_output.path.read_bytes() == expected.tobytes()
