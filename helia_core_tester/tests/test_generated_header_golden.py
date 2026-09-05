"""Pin the emitted header for one finite float case against a checked-in fixture.

The non-finite work reaches into shared sampling and serialization paths that every
float descriptor uses, so a change there can move the goldens of cases that carry no
non-finite token at all. This locks one such case byte for byte. The chosen case is a
tanh f16 on cortex-m55: it runs the MVE LUT reference, which is the reference model most
likely to be perturbed by a change made for a NaN lane.

Regenerating the fixture is a deliberate act. If this fails, either the change was not
meant to move finite goldens, or the fixture is stale and the diff belongs in the commit
message.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from helia_core_tester.generation.ops.ActivationFunctions.nn_activation_float import (
    OpNNActivationFloat,
)

GOLDEN_CASE = "nn_activation_float_tanh_f16"
GOLDEN_CPU = "cortex-m55"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _fixture_path() -> Path:
    return _repo_root() / "helia_core_tester" / "tests" / "fixtures" / f"{GOLDEN_CASE}_nn_activation_float.h"


def _descriptor() -> dict:
    path = _repo_root() / "assets" / "descriptors" / "ActivationFunctions" / "nn_activation_float.yaml"
    for doc in yaml.safe_load_all(path.read_text()):
        if isinstance(doc, dict) and doc.get("name") == GOLDEN_CASE:
            return doc
    raise AssertionError(f"descriptor {GOLDEN_CASE} not found")


def _emit_header(output_dir: Path) -> str:
    desc = _descriptor()
    # Mirrors the seed derivation in generation/test_ops.py so the fixture is the same
    # text a real `helia_core_tester generate` run produces, not a test-only artifact.
    seed = int.from_bytes(hashlib.sha256(GOLDEN_CASE.encode("utf-8")).digest()[:4], "little")
    op = OpNNActivationFloat(desc, seed, target_cpu=GOLDEN_CPU)
    (output_dir / f"{GOLDEN_CASE}.tflite").touch()
    op.generate_c_files(output_dir)
    return (output_dir / "includes" / f"{GOLDEN_CASE}_nn_activation_float.h").read_text()


def test_finite_case_header_matches_the_checked_in_fixture(tmp_path: Path) -> None:
    emitted = _emit_header(tmp_path)
    expected = _fixture_path().read_text()

    if emitted != expected:
        pytest.fail(
            f"generated header for {GOLDEN_CASE} drifted from the fixture "
            f"({_fixture_path()}); sha256 {hashlib.sha256(emitted.encode()).hexdigest()} "
            f"vs {hashlib.sha256(expected.encode()).hexdigest()}"
        )
