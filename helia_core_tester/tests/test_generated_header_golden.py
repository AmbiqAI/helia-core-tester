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
from helia_core_tester.generation.test_ops import default_seed_for_case

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
    # The generator's own seed derivation, so the fixture is the same text a real
    # `helia_core_tester generate` run produces, not a test-only artifact.
    op = OpNNActivationFloat(desc, default_seed_for_case(GOLDEN_CASE), target_cpu=GOLDEN_CPU)
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


# Phase 2a of #74 routes nine more operators through the shared sampler and moves
# six validation call sites onto a shared Jinja macro. Both are refactors that must
# not move a single finite golden, so one case per operator is pinned here -- .h for
# the tensor data the sampling change could perturb, .c for the call site the macro
# change could perturb. The fixtures were generated from the branch point before the
# change; regenerating them is a deliberate act, not a way to make this pass.
ROUTED_CASES = [
    ("abs_float_default_f32", "abs"),
    ("avg_pool_float_default_f32", "avg_pool"),
    ("concatenation_axis_x_f32", "concatenation"),
    ("max_pool_float_default_f32", "max_pool"),
    ("reduce_sum_float_axis_c_f32", "reduce_sum"),
    ("softmax_float_default_f32", "softmax"),
    ("split_float_channels_pairs_f16", "split"),
    ("strided_slice_float_whole_slab_f32", "strided_slice"),
    ("sub_float_default_f32", "sub"),
    ("transpose_float_default_f32", "transpose"),
]
ROUTED_CPU = "cortex-m55"


def _finite_golden_dir() -> Path:
    return _repo_root() / "helia_core_tester" / "tests" / "fixtures" / "finite_goldens"


@pytest.fixture(scope="module")
def routed_case_outputs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    from helia_core_tester.core.discovery import find_descriptors_dir
    from helia_core_tester.generation.io.descriptors import load_all_descriptors
    import helia_core_tester.generation.test_ops as generation_module

    descriptors = {
        desc["name"]: desc for desc in load_all_descriptors(str(find_descriptors_dir()))
    }
    out_dir = tmp_path_factory.mktemp("routed_goldens")
    emitted: dict[str, Path] = {}
    for case_name, _suffix in ROUTED_CASES:
        desc = descriptors[case_name]
        generation_module.generate_test(desc, str(out_dir), cpu=ROUTED_CPU)
        emitted[case_name] = out_dir / desc["_family"] / case_name
    return emitted


@pytest.mark.parametrize("case_name,op_suffix", ROUTED_CASES)
@pytest.mark.parametrize("kind", ["h", "c"])
def test_routed_operator_keeps_its_finite_golden(
    routed_case_outputs: dict[str, Path], case_name: str, op_suffix: str, kind: str
) -> None:
    test_dir = routed_case_outputs[case_name]
    filename = f"{case_name}_{op_suffix}.{kind}"
    emitted_path = test_dir / "includes" / filename if kind == "h" else test_dir / filename
    emitted = emitted_path.read_text()
    fixture = _finite_golden_dir() / filename
    expected = fixture.read_text()

    if emitted != expected:
        pytest.fail(
            f"generated {filename} drifted from the fixture ({fixture}); sha256 "
            f"{hashlib.sha256(emitted.encode()).hexdigest()} vs "
            f"{hashlib.sha256(expected.encode()).hexdigest()}"
        )
