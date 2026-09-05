"""Pin the emitted header for one finite float case against a checked-in fixture.

The non-finite work reaches into shared sampling and serialization paths that every
float descriptor uses, so a change there can move the goldens of cases that carry no
non-finite token at all. This locks one such case. The chosen case is a tanh f16 on
cortex-m55: it runs the MVE LUT reference, which is the reference model most likely to
be perturbed by a change made for a NaN lane.

A header is locked byte for byte everywhere except the bodies of the
`*expected_output*` arrays. Those carry the reference model's own arithmetic, and
operators such as softmax differ in the last bits between host libms, so a byte hash of
a golden is host-dependent while the input draw and every other line is not. The golden
bodies are therefore compared numerically, tightly enough that any change of substance
still fails.

Regenerating the fixture is a deliberate act. If this fails, either the change was not
meant to move finite goldens, or the fixture is stale and the diff belongs in the commit
message.
"""

from __future__ import annotations

import hashlib
import math
import re
import struct
from pathlib import Path

import pytest
import yaml

from helia_core_tester.generation.ops.ActivationFunctions.nn_activation_float import (
    OpNNActivationFloat,
)
from helia_core_tester.generation.test_ops import default_seed_for_case

GOLDEN_CASE = "nn_activation_float_tanh_f16"
GOLDEN_CPU = "cortex-m55"

# Multi-output operators emit one array per tensor (`..._out_0_expected_output`), so the
# name is matched by substring rather than by a per-operator suffix list.
_GOLDEN_ARRAY_RE = re.compile(
    r"(?P<head>static const (?P<ctype>[A-Za-z_][A-Za-z0-9_ ]*?) "
    r"(?P<name>[A-Za-z0-9_]*expected_output[A-Za-z0-9_]*)\[\]\s*=\s*\{)"
    r"(?P<body>[^{}]*)\}",
    re.MULTILINE,
)
_CAST_RE = re.compile(r"^\(\s*[A-Za-z_][A-Za-z0-9_]*\s*\)\s*")

# f16 carries ~11 bits of mantissa, so one ulp is already ~1e-3 relative; f32 gets the
# tighter of a 4 ulp and a relative bound so near-zero goldens are not given a free pass.
_REL_TOL = {"float16_t": 1e-3}
_DEFAULT_REL_TOL = 1e-6
_ULP_TOL = 4


def _f32_ulp(value: float) -> float:
    bits = struct.unpack("<I", struct.pack("<f", value))[0] & 0x7FFFFFFF
    (lo,) = struct.unpack("<f", struct.pack("<I", bits))
    (hi,) = struct.unpack("<f", struct.pack("<I", bits + 1))
    step = hi - lo
    return step if math.isfinite(step) else 0.0


def _mask_golden_arrays(text: str) -> tuple[str, dict[str, tuple[str, str]]]:
    """Replace each golden array body with its name, keeping the bodies aside."""
    bodies: dict[str, tuple[str, str]] = {}

    def _replace(match: re.Match[str]) -> str:
        name = match.group("name")
        bodies[name] = (match.group("ctype").strip(), match.group("body"))
        return f"{match.group('head')}<{name} body>}}"

    return _GOLDEN_ARRAY_RE.sub(_replace, text), bodies


def _parse_c_literals(body: str) -> list[float | str]:
    values: list[float | str] = []
    for raw in body.split(","):
        token = _CAST_RE.sub("", raw.strip())
        if not token:
            continue
        # NAN and INFINITY are symbolic: a golden that flips finiteness is a real drift,
        # never a libm last-bit difference, so they are held to an exact match.
        if re.fullmatch(r"[-+]?(NAN|INFINITY)", token):
            values.append(token.lstrip("+"))
            continue
        values.append(float(token.rstrip("fF")))
    return values


def _golden_array_deviation(name: str, ctype: str, emitted: str, expected: str) -> str | None:
    got = _parse_c_literals(emitted)
    want = _parse_c_literals(expected)
    if len(got) != len(want):
        return f"{name}: {len(got)} values emitted, fixture has {len(want)}"

    rel_tol = _REL_TOL.get(ctype, _DEFAULT_REL_TOL)
    worst: tuple[float, str] | None = None
    for index, (a, b) in enumerate(zip(got, want)):
        if isinstance(a, str) or isinstance(b, str):
            if a != b:
                return f"{name}[{index}]: emitted {a!r}, fixture has {b!r}"
            continue
        tol = max(abs(b) * rel_tol, _ULP_TOL * _f32_ulp(b))
        delta = abs(a - b)
        if delta <= tol:
            continue
        relative = delta / abs(b) if b else math.inf
        detail = (
            f"{name}[{index}]: emitted {a!r}, fixture has {b!r} "
            f"(abs {delta:.3e}, rel {relative:.3e}, tol {tol:.3e})"
        )
        if worst is None or delta > worst[0]:
            worst = (delta, detail)
    return worst[1] if worst else None


def _assert_header_matches(emitted: str, expected: str, filename: str, fixture: Path) -> None:
    emitted_skeleton, emitted_goldens = _mask_golden_arrays(emitted)
    expected_skeleton, expected_goldens = _mask_golden_arrays(expected)

    if emitted_skeleton != expected_skeleton:
        pytest.fail(
            f"generated {filename} drifted from the fixture ({fixture}) outside its golden "
            f"arrays; sha256 {hashlib.sha256(emitted_skeleton.encode()).hexdigest()} vs "
            f"{hashlib.sha256(expected_skeleton.encode()).hexdigest()}"
        )

    failures = [
        detail
        for name, (ctype, body) in expected_goldens.items()
        if (detail := _golden_array_deviation(name, ctype, emitted_goldens[name][1], body))
    ]
    if failures:
        pytest.fail(
            f"golden values in {filename} moved beyond host-FPU tolerance vs the fixture "
            f"({fixture}):\n  " + "\n  ".join(failures)
        )


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
    fixture = _fixture_path()
    _assert_header_matches(
        _emit_header(tmp_path), fixture.read_text(), f"{GOLDEN_CASE}_nn_activation_float.h", fixture
    )


# Phase 2a of #74 routes more operators through the shared sampler and moves their
# validation call sites onto a shared Jinja macro. Both are refactors that must not
# move a single finite golden, so one case per operator is pinned here -- .h for the
# tensor data the sampling change could perturb, .c for the call site the macro change
# could perturb. Every fixture was generated from the state before the change it
# guards; regenerating one is a deliberate act, not a way to make this pass.
ROUTED_CASES = [
    ("abs_float_default_f32", "abs"),
    ("avg_pool_float_default_f32", "avg_pool"),
    ("batch_norm_default_f32", "batch_norm"),
    ("concatenation_axis_x_f32", "concatenation"),
    ("max_pool_float_default_f32", "max_pool"),
    ("minimum_float_default_f32", "minmax"),
    ("reduce_sum_float_axis_c_f32", "reduce_sum"),
    ("softmax_float_default_f32", "softmax"),
    ("split_float_channels_pairs_f16", "split"),
    ("strided_slice_float_whole_slab_f32", "strided_slice"),
    ("sub_float_default_f32", "sub"),
    ("transpose_float_default_f32", "transpose"),
    # Phase 2c adds the recurrent float families. The streaming cases stand in for LSTM
    # and GRU because the single-shot templates branch on a generation-time probe of the
    # configured ns-cmsis-nn checkout (the temp-buffer sizers of ns-cmsis-nn#381), which
    # would make their .c text a property of the machine rather than of this repo. The
    # streaming and single-shot paths share the sampling and the header template, so the
    # tensors and goldens under test are the same ones.
    ("lstm_unidirectional_float_stream_f32", "lstm_unidirectional"),
    ("gru_unidirectional_float_stream_f32", "gru_unidirectional"),
    ("svdf_float_default_f32", "svdf"),
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

    if kind == "h":
        _assert_header_matches(emitted, expected, filename, fixture)
        return

    if emitted != expected:
        pytest.fail(
            f"generated {filename} drifted from the fixture ({fixture}); sha256 "
            f"{hashlib.sha256(emitted.encode()).hexdigest()} vs "
            f"{hashlib.sha256(expected.encode()).hexdigest()}"
        )
