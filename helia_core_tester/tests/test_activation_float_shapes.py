"""Hidden float sigmoid shapes: draws, edges, streams."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
import yaml

from helia_core_tester.generation import random_shapes as rs

SIG32, SIG16 = ("NNActivationFloat", "FP32"), ("NNActivationFloat", "FP16")
SECRET = b"s3cret-hidden-seed-0123"
# Every hidden set drawn before sigmoid joined.
PRIOR = tuple(key for key in rs.OPS if key[0] != "NNActivationFloat")


def _digest(cases: list[dict]) -> str:
    return hashlib.sha256(yaml.safe_dump_all(cases, sort_keys=False).encode()).hexdigest()


def test_sigmoid_keys_and_streams() -> None:
    assert rs.select_ops("NNActivationFloat", "fp32") == (SIG32,)
    assert rs.select_ops("nn_activation_float", None) == (SIG32, SIG16)
    assert rs.op_suites("NNActivationFloat", None) == {"float"}
    assert [rs.GENERATORS[key].stream for key in (SIG32, SIG16)] == [6, 7]
    assert not any(rs.GENERATORS[key].layerless for key in PRIOR)


@pytest.mark.parametrize(("cpu", "n", "digest"), [
    ("cortex-m55", 4, "8ef5566124719b8b66221c49f9217a7b661578ae9f692d5f29fe3ce37a8dc6be"),
    ("cortex-m55", 12, "a7dbe60de604e71c3c3a59849dfee0c3e5512a97b0fedcad03a5f36f622e7073"),
    ("cortex-m4", 4, "e5cfad80f35b8cb71464932f08c3d4e09f8abd67e5a9f0ccbc9e1a0bb13b3aee"),
    ("cortex-m4", 12, "75cdebefb65732259f51fef7f2e8adedaf544c314bbdb20e0f43f4b9b8ce9d60"),
])
def test_prior_hidden_sets_unchanged(cpu: str, n: int, digest: str) -> None:
    # Pinned before sigmoid joined.
    assert _digest(rs.hidden_cases(n, SECRET, cpu, PRIOR)) == digest


def test_sigmoid_streams_differ() -> None:
    # Same plan, other stream: other data.
    f32, f16 = (rs.sample_cases(12, 9, ops=(key,)) for key in (SIG32, SIG16))
    assert [c["input_shape"] for c in f32] != [c["input_shape"] for c in f16]
    assert rs.sample_cases(12, 5, ops=(SIG32,)) != rs.sample_cases(12, 6, ops=(SIG32,))


@pytest.mark.parametrize("n", [1, 2, 3, 6, 9, 12, 24, 50])
def test_sigmoid_edge_slots(n: int) -> None:
    cases = rs.sample_cases(n, 4, ops=(SIG32,))
    count = rs.edge_count(n, len(rs.SIG_EDGES))
    assert [c.get("edge") for c in cases[: n - count]] == [None] * (n - count)
    assert [c["edge"] for c in cases[n - count:]] == [rs.SIG_EDGES[i % len(rs.SIG_EDGES)] for i in range(count)]


@pytest.mark.parametrize("key", [SIG32, SIG16])
@pytest.mark.parametrize("seed", range(6))
def test_sigmoid_edges_hold(key: tuple[str, str], seed: int) -> None:
    dtype = key[1]
    lanes, top = rs.TCF_LANES[dtype], rs.ACT_MAX_ABS[dtype]
    cases = rs.sample_cases(14, seed, ops=(key,))
    assert [c["edge"] for c in cases if "edge" in c] == list(rs.SIG_EDGES)
    for case in cases:
        length, low, high = case["input_shape"][1], case["input_min"], case["input_max"]
        edge = case.get("edge")
        assert case["activation_type"] == "ARM_NN_FLT_ACT_SIGMOID" and case["suite"] == "float"
        assert case["tensor_dtypes"] == {"input": dtype, "output": dtype}
        assert case["expected_route"] == rs.ACT_ROUTES[dtype]
        assert 1 <= length <= rs.ACT_MAX_LEN and -top <= low < high <= top
        assert ("input_mode" in case) == (edge == "nonfinite")
        if edge == "saturate":
            assert low == -high and high >= 4 * rs.SIG_ROUNDS[dtype]
        elif edge == "knee":
            assert low * high > 0 and max(-low, high) >= rs.SIG_ROUNDS[dtype] > min(abs(low), abs(high))
        elif edge == "nonfinite":
            tokens, positions = case["nonfinite_tokens"], case["nonfinite_positions"]
            # NaN is outside the kernel contract.
            assert set(tokens) <= {"inf", "-inf"} and case["nonfinite_policy"] == "strict"
            assert len(set(positions)) == len(tokens) and max(positions) < length
        elif edge == "odd_tail":
            assert length % lanes
        elif edge == "underflow":
            sub_low, sub_high = rs.SIG_SUBNORMAL[dtype]
            assert low <= sub_low and sub_high <= high < 0
        elif edge == "near_zero":
            assert low == -high and high <= 10 ** rs.SIG_TINY[dtype][1]
        elif edge == "long":
            assert length >= 2048


def test_sigmoid_bounds_match_dtype() -> None:
    for dtype, cast, tiny in (("FP32", np.float32, 2.0 ** -149), ("FP16", np.float16, 2.0 ** -24)):
        info = np.finfo(cast)
        # Past the bound, 1 + e^-x rounds to 1.
        assert cast(1) + cast(np.exp(-rs.SIG_ROUNDS[dtype])) == cast(1)
        low, high = rs.SIG_SUBNORMAL[dtype]
        assert np.exp(high) < info.tiny and np.exp(low) >= tiny / 2
        assert rs.ACT_MAX_ABS[dtype] < info.max


@pytest.mark.parametrize("cpu", ["cortex-m55", "cortex-m4"])
def test_sigmoid_cases_fit_the_board(cpu: str) -> None:
    workspace = rs.min_workspace(cpu)
    for case in rs.sample_cases(24, 2, cpu, (SIG32, SIG16)):
        item = 4 if case["activation_dtype"] == "FP32" else 2
        assert 2 * item * case["input_shape"][1] <= workspace
