"""Float comparison policy: assert_close's defaults everywhere, descriptors may only tighten them, and only the
shipped build (-Ofast, no coverage) gets kernel-faithful goldens; every other build widens an exact golden by a
bound the generator derives."""

from __future__ import annotations

import math
from glob import glob
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from helia_core_tester.core.steps.generate import kernel_numerics
from helia_core_tester.generation.io.dtypes import default_comparison_for_dtype, resolve_comparison
from helia_core_tester.generation.ops._shared import kernel_faithful as kf
from helia_core_tester.generation.reuse import case_stamp
from helia_core_tester.hardware import generated_test_bridge as bridge
from helia_core_tester.hardware.session import tolerance_q16

ROOT = Path(__file__).resolve().parents[2]
ASSERT_CLOSE = {"FP32": (1e-5, 1.3e-6), "FP16": (1e-5, 1e-3)}


def _float_descriptors():
    for path in sorted(glob(str(ROOT / "assets/descriptors/**/*.yaml"), recursive=True)):
        for desc in yaml.safe_load_all(Path(path).read_text()):
            if not isinstance(desc, dict):
                continue
            out = (desc.get("tensor_dtypes") or {}).get("output") or desc.get("activation_dtype")
            if out in ASSERT_CLOSE:
                yield desc, out


@pytest.mark.parametrize("dtype", ["FP32", "FP16"])
def test_defaults_are_assert_close(dtype):
    cmp = default_comparison_for_dtype(dtype)
    assert (cmp["mode"], cmp["atol"], cmp["rtol"]) == ("float", *ASSERT_CLOSE[dtype])
    np_dtype = np.float32 if dtype == "FP32" else np.float16
    assert kf.PYTORCH_TOLERANCE[np.dtype(np_dtype)] == ASSERT_CLOSE[dtype]


def test_descriptors_only_tighten_the_defaults():
    loosened = []
    tightened = 0
    for desc, out in _float_descriptors():
        stated = desc.get("comparison")
        if not isinstance(stated, dict):
            continue
        atol, rtol = ASSERT_CLOSE[out]
        if float(stated.get("atol", 0.0)) > atol or float(stated.get("rtol", 0.0)) > rtol:
            loosened.append((desc["name"], stated))
        tightened += 1
    assert not loosened, f"descriptors may only tighten the float comparison: {loosened}"
    assert tightened > 0


def test_resolved_comparison_never_exceeds_the_defaults():
    for desc, out in _float_descriptors():
        cmp = resolve_comparison(desc)
        if cmp["mode"] != "float":
            continue
        atol, rtol = ASSERT_CLOSE[out]
        assert cmp["atol"] <= atol and cmp["rtol"] <= rtol, desc["name"]


@pytest.mark.parametrize("opt,coverage,expected", [
    ("-Ofast", False, "shipped"),
    ("-Ofast", True, "bounded"),
    ("-O2", False, "bounded"),
    ("-O0", False, "bounded"),
    ("Ofast", False, "bounded"),
])
def test_only_the_shipped_build_gets_kernel_faithful_goldens(opt, coverage, expected):
    assert kernel_numerics(SimpleNamespace(optimization=opt, coverage=coverage)) == expected


def test_kernel_numerics_is_part_of_the_reuse_stamp():
    args = dict(case_name="c", cpu="cortex-m55", suite="float", seed=1, version_hash="v")
    desc = {"name": "c", "operator": "Add"}
    shipped = case_stamp(desc, **args)
    assert shipped == case_stamp(desc, **args, kernel_numerics="shipped")
    assert shipped != case_stamp(desc, **args, kernel_numerics="bounded")


def _op(numerics="shipped", cpu="cortex-m55", desc=None):
    return SimpleNamespace(kernel_numerics=numerics, target_cpu=cpu, desc=desc or {})


@pytest.mark.parametrize("cpu,f16,f32", [
    ("cortex-m55", True, 0), ("cortex-m4", False, 1), ("cortex-m0", False, 2),
])
def test_routes_follow_the_build(cpu, f16, f32):
    assert kf.mve_float16_route(_op(cpu=cpu)) is f16
    assert kf.float_route(_op(cpu=cpu)) == f32
    assert kf.mve_float16_route(_op("bounded", cpu)) is False
    assert kf.float_route(_op("bounded", cpu)) is None


@pytest.mark.parametrize("value", ["", "fast", None, "SHIPPED"])
def test_unknown_kernel_numerics_is_rejected(value):
    with pytest.raises(ValueError, match="kernel_numerics"):
        kf.shipped(_op(value))


def test_ops_default_to_the_shipped_build():
    assert kf.shipped(SimpleNamespace(target_cpu="cortex-m55"))


def test_bounded_context_widens_the_defaults():
    ctx = kf.bounded_context(np.float16, 3e-3, 1e-3)
    assert ctx == {"validation_atol": 1e-5 + 3e-3, "validation_rtol": 2e-3, "kernel_faithful": False}
    assert kf.bounded_context(np.float32, 0.0) == {"validation_atol": 1e-5, "validation_rtol": 1.3e-6,
                                                    "kernel_faithful": False}


@pytest.mark.parametrize("atol,rtol", [(-1e-6, 0.0), (0.0, -1e-9), (math.nan, 0.0), (0.0, math.inf)])
def test_bounded_context_rejects_bad_bounds(atol, rtol):
    with pytest.raises(ValueError, match="bound"):
        kf.bounded_context(np.float32, atol, rtol)


def test_linear_bound_covers_every_summation_order():
    rng = np.random.default_rng(3)
    a = rng.uniform(-1, 1, (50, 37)).astype(np.float16)
    b = rng.uniform(-1, 1, 37).astype(np.float16)
    exact = a.astype(np.float64) @ b.astype(np.float64)
    golden = exact.astype(np.float16).astype(np.float64)
    atol, rtol = kf.linear_bound(np.abs(a.astype(np.float64)) @ np.abs(b.astype(np.float64)), 37, np.float16)
    for order in (np.arange(37), np.arange(37)[::-1], rng.permutation(37)):
        acc = np.zeros(50, np.float16)
        for k in order:
            acc = (acc.astype(np.float32) + a[:, k].astype(np.float32) * b[k].astype(np.float32)).astype(np.float16)
        assert np.all(np.abs(acc - golden) <= atol + rtol * np.abs(golden))
    assert kf.linear_bound(np.zeros(4), 0, np.float32) == (0.0, 2 * 2.0 ** -24)
    assert kf.linear_bound(np.array([np.inf, 2.0]), 3, np.float32)[0] == 5 * 2.0 ** -24 * 2.0


def test_linear_bound_rejects_a_negative_term_count():
    with pytest.raises(ValueError, match="term count"):
        kf.linear_bound(np.ones(2), -1, np.float32)


def test_a_descriptor_tightening_wins_on_every_build():
    desc = {"comparison": {"atol": 2e-6, "rtol": 0.0}}
    for numerics, extra in (("shipped", 0.0), ("bounded", 3e-5)):
        ctx = kf.comparison_context(_op(numerics, desc=desc), np.float32, extra)
        assert (ctx["validation_atol"], ctx["validation_rtol"]) == (2e-6, 0.0)
    assert kf.comparison_context(_op(desc=desc), np.float32)["kernel_faithful"] is True
    assert kf.comparison_context(_op("bounded", desc=desc), np.float32)["kernel_faithful"] is False


def test_comparison_context_without_a_tightening():
    assert kf.comparison_context(_op(), np.float16) == kf.tolerance_context(np.float16)
    assert kf.comparison_context(_op("bounded"), np.float16, 1e-3) == kf.bounded_context(np.float16, 1e-3)


@pytest.mark.parametrize("value,q16", [(0.0, 0), (1.3e-6, 1), (1e-5, 1), (1.0, 65536), (2.0 ** -16, 1),
                                       (2.0 ** -16 + 1e-12, 2), (1e-3, 66)])
def test_q16_tolerance_never_encodes_tighter_than_asked(value, q16):
    assert tolerance_q16(value) == q16
    assert q16 / 65536 >= value


@pytest.mark.parametrize("value", [-1e-9, math.nan, math.inf, 70000.0])
def test_q16_tolerance_rejects_unencodable_values(value):
    with pytest.raises(ValueError, match="tolerance"):
        tolerance_q16(value)


def _write_case(tmp_path, call):
    (tmp_path / "case.c").write_text(f"void run(void) {{\n    {call};\n}}\n")
    return tmp_path


def test_bridge_uses_the_emitted_float_comparison(tmp_path):
    case = _write_case(tmp_path, "HELIA_VALIDATE_OUTPUTS(FLOAT, out, exp, n, 0, 2.5e-05f, 1.3e-06f, 8, \"case\")")
    resolved = {"mode": "float", "atol": 1e-5, "rtol": 1.3e-6}
    assert bridge._emitted_float_comparison(case, resolved) == {"mode": "float", "atol": 2.5e-5, "rtol": 1.3e-6}


def test_bridge_keeps_the_resolved_comparison_without_a_float_validation(tmp_path):
    resolved = {"mode": "float", "atol": 1e-5, "rtol": 1.3e-6}
    assert bridge._emitted_float_comparison(tmp_path, resolved) == resolved
    case = _write_case(tmp_path, "HELIA_VALIDATE_OUTPUTS(NONE, out, exp, n, 0, 0.0f, 0.0f, 8, \"case\")")
    assert bridge._emitted_float_comparison(case, resolved) == resolved
    assert bridge._emitted_float_comparison(case, {"mode": "exact_int"}) == {"mode": "exact_int"}


def test_bridge_reads_a_masked_float_validation(tmp_path):
    case = _write_case(tmp_path, "HELIA_VALIDATE_FLOATS_MASKED(out, exp, mask, n, 3.1e-03f, 2e-03f, 20, failures)")
    resolved = {"mode": "float", "atol": 1e-5, "rtol": 1e-3, "nonfinite_mask": [0, 1]}
    assert bridge._emitted_float_comparison(case, resolved) == {**resolved, "atol": 3.1e-3, "rtol": 2e-3}


def test_bridge_keeps_the_resolved_comparison_for_an_unparsable_masked_call(tmp_path):
    case = _write_case(tmp_path, "HELIA_VALIDATE_FLOATS_MASKED(out, exp, mask, n, ATOL, 2e-03f, 20, failures)")
    resolved = {"mode": "float", "atol": 1e-5, "rtol": 1e-3}
    assert bridge._emitted_float_comparison(case, resolved) == resolved
