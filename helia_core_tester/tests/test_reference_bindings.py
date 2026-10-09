"""Bindings and ReferenceCall: strict argument checking, status codes, provenance records."""

from __future__ import annotations

import ctypes
import json
import math

import numpy as np
import pytest

from helia_core_tester.generation.reference import abi
from helia_core_tester.generation.reference.bindings import ReferenceKernelError, get_bindings, output_shape_for
from helia_core_tester.generation.reference.call import ReferenceCall

FLOAT_PARAMS = {"activation_min": -math.inf, "activation_max": math.inf}


@pytest.fixture(scope="module")
def lib():
    return get_bindings()


def _code(fn) -> str:
    with pytest.raises(ReferenceKernelError) as info:
        fn()
    return info.value.status


def _f32(*shape):
    return np.zeros(shape, np.float32)


def test_library_reports_the_spec_abi(lib) -> None:
    assert lib.raw("add_f32") is not None
    assert int(lib._lib.hct_ref_abi_version()) == abi.load_spec().abi_version


def test_unknown_entries_and_tensor_names_are_rejected(lib) -> None:
    with pytest.raises(KeyError, match="unknown kernel"):
        lib.run("nope_s8", {}, {}, {})
    with pytest.raises(KeyError, match="unknown prepare"):
        lib.prepare("nope", {})
    with pytest.raises(KeyError, match="inputs"):
        lib.run("add_f32", FLOAT_PARAMS, {"input1": _f32(2)}, {"output": (2,)})
    with pytest.raises(KeyError, match="outputs"):
        lib.run("add_f32", FLOAT_PARAMS, {"input1": _f32(2), "input2": _f32(2)}, {"out": (2,)})


def test_params_must_name_every_field_exactly(lib) -> None:
    with pytest.raises(KeyError, match="missing fields"):
        lib.run("add_f32", {"activation_min": 0.0}, {"input1": _f32(2), "input2": _f32(2)}, {"output": (2,)})
    with pytest.raises(KeyError, match="unknown fields"):
        lib.run("add_f32", {**FLOAT_PARAMS, "extra": 1}, {"input1": _f32(2), "input2": _f32(2)}, {"output": (2,)})
    with pytest.raises(TypeError, match="integer"):
        lib.prepare("activation_range_quantized", {"activation": 0.5, "dtype": 1, "scale": 0.1, "zero_point": 0})
    with pytest.raises(TypeError, match="integer"):
        lib.prepare("activation_range_quantized", {"activation": True, "dtype": 1, "scale": 0.1, "zero_point": 0})


def test_tensors_are_never_cast_silently(lib) -> None:
    with pytest.raises(TypeError, match="float32"):
        lib.run("add_f32", FLOAT_PARAMS, {"input1": np.zeros(2, np.float64), "input2": _f32(2)}, {"output": (2,)})
    with pytest.raises(TypeError, match="C-contiguous"):
        lib.run("add_f32", FLOAT_PARAMS, {"input1": _f32(4, 4)[:, ::2], "input2": _f32(4, 2)}, {"output": (4, 2)})
    with pytest.raises(TypeError, match="numpy array"):
        lib.run("add_f32", FLOAT_PARAMS, {"input1": [1.0, 2.0], "input2": _f32(2)}, {"output": (2,)})
    with pytest.raises(ValueError, match="rank"):
        lib.run("add_f32", FLOAT_PARAMS, {"input1": _f32(*([1] * 9)), "input2": _f32(1)}, {"output": (1,) * 9})


def test_c_side_status_codes_surface(lib) -> None:
    ins = {"input1": _f32(2, 3), "input2": _f32(3)}
    assert _code(lambda: lib.run("add_f32", FLOAT_PARAMS, ins, {"output": (2, 4)})) == "E_SHAPE"
    assert _code(lambda: lib.run("add_f32", FLOAT_PARAMS, {"input1": _f32(2, 3), "input2": _f32(4)},
                                 {"output": (2, 3)})) == "E_SHAPE"
    bad = {"activation_min": 1.0, "activation_max": -1.0}
    assert _code(lambda: lib.run("add_f32", bad, ins, {"output": (2, 3)})) == "E_PARAM"
    nan = {"activation_min": math.nan, "activation_max": 1.0}
    assert _code(lambda: lib.run("add_f32", nan, ins, {"output": (2, 3)})) == "E_PARAM"


def test_raw_abi_rejects_null_and_count_errors(lib) -> None:
    spec = abi.load_spec()
    fn = lib.raw("add_f32")
    params = abi.ctypes_struct(spec, "HctFloatActivationParams")(-math.inf, math.inf)
    a, b, out = _f32(2), _f32(2), _f32(2)
    tensors = (abi.HctTensor * 2)(lib.tensor(a, "float32"), lib.tensor(b, "float32"))
    outs = (abi.HctTensor * 1)(lib.tensor(out, "float32"))
    assert fn(None, tensors, 2, outs, 1) == spec.status["E_NULL"]
    assert fn(ctypes.byref(params), None, 2, outs, 1) == spec.status["E_NULL"]
    assert fn(ctypes.byref(params), tensors, 1, outs, 1) == spec.status["E_COUNT"]
    assert fn(ctypes.byref(params), tensors, 2, outs, 2) == spec.status["E_COUNT"]
    tensors[1].data = None
    assert fn(ctypes.byref(params), tensors, 2, outs, 1) == spec.status["E_NULL"]
    tensors[1].data = b.ctypes.data
    tensors[1].dtype = spec.dtypes["int8"]
    assert fn(ctypes.byref(params), tensors, 2, outs, 1) == spec.status["E_DTYPE"]
    assert fn(ctypes.byref(params), tensors, 2, outs, 1) != 0 and not np.any(out)


def test_prepare_errors_surface(lib) -> None:
    assert _code(lambda: lib.prepare("quantize_multiplier", {"real_multiplier": -1.0})) == "E_PARAM"
    assert _code(lambda: lib.prepare("quantize_multiplier", {"real_multiplier": math.inf})) == "E_PARAM"


def test_output_shape_for_broadcasts_or_raises() -> None:
    assert output_shape_for("add", (2, 1, 4), (3, 1)) == (2, 3, 4)
    assert output_shape_for("add", (), (5,)) == (5,)
    with pytest.raises(ValueError, match="do not broadcast"):
        output_shape_for("add", (2, 3), (4,))


def _call(**overrides):
    args = dict(entry="add_f32", params=FLOAT_PARAMS,
                inputs={"input1": np.arange(6, dtype=np.float32).reshape(2, 3), "input2": np.ones(3, np.float32)},
                output_shapes={"output": (2, 3)})
    args.update(overrides)
    return ReferenceCall(**args)


def test_reference_call_validates_itself_on_construction() -> None:
    with pytest.raises(KeyError, match="unknown reference entry"):
        _call(entry="nope_f32")
    with pytest.raises(KeyError, match="inputs"):
        _call(inputs={"input1": _f32(3)})
    with pytest.raises(TypeError, match="float32"):
        _call(inputs={"input1": np.zeros(3, np.float64), "input2": _f32(3)})
    with pytest.raises(KeyError, match="missing output"):
        _call(output_shapes={})
    with pytest.raises(KeyError, match="unknown outputs"):
        _call(output_shapes={"output": (2, 3), "extra": (1,)})
    with pytest.raises(ValueError, match="invalid output shape"):
        _call(output_shapes={"output": (2, -3)})


def test_reference_call_runs_probes_and_records() -> None:
    call = _call()
    np.testing.assert_array_equal(call.output(), np.arange(6, dtype=np.float32).reshape(2, 3) + 1)
    probe = call.with_inputs(input2=np.full(3, np.inf, np.float32))
    assert np.isposinf(probe.output()).all()
    np.testing.assert_array_equal(call.output(), np.arange(6, dtype=np.float32).reshape(2, 3) + 1)
    with pytest.raises(KeyError, match="unknown inputs"):
        call.with_inputs(other=_f32(3))
    record = call.provenance(seeds={"run_seed": 7}, library_key="abc")
    assert record["entry"] == "add_f32" and record["library_key"] == "abc" and record["seeds"] == {"run_seed": 7}
    assert record["params"] == {"activation_min": "-inf", "activation_max": "inf"}
    assert record["inputs"]["input1"]["shape"] == [2, 3] and len(record["inputs"]["input1"]["sha256"]) == 64
    assert record["outputs"] == {"output": [2, 3]}


def test_reference_record_is_strict_json(tmp_path) -> None:
    path = _call().to_json(tmp_path / "r.json", seeds={"case_seed": 1}, library_key="k")
    record = json.loads(path.read_text(), parse_constant=lambda c: pytest.fail(f"non-strict JSON constant {c}"))
    assert record["abi_version"] == abi.load_spec().abi_version
