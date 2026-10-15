"""Check the float16 GELU contract intervals and the validator that asserts them."""

import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml

from helia_core_tester.generation.ops.ActivationFunctions.gelu import OpGelu, gelu_contract_reference, gelu_exact_reference
from helia_core_tester.generation.ops._shared.contract_interval import _key, contract_interval

ROOT = Path(__file__).resolve().parents[2]
CASES = list(yaml.safe_load_all((ROOT / "assets/descriptors/ActivationFunctions/gelu_float16.yaml").read_text()))
RTOL, ATOL = 2.0**-10, 2.0**-24
F16_FINITE = np.arange(0, 1 << 16, dtype=np.uint32).astype(np.uint16)
F16_FINITE = F16_FINITE[(F16_FINITE & 0x7C00) != 0x7C00]


def _outside(out: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    k = _key(out.view(np.uint16), 0x8000)
    return (k < _key(lo, 0x8000)) | (k > _key(hi, 0x8000))


def test_every_case_asserts_the_fp16_contract():
    # ns-cmsis-nn #743, FP16: |out - ref| <= 2^-10 * |ref| + 2^-24.
    assert all(desc["contract_interval"] == {"rtol": RTOL, "atol": ATOL} for desc in CASES)


def test_sweep_cases_hold_every_finite_input_once():
    ranges = [r for desc in CASES if desc["name"].startswith("gelu_float16_sweep_") for r in desc["input_bits"]]
    swept = np.concatenate([np.arange(start, stop) for start, stop in ranges])
    assert np.array_equal(np.sort(swept), F16_FINITE.astype(np.int64))


def test_fz16_case_holds_every_subnormal_result():
    fz16 = next(desc for desc in CASES if desc.get("fpscr_fz16"))
    held = np.concatenate([np.arange(start, stop) for start, stop in fz16["input_bits"]])
    ref = gelu_exact_reference(F16_FINITE.view(np.float16))
    subnormal = F16_FINITE[np.isfinite(ref) & (ref != 0) & (np.abs(ref) < 2.0**-14)]
    assert np.array_equal(np.sort(held), np.sort(subnormal.astype(np.int64)))


def test_contract_is_decidable_on_every_input_and_separates_double_narrowing():
    x16 = F16_FINITE.view(np.float16)
    ref = gelu_contract_reference(x16)
    lo, hi = contract_interval(ref, np.float16, RTOL, ATOL)
    deep = np.abs(gelu_exact_reference(x16)) < 2.0**-1000

    erfc = np.vectorize(math.erfc, otypes=[np.float64])
    xs, c = x16.astype(np.float32), np.float32(float.fromhex("-0x1.6a09e6p-1"))
    erfc32 = erfc((xs * c).astype(np.float64)).astype(np.float32)
    with np.errstate(invalid="ignore", over="ignore"):
        single = (np.float32(0.5) * xs * erfc32).astype(np.float16)
        double = (np.float32(0.5) * xs * erfc32.astype(np.float16).astype(np.float32)).astype(np.float16)
    # A float32 model of the kernel (correctly rounded erfcf) meets the contract everywhere.
    assert not _outside(single, lo, hi).any()
    assert int(_outside(double, lo, hi).sum()) == 72
    # An input zero allows only its own zero; a deeply underflowed reference allows either zero.
    assert lo[x16 == 0].tolist() == hi[x16 == 0].tolist() == [0x0000, 0x8000]
    assert deep.any() and set(zip(lo[deep & (x16 != 0)].tolist(), hi[deep & (x16 != 0)].tolist())) == {(0x8001, 0x0000)}


def test_nonfinite_references_pin_their_class():
    lo, hi = contract_interval(np.array([np.nan, np.inf, -np.inf]), np.float16, RTOL, ATOL)
    assert lo.tolist() == hi.tolist() == [0x7E00, 0x7C00, 0xFC00]


@pytest.mark.parametrize("desc", CASES, ids=lambda desc: desc["name"])
def test_generated_case_calls_the_kernel(tmp_path, desc):
    op = OpGelu(desc, seed=500, target_cpu="cortex-m55")
    op.generate_c_files(tmp_path)
    source = (tmp_path / f"{desc['name']}_gelu.c").read_text()
    assert "arm_nn_gelu_f16(" in source
    assert "helia_test_float_interval(" in source
    assert ("vmsr fpscr" in source) == bool(desc.get("fpscr_fz16"))


def test_interval_validator_detects_planted_faults(tmp_path):
    cc = shutil.which("gcc-12") or shutil.which("gcc")
    if cc is None:
        pytest.skip("host GCC required for generated validator probe")
    # +0, -0, 1.0, -4.0, 2^-15 (subnormal result away from zero), NaN, +Inf, -Inf
    inputs = [0x0000, 0x8000, 0x3C00, 0xC400, 0x0200, 0x7E00, 0x7C00, 0xFC00]
    desc = {
        "name": "probe",
        "operator": "Gelu",
        "suite": "float",
        "tensor_dtypes": {"input": "FP16", "output": "FP16"},
        "input_shape": [1, 1, 1, len(inputs)],
        "input_bits": [[b, b + 1] for b in inputs],
        "contract_interval": {"rtol": RTOL, "atol": ATOL},
        "fpscr_fz16": True,
    }
    OpGelu(desc, seed=500, target_cpu="cortex-m55").generate_c_files(tmp_path)
    lo, hi = contract_interval(gelu_contract_reference(np.array(inputs, np.uint16).view(np.float16)), np.float16, RTOL, ATOL)
    lo, hi = lo.tolist(), hi.tolist()
    (tmp_path / "arm_nnfunctions.h").write_text(
        "#pragma once\n#include <stdint.h>\ntypedef _Float16 float16_t;\n"
        "typedef int arm_cmsis_nn_status;\n#define ARM_CMSIS_NN_SUCCESS 0\n"
        "int arm_nn_gelu_f16(const float16_t *, float16_t *, int32_t);\n"
    )
    runtime = tmp_path / "runtime.c"
    runtime.write_text(
        "#define helia_test_finish helia_test_finish_on_target\n"
        f'#include "{ROOT / "src" / "test_runtime" / "helia_test_runtime.c"}"\n'
        "#undef helia_test_finish\n#include <stdlib.h>\n"
        "void helia_test_finish(int32_t n) { exit(n != 0); }\n"
    )

    def replace(values, index, bits):
        return [*values[:index], bits, *values[index + 1:]]

    mutations = {
        "lowest": (lo, False),
        "highest": (hi, False),
        "nan-payload": (replace(lo, 5, 0xFE01), False),
        "flushed-subnormal": (replace(lo, 4, 0x0000), False),
        "below-lowest": (replace(lo, 2, lo[2] - 1), True),
        "above-highest": (replace(hi, 3, hi[3] - 1), True),
        "zero-sign": (replace(lo, 1, 0x0000), True),
        "flush-wrong-sign": (replace(lo, 4, 0x8000), True),
        "nan-for-inf": (replace(lo, 6, 0x7E00), True),
        "neg-inf-not-nan": (replace(lo, 7, 0xFC00), True),
        "inf-for-finite": (replace(lo, 2, 0x7C00), True),
    }
    for label, (values, should_fail) in mutations.items():
        stub = tmp_path / "kernel.c"
        stub.write_text(
            '#include <string.h>\n#include "arm_nnfunctions.h"\n'
            "int arm_nn_gelu_f16(const float16_t *in, float16_t *out, int32_t n) {\n"
            f'const uint16_t bits[] = {{{", ".join(hex(v) for v in values)}}};\n'
            "(void)in; (void)n; memcpy(out, bits, sizeof(bits)); return 0; }\n"
        )
        binary = tmp_path / label
        subprocess.run(
            [cc, "-std=c11", "-O3", "-ffast-math", "-I", str(tmp_path), "-I", str(tmp_path / "includes"),
             "-I", str(ROOT / "src"), str(tmp_path / "probe_gelu.c"), str(runtime), str(stub), "-o", str(binary)],
            check=True,
            capture_output=True,
        )
        result = subprocess.run([str(binary)], capture_output=True)
        assert (result.returncode != 0) == should_fail, (label, result.stdout)


@pytest.mark.parametrize(
    "change",
    [
        {"input_bits": [[0x7C00, 0x10001]]},
        {"input_bits": [[0x10, 0x08]]},
        {"input_bits": []},
        {"input_mode": "nonfinite_sweep", "nonfinite_policy": "strict"},
        {"input_min": -1.0},
        {"tensor_dtypes": {"input": "FP32", "output": "FP32"}, "input_bits": [[0, 8]]},
    ],
    ids=["past-dtype", "reversed", "empty", "with-input-mode", "with-input-min", "fz16-on-fp32"],
)
def test_malformed_sweep_descriptors_are_rejected(tmp_path, change):
    fz16 = next(desc for desc in CASES if desc.get("fpscr_fz16"))
    desc = {**fz16, "name": "malformed", "input_shape": [1, 1, 1, 8], "input_bits": [[0, 8]], **change}
    with pytest.raises(ValueError, match="input_bits (needs|replaces)|fpscr_fz16 applies"):
        OpGelu(desc, seed=500, target_cpu="cortex-m55").generate_c_files(tmp_path)


def test_interval_case_sidecar_records_the_contract(tmp_path):
    import json

    desc = next(desc for desc in CASES if desc["name"] == "gelu_float16_tail_f16")
    OpGelu(desc, seed=500, target_cpu="cortex-m55").generate_c_files(tmp_path)
    sidecar = json.loads((tmp_path / "gelu_float16_tail_f16_gelu.sidecar.json").read_text())
    assert sidecar["comparison"] == {"mode": "interval", "rtol": RTOL, "atol": ATOL}
    assert (sidecar["scalars"]["validation_rtol"], sidecar["scalars"]["validation_atol"]) == (RTOL, ATOL)
