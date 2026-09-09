"""Float sqrt/rsqrt contract references, generation, and independent oracle checks."""

from fractions import Fraction
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from helia_core_tester.generation.io.descriptors import load_descriptor
from helia_core_tester.generation.ops.BasicMathFunctions.sqrt import OpSqrt
from helia_core_tester.generation.ops.BasicMathFunctions.rsqrt import OpRsqrt
from helia_core_tester.generation.ops._shared.sqrt_float import sqrt_float_reference

ROOT = Path(__file__).resolve().parents[2]


def _half_value(bits):
    exponent, mantissa = (bits >> 10) & 31, bits & 1023
    return (
        Fraction(mantissa, 1 << 24)
        if exponent == 0
        else Fraction((1024 + mantissa) * (1 << exponent), 1 << 25)
    )


@pytest.mark.parametrize("reciprocal", [False, True])
def test_all_half_patterns_against_exact_midpoints(reciprocal):
    bits = np.arange(65536, dtype=np.uint16)
    with np.errstate(invalid="ignore"):
        outputs = sqrt_float_reference(bits, reciprocal)
    for b, result in enumerate(outputs):
        result = int(result)
        magnitude = b & 0x7FFF
        if magnitude == 0:
            assert result == b | (0x7C00 if reciprocal else 0)
        elif magnitude > 0x7C00:
            assert result == b | 0x0200
        elif b == 0x7C00:
            assert result == (0 if reciprocal else 0x7C00)
        elif b & 0x8000:
            assert result == 0x7E00
        else:
            x = _half_value(b)
            lower = (_half_value(result - 1) + _half_value(result)) / 2
            upper = (_half_value(result) + _half_value(result + 1)) / 2
            if reciprocal:
                low, high, value = x * lower * lower, x * upper * upper, Fraction(1)
            else:
                low, high, value = lower * lower, upper * upper, x
            assert low <= value <= high
            if value in (low, high):
                assert result % 2 == 0


@pytest.mark.parametrize("reciprocal", [False, True])
def test_f32_special_bit_contract(reciprocal):
    bits = np.array(
        [0, 0x80000000, 0x7F800000, 0xFF800000, 0xBF800000, 0x7F800123, 0xFFC00567],
        dtype=np.uint32,
    )
    expected = [
        0x7F800000 if reciprocal else 0,
        0xFF800000 if reciprocal else 0x80000000,
        0 if reciprocal else 0x7F800000,
        0x7FC00000,
        0x7FC00000,
        0x7FC00123,
        0xFFC00567,
    ]
    with np.errstate(invalid="ignore"):
        np.testing.assert_array_equal(sqrt_float_reference(bits, reciprocal), expected)


@pytest.mark.parametrize("operator,cls", [("sqrt", OpSqrt), ("rsqrt", OpRsqrt)])
def test_generate_every_float_descriptor(tmp_path, operator, cls):
    descriptors = load_descriptor(
        str(ROOT / "assets/descriptors/BasicMathFunctions" / (operator + "_float.yaml"))
    )
    assert len(descriptors) == 26
    for desc in descriptors:
        out = tmp_path / desc["name"]
        out.mkdir()
        op = cls(desc, seed=500, target_cpu="cortex-m55")
        op.convert_to_tflite(None, str(out / (desc["name"] + ".tflite")), 500)
        op.generate_c_files(out)
        source = (out / (desc["name"] + "_" + operator + ".c")).read_text()
        assert desc["required_kernel_symbols"][0] in source
        assert "HELIA_VALIDATE_EXPECTED_STATUS" in source
        assert "HELIA_VALIDATE_RETURN_FAILURES" in source


@pytest.mark.parametrize("reciprocal,op_name", [(False, "SQRT"), (True, "RSQRT")])
def test_positive_f32_reference_against_tflite(tmp_path, reciprocal, op_name):
    from helia_core_tester.generation.utils.litert_builder import (
        build_unary_same_shape_op,
    )
    from ai_edge_litert.interpreter import Interpreter, OpResolverType

    values = np.array([0.25, 1, 2, 3, 4, 9, 17, 0.125, 65504], dtype=np.float32)
    model = build_unary_same_shape_op(
        op_name=op_name, input_shape=values.shape, dtype="float32"
    )
    interpreter = Interpreter(
        model_content=model, experimental_op_resolver_type=OpResolverType.BUILTIN_REF
    )
    interpreter.allocate_tensors()
    interpreter.set_tensor(interpreter.get_input_details()[0]["index"], values)
    interpreter.invoke()
    actual = (
        interpreter.get_tensor(interpreter.get_output_details()[0]["index"])
        .view(np.uint32)
        .astype(np.int64)
    )
    expected = sqrt_float_reference(values.view(np.uint32), reciprocal).astype(np.int64)
    assert np.max(np.abs(actual - expected)) <= int(reciprocal)


@pytest.mark.parametrize("half", [False, True])
@pytest.mark.parametrize("reciprocal", [False, True])
def test_generated_validator_detects_planted_faults(
    tmp_path, monkeypatch, half, reciprocal
):
    cc = shutil.which("gcc-12") or shutil.which("gcc")
    if cc is None:
        pytest.skip("host GCC required for generated validator probe")
    from helia_core_tester.generation.ops._shared import sqrt_float

    word = np.uint16 if half else np.uint32
    sign, inf, one, four, quiet = (
        (0x8000, 0x7C00, 0x3C00, 0x4400, 0x0200)
        if half
        else (0x80000000, 0x7F800000, 0x3F800000, 0x40800000, 0x00400000)
    )
    inputs = np.array([0, sign, inf, inf | sign, one | sign, inf | 7, four], dtype=word)
    expected = [
        inf if reciprocal else 0,
        inf | sign if reciprocal else sign,
        0 if reciprocal else inf,
        inf | quiet,
        inf | quiet,
        inf | quiet | 7,
        (
            (0x3800 if half else 0x3F000000)
            if reciprocal
            else (0x4000 if half else 0x40000000)
        ),
    ]
    monkeypatch.setattr(sqrt_float, "sqrt_float_inputs", lambda *args: inputs)
    op_name = "Rsqrt" if reciprocal else "Sqrt"
    dtype = "FP16" if half else "FP32"
    suffix = op_name.lower()
    kernel = ("arm_rsqrt_" if reciprocal else "arm_nn_sqrt_") + (
        "f16" if half else "f32"
    )
    ctype, utype = ("float16_t", "uint16_t") if half else ("float", "uint32_t")
    desc = {
        "name": "probe",
        "operator": op_name,
        "suite": "float",
        "tensor_dtypes": {"input": dtype, "output": dtype},
        "input_shape": [7],
    }
    op = (OpRsqrt if reciprocal else OpSqrt)(desc, seed=500, target_cpu="cortex-m55")
    op.generate_c_files(tmp_path)
    (tmp_path / "arm_nnfunctions.h").write_text(
        "#pragma once\n#include <stdint.h>\ntypedef _Float16 float16_t;\n"
        "typedef int arm_cmsis_nn_status;\n#define ARM_CMSIS_NN_SUCCESS 0\n"
        "#define ARM_CMSIS_NN_ARG_ERROR -1\n"
        f"int {kernel}(const {ctype} *, {ctype} *, int32_t);\n"
    )
    (tmp_path / "arm_nnfunctions_flt.h").write_text(
        f'#include "arm_nnfunctions.h"\nint {kernel}(const {ctype} *, {ctype} *, int32_t);\n'
    )
    runtime = tmp_path / "runtime.c"
    runtime.write_text(
        '#include <stdlib.h>\n#include "test_runtime/helia_test_runtime.h"\n'
        "void helia_test_platform_init(void) {}\n"
        "int helia_test_status_failure(const char *s, int a) { return 1; }\n"
        "void helia_test_finish(int32_t n) { exit(n != 0); }\n"
        "int helia_test_expected_status_failure(const char *s, int a, int b) { return 1; }\n"
        "int helia_test_finish_validation(int n) { return n; }\n"
    )
    mutations = {
        "baseline": (expected, "", False),
        "zero-sign": ([expected[0], expected[1] ^ sign, *expected[2:]], "", True),
        "negative-nan-sign": (
            [*expected[:4], expected[4] ^ sign, *expected[5:]],
            "",
            True,
        ),
        "nan-payload": ([*expected[:5], expected[5] ^ 1, expected[6]], "", True),
        "nan-not-quiet": ([*expected[:5], expected[5] ^ quiet, expected[6]], "", True),
        "finite-two-ulp": ([*expected[:6], expected[6] + 2], "", True),
        "finite-one-ulp": (
            [*expected[:6], expected[6] + 1],
            "",
            half or not reciprocal,
        ),
        "overrun": (expected, "memset(out + n, 0, sizeof(*out));", True),
        "input-write": (expected, "memset((void *)in, 0, n * sizeof(*in));", True),
    }
    for label, (values, fault, should_fail) in mutations.items():
        stub = tmp_path / "kernel.c"
        stub.write_text(
            '#include <string.h>\n#include "arm_nnfunctions_flt.h"\n'
            + f"int {kernel}(const {ctype} *in, {ctype} *out, int32_t n) {{\n"
            + f'const {utype} bits[] = {{{", ".join(hex(x) for x in values)}}};\n'
            + f"memcpy(out, bits, sizeof(bits)); {fault} return 0; }}\n"
        )
        binary = tmp_path / label
        subprocess.run(
            [
                cc,
                "-std=c11",
                "-O3",
                "-ffast-math",
                "-I",
                str(tmp_path),
                "-I",
                str(tmp_path / "includes"),
                "-I",
                str(ROOT / "src"),
                str(tmp_path / f"probe_{suffix}.c"),
                str(runtime),
                str(stub),
                "-o",
                str(binary),
            ],
            check=True,
            capture_output=True,
        )
        result = subprocess.run([str(binary)], capture_output=True)
        assert (result.returncode != 0) == should_fail, (label, result.stdout)
