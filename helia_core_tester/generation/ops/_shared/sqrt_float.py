"""Float sqrt/rsqrt cases: goldens from the C reference, bit-level contract checks (ns-cmsis-nn#295)."""

from pathlib import Path
from typing import Any, Dict

import numpy as np

from helia_core_tester.generation.harness import ArgumentPool, ArrayLiteral, Declaration, GuardedBuffer, HarnessInput


def sqrt_float_inputs(dtype: str, count: int, pattern: str, seed: int) -> np.ndarray:
    half = dtype == "FP16"
    word = np.uint16 if half else np.uint32
    inf, one, fraction, sign = (
        (0x7C00, 0x3C00, 10, 0x8000)
        if half
        else (0x7F800000, 0x3F800000, 23, 0x80000000)
    )
    if pattern == "special":
        base = [
            0,
            sign,
            inf,
            sign | inf,
            one | sign,
            one,
            1,
            (1 << fraction) - 1,
            1 << fraction,
            inf - 1,
            inf | 1,
            inf | 3 | sign,
            inf | (1 << (fraction - 1)) | 7,
            inf | (1 << (fraction - 1)) | 11 | sign,
            sign | 1,
            one - 1,
            one + 1,
        ]
        return np.resize(np.array(base, dtype=word), count)
    if pattern == "powers_of_four":
        bias = 15 if half else 127
        exponents = range(-14 if half else -126, 16 if half else 128, 2)
        return np.resize(
            np.array([(e + bias) << fraction for e in exponents], dtype=word), count
        )
    if pattern != "positive":
        raise ValueError(f"Unknown float sqrt input pattern: {pattern}")
    return np.random.default_rng(seed).integers(
        1 << fraction, inf, size=count, dtype=word
    )


_FLUSH_PROBE = """\
#if defined(__SSE__)
#include <xmmintrin.h>
#endif

static int __NAME___flushes_inputs(void)
{
#if defined(__arm__) && defined(__ARM_FP) && (__ARM_FP != 0)
    uint32_t fpscr;
    __asm__ volatile("vmrs %0, fpscr" : "=r"(fpscr));
    return (fpscr & (1u << 24)) != 0;
#elif defined(__SSE__)
    return (_mm_getcsr() & (1u << 6)) != 0;
#else
    return 0;
#endif
}"""


def sqrt_float_argument_pool(context: Dict[str, Any]) -> ArgumentPool:
    """A bit-exact property case: the kernel runs over a fenced copy of the input bits (in place
    when asked), every byte outside the block must survive, and each result word is compared with
    the reference within its ULP allowance, or left untouched by a rejected call."""
    n, ctype, word = context["name"], context["output_dtype"], context["word_type"]
    block, half, error = int(context["block_size"]), bool(context["half"]), context["api_error"]
    if block < 1:
        raise ValueError(f"{n}: a float sqrt case needs at least one element")
    limit = "0x7c00u" if half else "0x7f800000u"
    result = f"{n}_{'input' if context['in_place'] else 'output'} + 1"
    lines = [
        f"    HELIA_GUARD_ARM({n}_input, false /* edge elements are stamped below */);",
        f"    HELIA_GUARD_ARM({n}_output, false /* edge elements are stamped below */);",
        f"    memset({n}_input, 0xa5, sizeof({n}_input));",
        f"    memset({n}_output, 0xa5, sizeof({n}_output));",
        f"    memcpy({n}_input + 1, {n}_input_bits, sizeof({n}_input_bits));",
        f"    {ctype} *result = {result};",
        f"    int32_t status = {n}_run(",
        f"        {'NULL' if error == 'null_input' else n + '_input + 1'},",
        f"        {'NULL' if error == 'null_output' else 'result'});",
        "    int failures = 0;",
        f'    HELIA_GUARD_CHECK({n}_input, "{n} input", failures);',
        f'    HELIA_GUARD_CHECK({n}_output, "{n} output", failures);',
        f"    for (unsigned int i = 0; i < sizeof({n}_input); ++i) {{",
        f"        const unsigned int first = sizeof({ctype});",
        f"        const unsigned int last = ({block} + 1) * sizeof({ctype});",
        "        if (i < first || i >= last) {",
        f"            if (((unsigned char *){n}_input)[i] != 0xa5 || ((unsigned char *){n}_output)[i] != 0xa5) ++failures;",
        "        }",
        "    }",
        f'    HELIA_VALIDATE_EXPECTED_STATUS("{n}", status, {context["expected_status"]});',
    ]
    probe = not half and not error
    if probe:
        lines.append(f"    const int flush_inputs = {n}_flushes_inputs();")
    lines += [f"    for (int i = 0; i < {block}; ++i) {{", f"        {word} actual;",
              "        memcpy(&actual, result + i, sizeof(actual));"]
    if error:
        untouched = f"{n}_input_bits[i]" if context["in_place"] else ("0xa5a5" if half else "0xa5a5a5a5")
        lines += [f"        const {word} expected = {untouched};", "        const uint32_t allowed = 0;"]
    else:
        lines.append(f"        {word} expected = {n}_expected_bits[i];")
        if not half:
            lines += [f"        if (flush_inputs && {n}_input_bits[i] > 0 && {n}_input_bits[i] < 0x00800000u) {{",
                      f"            expected = {context['flushed_bits']};", "        }"]
        lines.append(f"        const uint32_t allowed = (expected > 0 && expected < {limit}) ? {context['max_ulp']} : 0;")
    lines += [f"        HELIA_VALIDATE_FLOAT_BITS(actual, expected, {limit}, allowed, i, 8, failures);", "    }"]
    if not context["in_place"]:
        lines.append(f"    if (memcmp({n}_input + 1, {n}_input_bits, sizeof({n}_input_bits)) != 0) ++failures;")
    bits = lambda key: ArrayLiteral("    " + ", ".join(context[key]))  # noqa: E731
    return ArgumentPool(
        name=n, values={"block_size": str(context["call_size"])},
        header=(Declaration(f"{n}_input_bits", word, bits("input_bits"), array=True),
                Declaration(f"{n}_expected_bits", word, bits("expected_bits"), array=True)),
        guarded=(GuardedBuffer(f"{n}_input", ctype, f"{block} + 2"), GuardedBuffer(f"{n}_output", ctype, f"{block} + 2")),
        inputs=(HarnessInput("input", "input", f"{n}_input", ctype),), output_param="output", output_ctype=ctype,
        includes=("<string.h>",), file_scope=_FLUSH_PROBE.replace("__NAME__", n) if probe else "",
        test_body="\n".join(lines), benchmark=False, scratch_buffer=False,
    )


def generate_sqrt_float(op, output_dir: Path, reciprocal: bool) -> None:
    dtype = op.tensor_dtype("input")
    if dtype not in ("FP16", "FP32") or op.tensor_dtype("output") != dtype:
        raise ValueError("Float sqrt/rsqrt requires matching FP16 or FP32 tensors")
    hint = op.desc.get("hint", {})
    count = int(np.prod(op.desc["input_shape"]))
    if count < 1:
        raise ValueError("Float sqrt descriptors need positive storage dimensions")
    pattern = hint.get("float_pattern", "positive")
    bits = sqrt_float_inputs(dtype, count, pattern, op.seed)
    from helia_core_tester.generation.reference.call import ReferenceCall

    float_dtype = np.float16 if dtype == "FP16" else np.float32
    entry = ("rsqrt_" if reciprocal else "sqrt_") + ("f16" if dtype == "FP16" else "f32")
    golden = op.reference_golden(ReferenceCall(
        entry, {"unused": 0}, {"input": np.ascontiguousarray(bits.view(float_dtype))}, {"output": bits.shape},
    ))
    expected = np.ascontiguousarray(golden).view(bits.dtype)
    error = hint.get("api_error", "")
    if error not in ("", "null_input", "null_output", "zero_block", "negative_block"):
        raise ValueError(f"Unknown float sqrt api_error: {error}")
    half = dtype == "FP16"
    suffix = "rsqrt" if reciprocal else "sqrt"
    kernel = ("arm_rsqrt_" if reciprocal else "arm_nn_sqrt_") + (
        "f16" if half else "f32"
    )
    context = {
        "name": op.desc["name"],
        "kernel_fn": kernel,
        "op_suffix": suffix,
        "input_dtype": "float16_t" if half else "float",
        "output_dtype": "float16_t" if half else "float",
        "word_type": "uint16_t" if half else "uint32_t",
        "half": half,
        "block_size": count,
        "input_bits": [hex(int(x)) for x in bits],
        "expected_bits": [hex(int(x)) for x in expected],
        "api_error": error,
        "in_place": bool(hint.get("in_place", False)),
        "call_size": (
            0 if error == "zero_block" else -1 if error == "negative_block" else count
        ),
        "expected_status": (
            "ARM_CMSIS_NN_ARG_ERROR" if error else "ARM_CMSIS_NN_SUCCESS"
        ),
        "max_ulp": int(reciprocal and not half and pattern != "powers_of_four"),
        "flushed_bits": "0x7f800000" if reciprocal else "0",
        "validation_mode": "float",
    }
    op.render_harness_case(
        output_dir, stem=suffix, context=context, pool=sqrt_float_argument_pool(context),
        validation_key="BasicMathFunctions/sqrt_float/sqrt_float.c.j2", label=op.desc["operator"],
        operator=op.desc["operator"], sidecar=True,
    )
