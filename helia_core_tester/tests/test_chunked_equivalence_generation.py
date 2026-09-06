"""
Generation-level tests for the chunked-equivalence property cases (issue #81).

These cases assert block-size invariance: one full-length elementwise call is
the bit-exact reference for a sequence of chunked calls over the same data.
The tests below pin down the properties that make the case type able to
discriminate packed-vs-tail defects like ns-cmsis-nn#343:

  - the generated C compares the chunked output buffer against the full-call
    output buffer with the EXACT_INT validator and contains no golden data;
  - the generated operands are sign-diverse post-offset inside the packed
    region (the #343 trigger was sign-dependent), and generation FAILS for
    data that is not;
  - chunk patterns are validated (>= 2 chunks, all >= 1, summing to the
    element count) so a descriptor typo cannot silently weaken the property;
  - the shipped descriptor corpus covers every sliceable int kernel family
    (add/sub/mul/squared_difference for s8 and s16, minimum/maximum for s8 and
    s16, requantize for s8) with both an all-singles pattern (pure tail vs
    packed) and a mixed "offcut" pattern (packed loops starting on odd
    boundaries), and every pattern leaves a tail and an unaligned chunk
    boundary for each vector width the kernel actually uses;
  - arm_elementwise_prelu_s8/_s16 stay out: they are a single scalar loop with
    no vectorised path, so block-size invariance holds trivially there.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from helia_core_tester.generation.io.descriptors import load_all_descriptors
from helia_core_tester.generation.ops.BasicMathFunctions.chunked_equivalence import (
    _KERNEL_TABLE,
    OpChunkedEquivalence,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _make_desc(**overrides) -> dict:
    desc = {
        "operator": "ChunkedEquivalence",
        "name": "chunked_equivalence_add_singles_s8",
        "activation_dtype": "S8",
        "weight_dtype": "S8",
        "kernel": "add",
        "element_count": 23,
        "chunk_sizes": [1] * 23,
    }
    desc.update(overrides)
    return desc


_EXPECTED_KERNELS = {
    ("add", "S8"): "arm_elementwise_add_s8",
    ("sub", "S8"): "arm_elementwise_sub_s8",
    ("mul", "S8"): "arm_elementwise_mul_s8",
    ("squared_difference", "S8"): "arm_elementwise_squared_difference_s8",
    ("minimum", "S8"): "arm_minimum_s8",
    ("maximum", "S8"): "arm_maximum_s8",
    ("requantize", "S8"): "arm_requantize_s8_s8",
    ("add", "S16"): "arm_elementwise_add_s16",
    ("sub", "S16"): "arm_elementwise_sub_s16",
    ("mul", "S16"): "arm_elementwise_mul_s16",
    ("squared_difference", "S16"): "arm_elementwise_squared_difference_s16",
    ("minimum", "S16"): "arm_minimum_s16",
    ("maximum", "S16"): "arm_maximum_s16",
}

# Kernels with exactly one operand: the header must not emit a second input
# array and the call must not pass one.
_UNARY_KERNELS = {("requantize", "S8")}


@pytest.mark.parametrize(("kernel", "dtype"), sorted(_EXPECTED_KERNELS))
def test_generates_full_vs_chunked_comparison_without_golden_data(
    kernel: str, dtype: str, tmp_path: Path
) -> None:
    suffix = dtype.lower()
    name = f"chunked_equivalence_{kernel}_offcut_{suffix}"
    desc = _make_desc(
        name=name,
        kernel=kernel,
        activation_dtype=dtype,
        element_count=29,
        chunk_sizes=[3, 8, 7, 5, 4, 2],
    )
    op = OpChunkedEquivalence(desc, seed=1, target_cpu="cortex-m4")
    assert op.needs_keras_model() is False
    assert op.allow_no_tflite() is True
    op.generate_c_files(tmp_path)

    c_text = (tmp_path / f"{name}_chunked_equivalence.c").read_text()
    h_text = (tmp_path / "includes" / f"{name}_chunked_equivalence.h").read_text()

    # The kernel under test is called for both passes with block_size varying.
    assert _EXPECTED_KERNELS[(kernel, dtype)] in c_text
    assert f"{name}_output_full" in c_text
    assert f"{name}_output_chunked" in c_text
    # Bit-exact comparison of chunked against full: the full call IS the
    # reference. No golden/expected array exists anywhere.
    assert "EXACT_INT" in c_text
    assert "expected_output" not in c_text
    assert "expected_output" not in h_text
    # Chunk sizes land in the header and sum to the element count.
    assert "3, 8, 7, 5, 4, 2" in h_text
    assert "(29)" in h_text
    # Both element types match the dtype.
    c_type = "int8_t" if dtype == "S8" else "int16_t"
    assert f"static const {c_type} {name}_input1" in h_text
    if (kernel, dtype) in _UNARY_KERNELS:
        assert f"{name}_input2" not in h_text
        assert f"{name}_input2" not in c_text
    else:
        assert f"static const {c_type} {name}_input2" in h_text

    cmake_text = (tmp_path / "CMakeLists.txt").read_text()
    assert f"{name}_chunked_equivalence.c" in cmake_text


def test_generated_operands_are_sign_diverse_in_packed_region() -> None:
    for dtype in ("S8", "S16"):
        desc = _make_desc(activation_dtype=dtype)
        op = OpChunkedEquivalence(desc, seed=1)
        input_1, input_2 = op._generate_operands(23, dtype)
        for data in (input_1, input_2):
            packed = data.astype(np.int64)[: (23 // 4) * 4]
            assert (packed < 0).any(), dtype
            assert (packed > 0).any(), dtype


def test_generation_is_deterministic() -> None:
    desc = _make_desc()
    a1, a2 = OpChunkedEquivalence(desc, seed=7)._generate_operands(23, "S8")
    b1, b2 = OpChunkedEquivalence(desc, seed=7)._generate_operands(23, "S8")
    np.testing.assert_array_equal(a1, b1)
    np.testing.assert_array_equal(a2, b2)


def test_sign_coverage_check_rejects_non_negative_packed_region() -> None:
    # All-positive post-offset data cannot discriminate the ns-cmsis-nn#343
    # class of defect; generation must refuse it rather than emit a case that
    # can only ever pass.
    data = np.arange(1, 24, dtype=np.int8)
    with pytest.raises(ValueError, match="sign-dependent"):
        OpChunkedEquivalence._check_sign_coverage("case", "input_1", data, 0, 23)
    # Sign-diverse data passes.
    OpChunkedEquivalence._check_sign_coverage(
        "case", "input_1", np.array([-5, 3, -1, 7] * 6, dtype=np.int8)[:23], 0, 23
    )


def test_sign_coverage_accounts_for_input_offset() -> None:
    # Values are negative pre-offset but the offset pushes every packed lane
    # non-negative: the case is vacuous and must be rejected.
    data = np.full(23, -3, dtype=np.int8)
    with pytest.raises(ValueError, match="sign-dependent"):
        OpChunkedEquivalence._check_sign_coverage("case", "input_1", data, 100, 23)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"chunk_sizes": [10, 10]}, "sum to 20"),
        ({"chunk_sizes": [23]}, "at least 2 chunks"),
        ({"chunk_sizes": [0, 23]}, "chunk size must be >= 1"),
        ({"kernel": "div"}, "unsupported"),
    ],
)
def test_invalid_descriptors_are_rejected(overrides: dict, match: str, tmp_path: Path) -> None:
    desc = _make_desc(**overrides)
    op = OpChunkedEquivalence(desc, seed=1)
    with pytest.raises(ValueError, match=match):
        op.generate_c_files(tmp_path)


def test_shipped_descriptors_cover_all_families_and_patterns(tmp_path: Path) -> None:
    descriptors = load_all_descriptors(str(_repo_root() / "assets" / "descriptors"))
    ce = [d for d in descriptors if d["operator"] == "ChunkedEquivalence"]
    assert len(ce) == 26

    combos = {(d["kernel"], d["activation_dtype"]) for d in ce}
    assert combos == set(_EXPECTED_KERNELS)

    for desc in ce:
        chunks = list(desc["chunk_sizes"])
        element_count = int(desc["element_count"])
        assert sum(chunks) == element_count
        assert len(chunks) >= 2
        # Every shipped case must force a tail path and an unaligned chunk
        # boundary for every vector width its kernel actually uses (16 for the
        # s8 min/max wlstp.8 loop, 8 for the s16 one, 4 for the MVE 32-bit-lane
        # loops, 2 for the s16 DSP halfword pairs).
        widths = _KERNEL_TABLE[(desc["kernel"], desc["activation_dtype"])]["vector_widths"]
        boundaries = list(np.cumsum(chunks[:-1]))
        for width in widths:
            assert element_count % width != 0, (desc["name"], width)
            assert any(b % width for b in boundaries), (desc["name"], width)
        # The descriptor stays in the int suite.
        assert desc["_descriptor_suite"] == "default"
        # And it must actually generate.
        out_dir = tmp_path / desc["name"]
        out_dir.mkdir()
        OpChunkedEquivalence(desc, seed=1, target_cpu="cortex-m4").generate_c_files(out_dir)
        assert (out_dir / f"{desc['name']}_chunked_equivalence.c").exists()

    singles = [d for d in ce if "singles" in d["name"]]
    assert len(singles) == 13
    for desc in singles:
        # All-singles pattern: pure scalar tail on the chunked pass.
        assert set(desc["chunk_sizes"]) == {1}


def test_unary_kernel_emits_one_operand_and_no_left_shift(tmp_path: Path) -> None:
    # arm_requantize_s8_s8 takes a single operand and a `size`; the addsub
    # left_shift/per-input multiplier arguments have no image here and must not
    # leak into the generated call.
    name = "chunked_equivalence_requantize_offcut_s8"
    desc = _make_desc(name=name, kernel="requantize", element_count=29, chunk_sizes=[3, 8, 7, 5, 4, 2])
    OpChunkedEquivalence(desc, seed=1, target_cpu="cortex-m55").generate_c_files(tmp_path)

    c_text = (tmp_path / f"{name}_chunked_equivalence.c").read_text()
    assert "arm_requantize_s8_s8(" in c_text
    assert "effective_scale_multiplier" in c_text
    assert "left_shift" not in c_text
    assert "input2" not in c_text


def test_minmax_kernel_slices_by_dims_and_keeps_the_context_argument(tmp_path: Path) -> None:
    # min/max take cmsis_nn_dims, not a block size. Identical dims on both
    # inputs and the output are what keep the broadcast walk on the
    # no-broadcast path, which is the one holding the vector loop.
    name = "chunked_equivalence_minimum_offcut_s8"
    desc = _make_desc(name=name, kernel="minimum", element_count=29, chunk_sizes=[3, 8, 7, 5, 4, 2])
    OpChunkedEquivalence(desc, seed=1, target_cpu="cortex-m55").generate_c_files(tmp_path)

    c_text = (tmp_path / f"{name}_chunked_equivalence.c").read_text()
    assert "cmsis_nn_context ctx" in c_text
    assert "cmsis_nn_dims dims = {1, 1, 1, block_size}" in c_text
    assert "arm_minimum_s8(&ctx, input1, &dims, input2, &dims, output, &dims)" in c_text


def test_chunk_pattern_must_leave_a_tail_and_an_unaligned_boundary() -> None:
    # 32 is a multiple of every width in play: the full call has no tail.
    with pytest.raises(ValueError, match="no tail"):
        OpChunkedEquivalence._check_chunk_discrimination("case", 32, [16, 16], (16, 4, 2))
    # 21 leaves a tail at width 4, but every boundary is 4-aligned, so the
    # chunked pass runs exactly the vector lanes the full call runs.
    with pytest.raises(ValueError, match="aligned to the kernel's vector width 4"):
        OpChunkedEquivalence._check_chunk_discrimination("case", 21, [8, 8, 5], (4,))
    # The shipped offcut pattern satisfies both at every width.
    OpChunkedEquivalence._check_chunk_discrimination("case", 29, [3, 8, 7, 5, 4, 2], (16, 8, 4, 2))


def test_prelu_is_not_a_chunked_equivalence_kernel() -> None:
    # arm_elementwise_prelu_s8/_s16 are a single scalar loop in the checkout:
    # there is no packed path for a chunk pattern to disagree with.
    assert not [k for k, _ in _KERNEL_TABLE if "prelu" in k]
    desc = _make_desc(kernel="prelu")
    with pytest.raises(ValueError, match="unsupported"):
        OpChunkedEquivalence(desc, seed=1)._kernel_key()


def test_minmax_operands_need_no_offset_but_still_span_sign() -> None:
    # min/max apply no input offset, so the raw stored values are what the
    # kernel compares and the span has to hold on them directly.
    desc = _make_desc(name="chunked_equivalence_minimum_singles_s8", kernel="minimum")
    op = OpChunkedEquivalence(desc, seed=1)
    assert op._operand_offsets("minmax", {}) == [0, 0]
    for data in op._generate_operands(23, "S8", 2):
        packed = data.astype(np.int64)[:16]
        assert (packed < 0).any()
        assert (packed > 0).any()


def test_requantize_sign_check_uses_the_negated_input_zero_point() -> None:
    # arm_requantize_s8_s8 computes value - input_zeropoint, so the offset the
    # span check has to add is -input_zeropoint, not +.
    quant = {"input_zeropoint": -3}
    assert OpChunkedEquivalence._operand_offsets("requantize", quant) == [3]


def test_squared_difference_output_params_do_not_saturate_every_lane() -> None:
    # Sharing the add/sub output requantization would clamp every lane of a
    # squared result to out_activation_max and make the case vacuous.
    s8 = OpChunkedEquivalence._quant_params("addsub", "S8", "squared_difference")
    assert s8["out_shift"] != OpChunkedEquivalence._quant_params("addsub", "S8", "add")["out_shift"]
    peak = ((127 + abs(s8["input_1_offset"])) // 4 + (128 + abs(s8["input_2_offset"])) // 4) ** 2
    assert peak >> (-s8["out_shift"] + 1) < s8["out_activation_max"] - s8["out_offset"]
