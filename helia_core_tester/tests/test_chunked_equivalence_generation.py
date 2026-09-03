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
  - the shipped descriptor corpus covers add/sub/mul for s8 and s16 with both
    an all-singles pattern (pure tail vs packed) and a mixed "offcut" pattern
    (packed loops starting on odd boundaries, every mod-4/mod-2 tail length).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from helia_core_tester.generation.io.descriptors import load_all_descriptors
from helia_core_tester.generation.ops.BasicMathFunctions.chunked_equivalence import (
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
    ("add", "S16"): "arm_elementwise_add_s16",
    ("sub", "S16"): "arm_elementwise_sub_s16",
    ("mul", "S16"): "arm_elementwise_mul_s16",
}


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
    assert len(ce) == 12

    combos = {(d["kernel"], d["activation_dtype"]) for d in ce}
    assert combos == set(_EXPECTED_KERNELS)

    for desc in ce:
        chunks = list(desc["chunk_sizes"])
        element_count = int(desc["element_count"])
        assert sum(chunks) == element_count
        assert len(chunks) >= 2
        # Every shipped case must force a tail path: the element count is not
        # a multiple of the widest packed stride (4), and the chunked pass
        # contains at least one chunk that is not a multiple of 4 either.
        assert element_count % 4 != 0
        assert any(c % 4 != 0 for c in chunks)
        # The descriptor stays in the int suite.
        assert desc["_descriptor_suite"] == "default"
        # And it must actually generate.
        out_dir = tmp_path / desc["name"]
        out_dir.mkdir()
        OpChunkedEquivalence(desc, seed=1, target_cpu="cortex-m4").generate_c_files(out_dir)
        assert (out_dir / f"{desc['name']}_chunked_equivalence.c").exists()

    singles = [d for d in ce if "singles" in d["name"]]
    assert len(singles) == 6
    for desc in singles:
        # All-singles pattern: pure scalar tail on the chunked pass.
        assert set(desc["chunk_sizes"]) == {1}
