"""SVDF input_ctx / output_ctx scratch sizing (issue #71).

heliaCORE publishes eight SVDF staging-buffer sizers
(``arm_svdf_{s8,state_s16_s8,f32,f16}_{input,output}_ctx_get_buffer_size``). The
generated harnesses must size their scratch through them rather than re-deriving
``n * feature_batches * sizeof(elem)`` locally, and must assert the sizer's answer
against a value measured independently on host (carried by the descriptor, never
computed in Python). A descriptor without the new keys renders none of the asserts.
"""
from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from helia_core_tester.generation.io.descriptors import load_all_descriptors
from helia_core_tester.generation.ops.SVDFunctions.svdf import OpSVDF
from helia_core_tester.generation.test_ops import default_seed_for_case


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _descriptor_dir() -> Path:
    return _repo_root() / "assets" / "descriptors" / "SVDFunctions"


def _template_dir() -> Path:
    return _repo_root() / "assets" / "templates" / "SVDFunctions" / "svdf"


# One checked-in case per kernel, used as the base descriptor for the render tests
# so the generator sees exactly what a real run sees.
KERNEL_CASES = {
    "arm_svdf_s8": "svdf_bias_rank1_s8",
    "arm_svdf_state_s16_s8": "svdf_state_s16_bias_rank1_s8",
    "arm_svdf_f32": "svdf_float_default_f32",
    "arm_svdf_f16": "svdf_float_default_f16",
}
INT_KERNELS = ("arm_svdf_s8", "arm_svdf_state_s16_s8")
FLOAT_KERNELS = ("arm_svdf_f32", "arm_svdf_f16")
CTX_KEYS = ("expected_input_ctx_size", "expected_output_ctx_size", "ctx_sizer_sentinels")


@pytest.fixture(scope="module")
def checked_in_descriptors() -> dict[str, dict]:
    descriptors = load_all_descriptors(str(_repo_root() / "assets" / "descriptors"))
    return {desc["name"]: desc for desc in descriptors if desc["operator"] == "SVDF"}


def _base_descriptor(checked_in: dict[str, dict], kernel: str, **overrides) -> dict:
    desc = copy.deepcopy(checked_in[KERNEL_CASES[kernel]])
    for key in CTX_KEYS:
        desc.pop(key, None)
    desc.update(overrides)
    return desc


def _emit(tmp_path: Path, desc: dict) -> str:
    out_dir = tmp_path / desc["name"]
    out_dir.mkdir()
    op = OpSVDF(desc, default_seed_for_case(desc["name"]), target_cpu="cortex-m55")
    op.generate_c_files(out_dir)
    return (out_dir / f"{desc['name']}_svdf.c").read_text()


# ---------------------------------------------------------------------------
# Sizer wiring
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kernel", list(KERNEL_CASES))
def test_each_kernel_sizes_scratch_through_its_own_sizers(
    tmp_path: Path, checked_in_descriptors: dict[str, dict], kernel: str
) -> None:
    rendered = _emit(tmp_path, _base_descriptor(checked_in_descriptors, kernel))

    assert f"{kernel}_input_ctx_get_buffer_size(" in rendered
    assert f"{kernel}_output_ctx_get_buffer_size(" in rendered
    for other in KERNEL_CASES:
        if other != kernel:
            assert f"{other}_input_ctx_get_buffer_size(" not in rendered
            assert f"{other}_output_ctx_get_buffer_size(" not in rendered
    # A negative sizer result must be rejected before any allocation or kernel call.
    assert "return ARM_CMSIS_NN_ARG_ERROR;" in rendered


def test_int_template_no_longer_hardcodes_scratch_arithmetic(
    tmp_path: Path, checked_in_descriptors: dict[str, dict]
) -> None:
    template = (_template_dir() / "svdf.c.j2").read_text()
    assert "* sizeof(int32_t)" not in template
    assert "_FEATURE_BATCHES * sizeof" not in template
    assert "_UNIT_COUNT * sizeof" not in template

    for kernel in INT_KERNELS:
        rendered = _emit(tmp_path, _base_descriptor(checked_in_descriptors, kernel))
        assert "* sizeof(int32_t)" not in rendered
        assert "malloc(scratch_size);" in rendered
        assert "malloc(scratch_size_out);" in rendered
        # rank feeds the output sizer, so the params must be populated ahead of it.
        assert rendered.index("svdf_params.rank =") < rendered.index(
            f"{kernel}_output_ctx_get_buffer_size(&svdf_params"
        )


@pytest.mark.parametrize("kernel", FLOAT_KERNELS)
def test_float_template_keeps_static_arrays_as_upper_bound(
    tmp_path: Path, checked_in_descriptors: dict[str, dict], kernel: str
) -> None:
    name = KERNEL_CASES[kernel]
    rendered = _emit(tmp_path, _base_descriptor(checked_in_descriptors, kernel))
    assert f"scratch_input_bytes > (int32_t)sizeof({name}_scratch_input)" in rendered
    assert f"scratch_output_bytes > (int32_t)sizeof({name}_scratch_output)" in rendered
    assert f".buf = {name}_scratch_input, .size = scratch_input_bytes" in rendered
    assert f".buf = {name}_scratch_output, .size = scratch_output_bytes" in rendered
    assert f".size = sizeof({name}_scratch_input)" not in rendered


# ---------------------------------------------------------------------------
# expected_* gating
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kernel", list(KERNEL_CASES))
def test_descriptor_without_expected_keys_renders_no_ctx_size_assert(
    tmp_path: Path, checked_in_descriptors: dict[str, dict], kernel: str
) -> None:
    rendered = _emit(tmp_path, _base_descriptor(checked_in_descriptors, kernel))
    assert '"input ctx size"' not in rendered
    assert '"output ctx size"' not in rendered
    assert "probe_input_dims" not in rendered
    assert "issue #71), measured" not in rendered


@pytest.mark.parametrize("kernel", list(KERNEL_CASES))
def test_expected_keys_render_assert_with_descriptor_values(
    tmp_path: Path, checked_in_descriptors: dict[str, dict], kernel: str
) -> None:
    desc = _base_descriptor(
        checked_in_descriptors, kernel, expected_input_ctx_size=12345, expected_output_ctx_size=678
    )
    rendered = _emit(tmp_path, desc)
    assert 'HELIA_VALIDATE_SCALAR_EQ_INT("' in rendered
    assert '"input ctx size", 12345,' in rendered
    assert '"output ctx size", 678,' in rendered
    assert f"{kernel}_input_ctx_get_buffer_size(&{desc['name']}_input_dims" in rendered
    # Target-invariant: no ISA split around the ctx asserts.
    start = rendered.index('"input ctx size"')
    end = rendered.index("int32_t status = run_svdf();")
    assert "#if defined(ARM_MATH" not in rendered[start:end]
    assert "probe_input_dims" not in rendered


@pytest.mark.parametrize("kernel", list(KERNEL_CASES))
def test_only_one_expected_key_renders_only_that_assert(
    tmp_path: Path, checked_in_descriptors: dict[str, dict], kernel: str
) -> None:
    rendered = _emit(
        tmp_path, _base_descriptor(checked_in_descriptors, kernel, expected_output_ctx_size=99)
    )
    assert '"input ctx size"' not in rendered
    assert '"output ctx size", 99,' in rendered


# ---------------------------------------------------------------------------
# Sentinel probes
# ---------------------------------------------------------------------------

_SENTINEL_SUBJECTS = (
    '"input ctx size (input n == 0)", 0,',
    '"output ctx size (input n == 0)", 0,',
    '"input ctx size (weights_feature n < 0)", -1,',
    '"output ctx size (weights_feature n < 0)", -1,',
    '"input ctx size (NULL input_dims)", -1,',
    '"output ctx size (NULL input_dims)", -1,',
    '"output ctx size (NULL params)", -1,',
    '"output ctx size (rank 0)", -1,',
    '"output ctx size (rank > weights_feature n)", 0,',
)
_INT_ONLY_SENTINEL = '"output ctx size (rank 65538)", -1,'


@pytest.mark.parametrize("kernel", list(KERNEL_CASES))
def test_sentinels_off_renders_no_probe(
    tmp_path: Path, checked_in_descriptors: dict[str, dict], kernel: str
) -> None:
    rendered = _emit(
        tmp_path, _base_descriptor(checked_in_descriptors, kernel, ctx_sizer_sentinels=False)
    )
    assert "probe_input_dims" not in rendered
    assert "65538" not in rendered
    for subject in _SENTINEL_SUBJECTS:
        assert subject not in rendered


@pytest.mark.parametrize("kernel", list(KERNEL_CASES))
def test_sentinels_on_renders_sizer_only_probes(
    tmp_path: Path, checked_in_descriptors: dict[str, dict], kernel: str
) -> None:
    rendered = _emit(
        tmp_path, _base_descriptor(checked_in_descriptors, kernel, ctx_sizer_sentinels=True)
    )
    for subject in _SENTINEL_SUBJECTS:
        assert subject in rendered, subject
    assert f"{kernel}_input_ctx_get_buffer_size(NULL, &probe_weights_feature_dims)" in rendered
    assert f"{kernel}_output_ctx_get_buffer_size(NULL, &probe_input_dims" in rendered

    # The int16 narrowing rule only exists in the integer sizers; the float sizers return
    # 0 for rank 65538 (truncating division), so the probe must not be emitted there.
    if kernel in INT_KERNELS:
        assert _INT_ONLY_SENTINEL in rendered
        assert "probe_params.rank = 65538;" in rendered
    else:
        assert "65538" not in rendered

    # Sizer-only: the probe structs never reach the kernel call.
    kernel_call = rendered[rendered.index(f"= {kernel}(") :]
    kernel_call = kernel_call[: kernel_call.index(";")]
    assert "probe_" not in kernel_call


# ---------------------------------------------------------------------------
# Checked-in descriptors
# ---------------------------------------------------------------------------

# (case, input_ctx bytes, output_ctx bytes): measured by calling the real sizers on host
# for each case's (input_batches, feature_batches, rank); see the YAML comments.
DOCUMENTED_SIZES = {
    "svdf_bias_rank1_s8": (32, 32),
    "svdf_bias_rank2_s8": (32, 16),
    "svdf_no_bias_rank2_s8": (32, 16),
    "svdf_state_s16_bias_rank1_s8": (32, 32),
    "svdf_state_s16_bias_rank2_s8": (32, 16),
    "svdf_state_s16_no_bias_rank2_s8": (32, 16),
    "svdf_bias_rank1_rows3_cols17_s8": (24, 24),
    "svdf_no_bias_rank2_rows6_cols17_s8": (48, 24),
    "svdf_state_s16_bias_rank1_rows5_cols17_s8": (40, 40),
    "svdf_state_s16_no_bias_rank2_rows6_cols33_s8": (48, 24),
    "svdf_float_default_f32": (16, 16),
    "svdf_float_no_bias_f32": (16, 16),
    "svdf_float_rank2_f32": (16, 8),
    "svdf_float_default_f16": (8, 8),
    "svdf_float_no_bias_f16": (8, 8),
    "svdf_float_rank2_f16": (8, 4),
    "svdf_float_nonfinite_nan_f32": (32, 32),
    "svdf_float_nonfinite_inf_f32": (32, 32),
    "svdf_float_nonfinite_neginf_f32": (32, 32),
    "svdf_float_nonfinite_nan_f16": (16, 16),
    "svdf_float_nonfinite_inf_f16": (16, 16),
    "svdf_float_nonfinite_neginf_f16": (16, 16),
}
SENTINEL_CASES = {
    "svdf_bias_rank2_s8",
    "svdf_state_s16_bias_rank2_s8",
    "svdf_float_rank2_f32",
    "svdf_float_rank2_f16",
}


def _raw_svdf_documents() -> dict[str, dict]:
    docs: dict[str, dict] = {}
    for filename in ("svdf.yaml", "svdf_float.yaml"):
        for doc in yaml.safe_load_all((_descriptor_dir() / filename).read_text()):
            if isinstance(doc, dict):
                docs[doc["name"]] = doc
    return docs


def test_checked_in_yaml_carries_the_documented_sizes() -> None:
    docs = _raw_svdf_documents()
    assert set(docs) == set(DOCUMENTED_SIZES)
    for name, (input_bytes, output_bytes) in DOCUMENTED_SIZES.items():
        assert docs[name]["expected_input_ctx_size"] == input_bytes, name
        assert docs[name]["expected_output_ctx_size"] == output_bytes, name


def test_checked_in_yaml_arms_sentinels_once_per_kernel() -> None:
    docs = _raw_svdf_documents()
    armed = {name for name, doc in docs.items() if doc.get("ctx_sizer_sentinels")}
    assert armed == SENTINEL_CASES


def test_loaded_descriptors_reach_the_generator_context(
    checked_in_descriptors: dict[str, dict],
) -> None:
    for name, (input_bytes, output_bytes) in DOCUMENTED_SIZES.items():
        op = OpSVDF(checked_in_descriptors[name], default_seed_for_case(name))
        context = op._ctx_sizer_context("arm_svdf_x")
        assert context["expected_input_ctx_size"] == input_bytes
        assert context["expected_output_ctx_size"] == output_bytes
        assert context["ctx_sizer_sentinels"] is (name in SENTINEL_CASES)
        assert context["input_ctx_sizer_fn"] == "arm_svdf_x_input_ctx_get_buffer_size"
        assert context["output_ctx_sizer_fn"] == "arm_svdf_x_output_ctx_get_buffer_size"
