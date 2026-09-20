"""Assembly preserves storage, ordered metadata and the existing failure boundary."""

from pathlib import Path
import struct

import numpy as np
import pytest

from helia_core_tester.hardware import generated_test_bridge as bridge
from helia_core_tester.hardware.case_bundle import load_case_bundle
from helia_core_tester.hardware.session import case_meta_for_bundle

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("cast,expected", [
    (None, struct.pack("<dd", 1.25, -2.5)),
    (np.float16, struct.pack("<ee", 1.25, -2.5)),
    (np.float32, struct.pack("<ff", 1.25, -2.5)),
])
def test_blob_storage_cast_is_explicit(tmp_path, cast, expected):
    # A declared wire dtype must not silently become a storage conversion policy.
    arrays = [(7, "input_0", "FP16", (2,), np.array([1.25, -2.5]), False, False)]
    blob, = bridge._write_generated_blobs(tmp_path, arrays, numpy_dtype=cast)
    assert blob.path.read_bytes() == expected
    assert (blob.blob_id, blob.dtype, blob.dimensions) == (7, "FP16", (2,))


def test_packed_storage_and_host_only_flags_are_not_inferred(tmp_path):
    arrays = [(3, "weights", "S4", (2,), np.array([0x8F], dtype=np.uint8), True, True)]
    blob, = bridge._write_generated_blobs(tmp_path, arrays)
    assert blob.path.read_bytes() == b"\x8f"
    assert (blob.dimensions, blob.byte_length, blob.mutable_data, blob.host_only) == ((2,), 1, True, True)


@pytest.fixture
def abs_case(tmp_path):
    directory = tmp_path / "generated"
    directory.mkdir()
    (directory / "sample.h").write_text("""
const cmsis_nn_dims sample_input_dims = {.n=1, .h=1, .w=1, .c=2};
const cmsis_nn_dims sample_output_dims = {.n=1, .h=1, .w=1, .c=2};
const int16_t sample_input[2] = {-32768, 2};
const int16_t sample_expected_output[2] = {32767, 2};
""")
    (directory / "sample.c").write_text(
        "arm_abs_s16(sample_input, 0, sample_output, 0, 1073741824, 1, 0, -32768, 32767, 2);"
    )
    (directory / "descriptor.yaml").write_text("operator: Abs\nactivation_dtype: S16\n")
    return bridge.GeneratedTestCase(
        "sample", "cortex-m55", "BasicMathFunctions", directory,
        {"operator": "Abs", "activation_dtype": "S16"},
    )


def test_bundle_order_bytes_and_per_case_defaults(abs_case, tmp_path):
    first = bridge._build_abs_case(ROOT, abs_case, output_root=tmp_path / "first")
    second = bridge._build_abs_case(ROOT, abs_case, output_root=tmp_path / "second")
    loaded = load_case_bundle(first.manifest_path)
    assert list(loaded.manifest) == [
        "schema_name", "schema_version", "case_id", "descriptor_name", "descriptor_path",
        "descriptor_sha256", "operator", "family", "target_cpu", "kernel_id",
        "adapter_metadata_schema", "source", "serialized_scalar_parameters", "tensor_dtypes",
        "blob_roles", "expected_output", "correctness_comparison", "scratch_buffer",
        "required_target_capabilities", "repeated_invocation_safe", "timing",
    ]
    assert first.input_blob.path.read_bytes() == struct.pack("<hh", -32768, 2)
    assert first.expected_output.path.read_bytes() == struct.pack("<hh", 32767, 2)
    assert [blob.role for blob in loaded.blobs] == ["input_0", "expected_output"]
    meta = case_meta_for_bundle(loaded)
    assert [(b.blob_id, b.role, b.dtype) for b in meta.blobs] == [(1, "input_0", "S16")]
    assert [key for key, _ in meta.scalar_parameters] == [
        "input_offset", "output_offset", "out_mult", "out_shift", "needs_rescale",
        "activation_min", "activation_max", "output_capacity_bytes",
    ]
    first.manifest["timing"]["samples"] = 99
    assert second.manifest["timing"]["samples"] == 5


@pytest.mark.parametrize("missing_descriptor", [False, True])
def test_blob_writes_then_descriptor_read_precede_kernel_lookup(abs_case, tmp_path, monkeypatch, missing_descriptor):
    calls = []
    def reject_kernel(*args, **kwargs):
        calls.append("kernel")
        raise LookupError("kernel rejected")
    monkeypatch.setattr(bridge, "_kernel_id", reject_kernel)
    if missing_descriptor:
        (abs_case.directory / "descriptor.yaml").unlink()
    with pytest.raises(FileNotFoundError if missing_descriptor else LookupError):
        bridge._build_abs_case(ROOT, abs_case, output_root=tmp_path / "failed")
    assert calls == ([] if missing_descriptor else ["kernel"])
    assert sorted(p.name for p in (tmp_path / "failed").rglob("*.bin")) == ["expected_output.bin", "input_0.bin"]
