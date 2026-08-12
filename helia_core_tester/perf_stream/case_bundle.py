"""Host-side streamable case bundle helpers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any
import zlib

import numpy as np

from helia_core_tester.generation.io.descriptors import load_descriptor
from helia_core_tester.generation.utils.template_context import TemplateContextBuilder
from helia_core_tester.generation.utils.tflite_utils import calculate_per_channel_multiplier_shift, requantize_np


_DTYPE_TO_NUMPY = {
    "S8": np.int8,
    "S16": np.int16,
    "S32": np.int32,
    "S64": np.int64,
    "BOOL": np.bool_,
    "FP32": np.float32,
    "FP16": np.float16,
}


@dataclass(frozen=True)
class BlobInfo:
    blob_id: int
    role: str
    dtype: str
    rank: int
    dimensions: tuple[int, ...]
    byte_length: int
    required_alignment: int
    expected_crc32: int
    sha256: str
    path: Path
    mutable_data: bool = False
    host_only: bool = False


@dataclass(frozen=True)
class CaseBundle:
    root_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]
    blobs: tuple[BlobInfo, ...]

    @property
    def case_id(self) -> str:
        return str(self.manifest["case_id"])

    @property
    def comparison(self) -> dict[str, Any]:
        return dict(self.manifest["correctness_comparison"])

    @property
    def kernel_id(self) -> int:
        return int(self.manifest["kernel_id"])

    @property
    def streamable_blobs(self) -> tuple[BlobInfo, ...]:
        return tuple(blob for blob in self.blobs if not blob.host_only)

    @property
    def input_blob(self) -> BlobInfo:
        return self.blob_by_role("input_0")

    @property
    def expected_output(self) -> BlobInfo:
        return self.blob_by_role("expected_output")

    def blob_by_role(self, role: str) -> BlobInfo:
        return next(blob for blob in self.blobs if blob.role == role)



def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()



def _blob_info(
    path: Path,
    *,
    blob_id: int,
    role: str,
    dtype: str,
    dimensions: tuple[int, ...],
    mutable_data: bool = False,
    host_only: bool = False,
) -> BlobInfo:
    payload = path.read_bytes()
    return BlobInfo(
        blob_id=blob_id,
        role=role,
        dtype=dtype,
        rank=len(dimensions),
        dimensions=dimensions,
        byte_length=len(payload),
        required_alignment=np.dtype(_DTYPE_TO_NUMPY[dtype]).itemsize,
        expected_crc32=zlib.crc32(payload) & 0xFFFFFFFF,
        sha256=_sha256_bytes(payload),
        path=path,
        mutable_data=mutable_data,
        host_only=host_only,
    )



def _write_blob(path: Path, array: np.ndarray) -> None:
    path.write_bytes(np.asarray(array).tobytes(order="C"))



def _padding_same(input_size: int, filter_size: int, stride: int) -> tuple[int, int]:
    output_size = (input_size + stride - 1) // stride
    pad_total = max((output_size - 1) * stride + filter_size - input_size, 0)
    return pad_total // 2, pad_total - (pad_total // 2)



def _conv2d_s8(
    input_data: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    *,
    stride_h: int,
    stride_w: int,
    padding: str,
    multiplier: np.ndarray,
    shift: np.ndarray,
    output_offset: int = 0,
    activation_min: int = -128,
    activation_max: int = 127,
) -> np.ndarray:
    batch, in_h, in_w, in_c = input_data.shape
    filt_h, filt_w, filt_c, out_c = weights.shape
    assert batch == 1
    assert filt_c == in_c
    if padding == "SAME":
        pad_top, pad_bottom = _padding_same(in_h, filt_h, stride_h)
        pad_left, pad_right = _padding_same(in_w, filt_w, stride_w)
    else:
        pad_top = pad_bottom = pad_left = pad_right = 0
    padded = np.pad(
        input_data.astype(np.int32),
        ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode="constant",
    )
    out_h = ((padded.shape[1] - filt_h) // stride_h) + 1
    out_w = ((padded.shape[2] - filt_w) // stride_w) + 1
    output = np.zeros((batch, out_h, out_w, out_c), dtype=np.int8)
    for oy in range(out_h):
        for ox in range(out_w):
            window = padded[0, oy * stride_h : oy * stride_h + filt_h, ox * stride_w : ox * stride_w + filt_w, :]
            for oc in range(out_c):
                acc = int(np.sum(window * weights[:, :, :, oc].astype(np.int32))) + int(bias[oc])
                requantized = int(requantize_np(np.array([acc], dtype=np.int32), int(multiplier[oc]), int(shift[oc]))[0])
                output[0, oy, ox, oc] = np.int8(np.clip(requantized + output_offset, activation_min, activation_max))
    return output



def _manifest_blob_entry(blob: BlobInfo) -> dict[str, Any]:
    return {
        "blob_id": blob.blob_id,
        "role": blob.role,
        "dtype": blob.dtype,
        "byte_length": blob.byte_length,
        "dimensions": list(blob.dimensions),
        "alignment": blob.required_alignment,
        "crc32": blob.expected_crc32,
        "sha256": blob.sha256,
        "file_name": blob.path.name,
        "mutable_data": blob.mutable_data,
        "host_only": blob.host_only,
    }



def _write_manifest(case_root: Path, manifest: dict[str, Any]) -> Path:
    manifest_path = case_root / "case_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8", newline="\n")
    return manifest_path



def _case_root(bundle_root: Path, family: str, case_id: str) -> Path:
    return bundle_root / "artifacts" / "stream_cases" / "int" / "cortex-m55" / family / case_id



def build_abs_s8_case_bundle(
    project_root: Path,
    *,
    output_root: Path | None = None,
    case_id: str = "abs_default_s8_stream_demo",
) -> CaseBundle:
    descriptor_path = Path("assets/descriptors/BasicMathFunctions/abs.yaml")
    descriptors = load_descriptor(str(project_root / descriptor_path))
    descriptor = next(desc for desc in descriptors if desc["name"] == "abs_default_s8")

    rng = np.random.default_rng(500)
    input_shape = tuple(int(v) for v in descriptor["input_shape"])
    input_data = rng.integers(-127, 128, size=input_shape, dtype=np.int16).astype(np.int8)
    expected_output = np.abs(input_data.astype(np.int16)).astype(np.int8)

    bundle_root = output_root if output_root is not None else project_root
    case_root = _case_root(bundle_root, "BasicMathFunctions", case_id)
    blobs_dir = case_root / "blobs"
    blobs_dir.mkdir(parents=True, exist_ok=True)

    input_blob_path = blobs_dir / "input_0.bin"
    expected_blob_path = blobs_dir / "expected_output.bin"
    _write_blob(input_blob_path, input_data)
    _write_blob(expected_blob_path, expected_output)

    blobs = (
        _blob_info(input_blob_path, blob_id=1, role="input_0", dtype="S8", dimensions=input_shape),
        _blob_info(expected_blob_path, blob_id=2, role="expected_output", dtype="S8", dimensions=input_shape, host_only=True),
    )

    descriptor_text = (project_root / descriptor_path).read_text(encoding="utf-8")
    manifest = {
        "schema_name": "hct.case_manifest",
        "schema_version": 1,
        "case_id": case_id,
        "descriptor_name": descriptor["name"],
        "descriptor_path": descriptor_path.as_posix(),
        "descriptor_sha256": hashlib.sha256(descriptor_text.encode("utf-8")).hexdigest(),
        "operator": descriptor["operator"],
        "family": "BasicMathFunctions",
        "target_cpu": "cortex-m55",
        "kernel_id": 1,
        "adapter_metadata_schema": 1,
        "serialized_scalar_parameters": {},
        "tensor_dtypes": {"input": "S8", "output": "S8"},
        "blob_roles": [_manifest_blob_entry(blob) for blob in blobs],
        "expected_output": {"dtype": "S8", "byte_length": blobs[1].byte_length, "blob_id": blobs[1].blob_id},
        "correctness_comparison": dict(descriptor["resolved_comparison"]),
        "scratch_buffer": {"bytes": 0},
        "required_target_capabilities": [],
        "repeated_invocation_safe": True,
        "timing": {"warmups": 2, "samples": 3, "iterations_per_sample": 4, "min_cycles": 512, "max_iterations": 128},
    }
    return CaseBundle(root_dir=case_root, manifest_path=_write_manifest(case_root, manifest), manifest=manifest, blobs=blobs)



def build_convolve_s8_case_bundle(
    project_root: Path,
    *,
    output_root: Path | None = None,
    case_id: str = "convolve_default_s8_stream_demo",
) -> CaseBundle:
    descriptor_path = Path("assets/descriptors/ConvolutionFunctions/convolve.yaml")
    descriptors = load_descriptor(str(project_root / descriptor_path))
    descriptor = next(desc for desc in descriptors if desc["name"] == "convolve_default_s8")

    rng = np.random.default_rng(501)
    input_shape = tuple(int(v) for v in descriptor["input_shape"])
    filter_shape = tuple(int(v) for v in descriptor["filter_shape"])
    strides = tuple(int(v) for v in descriptor["strides"])
    padding = str(descriptor["padding"])
    output_channels = filter_shape[3]

    input_data = rng.integers(-8, 9, size=input_shape, dtype=np.int16).astype(np.int8)
    weights = rng.integers(-4, 5, size=filter_shape, dtype=np.int16).astype(np.int8)
    bias = rng.integers(-32, 33, size=(output_channels,), dtype=np.int32)
    multiplier, shift = calculate_per_channel_multiplier_shift(np.ones(output_channels, dtype=np.float32))
    expected_output = _conv2d_s8(
        input_data,
        weights,
        bias,
        stride_h=strides[0],
        stride_w=strides[1],
        padding=padding,
        multiplier=multiplier,
        shift=shift,
    )
    # Ground-truth "before" padding matching _conv2d_s8's own internal computation, sent
    # explicitly to firmware (see serialized_scalar_parameters below) so it doesn't have to
    # re-derive padding/output size from a SAME/VALID heuristic that could silently diverge.
    if padding == "SAME":
        pad_h, _ = _padding_same(input_shape[1], filter_shape[0], strides[0])
        pad_w, _ = _padding_same(input_shape[2], filter_shape[1], strides[1])
    else:
        pad_h = pad_w = 0

    input_dims = {"n": input_shape[0], "h": input_shape[1], "w": input_shape[2], "c": input_shape[3]}
    filter_dims = {"h": filter_shape[0], "w": filter_shape[1], "c": filter_shape[2], "n": filter_shape[3]}
    output_dims = {"n": expected_output.shape[0], "h": expected_output.shape[1], "w": expected_output.shape[2], "c": expected_output.shape[3]}
    scratch_bytes = TemplateContextBuilder.calculate_buffer_size_max(input_dims, filter_dims, output_dims, output_dtype="S8")

    bundle_root = output_root if output_root is not None else project_root
    case_root = _case_root(bundle_root, "ConvolutionFunctions", case_id)
    blobs_dir = case_root / "blobs"
    blobs_dir.mkdir(parents=True, exist_ok=True)

    arrays = [
        (1, "input_0", "S8", input_shape, input_data, False, False),
        (2, "weights", "S8", filter_shape, weights, False, False),
        (3, "bias", "S32", (output_channels,), bias, False, False),
        (4, "multiplier", "S32", (output_channels,), multiplier.astype(np.int32), False, False),
        (5, "shift", "S32", (output_channels,), shift.astype(np.int32), False, False),
        (6, "expected_output", "S8", tuple(int(v) for v in expected_output.shape), expected_output, False, True),
    ]
    blobs: list[BlobInfo] = []
    for blob_id, role, dtype, dims, array, mutable_data, host_only in arrays:
        path = blobs_dir / f"{role}.bin"
        _write_blob(path, np.asarray(array))
        blobs.append(
            _blob_info(path, blob_id=blob_id, role=role, dtype=dtype, dimensions=tuple(int(v) for v in dims), mutable_data=mutable_data, host_only=host_only)
        )

    descriptor_text = (project_root / descriptor_path).read_text(encoding="utf-8")
    manifest = {
        "schema_name": "hct.case_manifest",
        "schema_version": 1,
        "case_id": case_id,
        "descriptor_name": descriptor["name"],
        "descriptor_path": descriptor_path.as_posix(),
        "descriptor_sha256": hashlib.sha256(descriptor_text.encode("utf-8")).hexdigest(),
        "operator": descriptor["operator"],
        "family": "ConvolutionFunctions",
        "target_cpu": "cortex-m55",
        "kernel_id": 2,
        "adapter_metadata_schema": 1,
        "serialized_scalar_parameters": {
            "stride_h": strides[0],
            "stride_w": strides[1],
            "padding": padding,
            "pad_h": pad_h,
            "pad_w": pad_w,
            "dilation_h": 1,
            "dilation_w": 1,
            "output_h": output_dims["h"],
            "output_w": output_dims["w"],
            "output_c": output_dims["c"],
            "input_offset": 0,
            "output_offset": 0,
            "activation_min": -128,
            "activation_max": 127,
        },
        "tensor_dtypes": {"input": "S8", "weights": "S8", "bias": "S32", "output": "S8"},
        "blob_roles": [_manifest_blob_entry(blob) for blob in blobs],
        "expected_output": {"dtype": "S8", "byte_length": blobs[-1].byte_length, "blob_id": blobs[-1].blob_id},
        "correctness_comparison": dict(descriptor["resolved_comparison"]),
        "scratch_buffer": {"bytes": int(scratch_bytes)},
        "required_target_capabilities": ["convolve_s8"],
        "repeated_invocation_safe": True,
        "timing": {"warmups": 2, "samples": 5, "iterations_per_sample": 4, "min_cycles": 1024, "max_iterations": 256},
    }
    return CaseBundle(root_dir=case_root, manifest_path=_write_manifest(case_root, manifest), manifest=manifest, blobs=tuple(blobs))



def load_case_bundle(manifest_path: Path) -> CaseBundle:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    case_root = manifest_path.parent
    blobs = []
    for entry in manifest["blob_roles"]:
        blobs.append(
            _blob_info(
                case_root / "blobs" / entry["file_name"],
                blob_id=int(entry["blob_id"]),
                role=str(entry["role"]),
                dtype=str(entry["dtype"]),
                dimensions=tuple(int(v) for v in entry["dimensions"]),
                mutable_data=bool(entry.get("mutable_data", False)),
                host_only=bool(entry.get("host_only", False)),
            )
        )
    return CaseBundle(root_dir=case_root, manifest_path=manifest_path, manifest=manifest, blobs=tuple(blobs))



def blob_numpy(blob: BlobInfo) -> np.ndarray:
    dtype = _DTYPE_TO_NUMPY[blob.dtype]
    data = np.frombuffer(blob.path.read_bytes(), dtype=dtype)
    return data.reshape(blob.dimensions)
