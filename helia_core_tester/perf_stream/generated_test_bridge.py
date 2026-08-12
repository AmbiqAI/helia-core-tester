"""Bridge real generated CMSIS-NN kernel tests (descriptor.yaml + golden C arrays
produced by `helia_core_tester generate`) into perf-stream CaseBundles so they can
be streamed to and executed on real hardware over HCTP/RTT, instead of only the
hand-authored synthetic demo cases in case_bundle.py.

Bridged (family, operator) pairs are registered in `_BUILDERS` below; each builder is
responsible for extracting its own header/source format and producing a CaseBundle. The
kernel_id sent to firmware for each is looked up from the shared registry in
`assets/kernel_registry.yaml` via `kernel_registry.py` -- see that file's header comment
for the full list of currently-bridged kernels and how to add new ones.

Anything not in `_BUILDERS` (or that fails dtype/shape preconditions inside a builder)
raises UnsupportedGeneratedTestError with a clear reason so callers can skip/report it
instead of silently fabricating results.
"""

from __future__ import annotations

import hashlib
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import yaml

from .case_bundle import BlobInfo, CaseBundle, _blob_info, _case_root, _manifest_blob_entry, _write_blob, _write_manifest
from .kernel_registry import lookup_kernel_id
from helia_core_tester.generation.utils.template_context import TemplateContextBuilder


class UnsupportedGeneratedTestError(Exception):
    """Raised when a generated test's operator/dtype isn't bridgeable to real firmware dispatch yet."""


# Must match HCT_SERVER_MAX_ARENA_BYTES in cmake/perf_stream/benchmark_server_session.h.
# The firmware's `session->case_arena` is a single fixed-size buffer shared by every
# streamed (non-host-only) blob *and* any scratch/weight-sum buffer for the case; a case
# whose total footprint exceeds this will fail `allocate_blob()` inside `handle_case_meta()`
# with HCTP_STATUS_INVALID_ARGUMENT (surfaced to the host as a CASE_META ERROR frame) --
# so it must be rejected here at bridge time with a clear skip reason instead of silently
# producing a CaseBundle the firmware will reject at runtime.
_ARENA_CAPACITY_BYTES = 49152

# Must match HCT_SERVER_MAX_OUTPUT_BYTES in cmake/perf_stream/benchmark_server_session.h.
# The firmware's `session->output_buffer` is a separate fixed-size buffer (not part of
# `case_arena`) that every kernel dispatch writes its correctness output into; a case
# whose output byte-length exceeds this is rejected deep inside the kernel-dispatch adapter
# itself (e.g. `output_length > sizeof(session->output_buffer)` in run_convolve_once/
# run_depthwise_conv_once/run_pooling_once) with a generic ARM_CMSIS_NN_ARG_ERROR that
# `handle_run_correctness` collapses to HCTP_STATUS_INVALID_ARGUMENT with no detail -- so,
# like the arena check above, this must be caught here at bridge time with a clear reason.
_OUTPUT_BUFFER_CAPACITY_BYTES = 8192


def _align_up(value: int, alignment: int) -> int:
    if alignment <= 1:
        return value
    return (value + alignment - 1) // alignment * alignment


def _check_case_arena_capacity(generated_test: GeneratedTestCase, manifest: dict, blobs: tuple[BlobInfo, ...]) -> None:
    """Simulate the firmware's `allocate_blob()` bump-allocator over every streamed
    (non-host-only) blob, plus any scratch buffer, plus (for ConvolutionFunctions/Convolve
    specifically) the extra weight-sum buffer `arm_convolve_weight_sum()` needs -- and raise
    UnsupportedGeneratedTestError if the simulated total would exceed the firmware's fixed
    `HCT_SERVER_MAX_ARENA_BYTES` case arena.
    """
    used = 0
    for blob in blobs:
        if blob.host_only:
            continue
        used = _align_up(used, max(int(blob.required_alignment), 1)) + int(blob.byte_length)
    scratch_bytes = int(manifest.get("scratch_buffer", {}).get("bytes", 0))
    if scratch_bytes > 0:
        used = _align_up(used, 16) + scratch_bytes
    if manifest.get("family") == "ConvolutionFunctions" and manifest.get("operator") == "Convolve":
        output_c = int(manifest.get("serialized_scalar_parameters", {}).get("output_c", 0))
        used = _align_up(used, 16) + output_c * 4
    if used > _ARENA_CAPACITY_BYTES:
        raise UnsupportedGeneratedTestError(
            f"{generated_test.name}: estimated case-arena footprint ({used} bytes) exceeds the "
            f"firmware's fixed HCT_SERVER_MAX_ARENA_BYTES ({_ARENA_CAPACITY_BYTES} bytes) -- this "
            f"case's blobs/scratch would not fit in a single hardware session and would be "
            f"rejected by allocate_blob() at CASE_META time."
        )
    output_bytes = int(manifest.get("expected_output", {}).get("byte_length", 0))
    if output_bytes > _OUTPUT_BUFFER_CAPACITY_BYTES:
        raise UnsupportedGeneratedTestError(
            f"{generated_test.name}: expected output byte-length ({output_bytes} bytes) exceeds the "
            f"firmware's fixed HCT_SERVER_MAX_OUTPUT_BYTES ({_OUTPUT_BUFFER_CAPACITY_BYTES} bytes) -- "
            f"this case's output would not fit in the firmware's single output_buffer and would be "
            f"rejected by the kernel-dispatch adapter's output_length bounds check at runtime."
        )


@dataclass(frozen=True)
class GeneratedTestCase:
    """A discovered generated-test directory paired with its parsed descriptor."""

    name: str
    cpu: str
    family: str
    directory: Path
    descriptor: dict


_INT_ARRAY_RE = re.compile(r"=\s*\{([^}]*)\}")


def _find_header_file(directory: Path) -> Path:
    includes_dir = directory / "includes"
    candidates = sorted(includes_dir.glob("*.h")) if includes_dir.is_dir() else []
    if not candidates:
        candidates = sorted(directory.glob("*.h"))
    if not candidates:
        raise UnsupportedGeneratedTestError(f"No generated header (.h) found under {directory}")
    return candidates[0]


def _find_source_file(directory: Path) -> Path:
    candidates = sorted(directory.glob("*.c"))
    if not candidates:
        raise UnsupportedGeneratedTestError(f"No generated source (.c) found under {directory}")
    return candidates[0]


def _extract_call_args(source_text: str, function_name: str, *, expected_count: int) -> list[str]:
    """Extract the positional argument expressions from the first `function_name(...)` call
    in `source_text` (as generated by the standalone test-harness templates).

    Elementwise ops (unlike Convolve's `cmsis_nn_conv_params` struct) don't have a named
    scalar-params struct in the generated header -- their quant scalars are inlined directly
    as call arguments (with `// name` comments) in the generated `.c` file. This is
    positional/fragile by nature (relies on the generator's fixed CMSIS-NN argument order),
    so callers must pass `expected_count` to fail loudly on any drift instead of silently
    misreading arguments.
    """
    pattern = re.compile(rf"\b{re.escape(function_name)}\s*\((.*?)\);", re.DOTALL)
    match = pattern.search(source_text)
    if match is None:
        raise UnsupportedGeneratedTestError(f"Could not find call to `{function_name}(...)` in generated source")
    body = re.sub(r"//[^\n]*", "", match.group(1))
    args = [a.strip() for a in body.split(",") if a.strip() != ""]
    if len(args) != expected_count:
        raise UnsupportedGeneratedTestError(
            f"Call to `{function_name}(...)` has {len(args)} arguments, expected {expected_count} -- "
            f"the generator's argument order/signature may have changed; positional extraction is unsafe."
        )
    return args


def _extract_array(header_text: str, array_name: str) -> list[int]:
    pattern = re.compile(rf"\b{re.escape(array_name)}\s*(?:\[[^\]]*\])?\s*=\s*\{{(.*?)\}}\s*;", re.DOTALL)
    match = pattern.search(header_text)
    if match is None:
        raise UnsupportedGeneratedTestError(f"Could not find array `{array_name}` in generated header")
    raw = match.group(1)
    values = [v.strip() for v in raw.replace("\n", " ").split(",") if v.strip() != ""]
    return [int(v) for v in values]


def _extract_float_array(header_text: str, array_name: str) -> list[float]:
    """Same as `_extract_array()` but for float32 arrays (values written with an `f` suffix,
    e.g. `0.133486f`) -- used by Quantize (float input) and Dequantize (float expected
    output)."""
    pattern = re.compile(rf"\b{re.escape(array_name)}\s*(?:\[[^\]]*\])?\s*=\s*\{{(.*?)\}}\s*;", re.DOTALL)
    match = pattern.search(header_text)
    if match is None:
        raise UnsupportedGeneratedTestError(f"Could not find array `{array_name}` in generated header")
    raw = match.group(1)
    values = [v.strip() for v in raw.replace("\n", " ").split(",") if v.strip() != ""]
    return [float(v.rstrip("fF")) for v in values]


def _extract_scalar(header_text: str, struct_name: str, field: str) -> int:
    struct_pattern = re.compile(rf"\b{re.escape(struct_name)}\s*=\s*\{{(.*?)\}}\s*;", re.DOTALL)
    struct_match = struct_pattern.search(header_text)
    if struct_match is None:
        raise UnsupportedGeneratedTestError(f"Could not find struct `{struct_name}` in generated header")
    body = struct_match.group(1)
    field_pattern = re.compile(rf"\.{re.escape(field)}\s*=\s*(-?\d+)")
    field_match = field_pattern.search(body)
    if field_match is None:
        raise UnsupportedGeneratedTestError(f"Could not find field `.{field}` on struct `{struct_name}`")
    return int(field_match.group(1))


def _extract_bare_scalar(header_text: str, variable_name: str) -> int | None:
    """Extract a plain `static int32_t <variable_name> = <value>;` declaration (as opposed
    to `_extract_array()`'s `<name>[] = {...}` or `_extract_scalar()`'s struct-field form).
    Used by FullyConnected, whose per-tensor-quantized descriptors emit a bare
    `..._multiplier_val`/`..._shift_val` scalar instead of a `..._multiplier[]`/`..._shift[]`
    array. Returns None (rather than raising) when not found, so callers can fall back to
    `_extract_array()` for the per-channel case.
    """
    pattern = re.compile(rf"\b{re.escape(variable_name)}\s*=\s*(-?\d+)\s*;")
    match = pattern.search(header_text)
    return int(match.group(1)) if match is not None else None


def _extract_define_int(source_text: str, name: str) -> int:
    pattern = re.compile(rf"^\s*#define\s+{re.escape(name)}\s+\(?(-?\d+)\)?\s*$", re.MULTILINE)
    match = pattern.search(source_text)
    if match is None:
        raise UnsupportedGeneratedTestError(f"Could not find #define `{name}` in generated source")
    return int(match.group(1))


def _extract_null_pointer_decl(header_text: str, variable_name: str) -> bool:
    pattern = re.compile(rf"\b{re.escape(variable_name)}\s*=\s*NULL\s*;")
    return pattern.search(header_text) is not None


def _extract_dims(header_text: str, struct_name: str) -> dict[str, int]:
    struct_pattern = re.compile(rf"\b{re.escape(struct_name)}\s*=\s*\{{(.*?)\}}\s*;", re.DOTALL)
    match = struct_pattern.search(header_text)
    if match is None:
        raise UnsupportedGeneratedTestError(f"Could not find dims struct `{struct_name}` in generated header")
    body = match.group(1)
    dims: dict[str, int] = {}
    for field in ("n", "h", "w", "c"):
        field_match = re.search(rf"\.{field}\s*=\s*(-?\d+)", body)
        if field_match is None:
            raise UnsupportedGeneratedTestError(f"Dims struct `{struct_name}` missing field `.{field}`")
        dims[field] = int(field_match.group(1))
    return dims


def _extract_nested_scalar(header_text: str, struct_name: str, nested_field: str, field: str) -> int:
    """Extract `.field` from a nested sub-struct inside `struct_name`. Two nested-struct
    initializer styles are seen across generated headers: braced (e.g.
    `.padding = {.w = 0, .h = 0}`, used by Convolve/DepthwiseConv's cmsis_nn_conv_params/
    cmsis_nn_dw_conv_params) and flattened-dotted (e.g. `.padding.w = 0`, used by
    Pooling's cmsis_nn_pool_params) -- both are tried here since `struct_name` may have
    multiple nested `{w, h}`/`{min, max}` sub-fields, so a flat search for `.h`/`.w`/`.min`
    across the whole struct body could match the wrong one either way.
    """
    struct_pattern = re.compile(rf"\b{re.escape(struct_name)}\s*=\s*\{{(.*?)\}}\s*;", re.DOTALL)
    struct_match = struct_pattern.search(header_text)
    if struct_match is None:
        raise UnsupportedGeneratedTestError(f"Could not find struct `{struct_name}` in generated header")
    body = struct_match.group(1)

    dotted_pattern = re.compile(rf"\.{re.escape(nested_field)}\.{re.escape(field)}\s*=\s*(-?\d+)")
    dotted_match = dotted_pattern.search(body)
    if dotted_match is not None:
        return int(dotted_match.group(1))

    nested_pattern = re.compile(rf"\.{re.escape(nested_field)}\s*=\s*\{{([^}}]*)\}}")
    nested_match = nested_pattern.search(body)
    if nested_match is None:
        raise UnsupportedGeneratedTestError(f"Could not find nested field `.{nested_field}` on struct `{struct_name}`")
    nested_body = nested_match.group(1)
    field_pattern = re.compile(rf"\.{re.escape(field)}\s*=\s*(-?\d+)")
    field_match = field_pattern.search(nested_body)
    if field_match is None:
        raise UnsupportedGeneratedTestError(
            f"Could not find field `.{field}` on nested `.{nested_field}` of struct `{struct_name}`"
        )
    return int(field_match.group(1))


def discover_generated_tests(
    project_root: Path,
    *,
    cpu: str = "cortex-m55",
    family: str = "ConvolutionFunctions",
    name_filter: str | None = None,
    limit: int | None = None,
) -> list[GeneratedTestCase]:
    """Discover generated-test directories with a parseable descriptor.yaml under artifacts/generated_tests."""
    root = project_root / "artifacts" / "generated_tests" / "int" / cpu / family
    if not root.is_dir():
        return []
    results: list[GeneratedTestCase] = []
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        descriptor_path = directory / "descriptor.yaml"
        if not descriptor_path.is_file():
            continue
        name = directory.name
        if name_filter is not None and name_filter not in name:
            continue
        descriptor = yaml.safe_load(descriptor_path.read_text(encoding="utf-8"))
        results.append(GeneratedTestCase(name=name, cpu=cpu, family=family, directory=directory, descriptor=descriptor))
        if limit is not None and len(results) >= limit:
            break
    return results


def build_case_bundle_from_generated_test(
    project_root: Path,
    generated_test: GeneratedTestCase,
    *,
    output_root: Path | None = None,
) -> CaseBundle:
    """Convert one real generated CMSIS-NN kernel test (with its real golden data) into a
    streamable perf-stream CaseBundle. Dispatches to a per-(family, operator) builder
    registered in `_BUILDERS` -- each builder owns its own header/source extraction logic.
    """
    descriptor = generated_test.descriptor
    operator = str(descriptor.get("operator", ""))
    builder = _BUILDERS.get((generated_test.family, operator))
    if builder is None:
        raise UnsupportedGeneratedTestError(
            f"{generated_test.name}: family/operator '{generated_test.family}/{operator}' has no real "
            f"perf-stream firmware dispatch yet (bridged today: "
            f"{sorted(f'{fam}/{op}' for fam, op in _BUILDERS)})."
        )
    bundle = builder(project_root, generated_test, output_root=output_root)
    _check_case_arena_capacity(generated_test, bundle.manifest, bundle.blobs)
    return bundle


def _build_convolve_case(
    project_root: Path,
    generated_test: GeneratedTestCase,
    *,
    output_root: Path | None = None,
) -> CaseBundle:
    """Convert one real generated CMSIS-NN Convolve test (with its real golden data) into a
    streamable perf-stream CaseBundle, reusing the actual generator-produced input/weights/
    bias/expected_output arrays rather than fabricating new ones.
    """
    descriptor = generated_test.descriptor
    operator = str(descriptor.get("operator", ""))
    weight_dtype = str(descriptor.get("weight_dtype", descriptor.get("resolved_tensor_dtypes", {}).get("weights", "")))
    activation_dtype = str(descriptor.get("activation_dtype", ""))

    if weight_dtype != "S8" or activation_dtype not in ("S8", "S16"):
        raise UnsupportedGeneratedTestError(
            f"{generated_test.name}: weight_dtype={weight_dtype!r} activation_dtype={activation_dtype!r} "
            f"is not bridgeable -- perf-stream firmware only dispatches arm_convolve_s8/arm_convolve_wrapper_s16 "
            f"(S8 weight + S8 or S16 activation)."
        )

    header_path = _find_header_file(generated_test.directory)
    header_text = header_path.read_text(encoding="utf-8")
    prefix = generated_test.name

    input_dims = _extract_dims(header_text, f"{prefix}_input_dims")
    filter_dims = _extract_dims(header_text, f"{prefix}_filter_dims")
    output_dims = _extract_dims(header_text, f"{prefix}_output_dims")
    if input_dims["n"] != 1:
        raise UnsupportedGeneratedTestError(
            f"{generated_test.name}: batch size {input_dims['n']} > 1 is not yet supported by the "
            f"perf-stream hardware bridge (firmware dispatches a single arm_convolve_s8 invocation per case)."
        )
    input_shape = (input_dims["n"], input_dims["h"], input_dims["w"], input_dims["c"])
    filter_shape = (filter_dims["h"], filter_dims["w"], filter_dims["c"], filter_dims["n"])
    output_shape = (output_dims["n"], output_dims["h"], output_dims["w"], output_dims["c"])
    output_channels = filter_dims["n"]

    activation_numpy_dtype = np.int16 if activation_dtype == "S16" else np.int8
    bias_numpy_dtype = np.int64 if activation_dtype == "S16" else np.int32
    bias_wire_dtype = "S64" if activation_dtype == "S16" else "S32"

    weights_flat = np.array(_extract_array(header_text, f"{prefix}_weights"), dtype=np.int8)
    biases = np.array(_extract_array(header_text, f"{prefix}_biases"), dtype=bias_numpy_dtype)
    input_flat = np.array(_extract_array(header_text, f"{prefix}_input"), dtype=activation_numpy_dtype)
    expected_flat = np.array(_extract_array(header_text, f"{prefix}_expected_output"), dtype=activation_numpy_dtype)
    multiplier = np.array(_extract_array(header_text, f"{prefix}_multiplier"), dtype=np.int32)
    shift = np.array(_extract_array(header_text, f"{prefix}_shift"), dtype=np.int32)

    if input_flat.size != int(np.prod(input_shape)) or weights_flat.size != int(np.prod(filter_shape)):
        raise UnsupportedGeneratedTestError(
            f"{generated_test.name}: generated array sizes (input={input_flat.size}, weights={weights_flat.size}) "
            f"don't match header dims (input_shape={input_shape}, filter_shape={filter_shape}) -- the standalone "
            f"test harness likely loops over an outer batch/tiling dimension not reflected in these dims structs, "
            f"which the perf-stream single-invocation bridge doesn't replicate yet."
        )
    weights = weights_flat.reshape(filter_shape)
    input_data = input_flat.reshape(input_shape)

    input_offset = _extract_scalar(header_text, f"{prefix}_conv_params", "input_offset")
    output_offset = _extract_scalar(header_text, f"{prefix}_conv_params", "output_offset")
    activation_min = _extract_scalar(header_text, f"{prefix}_conv_params", "min")
    activation_max = _extract_scalar(header_text, f"{prefix}_conv_params", "max")
    strides = tuple(int(v) for v in descriptor["strides"])
    padding = str(descriptor["padding"])
    # Ground-truth padding actually used to generate the reference output, read directly
    # from the generated header rather than re-derived from the VALID/SAME keyword above.
    # Firmware must use these (and output_dims below) verbatim -- re-deriving them from a
    # SAME/VALID formula can silently diverge from the real generator's padding/output-size
    # convention (asymmetric splits, rounding, etc.), corrupting both output size and values.
    pad_h = _extract_nested_scalar(header_text, f"{prefix}_conv_params", "padding", "h")
    pad_w = _extract_nested_scalar(header_text, f"{prefix}_conv_params", "padding", "w")
    # Ground-truth dilation, likewise read directly from the generated header rather than
    # assumed to be 1 -- firmware previously hardcoded dilation.w/h=1 unconditionally, which
    # silently produced wrong output for any real generated test with dilation != 1.
    dilation_h = _extract_nested_scalar(header_text, f"{prefix}_conv_params", "dilation", "h")
    dilation_w = _extract_nested_scalar(header_text, f"{prefix}_conv_params", "dilation", "w")

    if expected_flat.size != int(np.prod(output_shape)):
        raise UnsupportedGeneratedTestError(
            f"{generated_test.name}: expected_output size ({expected_flat.size}) does not match header "
            f"output_dims product ({int(np.prod(output_shape))})"
        )
    expected_output = expected_flat.reshape(output_shape)

    input_dims_dict = input_dims
    filter_dims_dict = {"h": filter_dims["h"], "w": filter_dims["w"], "c": filter_dims["c"], "n": filter_dims["n"]}
    output_dims_dict = output_dims
    scratch_bytes = TemplateContextBuilder.calculate_buffer_size_max(
        input_dims_dict, filter_dims_dict, output_dims_dict, output_dtype=activation_dtype
    )

    case_id = f"{generated_test.name}_hw_generated"
    bundle_root = output_root if output_root is not None else project_root
    case_root = _case_root(bundle_root, "ConvolutionFunctions", case_id)
    blobs_dir = case_root / "blobs"
    blobs_dir.mkdir(parents=True, exist_ok=True)

    arrays = [
        (1, "input_0", activation_dtype, input_shape, input_data, False, False),
        (2, "weights", "S8", filter_shape, weights, False, False),
        (3, "bias", bias_wire_dtype, (output_channels,), biases, False, False),
        (4, "multiplier", "S32", (output_channels,), multiplier, False, False),
        (5, "shift", "S32", (output_channels,), shift, False, False),
        (6, "expected_output", activation_dtype, tuple(int(v) for v in expected_output.shape), expected_output, False, True),
    ]
    blobs: list[BlobInfo] = []
    for blob_id, role, dtype, dims, array, mutable_data, host_only in arrays:
        path = blobs_dir / f"{role}.bin"
        _write_blob(path, np.asarray(array))
        blobs.append(_blob_info(path, blob_id=blob_id, role=role, dtype=dtype, dimensions=dims, mutable_data=mutable_data, host_only=host_only))

    descriptor_path = generated_test.directory / "descriptor.yaml"
    descriptor_text = descriptor_path.read_text(encoding="utf-8")
    comparison = {"mode": "tolerant_int", "tolerance": 1} if operator == "Mean" else dict(
        descriptor.get("resolved_comparison", {"mode": "exact_int"})
    )
    manifest = {
        "schema_name": "hct.case_manifest",
        "schema_version": 1,
        "case_id": case_id,
        "descriptor_name": generated_test.name,
        "descriptor_path": str(descriptor_path.relative_to(project_root)) if descriptor_path.is_relative_to(project_root) else str(descriptor_path),
        "descriptor_sha256": hashlib.sha256(descriptor_text.encode("utf-8")).hexdigest(),
        "operator": operator,
        "family": "ConvolutionFunctions",
        "target_cpu": generated_test.cpu,
        "kernel_id": lookup_kernel_id(project_root, family="ConvolutionFunctions", operator="Convolve", dtype=activation_dtype),
        "adapter_metadata_schema": 1,
        "source": "generated_test_bridge",
        "serialized_scalar_parameters": {
            "stride_h": strides[0],
            "stride_w": strides[1],
            "padding": padding,
            "pad_h": pad_h,
            "pad_w": pad_w,
            "dilation_h": dilation_h,
            "dilation_w": dilation_w,
            "output_h": output_dims["h"],
            "output_w": output_dims["w"],
            "output_c": output_dims["c"],
            "input_offset": input_offset,
            "output_offset": output_offset,
            "activation_min": activation_min,
            "activation_max": activation_max,
        },
        "tensor_dtypes": {"input": activation_dtype, "weights": "S8", "bias": bias_wire_dtype, "output": activation_dtype},
        "blob_roles": [_manifest_blob_entry(blob) for blob in blobs],
        "expected_output": {"dtype": activation_dtype, "byte_length": blobs[-1].byte_length, "blob_id": blobs[-1].blob_id},
        "correctness_comparison": comparison,
        "scratch_buffer": {"bytes": int(scratch_bytes)},
        "required_target_capabilities": ["convolve_s8" if activation_dtype == "S8" else "convolve_s16"],
        "repeated_invocation_safe": True,
        "timing": {"warmups": 2, "samples": 5, "iterations_per_sample": 4, "min_cycles": 1024, "max_iterations": 256},
    }
    return CaseBundle(root_dir=case_root, manifest_path=_write_manifest(case_root, manifest), manifest=manifest, blobs=tuple(blobs))


def _build_depthwise_conv_case(
    project_root: Path,
    generated_test: GeneratedTestCase,
    *,
    output_root: Path | None = None,
) -> CaseBundle:
    """Convert one real generated CMSIS-NN DepthwiseConv test (S8 or S16 activation) into
    a streamable perf-stream CaseBundle. The S8 activation path uses the low-level
    `arm_depthwise_conv_s8` (not the `_wrapper_s8` variant) since it needs no scratch
    buffer at all (`ctx` is unused -- see
    `Source/ConvolutionFunctions/arm_depthwise_conv_s8.c`'s `(void)ctx;`) and no
    weight-sum precomputation, unlike Convolve's `arm_convolve_s8`. The S16 activation
    path uses `arm_depthwise_conv_wrapper_s16`, which does need a real scratch buffer
    (sized via `arm_depthwise_conv_wrapper_s16_get_buffer_size`) and takes a plain
    `int64_t*` bias pointer (not S16 Convolve's `cmsis_nn_bias_data`-wrapped bias). Filter
    dims are kept in the header's native (N, H, W, C_OUT) order (unlike Convolve's
    builder, which reorders filter dims to (H, W, C, N) to match `cmsis_nn_dims`
    filter-dims convention for the full-conv kernel -- depthwise's
    `cmsis_nn_dw_conv_params`-based filter format is already (1, H, W, C_OUT), so no
    reordering is needed).
    """
    descriptor = generated_test.descriptor
    operator = str(descriptor.get("operator", ""))
    weight_dtype = str(descriptor.get("weight_dtype", descriptor.get("resolved_tensor_dtypes", {}).get("weights", "")))
    activation_dtype = str(descriptor.get("activation_dtype", ""))

    if weight_dtype != "S8" or activation_dtype not in ("S8", "S16"):
        raise UnsupportedGeneratedTestError(
            f"{generated_test.name}: weight_dtype={weight_dtype!r} activation_dtype={activation_dtype!r} "
            f"is not bridgeable -- perf-stream firmware only dispatches "
            f"arm_depthwise_conv_s8/arm_depthwise_conv_wrapper_s16 (S8 weight + S8 or S16 activation)."
        )

    header_path = _find_header_file(generated_test.directory)
    header_text = header_path.read_text(encoding="utf-8")
    prefix = generated_test.name

    input_dims = _extract_dims(header_text, f"{prefix}_input_dims")
    filter_dims = _extract_dims(header_text, f"{prefix}_filter_dims")
    output_dims = _extract_dims(header_text, f"{prefix}_output_dims")
    if input_dims["n"] != 1:
        raise UnsupportedGeneratedTestError(
            f"{generated_test.name}: batch size {input_dims['n']} > 1 is not yet supported by the "
            f"perf-stream hardware bridge (firmware dispatches a single arm_depthwise_conv_s8 "
            f"invocation per case)."
        )
    input_shape = (input_dims["n"], input_dims["h"], input_dims["w"], input_dims["c"])
    # Native (N, H, W, C_OUT) order, per arm_depthwise_conv_s8's filter_dims docstring.
    filter_shape = (filter_dims["n"], filter_dims["h"], filter_dims["w"], filter_dims["c"])
    output_shape = (output_dims["n"], output_dims["h"], output_dims["w"], output_dims["c"])
    output_channels = output_dims["c"]

    weights_flat = np.array(_extract_array(header_text, f"{prefix}_weights"), dtype=np.int8)
    activation_numpy_dtype = np.int16 if activation_dtype == "S16" else np.int8
    bias_numpy_dtype = np.int64 if activation_dtype == "S16" else np.int32
    bias_wire_dtype = "S64" if activation_dtype == "S16" else "S32"
    biases = np.array(_extract_array(header_text, f"{prefix}_biases"), dtype=bias_numpy_dtype)
    input_flat = np.array(_extract_array(header_text, f"{prefix}_input"), dtype=activation_numpy_dtype)
    expected_flat = np.array(_extract_array(header_text, f"{prefix}_expected_output"), dtype=activation_numpy_dtype)
    multiplier = np.array(_extract_array(header_text, f"{prefix}_multiplier"), dtype=np.int32)
    shift = np.array(_extract_array(header_text, f"{prefix}_shift"), dtype=np.int32)

    if input_flat.size != int(np.prod(input_shape)) or weights_flat.size != int(np.prod(filter_shape)):
        raise UnsupportedGeneratedTestError(
            f"{generated_test.name}: generated array sizes (input={input_flat.size}, weights={weights_flat.size}) "
            f"don't match header dims (input_shape={input_shape}, filter_shape={filter_shape})."
        )
    weights = weights_flat.reshape(filter_shape)
    input_data = input_flat.reshape(input_shape)

    if expected_flat.size != int(np.prod(output_shape)):
        raise UnsupportedGeneratedTestError(
            f"{generated_test.name}: expected_output size ({expected_flat.size}) does not match header "
            f"output_dims product ({int(np.prod(output_shape))})"
        )
    expected_output = expected_flat.reshape(output_shape)

    params_struct = f"{prefix}_dw_conv_params"
    input_offset = _extract_scalar(header_text, params_struct, "input_offset")
    output_offset = _extract_scalar(header_text, params_struct, "output_offset")
    ch_mult = _extract_scalar(header_text, params_struct, "ch_mult")
    activation_min = _extract_nested_scalar(header_text, params_struct, "activation", "min")
    activation_max = _extract_nested_scalar(header_text, params_struct, "activation", "max")
    stride_h = _extract_nested_scalar(header_text, params_struct, "stride", "h")
    stride_w = _extract_nested_scalar(header_text, params_struct, "stride", "w")
    pad_h = _extract_nested_scalar(header_text, params_struct, "padding", "h")
    pad_w = _extract_nested_scalar(header_text, params_struct, "padding", "w")
    dilation_h = _extract_nested_scalar(header_text, params_struct, "dilation", "h")
    dilation_w = _extract_nested_scalar(header_text, params_struct, "dilation", "w")

    case_id = f"{generated_test.name}_hw_generated"
    bundle_root = output_root if output_root is not None else project_root
    case_root = _case_root(bundle_root, "ConvolutionFunctions", case_id)
    blobs_dir = case_root / "blobs"
    blobs_dir.mkdir(parents=True, exist_ok=True)

    # arm_depthwise_conv_s8 needs no scratch buffer at all (see docstring above); the S16
    # wrapper does, and its size depends on activation_dtype -- see
    # calculate_depthwise_buffer_size_max()'s S8/S16 branches.
    scratch_bytes = (
        TemplateContextBuilder.calculate_depthwise_buffer_size_max(input_dims, filter_dims, output_dims, output_dtype="S16")
        if activation_dtype == "S16"
        else 0
    )

    arrays = [
        (1, "input_0", activation_dtype, input_shape, input_data, False, False),
        (2, "weights", "S8", filter_shape, weights, False, False),
        (3, "bias", bias_wire_dtype, (output_channels,), biases, False, False),
        (4, "multiplier", "S32", (output_channels,), multiplier, False, False),
        (5, "shift", "S32", (output_channels,), shift, False, False),
        (6, "expected_output", activation_dtype, tuple(int(v) for v in expected_output.shape), expected_output, False, True),
    ]
    blobs: list[BlobInfo] = []
    for blob_id, role, dtype, dims, array, mutable_data, host_only in arrays:
        path = blobs_dir / f"{role}.bin"
        _write_blob(path, np.asarray(array))
        blobs.append(_blob_info(path, blob_id=blob_id, role=role, dtype=dtype, dimensions=dims, mutable_data=mutable_data, host_only=host_only))

    descriptor_path = generated_test.directory / "descriptor.yaml"
    descriptor_text = descriptor_path.read_text(encoding="utf-8")
    comparison = {"mode": "tolerant_int", "tolerance": 1} if operator == "Mean" else dict(
        descriptor.get("resolved_comparison", {"mode": "exact_int"})
    )
    manifest = {
        "schema_name": "hct.case_manifest",
        "schema_version": 1,
        "case_id": case_id,
        "descriptor_name": generated_test.name,
        "descriptor_path": str(descriptor_path.relative_to(project_root)) if descriptor_path.is_relative_to(project_root) else str(descriptor_path),
        "descriptor_sha256": hashlib.sha256(descriptor_text.encode("utf-8")).hexdigest(),
        "operator": operator,
        "family": "ConvolutionFunctions",
        "target_cpu": generated_test.cpu,
        "kernel_id": lookup_kernel_id(project_root, family="ConvolutionFunctions", operator="DepthwiseConv", dtype=activation_dtype),
        "adapter_metadata_schema": 1,
        "source": "generated_test_bridge",
        "serialized_scalar_parameters": {
            "stride_h": stride_h,
            "stride_w": stride_w,
            "pad_h": pad_h,
            "pad_w": pad_w,
            "dilation_h": dilation_h,
            "dilation_w": dilation_w,
            "output_h": output_dims["h"],
            "output_w": output_dims["w"],
            "output_c": output_dims["c"],
            "input_offset": input_offset,
            "output_offset": output_offset,
            "activation_min": activation_min,
            "activation_max": activation_max,
            "ch_mult": ch_mult,
        },
        "tensor_dtypes": {"input": activation_dtype, "weights": "S8", "bias": bias_wire_dtype, "output": activation_dtype},
        "blob_roles": [_manifest_blob_entry(blob) for blob in blobs],
        "expected_output": {"dtype": activation_dtype, "byte_length": blobs[-1].byte_length, "blob_id": blobs[-1].blob_id},
        "correctness_comparison": comparison,
        "scratch_buffer": {"bytes": int(scratch_bytes)},
        "required_target_capabilities": ["depthwise_conv_s8" if activation_dtype == "S8" else "depthwise_conv_s16"],
        "repeated_invocation_safe": True,
        "timing": {"warmups": 2, "samples": 5, "iterations_per_sample": 4, "min_cycles": 1024, "max_iterations": 256},
    }
    return CaseBundle(root_dir=case_root, manifest_path=_write_manifest(case_root, manifest), manifest=manifest, blobs=tuple(blobs))


def _build_transpose_conv_case(
    project_root: Path,
    generated_test: GeneratedTestCase,
    *,
    output_root: Path | None = None,
) -> CaseBundle:
    descriptor = generated_test.descriptor
    operator = str(descriptor.get("operator", ""))
    activation_dtype = str(descriptor.get("activation_dtype", ""))
    weight_dtype = str(descriptor.get("weight_dtype", descriptor.get("resolved_tensor_dtypes", {}).get("weights", "")))
    if operator != "TransposeConv" or activation_dtype != "S8" or weight_dtype != "S8":
        raise UnsupportedGeneratedTestError(
            f"{generated_test.name}: weight_dtype={weight_dtype!r} activation_dtype={activation_dtype!r} is not "
            "bridgeable -- perf-stream firmware only dispatches arm_transpose_conv_wrapper_s8."
        )

    header_path = _find_header_file(generated_test.directory)
    header_text = header_path.read_text(encoding="utf-8")
    source_path = _find_source_file(generated_test.directory)
    source_text = source_path.read_text(encoding="utf-8")
    prefix = generated_test.name
    upper_prefix = prefix.upper()

    input_dims = _extract_dims(header_text, f"{prefix}_input_dims")
    filter_dims = _extract_dims(header_text, f"{prefix}_filter_dims")
    output_dims = _extract_dims(header_text, f"{prefix}_output_dims")
    if input_dims["n"] != 1 or output_dims["n"] != 1:
        raise UnsupportedGeneratedTestError(
            f"{generated_test.name}: batch size > 1 is not yet supported by the perf-stream hardware bridge."
        )
    input_shape = (input_dims["n"], input_dims["h"], input_dims["w"], input_dims["c"])
    filter_shape = (filter_dims["n"], filter_dims["h"], filter_dims["w"], filter_dims["c"])
    output_shape = (output_dims["n"], output_dims["h"], output_dims["w"], output_dims["c"])
    output_channels = output_dims["c"]

    input_flat = np.array(_extract_array(header_text, f"{prefix}_input"), dtype=np.int8)
    weights_flat = np.array(_extract_array(header_text, f"{prefix}_weights"), dtype=np.int8)
    expected_flat = np.array(_extract_array(header_text, f"{prefix}_expected_output"), dtype=np.int8)
    multiplier = np.array(_extract_array(header_text, f"{prefix}_multiplier"), dtype=np.int32)
    shift = np.array(_extract_array(header_text, f"{prefix}_shift"), dtype=np.int32)
    expected_input_size = int(np.prod(input_shape))
    expected_filter_size = int(np.prod(filter_shape))
    expected_output_size = int(np.prod(output_shape))
    if input_flat.size < expected_input_size or weights_flat.size < expected_filter_size:
        raise UnsupportedGeneratedTestError(
            f"{generated_test.name}: generated array sizes (input={input_flat.size}, weights={weights_flat.size}) "
            f"don't match header dims (input_shape={input_shape}, filter_shape={filter_shape})."
        )
    if expected_flat.size < expected_output_size:
        raise UnsupportedGeneratedTestError(
            f"{generated_test.name}: expected_output size ({expected_flat.size}) does not match output dims ({output_shape})."
        )
    # Some generated TransposeConv headers currently emit a trailing duplicate block after the
    # real tensor payload while still validating only the first NHWC-sized slice in the
    # standalone harness (the output-size macros in the generated C source remain the true
    # dims product). Mirror that real harness behavior here by truncating any oversized
    # arrays to the dims-declared payload instead of rejecting otherwise-dispatchable cases.
    input_flat = input_flat[:expected_input_size]
    weights_flat = weights_flat[:expected_filter_size]
    expected_flat = expected_flat[:expected_output_size]
    if multiplier.size != output_channels or shift.size != output_channels:
        raise UnsupportedGeneratedTestError(
            f"{generated_test.name}: quant array sizes (multiplier={multiplier.size}, shift={shift.size}) do not "
            f"match output channels ({output_channels})."
        )

    params_struct = f"{prefix}_transpose_conv_params"
    input_offset = _extract_scalar(header_text, params_struct, "input_offset")
    output_offset = _extract_scalar(header_text, params_struct, "output_offset")
    stride_h = _extract_nested_scalar(header_text, params_struct, "stride", "h")
    stride_w = _extract_nested_scalar(header_text, params_struct, "stride", "w")
    dilation_h = _extract_nested_scalar(header_text, params_struct, "dilation", "h")
    dilation_w = _extract_nested_scalar(header_text, params_struct, "dilation", "w")
    pad_h = _extract_nested_scalar(header_text, params_struct, "padding", "h")
    pad_w = _extract_nested_scalar(header_text, params_struct, "padding", "w")
    pad_offset_h = _extract_nested_scalar(header_text, params_struct, "padding_offsets", "h")
    pad_offset_w = _extract_nested_scalar(header_text, params_struct, "padding_offsets", "w")
    activation_min = _extract_nested_scalar(header_text, params_struct, "activation", "min")
    activation_max = _extract_nested_scalar(header_text, params_struct, "activation", "max")

    has_bias = not _extract_null_pointer_decl(header_text, f"{prefix}_biases")
    biases = np.array(_extract_array(header_text, f"{prefix}_biases"), dtype=np.int32) if has_bias else None
    if has_bias and biases.size != output_channels:
        raise UnsupportedGeneratedTestError(
            f"{generated_test.name}: bias array size ({biases.size}) does not match output channels ({output_channels})."
        )

    ctx_upper = _extract_define_int(source_text, f"{upper_prefix}_BUFFER_SIZE_MAX")
    reverse_upper = _extract_define_int(source_text, f"{upper_prefix}_REVERSE_CONV_CTX_SIZE")
    weight_sum_bytes = output_channels * 4
    scratch_bytes = int(_align_up(_align_up(ctx_upper, 16) + reverse_upper, 16) + weight_sum_bytes)

    input_data = input_flat.reshape(input_shape)
    weights = weights_flat.reshape(filter_shape)
    expected_output = expected_flat.reshape(output_shape)

    case_id = f"{generated_test.name}_hw_generated"
    bundle_root = output_root if output_root is not None else project_root
    case_root = _case_root(bundle_root, "ConvolutionFunctions", case_id)
    blobs_dir = case_root / "blobs"
    blobs_dir.mkdir(parents=True, exist_ok=True)

    arrays: list[tuple[int, str, str, tuple[int, ...], np.ndarray, bool, bool]] = [
        (1, "input_0", "S8", input_shape, input_data, False, False),
        (2, "weights", "S8", filter_shape, weights, False, False),
        (3, "multiplier", "S32", (output_channels,), multiplier, False, False),
        (4, "shift", "S32", (output_channels,), shift, False, False),
    ]
    if has_bias and biases is not None:
        arrays.append((5, "bias", "S32", (output_channels,), biases, False, False))
        expected_blob_id = 6
    else:
        expected_blob_id = 5
    arrays.append((expected_blob_id, "expected_output", "S8", output_shape, expected_output, False, True))

    blobs: list[BlobInfo] = []
    for blob_id, role, dtype, dims, array, mutable_data, host_only in arrays:
        path = blobs_dir / f"{role}.bin"
        _write_blob(path, np.asarray(array))
        blobs.append(_blob_info(path, blob_id=blob_id, role=role, dtype=dtype, dimensions=dims, mutable_data=mutable_data, host_only=host_only))

    descriptor_path = generated_test.directory / "descriptor.yaml"
    descriptor_text = descriptor_path.read_text(encoding="utf-8")
    manifest = {
        "schema_name": "hct.case_manifest",
        "schema_version": 1,
        "case_id": case_id,
        "descriptor_name": generated_test.name,
        "descriptor_path": str(descriptor_path.relative_to(project_root)) if descriptor_path.is_relative_to(project_root) else str(descriptor_path),
        "descriptor_sha256": hashlib.sha256(descriptor_text.encode("utf-8")).hexdigest(),
        "operator": operator,
        "family": generated_test.family,
        "target_cpu": generated_test.cpu,
        "kernel_id": lookup_kernel_id(project_root, family=generated_test.family, operator=operator, dtype="S8"),
        "adapter_metadata_schema": 1,
        "source": "generated_test_bridge",
        "serialized_scalar_parameters": {
            "stride_h": stride_h,
            "stride_w": stride_w,
            "pad_h": pad_h,
            "pad_w": pad_w,
            "pad_offset_h": pad_offset_h,
            "pad_offset_w": pad_offset_w,
            "output_n": output_dims["n"],
            "output_h": output_dims["h"],
            "output_w": output_dims["w"],
            "output_c": output_dims["c"],
            "dilation_h": dilation_h,
            "dilation_w": dilation_w,
            "input_offset": input_offset,
            "output_offset": output_offset,
            "activation_min": activation_min,
            "activation_max": activation_max,
        },
        "tensor_dtypes": {"input": "S8", "weights": "S8", **({"bias": "S32"} if has_bias else {}), "output": "S8"},
        "blob_roles": [_manifest_blob_entry(blob) for blob in blobs],
        "expected_output": {"dtype": "S8", "byte_length": blobs[-1].byte_length, "blob_id": blobs[-1].blob_id},
        "correctness_comparison": {"mode": "tolerant_int", "tolerance": 1},
        "scratch_buffer": {"bytes": scratch_bytes},
        "required_target_capabilities": ["arm_transpose_conv_wrapper_s8"],
        "repeated_invocation_safe": True,
        "timing": {"warmups": 2, "samples": 5, "iterations_per_sample": 4, "min_cycles": 1024, "max_iterations": 256},
    }
    return CaseBundle(root_dir=case_root, manifest_path=_write_manifest(case_root, manifest), manifest=manifest, blobs=tuple(blobs))


def _build_pooling_case(
    project_root: Path,
    generated_test: GeneratedTestCase,
    *,
    output_root: Path | None = None,
) -> CaseBundle:
    """Convert one real generated CMSIS-NN AvgPool/MaxPool test (S8 or S16 activation)
    into a streamable perf-stream CaseBundle. Unlike Convolve/DepthwiseConv, pooling has
    no weights/bias/quant-multiplier blobs at all (a pool window has no learned
    parameters) and no input/output zero-point offsets (`cmsis_nn_pool_params` has only
    stride/padding/activation -- see arm_nn_types.h). The pool window size (pool_h/w) is
    sent as new scalar fields since there is no weights blob to read filter dims off of.
    MaxPool never needs a scratch buffer; AvgPool needs one sized via
    `arm_avgpool_{s8,s16}_get_buffer_size()`, which can legitimately be 0 for many cases.
    """
    descriptor = generated_test.descriptor
    operator = str(descriptor.get("operator", ""))
    activation_dtype = str(descriptor.get("activation_dtype", ""))
    if operator not in ("AvgPool", "MaxPool") or activation_dtype not in ("S8", "S16"):
        raise UnsupportedGeneratedTestError(
            f"{generated_test.name}: operator={operator!r} activation_dtype={activation_dtype!r} is not "
            f"bridgeable -- perf-stream firmware only dispatches arm_avgpool_s8/s16 and "
            f"arm_max_pool_s8/s16 (S8 or S16 activation)."
        )

    header_path = _find_header_file(generated_test.directory)
    header_text = header_path.read_text(encoding="utf-8")
    prefix = generated_test.name

    input_dims = _extract_dims(header_text, f"{prefix}_input_dims")
    filter_dims = _extract_dims(header_text, f"{prefix}_filter_dims")
    output_dims = _extract_dims(header_text, f"{prefix}_output_dims")
    if input_dims["n"] != 1:
        raise UnsupportedGeneratedTestError(
            f"{generated_test.name}: batch size {input_dims['n']} > 1 is not yet supported by the "
            f"perf-stream hardware bridge (firmware dispatches a single pooling invocation per case)."
        )
    input_shape = (input_dims["n"], input_dims["h"], input_dims["w"], input_dims["c"])
    output_shape = (output_dims["n"], output_dims["h"], output_dims["w"], output_dims["c"])

    activation_numpy_dtype = np.int16 if activation_dtype == "S16" else np.int8
    input_flat = np.array(_extract_array(header_text, f"{prefix}_input"), dtype=activation_numpy_dtype)
    expected_flat = np.array(_extract_array(header_text, f"{prefix}_expected_output"), dtype=activation_numpy_dtype)

    if input_flat.size != int(np.prod(input_shape)):
        raise UnsupportedGeneratedTestError(
            f"{generated_test.name}: generated input array size ({input_flat.size}) doesn't match header "
            f"input_dims (input_shape={input_shape})."
        )
    input_data = input_flat.reshape(input_shape)
    if expected_flat.size != int(np.prod(output_shape)):
        raise UnsupportedGeneratedTestError(
            f"{generated_test.name}: expected_output size ({expected_flat.size}) does not match header "
            f"output_dims product ({int(np.prod(output_shape))})"
        )
    expected_output = expected_flat.reshape(output_shape)

    params_struct = f"{prefix}_pool_params"
    activation_min = _extract_nested_scalar(header_text, params_struct, "activation", "min")
    activation_max = _extract_nested_scalar(header_text, params_struct, "activation", "max")
    stride_h = _extract_nested_scalar(header_text, params_struct, "stride", "h")
    stride_w = _extract_nested_scalar(header_text, params_struct, "stride", "w")
    pad_h = _extract_nested_scalar(header_text, params_struct, "padding", "h")
    pad_w = _extract_nested_scalar(header_text, params_struct, "padding", "w")

    case_id = f"{generated_test.name}_hw_generated"
    bundle_root = output_root if output_root is not None else project_root
    case_root = _case_root(bundle_root, "PoolingFunctions", case_id)
    blobs_dir = case_root / "blobs"
    blobs_dir.mkdir(parents=True, exist_ok=True)

    scratch_bytes = (
        int(TemplateContextBuilder.calculate_pooling_buffer_size_max(input_dims, output_dims, pooling_type="AVERAGE", output_dtype=activation_dtype))
        if operator == "AvgPool"
        else 0
    )

    arrays = [
        (1, "input_0", activation_dtype, input_shape, input_data, False, False),
        (6, "expected_output", activation_dtype, tuple(int(v) for v in expected_output.shape), expected_output, False, True),
    ]
    blobs: list[BlobInfo] = []
    for blob_id, role, dtype, dims, array, mutable_data, host_only in arrays:
        path = blobs_dir / f"{role}.bin"
        _write_blob(path, np.asarray(array))
        blobs.append(_blob_info(path, blob_id=blob_id, role=role, dtype=dtype, dimensions=dims, mutable_data=mutable_data, host_only=host_only))

    descriptor_path = generated_test.directory / "descriptor.yaml"
    descriptor_text = descriptor_path.read_text(encoding="utf-8")
    comparison = {"mode": "tolerant_int", "tolerance": 1} if operator == "Mean" else dict(
        descriptor.get("resolved_comparison", {"mode": "exact_int"})
    )
    manifest = {
        "schema_name": "hct.case_manifest",
        "schema_version": 1,
        "case_id": case_id,
        "descriptor_name": generated_test.name,
        "descriptor_path": str(descriptor_path.relative_to(project_root)) if descriptor_path.is_relative_to(project_root) else str(descriptor_path),
        "descriptor_sha256": hashlib.sha256(descriptor_text.encode("utf-8")).hexdigest(),
        "operator": operator,
        "family": "PoolingFunctions",
        "target_cpu": generated_test.cpu,
        "kernel_id": lookup_kernel_id(project_root, family="PoolingFunctions", operator=operator, dtype=activation_dtype),
        "adapter_metadata_schema": 1,
        "source": "generated_test_bridge",
        "serialized_scalar_parameters": {
            "stride_h": stride_h,
            "stride_w": stride_w,
            "pad_h": pad_h,
            "pad_w": pad_w,
            "output_h": output_dims["h"],
            "output_w": output_dims["w"],
            "output_c": output_dims["c"],
            "activation_min": activation_min,
            "activation_max": activation_max,
            "pool_h": filter_dims["h"],
            "pool_w": filter_dims["w"],
        },
        "tensor_dtypes": {"input": activation_dtype, "output": activation_dtype},
        "blob_roles": [_manifest_blob_entry(blob) for blob in blobs],
        "expected_output": {"dtype": activation_dtype, "byte_length": blobs[-1].byte_length, "blob_id": blobs[-1].blob_id},
        "correctness_comparison": comparison,
        "scratch_buffer": {"bytes": int(scratch_bytes)},
        "required_target_capabilities": [f"{operator.lower()}_{activation_dtype.lower()}"],
        "repeated_invocation_safe": True,
        "timing": {"warmups": 2, "samples": 5, "iterations_per_sample": 4, "min_cycles": 1024, "max_iterations": 256},
    }
    return CaseBundle(root_dir=case_root, manifest_path=_write_manifest(case_root, manifest), manifest=manifest, blobs=tuple(blobs))


# ActivationFunctions unary ops (single input tensor -> same-shape output, no weights/bias
# blob, no named scalar-params struct in the generated header -- quant scalars are inlined
# as call arguments in the generated .c file, same style as BasicMathFunctions elementwise
# ops). Each entry maps operator -> (cmsis_function_by_dtype, positional arg count, and a
# function mapping the extracted args list to a scalar_parameters dict).
_ACTIVATION_CMSIS_FUNCTION = {
    ("Relu", "S8"): "arm_relu_s8",
    ("Relu", "S16"): "arm_relu_s16",
    ("Relu6", "S8"): "arm_relu_generic_s8",
    ("Relu6", "S16"): "arm_relu_generic_s16",
    ("Clamp", "S8"): "arm_clamp_s8",
    ("Clamp", "S16"): "arm_clamp_s16",
    ("LeakyRelu", "S8"): "arm_leaky_relu_s8",
    ("LeakyRelu", "S16"): "arm_leaky_relu_s16",
    ("Logistic", "S16"): "arm_logistic_s16",
    ("Tanh", "S16"): "arm_tanh_s16",
    ("HardSwishCompat", "S8"): "arm_hard_swish_compat_s8",
    ("HardSwishPrecise", "S8"): "arm_hard_swish_precise_s8",
    ("HardSwishPrecise", "S16"): "arm_hard_swish_precise_s16",
}
# CMSIS-NN only implements these two ops in S16 -- the generator forces S16 even when the
# descriptor's activation_dtype says S8 (see OpLogistic/OpTanh generate_c_files()).
_ACTIVATION_FORCE_S16_OPERATORS = ("Logistic", "Tanh")
_ACTIVATION_ARG_COUNT = {
    "Relu": 7,
    "Relu6": 9,
    "Clamp": 5,
    "LeakyRelu": 9,
    "Logistic": 5,
    "Tanh": 5,
    "HardSwishCompat": 9,
    "HardSwishPrecise": 10,
}


def _activation_scalar_parameters(operator: str, args: list[str]) -> dict[str, int]:
    """Map the positionally-extracted call arguments to named scalar_parameters, per
    operator's CMSIS-NN signature (see _ACTIVATION_CMSIS_FUNCTION docstring above and
    arm_nnfunctions.h for each function's exact argument order). Only the scalar-parameter
    indices are read here -- the input/output pointer-argument slots (e.g. args[0], and
    args[1] for Logistic/Tanh) are never int()-converted since they're identifiers, not
    integer literals."""
    def as_int(index: int) -> int:
        return int(args[index])

    if operator == "Relu":
        return {
            "input_offset": as_int(1), "output_offset": as_int(2),
            "out_mult": as_int(3), "out_shift": as_int(4),
        }
    if operator == "Relu6":
        return {
            "input_offset": as_int(1), "output_offset": as_int(2), "out_mult": as_int(3), "out_shift": as_int(4),
            "activation_min": as_int(5), "activation_max": as_int(6),
        }
    if operator == "Clamp":
        return {"activation_min": as_int(1), "activation_max": as_int(2)}
    if operator == "LeakyRelu":
        return {
            "input_offset": as_int(1), "output_offset": as_int(2),
            "out_mult_alpha": as_int(3), "out_shift_alpha": as_int(4),
            "out_mult": as_int(5), "out_shift": as_int(6),
        }
    if operator in ("Logistic", "Tanh"):
        return {"input_mult": as_int(3), "input_left_shift": as_int(4)}
    if operator == "HardSwishCompat":
        return {
            "input_offset": as_int(1), "output_offset": as_int(2),
            "out_mult_fp": as_int(3), "out_mult_exp": as_int(4),
            "relu_mult_fp": as_int(5), "relu_mult_exp": as_int(6),
        }
    if operator == "HardSwishPrecise":
        return {
            "input_offset": as_int(1), "output_offset": as_int(2), "out_mult": as_int(3), "out_shift": as_int(4),
            "relu_q3": as_int(5), "relu_q6": as_int(6), "prescale": as_int(7),
        }
    raise UnsupportedGeneratedTestError(f"Unknown ActivationFunctions operator: {operator!r}")


def _build_activation_case(
    project_root: Path,
    generated_test: GeneratedTestCase,
    *,
    output_root: Path | None = None,
) -> CaseBundle:
    """Bridge an ActivationFunctions unary generated test (Relu/Relu6/Clamp/LeakyRelu/
    Logistic/Tanh/HardSwishCompat/HardSwishPrecise). All of these take one input tensor and
    produce a same-shape output, with no weights/bias blob and no scratch buffer -- their
    scalar quant params are inlined as call arguments in the generated .c file (positionally
    extracted via `_extract_call_args`, same technique as BasicMathFunctions elementwise ops)
    since there's no named scalar-params struct in the header for these ops."""
    descriptor = generated_test.descriptor
    operator = str(descriptor.get("operator", ""))
    activation_dtype = str(descriptor.get("activation_dtype", ""))
    if operator in _ACTIVATION_FORCE_S16_OPERATORS:
        activation_dtype = "S16"
    if (operator, activation_dtype) not in _ACTIVATION_CMSIS_FUNCTION:
        raise UnsupportedGeneratedTestError(
            f"{generated_test.name}: operator={operator!r} activation_dtype={activation_dtype!r} is not "
            f"bridgeable -- perf-stream firmware only dispatches {sorted(_ACTIVATION_CMSIS_FUNCTION)}."
        )

    header_path = _find_header_file(generated_test.directory)
    header_text = header_path.read_text(encoding="utf-8")
    source_path = _find_source_file(generated_test.directory)
    source_text = source_path.read_text(encoding="utf-8")
    prefix = generated_test.name

    input_dims = _extract_dims(header_text, f"{prefix}_input_dims")
    output_dims = _extract_dims(header_text, f"{prefix}_output_dims")
    if input_dims["n"] != 1:
        raise UnsupportedGeneratedTestError(
            f"{generated_test.name}: batch size {input_dims['n']} > 1 is not yet supported by the "
            f"perf-stream hardware bridge (firmware dispatches a single {operator} invocation per case)."
        )
    input_shape = (input_dims["n"], input_dims["h"], input_dims["w"], input_dims["c"])
    output_shape = (output_dims["n"], output_dims["h"], output_dims["w"], output_dims["c"])

    numpy_dtype = np.int16 if activation_dtype == "S16" else np.int8
    input_flat = np.array(_extract_array(header_text, f"{prefix}_input"), dtype=numpy_dtype)
    expected_flat = np.array(_extract_array(header_text, f"{prefix}_expected_output"), dtype=numpy_dtype)
    if input_flat.size != int(np.prod(input_shape)):
        raise UnsupportedGeneratedTestError(
            f"{generated_test.name}: generated input array size ({input_flat.size}) doesn't match header "
            f"input_dims (input_shape={input_shape})."
        )
    input_data = input_flat.reshape(input_shape)
    if expected_flat.size != int(np.prod(output_shape)):
        raise UnsupportedGeneratedTestError(
            f"{generated_test.name}: expected_output size ({expected_flat.size}) does not match header "
            f"output_dims product ({int(np.prod(output_shape))})"
        )
    expected_output = expected_flat.reshape(output_shape)

    cmsis_function = _ACTIVATION_CMSIS_FUNCTION[(operator, activation_dtype)]
    args = _extract_call_args(source_text, cmsis_function, expected_count=_ACTIVATION_ARG_COUNT[operator])
    scalar_parameters = _activation_scalar_parameters(operator, args)

    case_id = f"{generated_test.name}_hw_generated"
    bundle_root = output_root if output_root is not None else project_root
    case_root = _case_root(bundle_root, "ActivationFunctions", case_id)
    blobs_dir = case_root / "blobs"
    blobs_dir.mkdir(parents=True, exist_ok=True)

    arrays = [
        (1, "input_0", activation_dtype, input_shape, input_data, False, False),
        (6, "expected_output", activation_dtype, tuple(int(v) for v in expected_output.shape), expected_output, False, True),
    ]
    blobs: list[BlobInfo] = []
    for blob_id, role, dtype, dims, array, mutable_data, host_only in arrays:
        path = blobs_dir / f"{role}.bin"
        _write_blob(path, np.asarray(array))
        blobs.append(_blob_info(path, blob_id=blob_id, role=role, dtype=dtype, dimensions=dims, mutable_data=mutable_data, host_only=host_only))

    descriptor_path = generated_test.directory / "descriptor.yaml"
    descriptor_text = descriptor_path.read_text(encoding="utf-8")
    comparison = {"mode": "tolerant_int", "tolerance": 1} if operator == "Mean" else dict(
        descriptor.get("resolved_comparison", {"mode": "exact_int"})
    )
    manifest = {
        "schema_name": "hct.case_manifest",
        "schema_version": 1,
        "case_id": case_id,
        "descriptor_name": generated_test.name,
        "descriptor_path": str(descriptor_path.relative_to(project_root)) if descriptor_path.is_relative_to(project_root) else str(descriptor_path),
        "descriptor_sha256": hashlib.sha256(descriptor_text.encode("utf-8")).hexdigest(),
        "operator": operator,
        "family": "ActivationFunctions",
        "target_cpu": generated_test.cpu,
        "kernel_id": lookup_kernel_id(project_root, family="ActivationFunctions", operator=operator, dtype=activation_dtype),
        "adapter_metadata_schema": 1,
        "source": "generated_test_bridge",
        "serialized_scalar_parameters": {
            **scalar_parameters,
            "output_h": output_dims["h"],
            "output_w": output_dims["w"],
            "output_c": output_dims["c"],
        },
        "tensor_dtypes": {"input": activation_dtype, "output": activation_dtype},
        "blob_roles": [_manifest_blob_entry(blob) for blob in blobs],
        "expected_output": {"dtype": activation_dtype, "byte_length": blobs[-1].byte_length, "blob_id": blobs[-1].blob_id},
        "correctness_comparison": comparison,
        "scratch_buffer": {"bytes": 0},
        "required_target_capabilities": [cmsis_function],
        "repeated_invocation_safe": True,
        "timing": {"warmups": 2, "samples": 5, "iterations_per_sample": 4, "min_cycles": 1024, "max_iterations": 256},
    }
    return CaseBundle(root_dir=case_root, manifest_path=_write_manifest(case_root, manifest), manifest=manifest, blobs=tuple(blobs))


# arm_prelu_s8/s16(&input_dims, input, &alpha_dims, alpha, input_offset, alpha_offset,
#    output_offset, output_mult_identity, output_shift_identity, output_mult_alpha,
#    output_shift_alpha, &output_dims, output) -- alpha is broadcastable against input per
# cmsis_nn_dims (same broadcast semantics as elementwise binary Add/Sub), unlike the pure
# unary activations bridged earlier which have no second tensor at all.
_PRELU_ARG_COUNT = 13


def _quant_scale_to_bits(scale: float) -> int:
    """Reinterpret a float32 scale as the raw int32 bit pattern the firmware decodes back
    via `quant_scale_from_bits()` (see run_quantize_once()/run_dequantize_once() in
    benchmark_server_session.c). Scalar params travel the wire as int32 only, so this
    bit-cast is used instead of a fixed-point encoding (like atol_q16/rtol_q16) to avoid any
    precision loss -- Dequantize's FLOAT comparison tolerance (atol=5e-5) is far tighter
    than a Q16 encoding (~1.5e-5 resolution) could reliably support once multiplied through
    a tensor's full dequantized range."""
    return int(struct.unpack("<i", struct.pack("<f", np.float32(scale)))[0])


_ACTIVATION_KIND = {"NONE": 0, "": 0, "RELU": 1, "RELU6": 2}

# arm_quantize_f32_{s8,s16}(input, output, size, zero_point, scale) -- float input, quantized
# output. The generator's own template applies the descriptor's ReLU/ReLU6 activation (if
# any) to the float input BEFORE calling this kernel, entirely in float space -- so that
# activation is folded into the input blob here on the host, and the firmware-side adapter
# only ever needs to invoke the kernel itself (see _RUN_QUANTIZE_ONCE in adapter_specs.py).
_QUANTIZE_ARG_COUNT = 5


def _build_quantize_case(
    project_root: Path,
    generated_test: GeneratedTestCase,
    *,
    output_root: Path | None = None,
) -> CaseBundle:
    """Bridge a QuantizationFunctions Quantize generated test (S8 or S16 output, float32
    input). Both the `default` (per-op ReLU/ReLU6 pre-activation) and `float` (no
    activation) descriptor suites map to the same kernel_fn/builder."""
    descriptor = generated_test.descriptor
    operator = str(descriptor.get("operator", ""))
    activation_dtype = str(descriptor.get("activation_dtype", ""))
    if operator != "Quantize" or activation_dtype not in ("S8", "S16"):
        raise UnsupportedGeneratedTestError(
            f"{generated_test.name}: operator={operator!r} activation_dtype={activation_dtype!r} is not "
            f"bridgeable -- perf-stream firmware only dispatches arm_quantize_f32_s8/s16."
        )
    activation = str(descriptor.get("activation", "NONE"))
    if activation not in _ACTIVATION_KIND:
        raise UnsupportedGeneratedTestError(
            f"{generated_test.name}: activation={activation!r} is not one of NONE/RELU/RELU6."
        )
    input_shape = tuple(int(v) for v in descriptor.get("input_shape", []))
    if not input_shape or input_shape[0] != 1:
        raise UnsupportedGeneratedTestError(
            f"{generated_test.name}: batch size > 1 is not yet supported by the perf-stream hardware bridge."
        )
    size = int(np.prod(input_shape))

    header_path = _find_header_file(generated_test.directory)
    header_text = header_path.read_text(encoding="utf-8")
    source_path = _find_source_file(generated_test.directory)
    source_text = source_path.read_text(encoding="utf-8")
    prefix = generated_test.name

    input_flat = np.array(_extract_float_array(header_text, f"{prefix}_input"), dtype=np.float32)
    numpy_dtype = np.int16 if activation_dtype == "S16" else np.int8
    expected_flat = np.array(_extract_array(header_text, f"{prefix}_expected_output"), dtype=numpy_dtype)
    if input_flat.size != size or expected_flat.size != size:
        raise UnsupportedGeneratedTestError(
            f"{generated_test.name}: array sizes (input={input_flat.size}, "
            f"expected_output={expected_flat.size}) don't match input_shape product ({size})."
        )
    # TFLite applies activation in float space BEFORE quantization -- match the generator's
    # own template by pre-applying it here, so the firmware-side kernel call needs no
    # activation logic of its own (see _RUN_QUANTIZE_ONCE in adapter_specs.py).
    if activation == "RELU":
        input_flat = np.maximum(input_flat, np.float32(0.0))
    elif activation == "RELU6":
        input_flat = np.clip(input_flat, np.float32(0.0), np.float32(6.0))

    cmsis_function = "arm_quantize_f32_s16" if activation_dtype == "S16" else "arm_quantize_f32_s8"
    args = _extract_call_args(source_text, cmsis_function, expected_count=_QUANTIZE_ARG_COUNT)
    zero_point = int(args[3])
    scale = float(args[4].rstrip("fF"))

    case_id = f"{generated_test.name}_hw_generated"
    bundle_root = output_root if output_root is not None else project_root
    case_root = _case_root(bundle_root, "QuantizationFunctions", case_id)
    blobs_dir = case_root / "blobs"
    blobs_dir.mkdir(parents=True, exist_ok=True)

    arrays = [
        (1, "input_0", "FP32", (1, 1, 1, size), input_flat.reshape(1, 1, 1, size), False, False),
        (2, "expected_output", activation_dtype, (1, 1, 1, size), expected_flat.reshape(1, 1, 1, size), False, True),
    ]
    blobs: list[BlobInfo] = []
    for blob_id, role, dtype, dims, array, mutable_data, host_only in arrays:
        path = blobs_dir / f"{role}.bin"
        _write_blob(path, np.asarray(array))
        blobs.append(_blob_info(path, blob_id=blob_id, role=role, dtype=dtype, dimensions=dims, mutable_data=mutable_data, host_only=host_only))

    descriptor_path = generated_test.directory / "descriptor.yaml"
    descriptor_text = descriptor_path.read_text(encoding="utf-8")
    manifest = {
        "schema_name": "hct.case_manifest",
        "schema_version": 1,
        "case_id": case_id,
        "descriptor_name": generated_test.name,
        "descriptor_path": str(descriptor_path.relative_to(project_root)) if descriptor_path.is_relative_to(project_root) else str(descriptor_path),
        "descriptor_sha256": hashlib.sha256(descriptor_text.encode("utf-8")).hexdigest(),
        "operator": operator,
        "family": "QuantizationFunctions",
        "target_cpu": generated_test.cpu,
        "kernel_id": lookup_kernel_id(project_root, family="QuantizationFunctions", operator=operator, dtype=activation_dtype),
        "adapter_metadata_schema": 1,
        "source": "generated_test_bridge",
        "serialized_scalar_parameters": {
            "output_offset": zero_point,
            "scale_bits": _quant_scale_to_bits(scale),
            "output_h": 1,
            "output_w": 1,
            "output_c": size,
        },
        "tensor_dtypes": {"input": "FP32", "output": activation_dtype},
        "blob_roles": [_manifest_blob_entry(blob) for blob in blobs],
        "expected_output": {"dtype": activation_dtype, "byte_length": blobs[-1].byte_length, "blob_id": blobs[-1].blob_id},
        # Quantize's generated harness (quantize.c.j2's HELIA_VALIDATE_OUTPUTS call) always
        # validates with TOLERANT_INT tolerance=1, NOT the generic dtype-based exact_int
        # default resolve_comparison() computes for resolved_comparison below -- an
        # intentional, hardcoded template convention (no per-descriptor override exists)
        # that captures an inherent off-by-one rounding gap between the TFLite reference
        # graph used to produce the golden output and the CMSIS-NN kernel's own
        # value/scale+zero_point rounding. Match the template's real validation semantics
        # here rather than the generic descriptor metadata.
        "correctness_comparison": {"mode": "tolerant_int", "tolerance": 1},
        "scratch_buffer": {"bytes": 0},
        "required_target_capabilities": [cmsis_function],
        "repeated_invocation_safe": True,
        "timing": {"warmups": 2, "samples": 5, "iterations_per_sample": 4, "min_cycles": 1024, "max_iterations": 256},
    }
    return CaseBundle(root_dir=case_root, manifest_path=_write_manifest(case_root, manifest), manifest=manifest, blobs=tuple(blobs))


# arm_dequantize_{s8,s16}_f32(input, output, size, zero_point, scale) -- quantized input,
# float output. Unlike Quantize, the generator's own template applies the descriptor's
# ReLU/ReLU6 activation (if any) AFTER this kernel call, to the dequantized float output --
# so it must be replicated in firmware (see activation_kind in benchmark_server_session.h
# and _RUN_DEQUANTIZE_ONCE in adapter_specs.py) rather than folded into an input blob.
_DEQUANTIZE_ARG_COUNT = 5
_REQUANTIZE_ARG_COUNT = 7
_COMPARISON_ARG_COUNT = 14
_COMPARISON_FUNCTIONS = {
    ("equal", "S8"): "arm_equal_s8",
    ("equal", "S16"): "arm_equal_s16",
    ("not_equal", "S8"): "arm_not_equal_s8",
    ("not_equal", "S16"): "arm_not_equal_s16",
    ("greater", "S8"): "arm_greater_s8",
    ("greater", "S16"): "arm_greater_s16",
    ("greater_equal", "S8"): "arm_greater_equal_s8",
    ("greater_equal", "S16"): "arm_greater_equal_s16",
    ("less", "S8"): "arm_less_s8",
    ("less", "S16"): "arm_less_s16",
    ("less_equal", "S8"): "arm_less_equal_s8",
    ("less_equal", "S16"): "arm_less_equal_s16",
}
_COMPARISON_OPERATOR_NAMES = {
    "equal": "Equal",
    "not_equal": "NotEqual",
    "greater": "Greater",
    "greater_equal": "GreaterEqual",
    "less": "Less",
    "less_equal": "LessEqual",
}


def _build_dequantize_case(
    project_root: Path,
    generated_test: GeneratedTestCase,
    *,
    output_root: Path | None = None,
) -> CaseBundle:
    """Bridge a QuantizationFunctions Dequantize generated test (S8 or S16 input, float32
    output). Both the `default` (per-op ReLU/ReLU6 post-activation) and `float` (no
    activation) descriptor suites map to the same kernel_fn/builder."""
    descriptor = generated_test.descriptor
    operator = str(descriptor.get("operator", ""))
    activation_dtype = str(descriptor.get("activation_dtype", ""))
    if operator != "Dequantize" or activation_dtype not in ("S8", "S16"):
        raise UnsupportedGeneratedTestError(
            f"{generated_test.name}: operator={operator!r} activation_dtype={activation_dtype!r} is not "
            f"bridgeable -- perf-stream firmware only dispatches arm_dequantize_s8_f32/s16_f32."
        )
    activation = str(descriptor.get("activation", "NONE"))
    if activation not in _ACTIVATION_KIND:
        raise UnsupportedGeneratedTestError(
            f"{generated_test.name}: activation={activation!r} is not one of NONE/RELU/RELU6."
        )
    input_shape = tuple(int(v) for v in descriptor.get("input_shape", []))
    if not input_shape or input_shape[0] != 1:
        raise UnsupportedGeneratedTestError(
            f"{generated_test.name}: batch size > 1 is not yet supported by the perf-stream hardware bridge."
        )
    size = int(np.prod(input_shape))

    header_path = _find_header_file(generated_test.directory)
    header_text = header_path.read_text(encoding="utf-8")
    source_path = _find_source_file(generated_test.directory)
    source_text = source_path.read_text(encoding="utf-8")
    prefix = generated_test.name

    numpy_dtype = np.int16 if activation_dtype == "S16" else np.int8
    input_flat = np.array(_extract_array(header_text, f"{prefix}_input"), dtype=numpy_dtype)
    expected_flat = np.array(_extract_float_array(header_text, f"{prefix}_expected_output"), dtype=np.float32)
    if input_flat.size != size or expected_flat.size != size:
        raise UnsupportedGeneratedTestError(
            f"{generated_test.name}: array sizes (input={input_flat.size}, "
            f"expected_output={expected_flat.size}) don't match input_shape product ({size})."
        )

    cmsis_function = "arm_dequantize_s16_f32" if activation_dtype == "S16" else "arm_dequantize_s8_f32"
    args = _extract_call_args(source_text, cmsis_function, expected_count=_DEQUANTIZE_ARG_COUNT)
    zero_point = int(args[3])
    scale = float(args[4].rstrip("fF"))

    case_id = f"{generated_test.name}_hw_generated"
    bundle_root = output_root if output_root is not None else project_root
    case_root = _case_root(bundle_root, "QuantizationFunctions", case_id)
    blobs_dir = case_root / "blobs"
    blobs_dir.mkdir(parents=True, exist_ok=True)

    arrays = [
        (1, "input_0", activation_dtype, (1, 1, 1, size), input_flat.reshape(1, 1, 1, size), False, False),
        (2, "expected_output", "FP32", (1, 1, 1, size), expected_flat.reshape(1, 1, 1, size), False, True),
    ]
    blobs: list[BlobInfo] = []
    for blob_id, role, dtype, dims, array, mutable_data, host_only in arrays:
        path = blobs_dir / f"{role}.bin"
        _write_blob(path, np.asarray(array))
        blobs.append(_blob_info(path, blob_id=blob_id, role=role, dtype=dtype, dimensions=dims, mutable_data=mutable_data, host_only=host_only))

    descriptor_path = generated_test.directory / "descriptor.yaml"
    descriptor_text = descriptor_path.read_text(encoding="utf-8")
    manifest = {
        "schema_name": "hct.case_manifest",
        "schema_version": 1,
        "case_id": case_id,
        "descriptor_name": generated_test.name,
        "descriptor_path": str(descriptor_path.relative_to(project_root)) if descriptor_path.is_relative_to(project_root) else str(descriptor_path),
        "descriptor_sha256": hashlib.sha256(descriptor_text.encode("utf-8")).hexdigest(),
        "operator": operator,
        "family": "QuantizationFunctions",
        "target_cpu": generated_test.cpu,
        "kernel_id": lookup_kernel_id(project_root, family="QuantizationFunctions", operator=operator, dtype=activation_dtype),
        "adapter_metadata_schema": 1,
        "source": "generated_test_bridge",
        "serialized_scalar_parameters": {
            "input_offset": zero_point,
            "scale_bits": _quant_scale_to_bits(scale),
            "activation_kind": _ACTIVATION_KIND[activation],
            "output_h": 1,
            "output_w": 1,
            "output_c": size,
        },
        "tensor_dtypes": {"input": activation_dtype, "output": "FP32"},
        "blob_roles": [_manifest_blob_entry(blob) for blob in blobs],
        "expected_output": {"dtype": "FP32", "byte_length": blobs[-1].byte_length, "blob_id": blobs[-1].blob_id},
        "correctness_comparison": dict(descriptor.get("resolved_comparison", {"mode": "float"})),
        "scratch_buffer": {"bytes": 0},
        "required_target_capabilities": [cmsis_function],
        "repeated_invocation_safe": True,
        "timing": {"warmups": 2, "samples": 5, "iterations_per_sample": 4, "min_cycles": 1024, "max_iterations": 256},
    }
    return CaseBundle(root_dir=case_root, manifest_path=_write_manifest(case_root, manifest), manifest=manifest, blobs=tuple(blobs))


def _build_requantize_case(
    project_root: Path,
    generated_test: GeneratedTestCase,
    *,
    output_root: Path | None = None,
) -> CaseBundle:
    descriptor = generated_test.descriptor
    operator = str(descriptor.get("operator", ""))
    activation_dtype = str(descriptor.get("activation_dtype", ""))
    if operator != "Requantize" or activation_dtype not in ("S8", "S16"):
        raise UnsupportedGeneratedTestError(
            f"{generated_test.name}: operator={operator!r} activation_dtype={activation_dtype!r} is not "
            f"bridgeable -- perf-stream firmware only dispatches arm_requantize_s8_s8/s16_s16."
        )

    input_shape = tuple(int(v) for v in descriptor.get("input_shape", []))
    if not input_shape or input_shape[0] != 1:
        raise UnsupportedGeneratedTestError(
            f"{generated_test.name}: batch size > 1 is not yet supported by the perf-stream hardware bridge."
        )

    header_path = _find_header_file(generated_test.directory)
    header_text = header_path.read_text(encoding="utf-8")
    source_path = _find_source_file(generated_test.directory)
    source_text = source_path.read_text(encoding="utf-8")
    prefix = generated_test.name

    numpy_dtype = np.int16 if activation_dtype == "S16" else np.int8
    input_flat = np.array(_extract_array(header_text, f"{prefix}_input"), dtype=numpy_dtype)
    expected_flat = np.array(_extract_array(header_text, f"{prefix}_expected_output"), dtype=numpy_dtype)
    size = int(np.prod(input_shape))
    if input_flat.size != size or expected_flat.size != size:
        raise UnsupportedGeneratedTestError(
            f"{generated_test.name}: array sizes (input={input_flat.size}, expected_output={expected_flat.size}) "
            f"don't match input_shape product ({size})."
        )

    cmsis_function = "arm_requantize_s16_s16" if activation_dtype == "S16" else "arm_requantize_s8_s8"
    args = _extract_call_args(source_text, cmsis_function, expected_count=_REQUANTIZE_ARG_COUNT)

    input_data = input_flat.reshape(input_shape)
    expected_output = expected_flat.reshape(input_shape)
    case_id = f"{generated_test.name}_hw_generated"
    bundle_root = output_root if output_root is not None else project_root
    case_root = _case_root(bundle_root, generated_test.family, case_id)
    blobs_dir = case_root / "blobs"
    blobs_dir.mkdir(parents=True, exist_ok=True)

    arrays = [
        (1, "input_0", activation_dtype, input_shape, input_data, False, False),
        (2, "expected_output", activation_dtype, input_shape, expected_output, False, True),
    ]
    blobs: list[BlobInfo] = []
    for blob_id, role, dtype, dims, array, mutable_data, host_only in arrays:
        path = blobs_dir / f"{role}.bin"
        _write_blob(path, np.asarray(array))
        blobs.append(_blob_info(path, blob_id=blob_id, role=role, dtype=dtype, dimensions=dims, mutable_data=mutable_data, host_only=host_only))

    descriptor_path = generated_test.directory / "descriptor.yaml"
    descriptor_text = descriptor_path.read_text(encoding="utf-8")
    manifest = {
        "schema_name": "hct.case_manifest",
        "schema_version": 1,
        "case_id": case_id,
        "descriptor_name": generated_test.name,
        "descriptor_path": str(descriptor_path.relative_to(project_root)) if descriptor_path.is_relative_to(project_root) else str(descriptor_path),
        "descriptor_sha256": hashlib.sha256(descriptor_text.encode("utf-8")).hexdigest(),
        "operator": operator,
        "family": generated_test.family,
        "target_cpu": generated_test.cpu,
        "kernel_id": lookup_kernel_id(project_root, family=generated_test.family, operator=operator, dtype=activation_dtype),
        "adapter_metadata_schema": 1,
        "source": "generated_test_bridge",
        "serialized_scalar_parameters": {
            "out_mult": int(args[3]),
            "out_shift": int(args[4]),
            "input_offset": int(args[5]),
            "output_offset": int(args[6]),
        },
        "tensor_dtypes": {"input": activation_dtype, "output": activation_dtype},
        "blob_roles": [_manifest_blob_entry(blob) for blob in blobs],
        "expected_output": {"dtype": activation_dtype, "byte_length": blobs[-1].byte_length, "blob_id": blobs[-1].blob_id},
        "correctness_comparison": dict(descriptor.get("resolved_comparison", {"mode": "exact_int"})),
        "scratch_buffer": {"bytes": 0},
        "required_target_capabilities": [cmsis_function],
        "repeated_invocation_safe": True,
        "timing": {"warmups": 2, "samples": 5, "iterations_per_sample": 4, "min_cycles": 1024, "max_iterations": 256},
    }
    return CaseBundle(root_dir=case_root, manifest_path=_write_manifest(case_root, manifest), manifest=manifest, blobs=tuple(blobs))


def _build_comparison_case(
    project_root: Path,
    generated_test: GeneratedTestCase,
    *,
    output_root: Path | None = None,
) -> CaseBundle:
    descriptor = generated_test.descriptor
    activation_dtype = str(descriptor.get("activation_dtype", ""))
    operation = str(descriptor.get("operation", "")).lower()
    registry_operator = _COMPARISON_OPERATOR_NAMES.get(operation)
    cmsis_function = _COMPARISON_FUNCTIONS.get((operation, activation_dtype))
    if registry_operator is None or cmsis_function is None:
        raise UnsupportedGeneratedTestError(
            f"{generated_test.name}: operation={operation!r} activation_dtype={activation_dtype!r} is not "
            "bridgeable by the ComparisonFunctions adapter."
        )

    header_path = _find_header_file(generated_test.directory)
    header_text = header_path.read_text(encoding="utf-8")
    source_path = _find_source_file(generated_test.directory)
    source_text = source_path.read_text(encoding="utf-8")
    prefix = generated_test.name

    input_1_dims = _extract_dims(header_text, f"{prefix}_input_1_dims")
    input_2_dims = _extract_dims(header_text, f"{prefix}_input_2_dims")
    output_dims = _extract_dims(header_text, f"{prefix}_output_dims")
    input_1_shape = (input_1_dims["n"], input_1_dims["h"], input_1_dims["w"], input_1_dims["c"])
    input_2_shape = (input_2_dims["n"], input_2_dims["h"], input_2_dims["w"], input_2_dims["c"])
    output_shape = (output_dims["n"], output_dims["h"], output_dims["w"], output_dims["c"])

    numpy_dtype = np.int16 if activation_dtype == "S16" else np.int8
    input_1_flat = np.array(_extract_array(header_text, f"{prefix}_input_1"), dtype=numpy_dtype)
    input_2_flat = np.array(_extract_array(header_text, f"{prefix}_input_2"), dtype=numpy_dtype)
    expected_flat = np.array(_extract_array(header_text, f"{prefix}_expected_output"), dtype=np.bool_)
    if input_1_flat.size != int(np.prod(input_1_shape)) or input_2_flat.size != int(np.prod(input_2_shape)):
        raise UnsupportedGeneratedTestError(
            f"{generated_test.name}: generated array sizes (input_1={input_1_flat.size}, input_2={input_2_flat.size}) "
            f"don't match header dims (input_1_shape={input_1_shape}, input_2_shape={input_2_shape})."
        )
    if expected_flat.size != int(np.prod(output_shape)):
        raise UnsupportedGeneratedTestError(
            f"{generated_test.name}: expected_output size ({expected_flat.size}) does not match output dims ({output_shape})."
        )

    args = _extract_call_args(source_text, cmsis_function, expected_count=_COMPARISON_ARG_COUNT)
    scalar_parameters = {
        "output_n": output_dims["n"],
        "output_h": output_dims["h"],
        "output_w": output_dims["w"],
        "output_c": output_dims["c"],
        "input1_offset": int(args[7]),
        "input1_mult": int(args[8]),
        "input1_shift": int(args[9]),
        "input2_offset": int(args[10]),
        "input2_mult": int(args[11]),
        "input2_shift": int(args[12]),
        "left_shift": int(args[13]),
    }

    input_1_data = input_1_flat.reshape(input_1_shape)
    input_2_data = input_2_flat.reshape(input_2_shape)
    expected_output = expected_flat.reshape(output_shape)

    case_id = f"{generated_test.name}_hw_generated"
    bundle_root = output_root if output_root is not None else project_root
    case_root = _case_root(bundle_root, generated_test.family, case_id)
    blobs_dir = case_root / "blobs"
    blobs_dir.mkdir(parents=True, exist_ok=True)

    arrays = [
        (1, "input_0", activation_dtype, input_1_shape, input_1_data, False, False),
        (2, "input_1", activation_dtype, input_2_shape, input_2_data, False, False),
        (3, "expected_output", "BOOL", output_shape, expected_output, False, True),
    ]
    blobs: list[BlobInfo] = []
    for blob_id, role, dtype, dims, array, mutable_data, host_only in arrays:
        path = blobs_dir / f"{role}.bin"
        _write_blob(path, np.asarray(array))
        blobs.append(_blob_info(path, blob_id=blob_id, role=role, dtype=dtype, dimensions=dims, mutable_data=mutable_data, host_only=host_only))

    descriptor_path = generated_test.directory / "descriptor.yaml"
    descriptor_text = descriptor_path.read_text(encoding="utf-8")
    manifest = {
        "schema_name": "hct.case_manifest",
        "schema_version": 1,
        "case_id": case_id,
        "descriptor_name": generated_test.name,
        "descriptor_path": str(descriptor_path.relative_to(project_root)) if descriptor_path.is_relative_to(project_root) else str(descriptor_path),
        "descriptor_sha256": hashlib.sha256(descriptor_text.encode("utf-8")).hexdigest(),
        "operator": registry_operator,
        "family": generated_test.family,
        "target_cpu": generated_test.cpu,
        "kernel_id": lookup_kernel_id(project_root, family=generated_test.family, operator=registry_operator, dtype=activation_dtype),
        "adapter_metadata_schema": 1,
        "source": "generated_test_bridge",
        "serialized_scalar_parameters": scalar_parameters,
        "tensor_dtypes": {"input": activation_dtype, "output": "BOOL"},
        "blob_roles": [_manifest_blob_entry(blob) for blob in blobs],
        "expected_output": {"dtype": "BOOL", "byte_length": blobs[-1].byte_length, "blob_id": blobs[-1].blob_id},
        "correctness_comparison": {"mode": "bool"},
        "scratch_buffer": {"bytes": 0},
        "required_target_capabilities": [cmsis_function],
        "repeated_invocation_safe": True,
        "timing": {"warmups": 2, "samples": 5, "iterations_per_sample": 4, "min_cycles": 1024, "max_iterations": 256},
    }
    return CaseBundle(root_dir=case_root, manifest_path=_write_manifest(case_root, manifest), manifest=manifest, blobs=tuple(blobs))


def _build_prelu_case(
    project_root: Path,
    generated_test: GeneratedTestCase,
    *,
    output_root: Path | None = None,
) -> CaseBundle:
    """Bridge an ActivationFunctions PReLU generated test (S8 or S16). Unlike the pure unary
    activations, PReLU takes a second (broadcastable) alpha tensor input -- structurally
    closer to BasicMathFunctions elementwise binary ops, just with PReLU's own scalar-param
    set (input_offset/alpha_offset/output_offset + separate identity/alpha multiplier-shift
    pairs, no activation clamp)."""
    descriptor = generated_test.descriptor
    operator = str(descriptor.get("operator", ""))
    activation_dtype = str(descriptor.get("activation_dtype", ""))
    expected_status = str(descriptor.get("expected_status", "ARM_CMSIS_NN_SUCCESS"))
    if expected_status != "ARM_CMSIS_NN_SUCCESS":
        raise UnsupportedGeneratedTestError(
            f"{generated_test.name}: expects a non-success status ({expected_status}) -- the perf-stream "
            f"bridge only compares golden *output* arrays, it has no infrastructure to assert on a "
            f"kernel's returned status code, so intentional-argument-error cases are not bridgeable."
        )
    if operator != "PReLU" or activation_dtype not in ("S8", "S16"):
        raise UnsupportedGeneratedTestError(
            f"{generated_test.name}: operator={operator!r} activation_dtype={activation_dtype!r} is not "
            f"bridgeable -- perf-stream firmware only dispatches arm_prelu_s8/s16."
        )

    header_path = _find_header_file(generated_test.directory)
    header_text = header_path.read_text(encoding="utf-8")
    source_path = _find_source_file(generated_test.directory)
    source_text = source_path.read_text(encoding="utf-8")
    prefix = generated_test.name

    input_dims = _extract_dims(header_text, f"{prefix}_input_dims")
    alpha_dims = _extract_dims(header_text, f"{prefix}_alpha_dims")
    output_dims = _extract_dims(header_text, f"{prefix}_output_dims")
    if input_dims["n"] != 1 or alpha_dims["n"] != 1:
        raise UnsupportedGeneratedTestError(
            f"{generated_test.name}: batch size > 1 is not yet supported by the perf-stream hardware "
            f"bridge (firmware dispatches a single PReLU invocation per case)."
        )
    input_shape = (input_dims["n"], input_dims["h"], input_dims["w"], input_dims["c"])
    alpha_shape = (alpha_dims["n"], alpha_dims["h"], alpha_dims["w"], alpha_dims["c"])
    output_shape = (output_dims["n"], output_dims["h"], output_dims["w"], output_dims["c"])

    numpy_dtype = np.int16 if activation_dtype == "S16" else np.int8
    input_flat = np.array(_extract_array(header_text, f"{prefix}_input"), dtype=numpy_dtype)
    alpha_flat = np.array(_extract_array(header_text, f"{prefix}_alpha"), dtype=numpy_dtype)
    expected_flat = np.array(_extract_array(header_text, f"{prefix}_expected_output"), dtype=numpy_dtype)
    if input_flat.size != int(np.prod(input_shape)) or alpha_flat.size != int(np.prod(alpha_shape)):
        raise UnsupportedGeneratedTestError(
            f"{generated_test.name}: generated array sizes (input={input_flat.size}, "
            f"alpha={alpha_flat.size}) don't match header dims (input_shape={input_shape}, "
            f"alpha_shape={alpha_shape})."
        )
    if expected_flat.size != int(np.prod(output_shape)):
        raise UnsupportedGeneratedTestError(
            f"{generated_test.name}: expected_output size ({expected_flat.size}) does not match header "
            f"output_dims product ({int(np.prod(output_shape))})"
        )
    input_data = input_flat.reshape(input_shape)
    alpha_data = alpha_flat.reshape(alpha_shape)
    expected_output = expected_flat.reshape(output_shape)

    cmsis_function = "arm_prelu_s16" if activation_dtype == "S16" else "arm_prelu_s8"
    args = _extract_call_args(source_text, cmsis_function, expected_count=_PRELU_ARG_COUNT)
    input_offset, alpha_offset, output_offset = (int(args[4]), int(args[5]), int(args[6]))
    out_mult, out_shift = (int(args[7]), int(args[8]))
    out_mult_alpha, out_shift_alpha = (int(args[9]), int(args[10]))

    case_id = f"{generated_test.name}_hw_generated"
    bundle_root = output_root if output_root is not None else project_root
    case_root = _case_root(bundle_root, "ActivationFunctions", case_id)
    blobs_dir = case_root / "blobs"
    blobs_dir.mkdir(parents=True, exist_ok=True)

    arrays = [
        (1, "input_0", activation_dtype, input_shape, input_data, False, False),
        (2, "input_1", activation_dtype, alpha_shape, alpha_data, False, False),
        (3, "expected_output", activation_dtype, tuple(int(v) for v in expected_output.shape), expected_output, False, True),
    ]
    blobs: list[BlobInfo] = []
    for blob_id, role, dtype, dims, array, mutable_data, host_only in arrays:
        path = blobs_dir / f"{role}.bin"
        _write_blob(path, np.asarray(array))
        blobs.append(_blob_info(path, blob_id=blob_id, role=role, dtype=dtype, dimensions=dims, mutable_data=mutable_data, host_only=host_only))

    descriptor_path = generated_test.directory / "descriptor.yaml"
    descriptor_text = descriptor_path.read_text(encoding="utf-8")
    manifest = {
        "schema_name": "hct.case_manifest",
        "schema_version": 1,
        "case_id": case_id,
        "descriptor_name": generated_test.name,
        "descriptor_path": str(descriptor_path.relative_to(project_root)) if descriptor_path.is_relative_to(project_root) else str(descriptor_path),
        "descriptor_sha256": hashlib.sha256(descriptor_text.encode("utf-8")).hexdigest(),
        "operator": operator,
        "family": "ActivationFunctions",
        "target_cpu": generated_test.cpu,
        "kernel_id": lookup_kernel_id(project_root, family="ActivationFunctions", operator=operator, dtype=activation_dtype),
        "adapter_metadata_schema": 1,
        "source": "generated_test_bridge",
        "serialized_scalar_parameters": {
            "input_offset": input_offset,
            "alpha_offset": alpha_offset,
            "output_offset": output_offset,
            "out_mult": out_mult,
            "out_shift": out_shift,
            "out_mult_alpha": out_mult_alpha,
            "out_shift_alpha": out_shift_alpha,
            "output_h": output_dims["h"],
            "output_w": output_dims["w"],
            "output_c": output_dims["c"],
        },
        "tensor_dtypes": {"input": activation_dtype, "output": activation_dtype},
        "blob_roles": [_manifest_blob_entry(blob) for blob in blobs],
        "expected_output": {"dtype": activation_dtype, "byte_length": blobs[-1].byte_length, "blob_id": blobs[-1].blob_id},
        "correctness_comparison": dict(descriptor.get("resolved_comparison", {"mode": "exact_int"})),
        "scratch_buffer": {"bytes": 0},
        "required_target_capabilities": [cmsis_function],
        "repeated_invocation_safe": True,
        "timing": {"warmups": 2, "samples": 5, "iterations_per_sample": 4, "min_cycles": 1024, "max_iterations": 256},
    }
    return CaseBundle(root_dir=case_root, manifest_path=_write_manifest(case_root, manifest), manifest=manifest, blobs=tuple(blobs))


# arm_prelu_scalar_s8/s16(scalar_vect, non_scalar_vect, scalar_is_input, input_offset,
#    alpha_offset, output_offset, output_mult_identity, output_shift_identity,
#    output_mult_alpha, output_shift_alpha, output, block_size) -- a direct flat-vector API
# (no cmsis_nn_dims at all) used when one side is a true per-pixel scalar. Every real
# generated test calls this exactly once (num_pixels == 1 in this corpus), matching the
# perf-stream firmware's single-invocation-per-case model, so the multi-pixel loop the
# generator's own template supports (for num_pixels > 1) is out of scope here.
_PRELU_SCALAR_ARG_COUNT = 12


def _build_prelu_scalar_case(
    project_root: Path,
    generated_test: GeneratedTestCase,
    *,
    output_root: Path | None = None,
) -> CaseBundle:
    """Bridge an ActivationFunctions PReLUScalar generated test (S8 or S16). Unlike
    dims-based PReLU, this is a flat-vector API: one operand is a true scalar (1 element),
    the other a `block_size`-element vector -- there's no cmsis_nn_dims/broadcast machinery
    at all. `scalar_is_input` is always `true` in the generated harness, so only the
    single-pixel (num_pixels == 1) case is bridgeable with a 1:1 mapping to one firmware
    kernel call."""
    descriptor = generated_test.descriptor
    operator = str(descriptor.get("operator", ""))
    activation_dtype = str(descriptor.get("activation_dtype", ""))
    if operator != "PReLUScalar" or activation_dtype not in ("S8", "S16"):
        raise UnsupportedGeneratedTestError(
            f"{generated_test.name}: operator={operator!r} activation_dtype={activation_dtype!r} is not "
            f"bridgeable -- perf-stream firmware only dispatches arm_prelu_scalar_s8/s16."
        )
    num_pixels = int(np.prod(descriptor.get("input_shape", [1])))
    if num_pixels != 1:
        raise UnsupportedGeneratedTestError(
            f"{generated_test.name}: num_pixels={num_pixels} > 1 means the generated test calls "
            f"arm_prelu_scalar_{{s8,s16}} more than once -- the perf-stream firmware dispatches exactly "
            f"one kernel invocation per case, so multi-pixel PReLUScalar cases aren't bridgeable."
        )

    header_path = _find_header_file(generated_test.directory)
    header_text = header_path.read_text(encoding="utf-8")
    source_path = _find_source_file(generated_test.directory)
    source_text = source_path.read_text(encoding="utf-8")
    prefix = generated_test.name

    numpy_dtype = np.int16 if activation_dtype == "S16" else np.int8
    scalar_flat = np.array(_extract_array(header_text, f"{prefix}_scalar_input"), dtype=numpy_dtype)
    alpha_flat = np.array(_extract_array(header_text, f"{prefix}_alpha"), dtype=numpy_dtype)
    expected_flat = np.array(_extract_array(header_text, f"{prefix}_expected_output"), dtype=numpy_dtype)
    if scalar_flat.size != 1:
        raise UnsupportedGeneratedTestError(
            f"{generated_test.name}: scalar_input array has {scalar_flat.size} elements, expected exactly 1."
        )
    block_size = alpha_flat.size
    if expected_flat.size != block_size:
        raise UnsupportedGeneratedTestError(
            f"{generated_test.name}: expected_output size ({expected_flat.size}) does not match alpha "
            f"(block_size={block_size})."
        )

    cmsis_function = "arm_prelu_scalar_s16" if activation_dtype == "S16" else "arm_prelu_scalar_s8"
    args = _extract_call_args(source_text, cmsis_function, expected_count=_PRELU_SCALAR_ARG_COUNT)
    input_offset, alpha_offset, output_offset = (int(args[3]), int(args[4]), int(args[5]))
    out_mult, out_shift = (int(args[6]), int(args[7]))
    out_mult_alpha, out_shift_alpha = (int(args[8]), int(args[9]))

    case_id = f"{generated_test.name}_hw_generated"
    bundle_root = output_root if output_root is not None else project_root
    case_root = _case_root(bundle_root, "ActivationFunctions", case_id)
    blobs_dir = case_root / "blobs"
    blobs_dir.mkdir(parents=True, exist_ok=True)

    arrays = [
        (1, "input_0", activation_dtype, (1, 1, 1, 1), scalar_flat.reshape(1, 1, 1, 1), False, False),
        (2, "input_1", activation_dtype, (1, 1, 1, block_size), alpha_flat.reshape(1, 1, 1, block_size), False, False),
        (3, "expected_output", activation_dtype, (1, 1, 1, block_size), expected_flat.reshape(1, 1, 1, block_size), False, True),
    ]
    blobs: list[BlobInfo] = []
    for blob_id, role, dtype, dims, array, mutable_data, host_only in arrays:
        path = blobs_dir / f"{role}.bin"
        _write_blob(path, np.asarray(array))
        blobs.append(_blob_info(path, blob_id=blob_id, role=role, dtype=dtype, dimensions=dims, mutable_data=mutable_data, host_only=host_only))

    descriptor_path = generated_test.directory / "descriptor.yaml"
    descriptor_text = descriptor_path.read_text(encoding="utf-8")
    manifest = {
        "schema_name": "hct.case_manifest",
        "schema_version": 1,
        "case_id": case_id,
        "descriptor_name": generated_test.name,
        "descriptor_path": str(descriptor_path.relative_to(project_root)) if descriptor_path.is_relative_to(project_root) else str(descriptor_path),
        "descriptor_sha256": hashlib.sha256(descriptor_text.encode("utf-8")).hexdigest(),
        "operator": operator,
        "family": "ActivationFunctions",
        "target_cpu": generated_test.cpu,
        "kernel_id": lookup_kernel_id(project_root, family="ActivationFunctions", operator=operator, dtype=activation_dtype),
        "adapter_metadata_schema": 1,
        "source": "generated_test_bridge",
        "serialized_scalar_parameters": {
            "input_offset": input_offset,
            "alpha_offset": alpha_offset,
            "output_offset": output_offset,
            "out_mult": out_mult,
            "out_shift": out_shift,
            "out_mult_alpha": out_mult_alpha,
            "out_shift_alpha": out_shift_alpha,
            "block_size": block_size,
            "output_h": 1,
            "output_w": 1,
            "output_c": block_size,
        },
        "tensor_dtypes": {"input": activation_dtype, "output": activation_dtype},
        "blob_roles": [_manifest_blob_entry(blob) for blob in blobs],
        "expected_output": {"dtype": activation_dtype, "byte_length": blobs[-1].byte_length, "blob_id": blobs[-1].blob_id},
        "correctness_comparison": dict(descriptor.get("resolved_comparison", {"mode": "exact_int"})),
        "scratch_buffer": {"bytes": 0},
        "required_target_capabilities": [cmsis_function],
        "repeated_invocation_safe": True,
        "timing": {"warmups": 2, "samples": 5, "iterations_per_sample": 4, "min_cycles": 1024, "max_iterations": 256},
    }
    return CaseBundle(root_dir=case_root, manifest_path=_write_manifest(case_root, manifest), manifest=manifest, blobs=tuple(blobs))


# arm_softmax_s8/arm_softmax_s16/arm_softmax_s8_s16 all take a fixed
# (input, num_rows, row_size, mult, shift[, diff_min | softmax_params], output) call shape --
# num_rows/row_size/mult/shift are inlined literal call arguments (like the elementwise/
# activation ops above), so `_extract_call_args()` reads them positionally per kernel_fn.
_SOFTMAX_ARG_COUNT = 7


def _build_softmax_case(
    project_root: Path,
    generated_test: GeneratedTestCase,
    *,
    output_root: Path | None = None,
) -> CaseBundle:
    """Bridge a SoftmaxFunctions Softmax generated test. Three CMSIS-NN kernels are used
    depending on the descriptor's (activation_dtype, hint.force_cmsis, hint.output_dtype)
    combination -- detected here directly from which kernel function the generator actually
    emitted in the generated `.c` file (the descriptor's `operator` field is always
    "Softmax" for all three, so the kernel variant can't be read off it alone):
      - `arm_softmax_s8`      (S8 in, S8 out)   -- kernel_id lookup dtype S8
      - `arm_softmax_s16`     (S16 in, S16 out, needs the fixed CMSIS-NN LUT tables
                               embedded once in firmware) -- kernel_id lookup dtype S16
      - `arm_softmax_s8_s16`  (S8 in, S16 out, the `force_cmsis`+`output_dtype: S16` hint
                               combination) -- registered under operator "SoftmaxS8S16"
    """
    descriptor = generated_test.descriptor
    operator = str(descriptor.get("operator", ""))
    if operator != "Softmax":
        raise UnsupportedGeneratedTestError(
            f"{generated_test.name}: operator={operator!r} is not bridgeable -- perf-stream firmware "
            f"only dispatches arm_softmax_s8/s16/s8_s16."
        )

    source_path = _find_source_file(generated_test.directory)
    source_text = source_path.read_text(encoding="utf-8")
    header_path = _find_header_file(generated_test.directory)
    header_text = header_path.read_text(encoding="utf-8")
    prefix = generated_test.name

    if re.search(r"\barm_softmax_s8_s16\s*\(", source_text):
        cmsis_function = "arm_softmax_s8_s16"
        lookup_operator, lookup_dtype = "SoftmaxS8S16", "S8"
        input_dtype, output_dtype = "S8", "S16"
    elif re.search(r"\barm_softmax_s16\s*\(", source_text):
        cmsis_function = "arm_softmax_s16"
        lookup_operator, lookup_dtype = "Softmax", "S16"
        input_dtype = output_dtype = "S16"
    elif re.search(r"\barm_softmax_s8\s*\(", source_text):
        cmsis_function = "arm_softmax_s8"
        lookup_operator, lookup_dtype = "Softmax", "S8"
        input_dtype = output_dtype = "S8"
    else:
        raise UnsupportedGeneratedTestError(
            f"{generated_test.name}: no arm_softmax_s8/s16/s8_s16 call found in generated source."
        )

    args = _extract_call_args(source_text, cmsis_function, expected_count=_SOFTMAX_ARG_COUNT)
    num_rows, row_size, mult, shift = (int(args[1]), int(args[2]), int(args[3]), int(args[4]))
    diff_min = 0 if cmsis_function == "arm_softmax_s16" else int(args[5])
    size = num_rows * row_size

    input_numpy_dtype = np.int16 if input_dtype == "S16" else np.int8
    output_numpy_dtype = np.int16 if output_dtype == "S16" else np.int8
    input_flat = np.array(_extract_array(header_text, f"{prefix}_input"), dtype=input_numpy_dtype)
    expected_flat = np.array(_extract_array(header_text, f"{prefix}_expected_output"), dtype=output_numpy_dtype)
    if input_flat.size != size or expected_flat.size != size:
        raise UnsupportedGeneratedTestError(
            f"{generated_test.name}: array sizes (input={input_flat.size}, "
            f"expected_output={expected_flat.size}) don't match num_rows*row_size ({size})."
        )

    case_id = f"{generated_test.name}_hw_generated"
    bundle_root = output_root if output_root is not None else project_root
    case_root = _case_root(bundle_root, "SoftmaxFunctions", case_id)
    blobs_dir = case_root / "blobs"
    blobs_dir.mkdir(parents=True, exist_ok=True)

    arrays = [
        (1, "input_0", input_dtype, (1, 1, 1, size), input_flat.reshape(1, 1, 1, size), False, False),
        (2, "expected_output", output_dtype, (1, 1, 1, size), expected_flat.reshape(1, 1, 1, size), False, True),
    ]
    blobs: list[BlobInfo] = []
    for blob_id, role, dtype, dims, array, mutable_data, host_only in arrays:
        path = blobs_dir / f"{role}.bin"
        _write_blob(path, np.asarray(array))
        blobs.append(_blob_info(path, blob_id=blob_id, role=role, dtype=dtype, dimensions=dims, mutable_data=mutable_data, host_only=host_only))

    descriptor_path = generated_test.directory / "descriptor.yaml"
    descriptor_text = descriptor_path.read_text(encoding="utf-8")
    manifest = {
        "schema_name": "hct.case_manifest",
        "schema_version": 1,
        "case_id": case_id,
        "descriptor_name": generated_test.name,
        "descriptor_path": str(descriptor_path.relative_to(project_root)) if descriptor_path.is_relative_to(project_root) else str(descriptor_path),
        "descriptor_sha256": hashlib.sha256(descriptor_text.encode("utf-8")).hexdigest(),
        "operator": lookup_operator,
        "family": "SoftmaxFunctions",
        "target_cpu": generated_test.cpu,
        "kernel_id": lookup_kernel_id(project_root, family="SoftmaxFunctions", operator=lookup_operator, dtype=lookup_dtype),
        "adapter_metadata_schema": 1,
        "source": "generated_test_bridge",
        "serialized_scalar_parameters": {
            "num_rows": num_rows,
            "row_size": row_size,
            "out_mult": mult,
            "out_shift": shift,
            "diff_min": diff_min,
        },
        "tensor_dtypes": {"input": input_dtype, "output": output_dtype},
        "blob_roles": [_manifest_blob_entry(blob) for blob in blobs],
        "expected_output": {"dtype": output_dtype, "byte_length": blobs[-1].byte_length, "blob_id": blobs[-1].blob_id},
        # Softmax's generated harness (softmax.c.j2's HELIA_VALIDATE_OUTPUTS call) always
        # validates with TOLERANT_INT tolerance=1, NOT the generic dtype-based exact_int
        # default resolve_comparison() computes for resolved_comparison -- the same
        # template-vs-descriptor fidelity gap documented for Quantize (see
        # _build_quantize_case()): the non-force_cmsis golden output comes from a full
        # TFLite/LiteRT quantized graph, not the raw kernel math, so it can differ by up
        # to 1 ULP. Match the template's real validation semantics here.
        "correctness_comparison": {"mode": "tolerant_int", "tolerance": 1},
        "scratch_buffer": {"bytes": 0},
        "required_target_capabilities": [cmsis_function],
        "repeated_invocation_safe": True,
        "timing": {"warmups": 2, "samples": 5, "iterations_per_sample": 4, "min_cycles": 1024, "max_iterations": 256},
    }
    return CaseBundle(root_dir=case_root, manifest_path=_write_manifest(case_root, manifest), manifest=manifest, blobs=tuple(blobs))

# arm_abs_{s8,s16}(input, input_offset, output, out_offset, out_mult, out_shift,
#    needs_rescale, out_activation_min, out_activation_max, block_size)
_ABS_ARG_COUNT = 10


def _reduced_dims_from_axis(input_dims: dict[str, int], axis: int) -> dict[str, int]:
    if axis < 0 or axis > 3:
        raise UnsupportedGeneratedTestError(f"Unsupported reduction axis {axis}; expected a value in [0, 3].")
    output_dims = dict(input_dims)
    output_dims[("n", "h", "w", "c")[axis]] = 1
    return output_dims


def _build_abs_case(
    project_root: Path,
    generated_test: GeneratedTestCase,
    *,
    output_root: Path | None = None,
) -> CaseBundle:
    descriptor = generated_test.descriptor
    operator = str(descriptor.get("operator", ""))
    activation_dtype = str(descriptor.get("activation_dtype", ""))
    if operator != "Abs" or activation_dtype not in ("S8", "S16"):
        raise UnsupportedGeneratedTestError(
            f"{generated_test.name}: operator={operator!r} activation_dtype={activation_dtype!r} is not "
            f"bridgeable -- perf-stream firmware only dispatches arm_abs_s8/s16."
        )

    header_path = _find_header_file(generated_test.directory)
    header_text = header_path.read_text(encoding="utf-8")
    source_path = _find_source_file(generated_test.directory)
    source_text = source_path.read_text(encoding="utf-8")
    prefix = generated_test.name

    input_dims = _extract_dims(header_text, f"{prefix}_input_dims")
    output_dims = _extract_dims(header_text, f"{prefix}_output_dims")
    input_shape = (input_dims["n"], input_dims["h"], input_dims["w"], input_dims["c"])
    output_shape = (output_dims["n"], output_dims["h"], output_dims["w"], output_dims["c"])

    numpy_dtype = np.int16 if activation_dtype == "S16" else np.int8
    input_flat = np.array(_extract_array(header_text, f"{prefix}_input"), dtype=numpy_dtype)
    expected_flat = np.array(_extract_array(header_text, f"{prefix}_expected_output"), dtype=numpy_dtype)
    if input_flat.size != int(np.prod(input_shape)) or expected_flat.size != int(np.prod(output_shape)):
        raise UnsupportedGeneratedTestError(
            f"{generated_test.name}: generated array sizes (input={input_flat.size}, expected_output={expected_flat.size}) "
            f"don't match header dims (input_shape={input_shape}, output_shape={output_shape})."
        )
    input_data = input_flat.reshape(input_shape)
    expected_output = expected_flat.reshape(output_shape)

    cmsis_function = "arm_abs_s16" if activation_dtype == "S16" else "arm_abs_s8"
    args = _extract_call_args(source_text, cmsis_function, expected_count=_ABS_ARG_COUNT)
    scalar_parameters = {
        "input_offset": int(args[1]),
        "output_offset": int(args[3]),
        "out_mult": int(args[4]),
        "out_shift": int(args[5]),
        "needs_rescale": int(args[6]),
        "activation_min": int(args[7]),
        "activation_max": int(args[8]),
    }

    case_id = f"{generated_test.name}_hw_generated"
    bundle_root = output_root if output_root is not None else project_root
    case_root = _case_root(bundle_root, generated_test.family, case_id)
    blobs_dir = case_root / "blobs"
    blobs_dir.mkdir(parents=True, exist_ok=True)

    arrays = [
        (1, "input_0", activation_dtype, input_shape, input_data, False, False),
        (2, "expected_output", activation_dtype, output_shape, expected_output, False, True),
    ]
    blobs: list[BlobInfo] = []
    for blob_id, role, dtype, dims, array, mutable_data, host_only in arrays:
        path = blobs_dir / f"{role}.bin"
        _write_blob(path, np.asarray(array))
        blobs.append(_blob_info(path, blob_id=blob_id, role=role, dtype=dtype, dimensions=dims, mutable_data=mutable_data, host_only=host_only))

    descriptor_path = generated_test.directory / "descriptor.yaml"
    descriptor_text = descriptor_path.read_text(encoding="utf-8")
    manifest = {
        "schema_name": "hct.case_manifest",
        "schema_version": 1,
        "case_id": case_id,
        "descriptor_name": generated_test.name,
        "descriptor_path": str(descriptor_path.relative_to(project_root)) if descriptor_path.is_relative_to(project_root) else str(descriptor_path),
        "descriptor_sha256": hashlib.sha256(descriptor_text.encode("utf-8")).hexdigest(),
        "operator": operator,
        "family": generated_test.family,
        "target_cpu": generated_test.cpu,
        "kernel_id": lookup_kernel_id(project_root, family=generated_test.family, operator=operator, dtype=activation_dtype),
        "adapter_metadata_schema": 1,
        "source": "generated_test_bridge",
        "serialized_scalar_parameters": scalar_parameters,
        "tensor_dtypes": {"input": activation_dtype, "output": activation_dtype},
        "blob_roles": [_manifest_blob_entry(blob) for blob in blobs],
        "expected_output": {"dtype": activation_dtype, "byte_length": blobs[-1].byte_length, "blob_id": blobs[-1].blob_id},
        "correctness_comparison": dict(descriptor.get("resolved_comparison", {"mode": "exact_int"})),
        "scratch_buffer": {"bytes": 0},
        "required_target_capabilities": [cmsis_function],
        "repeated_invocation_safe": True,
        "timing": {"warmups": 2, "samples": 5, "iterations_per_sample": 4, "min_cycles": 1024, "max_iterations": 256},
    }
    return CaseBundle(root_dir=case_root, manifest_path=_write_manifest(case_root, manifest), manifest=manifest, blobs=tuple(blobs))


_ARG_REDUCTION_FUNCTIONS = {
    ("ArgMax", "S8"): "arm_argmax_s8",
    ("ArgMax", "S16"): "arm_argmax_s16",
    ("ArgMin", "S8"): "arm_argmin_s8",
    ("ArgMin", "S16"): "arm_argmin_s16",
}
_ARG_REDUCTION_ARG_COUNT = 4

_AXIS_REDUCTION_FUNCTIONS = {
    ("Mean", "S8"): "arm_mean_s8",
    ("Mean", "S16"): "arm_mean_s16",
    ("ReduceMax", "S8"): "arm_reduce_max_s8",
    ("ReduceMax", "S16"): "arm_reduce_max_s16",
    ("ReduceMin", "S8"): "arm_reduce_min_s8",
    ("ReduceMin", "S16"): "arm_reduce_min_s16",
}
_MEAN_ARG_COUNT = 9


def _build_basic_math_reduction_case(
    project_root: Path,
    generated_test: GeneratedTestCase,
    *,
    output_root: Path | None = None,
) -> CaseBundle:
    descriptor = generated_test.descriptor
    operator = str(descriptor.get("operator", ""))
    activation_dtype = str(descriptor.get("activation_dtype", ""))
    if (operator, activation_dtype) not in _ARG_REDUCTION_FUNCTIONS and (operator, activation_dtype) not in _AXIS_REDUCTION_FUNCTIONS:
        raise UnsupportedGeneratedTestError(
            f"{generated_test.name}: operator={operator!r} activation_dtype={activation_dtype!r} is not "
            f"bridgeable by the BasicMathFunctions reduction adapter."
        )

    header_path = _find_header_file(generated_test.directory)
    header_text = header_path.read_text(encoding="utf-8")
    prefix = generated_test.name
    input_dims = _extract_dims(header_text, f"{prefix}_input_dims")
    input_shape = (input_dims["n"], input_dims["h"], input_dims["w"], input_dims["c"])
    input_numpy_dtype = np.int16 if activation_dtype == "S16" else np.int8
    input_flat = np.array(_extract_array(header_text, f"{prefix}_input"), dtype=input_numpy_dtype)
    if input_flat.size != int(np.prod(input_shape)):
        raise UnsupportedGeneratedTestError(
            f"{generated_test.name}: generated input array size ({input_flat.size}) doesn't match header input_dims ({input_shape})."
        )
    input_data = input_flat.reshape(input_shape)

    scalar_parameters: dict[str, int]
    if (operator, activation_dtype) in _ARG_REDUCTION_FUNCTIONS:
        source_path = _find_source_file(generated_test.directory)
        source_text = source_path.read_text(encoding="utf-8")
        cmsis_function = _ARG_REDUCTION_FUNCTIONS[(operator, activation_dtype)]
        args = _extract_call_args(source_text, cmsis_function, expected_count=_ARG_REDUCTION_ARG_COUNT)
        axis = int(args[2])
        output_dims = _reduced_dims_from_axis(input_dims, axis)
        output_shape = (output_dims["n"], output_dims["h"], output_dims["w"], output_dims["c"])
        expected_flat = np.array(_extract_array(header_text, f"{prefix}_expected_output"), dtype=np.int32)
        scalar_parameters = {
            "axis": axis,
            "output_n": output_dims["n"],
            "output_h": output_dims["h"],
            "output_w": output_dims["w"],
            "output_c": output_dims["c"],
        }
        output_dtype = "S32"
    else:
        output_dims = _extract_dims(header_text, f"{prefix}_output_dims")
        axis_dims = _extract_dims(header_text, f"{prefix}_axis_dims")
        output_shape = (output_dims["n"], output_dims["h"], output_dims["w"], output_dims["c"])
        expected_flat = np.array(_extract_array(header_text, f"{prefix}_expected_output"), dtype=input_numpy_dtype)
        scalar_parameters = {
            "output_n": output_dims["n"],
            "output_h": output_dims["h"],
            "output_w": output_dims["w"],
            "output_c": output_dims["c"],
            "axis_n": axis_dims["n"],
            "axis_h": axis_dims["h"],
            "axis_w": axis_dims["w"],
            "axis_c": axis_dims["c"],
        }
        if operator == "Mean":
            source_path = _find_source_file(generated_test.directory)
            source_text = source_path.read_text(encoding="utf-8")
            cmsis_function = _AXIS_REDUCTION_FUNCTIONS[(operator, activation_dtype)]
            args = _extract_call_args(source_text, cmsis_function, expected_count=_MEAN_ARG_COUNT)
            scalar_parameters.update(
                {
                    "input_offset": int(args[2]),
                    "output_offset": int(args[6]),
                    "out_mult": int(args[7]),
                    "out_shift": int(args[8]),
                }
            )
        else:
            cmsis_function = _AXIS_REDUCTION_FUNCTIONS[(operator, activation_dtype)]
        output_dtype = activation_dtype

    if expected_flat.size != int(np.prod(output_shape)):
        raise UnsupportedGeneratedTestError(
            f"{generated_test.name}: expected_output size ({expected_flat.size}) does not match output dims ({output_shape})."
        )
    expected_output = expected_flat.reshape(output_shape)

    case_id = f"{generated_test.name}_hw_generated"
    bundle_root = output_root if output_root is not None else project_root
    case_root = _case_root(bundle_root, generated_test.family, case_id)
    blobs_dir = case_root / "blobs"
    blobs_dir.mkdir(parents=True, exist_ok=True)

    arrays = [
        (1, "input_0", activation_dtype, input_shape, input_data, False, False),
        (2, "expected_output", output_dtype, output_shape, expected_output, False, True),
    ]
    blobs: list[BlobInfo] = []
    for blob_id, role, dtype, dims, array, mutable_data, host_only in arrays:
        path = blobs_dir / f"{role}.bin"
        _write_blob(path, np.asarray(array))
        blobs.append(_blob_info(path, blob_id=blob_id, role=role, dtype=dtype, dimensions=dims, mutable_data=mutable_data, host_only=host_only))

    descriptor_path = generated_test.directory / "descriptor.yaml"
    descriptor_text = descriptor_path.read_text(encoding="utf-8")
    comparison = {"mode": "tolerant_int", "tolerance": 1} if operator == "Mean" else dict(
        descriptor.get("resolved_comparison", {"mode": "exact_int"})
    )
    manifest = {
        "schema_name": "hct.case_manifest",
        "schema_version": 1,
        "case_id": case_id,
        "descriptor_name": generated_test.name,
        "descriptor_path": str(descriptor_path.relative_to(project_root)) if descriptor_path.is_relative_to(project_root) else str(descriptor_path),
        "descriptor_sha256": hashlib.sha256(descriptor_text.encode("utf-8")).hexdigest(),
        "operator": operator,
        "family": generated_test.family,
        "target_cpu": generated_test.cpu,
        "kernel_id": lookup_kernel_id(project_root, family=generated_test.family, operator=operator, dtype=activation_dtype),
        "adapter_metadata_schema": 1,
        "source": "generated_test_bridge",
        "serialized_scalar_parameters": scalar_parameters,
        "tensor_dtypes": {"input": activation_dtype, "output": output_dtype},
        "blob_roles": [_manifest_blob_entry(blob) for blob in blobs],
        "expected_output": {"dtype": output_dtype, "byte_length": blobs[-1].byte_length, "blob_id": blobs[-1].blob_id},
        "correctness_comparison": comparison,
        "scratch_buffer": {"bytes": 0},
        "required_target_capabilities": [cmsis_function],
        "repeated_invocation_safe": True,
        "timing": {"warmups": 2, "samples": 5, "iterations_per_sample": 4, "min_cycles": 1024, "max_iterations": 256},
    }
    return CaseBundle(root_dir=case_root, manifest_path=_write_manifest(case_root, manifest), manifest=manifest, blobs=tuple(blobs))


_RSQRT_ARG_COUNTS = {"arm_rsqrt_s16_per_op": 8, "arm_rsqrt_s16_universal": 11}


def _build_basic_math_lut_case(
    project_root: Path,
    generated_test: GeneratedTestCase,
    *,
    output_root: Path | None = None,
) -> CaseBundle:
    descriptor = generated_test.descriptor
    operator = str(descriptor.get("operator", ""))
    activation_dtype = str(descriptor.get("activation_dtype", ""))
    if operator == "Rsqrt" and activation_dtype != "S16":
        raise UnsupportedGeneratedTestError(
            f"{generated_test.name}: Rsqrt is only bridgeable for activation_dtype='S16'."
        )
    if operator not in ("Sqrt", "Rsqrt") or activation_dtype not in ("S8", "S16"):
        raise UnsupportedGeneratedTestError(
            f"{generated_test.name}: operator={operator!r} activation_dtype={activation_dtype!r} is not "
            f"bridgeable by the BasicMathFunctions LUT adapter."
        )

    header_path = _find_header_file(generated_test.directory)
    header_text = header_path.read_text(encoding="utf-8")
    prefix = generated_test.name

    input_dims = _extract_dims(header_text, f"{prefix}_input_dims")
    input_shape = (input_dims["n"], input_dims["h"], input_dims["w"], input_dims["c"])
    numpy_dtype = np.int16 if activation_dtype == "S16" else np.int8
    input_flat = np.array(_extract_array(header_text, f"{prefix}_input"), dtype=numpy_dtype)
    if input_flat.size != int(np.prod(input_shape)):
        raise UnsupportedGeneratedTestError(
            f"{generated_test.name}: generated input array size ({input_flat.size}) doesn't match header input_dims ({input_shape})."
        )
    input_data = input_flat.reshape(input_shape)

    if operator == "Sqrt":
        output_shape = input_shape
        expected_flat = np.array(_extract_array(header_text, f"{prefix}_expected_output"), dtype=numpy_dtype)
        cmsis_function = f"arm_sqrt_{'s16' if activation_dtype == 'S16' else 's8'}"
        lut_dtype = activation_dtype
        lut_name = "sqrt_lut"
        scalar_parameters: dict[str, int] = {}
        registry_operator = operator
    else:
        output_dims = _extract_dims(header_text, f"{prefix}_output_dims")
        output_shape = (output_dims["n"], output_dims["h"], output_dims["w"], output_dims["c"])
        expected_flat = np.array(_extract_array(header_text, f"{prefix}_expected_output"), dtype=np.int16)
        call_style = str(descriptor.get("hint", {}).get("call_style", "per_op")).lower()
        if call_style == "universal":
            cmsis_function = "arm_rsqrt_s16_universal"
            lut_dtype = "S32"
            scalar_keys = ("input_offset", "output_offset", "out_mult", "out_shift", "needs_rescale", "activation_min", "activation_max")
            arg_indexes = (1, 3, 4, 5, 6, 7, 8)
            registry_operator = "RsqrtUniversal"
        else:
            cmsis_function = "arm_rsqrt_s16_per_op"
            lut_dtype = "S16"
            scalar_keys = ("input_offset", "output_offset", "activation_min", "activation_max")
            arg_indexes = (1, 3, 4, 5)
            registry_operator = operator
        source_path = _find_source_file(generated_test.directory)
        source_text = source_path.read_text(encoding="utf-8")
        args = _extract_call_args(source_text, cmsis_function, expected_count=_RSQRT_ARG_COUNTS[cmsis_function])
        scalar_parameters = {key: int(args[index]) for key, index in zip(scalar_keys, arg_indexes)}
        lut_name = f"{prefix}_rsqrt_lut"

    if expected_flat.size != int(np.prod(output_shape)):
        raise UnsupportedGeneratedTestError(
            f"{generated_test.name}: expected_output size ({expected_flat.size}) does not match output shape ({output_shape})."
        )
    expected_output = expected_flat.reshape(output_shape)

    lut_numpy_dtype = {"S8": np.int8, "S16": np.int16, "S32": np.int32}[lut_dtype]
    lut_flat = np.array(_extract_array(header_text, lut_name), dtype=lut_numpy_dtype)
    lut_shape = (int(lut_flat.size),)

    case_id = f"{generated_test.name}_hw_generated"
    bundle_root = output_root if output_root is not None else project_root
    case_root = _case_root(bundle_root, generated_test.family, case_id)
    blobs_dir = case_root / "blobs"
    blobs_dir.mkdir(parents=True, exist_ok=True)

    arrays = [
        (1, "input_0", activation_dtype, input_shape, input_data, False, False),
        (2, "weights", lut_dtype, lut_shape, lut_flat, False, False),
        (3, "expected_output", activation_dtype, output_shape, expected_output, False, True),
    ]
    blobs: list[BlobInfo] = []
    for blob_id, role, dtype, dims, array, mutable_data, host_only in arrays:
        path = blobs_dir / f"{role}.bin"
        _write_blob(path, np.asarray(array))
        blobs.append(_blob_info(path, blob_id=blob_id, role=role, dtype=dtype, dimensions=dims, mutable_data=mutable_data, host_only=host_only))

    descriptor_path = generated_test.directory / "descriptor.yaml"
    descriptor_text = descriptor_path.read_text(encoding="utf-8")
    manifest = {
        "schema_name": "hct.case_manifest",
        "schema_version": 1,
        "case_id": case_id,
        "descriptor_name": generated_test.name,
        "descriptor_path": str(descriptor_path.relative_to(project_root)) if descriptor_path.is_relative_to(project_root) else str(descriptor_path),
        "descriptor_sha256": hashlib.sha256(descriptor_text.encode("utf-8")).hexdigest(),
        "operator": operator,
        "family": generated_test.family,
        "target_cpu": generated_test.cpu,
        "kernel_id": lookup_kernel_id(project_root, family=generated_test.family, operator=registry_operator, dtype=activation_dtype),
        "adapter_metadata_schema": 1,
        "source": "generated_test_bridge",
        "serialized_scalar_parameters": scalar_parameters,
        "tensor_dtypes": {"input": activation_dtype, "weights": lut_dtype, "output": activation_dtype},
        "blob_roles": [_manifest_blob_entry(blob) for blob in blobs],
        "expected_output": {"dtype": activation_dtype, "byte_length": blobs[-1].byte_length, "blob_id": blobs[-1].blob_id},
        "correctness_comparison": dict(descriptor.get("resolved_comparison", {"mode": "exact_int"})),
        "scratch_buffer": {"bytes": 0},
        "required_target_capabilities": [cmsis_function],
        "repeated_invocation_safe": True,
        "timing": {"warmups": 2, "samples": 5, "iterations_per_sample": 4, "min_cycles": 1024, "max_iterations": 256},
    }
    return CaseBundle(root_dir=case_root, manifest_path=_write_manifest(case_root, manifest), manifest=manifest, blobs=tuple(blobs))


# arm_add_s8/arm_sub_s8 share an identical CMSIS-NN signature and argument order:
#   (input1, &input1_dims, input2, &input2_dims,
#    input1_offset, input1_mult, input1_shift,
#    input2_offset, input2_mult, input2_shift,
#    left_shift,
#    output, &output_dims,
#    out_offset, out_mult, out_shift,
#    out_activation_min, out_activation_max)
_ELEMENTWISE_BINARY_ARG_COUNT = 18
_ELEMENTWISE_BINARY_CMSIS_FUNCTION = {
    ("Add", "S8"): "arm_add_s8",
    ("Add", "S16"): "arm_add_s16",
    ("Sub", "S8"): "arm_sub_s8",
    ("Sub", "S16"): "arm_sub_s16",
}
_ELEMENTWISE_BINARY_SUPPORTED_DTYPES = ("S8", "S16")

# arm_mul_s8(input1_data, &input1_dims, input2_data, &input2_dims,
#    input1_offset, input2_offset,
#    output_data, &output_dims,
#    out_offset, out_mult, out_shift,
#    out_activation_min, out_activation_max)
# Unlike Add/Sub, Mul has no per-input mult/shift or left_shift arguments.
_MUL_ARG_COUNT = 13


def _extract_elementwise_binary_tensors(
    header_text: str,
    prefix: str,
    generated_test: GeneratedTestCase,
    operator: str,
    activation_dtype: str = "S8",
    *,
    allow_batch: bool = False,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], np.ndarray, np.ndarray, np.ndarray, dict]:
    """Shared dims/array extraction for BasicMathFunctions binary elementwise ops
    (Add/Sub/Mul all share the same input1/input2/output_dims + input1/input2/expected_output
    array naming convention in the generated header). Supports both S8 and S16 activations --
    CMSIS-NN's S16 elementwise kernels are argument-for-argument identical to their S8
    counterparts, just with int16_t data."""
    numpy_dtype = np.int16 if activation_dtype == "S16" else np.int8
    input1_dims = _extract_dims(header_text, f"{prefix}_input1_dims")
    input2_dims = _extract_dims(header_text, f"{prefix}_input2_dims")
    output_dims = _extract_dims(header_text, f"{prefix}_output_dims")
    if not allow_batch and (input1_dims["n"] != 1 or input2_dims["n"] != 1):
        raise UnsupportedGeneratedTestError(
            f"{generated_test.name}: batch size > 1 is not yet supported by the perf-stream "
            f"hardware bridge (firmware dispatches a single {operator} invocation per case)."
        )
    input1_shape = (input1_dims["n"], input1_dims["h"], input1_dims["w"], input1_dims["c"])
    input2_shape = (input2_dims["n"], input2_dims["h"], input2_dims["w"], input2_dims["c"])
    output_shape = (output_dims["n"], output_dims["h"], output_dims["w"], output_dims["c"])

    input1_flat = np.array(_extract_array(header_text, f"{prefix}_input1"), dtype=numpy_dtype)
    input2_flat = np.array(_extract_array(header_text, f"{prefix}_input2"), dtype=numpy_dtype)
    expected_flat = np.array(_extract_array(header_text, f"{prefix}_expected_output"), dtype=numpy_dtype)
    if input1_flat.size != int(np.prod(input1_shape)) or input2_flat.size != int(np.prod(input2_shape)):
        raise UnsupportedGeneratedTestError(
            f"{generated_test.name}: generated array sizes (input1={input1_flat.size}, "
            f"input2={input2_flat.size}) don't match header dims (input1_shape={input1_shape}, "
            f"input2_shape={input2_shape})."
        )
    if expected_flat.size != int(np.prod(output_shape)):
        raise UnsupportedGeneratedTestError(
            f"{generated_test.name}: expected_output size ({expected_flat.size}) does not match "
            f"header output_dims product ({int(np.prod(output_shape))})"
        )
    input1_data = input1_flat.reshape(input1_shape)
    input2_data = input2_flat.reshape(input2_shape)
    expected_output = expected_flat.reshape(output_shape)
    return input1_shape, input2_shape, input1_data, input2_data, expected_output, output_dims


def _write_elementwise_binary_bundle(
    project_root: Path,
    generated_test: GeneratedTestCase,
    *,
    operator: str,
    cmsis_function: str,
    activation_dtype: str,
    output_dims: dict,
    input1_shape: tuple[int, int, int, int],
    input2_shape: tuple[int, int, int, int],
    input1_data: np.ndarray,
    input2_data: np.ndarray,
    expected_output: np.ndarray,
    scalar_parameters: dict[str, int],
    output_root: Path | None,
    include_output_n: bool = False,
) -> CaseBundle:
    """Shared CaseBundle assembly for BasicMathFunctions binary elementwise ops
    (Add/Sub/Mul/Maximum/Minimum), parameterized over activation_dtype (S8 or S16)."""
    descriptor = generated_test.descriptor
    case_id = f"{generated_test.name}_hw_generated"
    bundle_root = output_root if output_root is not None else project_root
    case_root = _case_root(bundle_root, generated_test.family, case_id)
    blobs_dir = case_root / "blobs"
    blobs_dir.mkdir(parents=True, exist_ok=True)

    arrays = [
        (1, "input_0", activation_dtype, input1_shape, input1_data, False, False),
        (2, "input_1", activation_dtype, input2_shape, input2_data, False, False),
        (3, "expected_output", activation_dtype, tuple(int(v) for v in expected_output.shape), expected_output, False, True),
    ]
    blobs: list[BlobInfo] = []
    for blob_id, role, dtype, dims, array, mutable_data, host_only in arrays:
        path = blobs_dir / f"{role}.bin"
        _write_blob(path, np.asarray(array))
        blobs.append(_blob_info(path, blob_id=blob_id, role=role, dtype=dtype, dimensions=dims, mutable_data=mutable_data, host_only=host_only))

    descriptor_path = generated_test.directory / "descriptor.yaml"
    descriptor_text = descriptor_path.read_text(encoding="utf-8")
    manifest = {
        "schema_name": "hct.case_manifest",
        "schema_version": 1,
        "case_id": case_id,
        "descriptor_name": generated_test.name,
        "descriptor_path": str(descriptor_path.relative_to(project_root)) if descriptor_path.is_relative_to(project_root) else str(descriptor_path),
        "descriptor_sha256": hashlib.sha256(descriptor_text.encode("utf-8")).hexdigest(),
        "operator": operator,
        "family": generated_test.family,
        "target_cpu": generated_test.cpu,
        "kernel_id": lookup_kernel_id(project_root, family=generated_test.family, operator=operator, dtype=activation_dtype),
        "adapter_metadata_schema": 1,
        "source": "generated_test_bridge",
        "serialized_scalar_parameters": {
            **scalar_parameters,
            **({"output_n": output_dims["n"]} if include_output_n else {}),
            "output_h": output_dims["h"],
            "output_w": output_dims["w"],
            "output_c": output_dims["c"],
        },
        "tensor_dtypes": {"input": activation_dtype, "output": activation_dtype},
        "blob_roles": [_manifest_blob_entry(blob) for blob in blobs],
        "expected_output": {"dtype": activation_dtype, "byte_length": blobs[-1].byte_length, "blob_id": blobs[-1].blob_id},
        "correctness_comparison": dict(descriptor.get("resolved_comparison", {"mode": "exact_int"})),
        "scratch_buffer": {"bytes": 0},
        "required_target_capabilities": [f"{cmsis_function}"],
        "repeated_invocation_safe": True,
        "timing": {"warmups": 2, "samples": 5, "iterations_per_sample": 4, "min_cycles": 1024, "max_iterations": 256},
    }
    return CaseBundle(root_dir=case_root, manifest_path=_write_manifest(case_root, manifest), manifest=manifest, blobs=tuple(blobs))



def _build_elementwise_binary_case(
    project_root: Path,
    generated_test: GeneratedTestCase,
    *,
    output_root: Path | None = None,
) -> CaseBundle:
    """Bridge a BasicMathFunctions Add/Sub generated test (S8 or S16). Unlike Convolve's
    `cmsis_nn_conv_params` struct, these ops have no named scalar-params struct in the
    generated header -- their quant scalars are inlined as call arguments in the generated
    `.c` file, so they're extracted positionally via `_extract_call_args` instead of
    `_extract_scalar`/`_extract_nested_scalar`.
    """
    descriptor = generated_test.descriptor
    operator = str(descriptor.get("operator", ""))
    activation_dtype = str(descriptor.get("activation_dtype", ""))
    if activation_dtype not in _ELEMENTWISE_BINARY_SUPPORTED_DTYPES:
        raise UnsupportedGeneratedTestError(
            f"{generated_test.name}: activation_dtype={activation_dtype!r} is not bridgeable -- "
            f"perf-stream firmware only dispatches {_ELEMENTWISE_BINARY_SUPPORTED_DTYPES} {operator} kernels."
        )

    header_path = _find_header_file(generated_test.directory)
    header_text = header_path.read_text(encoding="utf-8")
    source_path = _find_source_file(generated_test.directory)
    source_text = source_path.read_text(encoding="utf-8")
    prefix = generated_test.name

    input1_shape, input2_shape, input1_data, input2_data, expected_output, output_dims = (
        _extract_elementwise_binary_tensors(header_text, prefix, generated_test, operator, activation_dtype)
    )

    cmsis_function = _ELEMENTWISE_BINARY_CMSIS_FUNCTION[(operator, activation_dtype)]
    args = _extract_call_args(source_text, cmsis_function, expected_count=_ELEMENTWISE_BINARY_ARG_COUNT)
    input1_offset, input1_mult, input1_shift = (int(args[4]), int(args[5]), int(args[6]))
    input2_offset, input2_mult, input2_shift = (int(args[7]), int(args[8]), int(args[9]))
    left_shift = int(args[10])
    out_offset, out_mult, out_shift = (int(args[13]), int(args[14]), int(args[15]))
    activation_min, activation_max = (int(args[16]), int(args[17]))

    scalar_parameters = {
        "input1_offset": input1_offset,
        "input1_mult": input1_mult,
        "input1_shift": input1_shift,
        "input2_offset": input2_offset,
        "input2_mult": input2_mult,
        "input2_shift": input2_shift,
        "left_shift": left_shift,
        "output_offset": out_offset,
        "out_mult": out_mult,
        "out_shift": out_shift,
        "activation_min": activation_min,
        "activation_max": activation_max,
    }
    return _write_elementwise_binary_bundle(
        project_root,
        generated_test,
        operator=operator,
        cmsis_function=cmsis_function,
        activation_dtype=activation_dtype,
        output_dims=output_dims,
        input1_shape=input1_shape,
        input2_shape=input2_shape,
        input1_data=input1_data,
        input2_data=input2_data,
        expected_output=expected_output,
        scalar_parameters=scalar_parameters,
        output_root=output_root,
    )


def _build_mul_case(
    project_root: Path,
    generated_test: GeneratedTestCase,
    *,
    output_root: Path | None = None,
) -> CaseBundle:
    """Bridge a BasicMathFunctions Mul generated test (S8 or S16). `arm_mul_s8`/`arm_mul_s16`
    have a different (shorter) signature than Add/Sub -- no per-input mult/shift, no
    left_shift -- so it gets its own positional-argument mapping while sharing the
    tensor-extraction and bundle-assembly helpers.
    """
    descriptor = generated_test.descriptor
    operator = str(descriptor.get("operator", ""))
    activation_dtype = str(descriptor.get("activation_dtype", ""))
    if activation_dtype not in _ELEMENTWISE_BINARY_SUPPORTED_DTYPES:
        raise UnsupportedGeneratedTestError(
            f"{generated_test.name}: activation_dtype={activation_dtype!r} is not bridgeable -- "
            f"perf-stream firmware only dispatches {_ELEMENTWISE_BINARY_SUPPORTED_DTYPES} {operator} kernels."
        )

    header_path = _find_header_file(generated_test.directory)
    header_text = header_path.read_text(encoding="utf-8")
    source_path = _find_source_file(generated_test.directory)
    source_text = source_path.read_text(encoding="utf-8")
    prefix = generated_test.name

    input1_shape, input2_shape, input1_data, input2_data, expected_output, output_dims = (
        _extract_elementwise_binary_tensors(header_text, prefix, generated_test, operator, activation_dtype)
    )

    cmsis_function = "arm_mul_s16" if activation_dtype == "S16" else "arm_mul_s8"
    args = _extract_call_args(source_text, cmsis_function, expected_count=_MUL_ARG_COUNT)
    input1_offset, input2_offset = (int(args[4]), int(args[5]))
    out_offset, out_mult, out_shift = (int(args[8]), int(args[9]), int(args[10]))
    activation_min, activation_max = (int(args[11]), int(args[12]))

    scalar_parameters = {
        "input1_offset": input1_offset,
        "input2_offset": input2_offset,
        "output_offset": out_offset,
        "out_mult": out_mult,
        "out_shift": out_shift,
        "activation_min": activation_min,
        "activation_max": activation_max,
    }
    return _write_elementwise_binary_bundle(
        project_root,
        generated_test,
        operator=operator,
        cmsis_function=cmsis_function,
        activation_dtype=activation_dtype,
        output_dims=output_dims,
        input1_shape=input1_shape,
        input2_shape=input2_shape,
        input1_data=input1_data,
        input2_data=input2_data,
        expected_output=expected_output,
        scalar_parameters=scalar_parameters,
        output_root=output_root,
    )


def _build_min_max_case(
    project_root: Path,
    generated_test: GeneratedTestCase,
    *,
    output_root: Path | None = None,
) -> CaseBundle:
    """Bridge a BasicMathFunctions Maximum/Minimum generated test (S8 or S16). Unlike
    Add/Sub/Mul, these ops have NO quantization scalar parameters at all --
    `arm_maximum_s8/s16`/`arm_minimum_s8/s16` take only a scratch `cmsis_nn_context` (always
    `{NULL, 0}` in the generated tests; no buffer-sizing helper exists for these ops) plus the
    two input tensors and dims. No `_extract_call_args` positional extraction is needed since
    there are no scalars to read.
    """
    descriptor = generated_test.descriptor
    operator = str(descriptor.get("operator", ""))
    activation_dtype = str(descriptor.get("activation_dtype", ""))
    if activation_dtype not in _ELEMENTWISE_BINARY_SUPPORTED_DTYPES:
        raise UnsupportedGeneratedTestError(
            f"{generated_test.name}: activation_dtype={activation_dtype!r} is not bridgeable -- "
            f"perf-stream firmware only dispatches {_ELEMENTWISE_BINARY_SUPPORTED_DTYPES} {operator} kernels."
        )

    header_path = _find_header_file(generated_test.directory)
    header_text = header_path.read_text(encoding="utf-8")
    prefix = generated_test.name

    input1_shape, input2_shape, input1_data, input2_data, expected_output, output_dims = (
        _extract_elementwise_binary_tensors(header_text, prefix, generated_test, operator, activation_dtype)
    )

    dtype_suffix = "s16" if activation_dtype == "S16" else "s8"
    cmsis_function = f"arm_maximum_{dtype_suffix}" if operator == "Maximum" else f"arm_minimum_{dtype_suffix}"
    return _write_elementwise_binary_bundle(
        project_root,
        generated_test,
        operator=operator,
        cmsis_function=cmsis_function,
        activation_dtype=activation_dtype,
        output_dims=output_dims,
        input1_shape=input1_shape,
        input2_shape=input2_shape,
        input1_data=input1_data,
        input2_data=input2_data,
        expected_output=expected_output,
        scalar_parameters={},
        output_root=output_root,
    )


_ELEMENTWISE_SQUARED_DIFFERENCE_ARG_COUNT = 16


def _build_squared_difference_case(
    project_root: Path,
    generated_test: GeneratedTestCase,
    *,
    output_root: Path | None = None,
) -> CaseBundle:
    descriptor = generated_test.descriptor
    operator = str(descriptor.get("operator", ""))
    activation_dtype = str(descriptor.get("activation_dtype", ""))
    if operator != "SquaredDifference" or activation_dtype not in _ELEMENTWISE_BINARY_SUPPORTED_DTYPES:
        raise UnsupportedGeneratedTestError(
            f"{generated_test.name}: operator={operator!r} activation_dtype={activation_dtype!r} is not "
            f"bridgeable -- perf-stream firmware only dispatches arm_squared_difference_s8/s16."
        )

    header_path = _find_header_file(generated_test.directory)
    header_text = header_path.read_text(encoding="utf-8")
    source_path = _find_source_file(generated_test.directory)
    source_text = source_path.read_text(encoding="utf-8")
    prefix = generated_test.name

    input1_shape, input2_shape, input1_data, input2_data, expected_output, output_dims = _extract_elementwise_binary_tensors(
        header_text,
        prefix,
        generated_test,
        operator,
        activation_dtype,
        allow_batch=True,
    )

    cmsis_function = f"arm_squared_difference_{'s16' if activation_dtype == 'S16' else 's8'}"
    try:
        args = _extract_call_args(source_text, cmsis_function, expected_count=_ELEMENTWISE_BINARY_ARG_COUNT)
        scalar_parameters = {
            "input1_offset": int(args[4]),
            "input1_mult": int(args[5]),
            "input1_shift": int(args[6]),
            "input2_offset": int(args[7]),
            "input2_mult": int(args[8]),
            "input2_shift": int(args[9]),
            "left_shift": int(args[10]),
            "output_offset": int(args[13]),
            "out_mult": int(args[14]),
            "out_shift": int(args[15]),
            "activation_min": int(args[16]),
            "activation_max": int(args[17]),
        }
    except UnsupportedGeneratedTestError:
        elementwise_cmsis_function = f"arm_elementwise_squared_difference_{'s16' if activation_dtype == 'S16' else 's8'}"
        args = _extract_call_args(source_text, elementwise_cmsis_function, expected_count=_ELEMENTWISE_SQUARED_DIFFERENCE_ARG_COUNT)
        scalar_parameters = {
            "input1_offset": int(args[2]),
            "input1_mult": int(args[3]),
            "input1_shift": int(args[4]),
            "input2_offset": int(args[5]),
            "input2_mult": int(args[6]),
            "input2_shift": int(args[7]),
            "left_shift": int(args[8]),
            "output_offset": int(args[10]),
            "out_mult": int(args[11]),
            "out_shift": int(args[12]),
            "activation_min": int(args[13]),
            "activation_max": int(args[14]),
        }

    return _write_elementwise_binary_bundle(
        project_root,
        generated_test,
        operator=operator,
        cmsis_function=cmsis_function,
        activation_dtype=activation_dtype,
        output_dims=output_dims,
        input1_shape=input1_shape,
        input2_shape=input2_shape,
        input1_data=input1_data,
        input2_data=input2_data,
        expected_output=expected_output,
        scalar_parameters=scalar_parameters,
        output_root=output_root,
        include_output_n=True,
    )


def _build_fully_connected_case(
    project_root: Path,
    generated_test: GeneratedTestCase,
    *,
    output_root: Path | None = None,
) -> CaseBundle:
    """Bridge a FullyConnectedFunctions FullyConnected generated test. Firmware always
    dispatches through the per-channel wrapper kernels (arm_fully_connected_wrapper_s8/
    _s16, see run_fully_connected_once() in benchmark_server_session.c), so genuinely
    per-tensor-quantized descriptors (a bare `..._multiplier_val`/`..._shift_val` scalar in
    the generated header, rather than a `..._multiplier[]`/`..._shift[]` array) have their
    scalar multiplier/shift broadcast across every output channel here -- mathematically
    identical to the real per-tensor kernel, since per-tensor quantization is simply the
    degenerate case of per-channel with every channel equal. S4 weight_dtype is out of
    scope (matches the established Convolve S4 precedent: a completely different
    quantization scheme with no separate weight-sum requirement)."""
    descriptor = generated_test.descriptor
    operator = str(descriptor.get("operator", ""))
    weight_dtype = str(descriptor.get("weight_dtype", descriptor.get("resolved_tensor_dtypes", {}).get("weights", "")))
    activation_dtype = str(descriptor.get("activation_dtype", ""))

    if operator != "FullyConnected" or weight_dtype != "S8" or activation_dtype not in ("S8", "S16"):
        raise UnsupportedGeneratedTestError(
            f"{generated_test.name}: operator={operator!r} weight_dtype={weight_dtype!r} "
            f"activation_dtype={activation_dtype!r} is not bridgeable -- perf-stream firmware only "
            f"dispatches arm_fully_connected_wrapper_s8/s16 (S8 weight + S8 or S16 activation)."
        )

    header_path = _find_header_file(generated_test.directory)
    header_text = header_path.read_text(encoding="utf-8")
    prefix = generated_test.name

    input_dims = _extract_dims(header_text, f"{prefix}_input_dims")
    filter_dims = _extract_dims(header_text, f"{prefix}_filter_dims")
    output_dims = _extract_dims(header_text, f"{prefix}_output_dims")
    batch = input_dims["n"]
    input_features = filter_dims["n"]
    output_units = filter_dims["c"]
    input_shape = (batch, input_features)
    filter_shape = (output_units, input_features)
    output_shape = (output_dims["n"], output_dims["c"])

    activation_numpy_dtype = np.int16 if activation_dtype == "S16" else np.int8
    bias_numpy_dtype = np.int64 if activation_dtype == "S16" else np.int32
    bias_wire_dtype = "S64" if activation_dtype == "S16" else "S32"

    weights_flat = np.array(_extract_array(header_text, f"{prefix}_weights"), dtype=np.int8)
    input_flat = np.array(_extract_array(header_text, f"{prefix}_input"), dtype=activation_numpy_dtype)
    expected_flat = np.array(_extract_array(header_text, f"{prefix}_expected_output"), dtype=activation_numpy_dtype)

    try:
        biases_list = _extract_array(header_text, f"{prefix}_biases")
        has_bias = True
    except UnsupportedGeneratedTestError:
        biases_list = []
        has_bias = False
    biases = np.array(biases_list if has_bias else [0] * output_units, dtype=bias_numpy_dtype)

    multiplier_scalar = _extract_bare_scalar(header_text, f"{prefix}_multiplier_val")
    if multiplier_scalar is not None:
        shift_scalar = _extract_bare_scalar(header_text, f"{prefix}_shift_val")
        multiplier = np.full((output_units,), multiplier_scalar, dtype=np.int32)
        shift = np.full((output_units,), shift_scalar, dtype=np.int32)
    else:
        multiplier = np.array(_extract_array(header_text, f"{prefix}_multiplier"), dtype=np.int32)
        shift = np.array(_extract_array(header_text, f"{prefix}_shift"), dtype=np.int32)

    input_required = int(np.prod(input_shape))
    filter_required = int(np.prod(filter_shape))
    output_required = int(np.prod(output_shape))
    # The generated test harness's own struct dims always declare input_dims.n=1 (even for
    # descriptors named "batchN") and its `test_case_run()` only ever validates the first
    # `FULLY_CONNECTED_..._OUTPUT_SIZE` (= output_required) elements against
    # `_expected_output` -- the full `_input`/`_expected_output` header arrays retain the
    # complete original (possibly multi-row) LiteRT tensor for documentation/traceability,
    # but only their leading slice is ever fed to/compared against the real single-invocation
    # kernel call. Slice to match that same ground-truth single-invocation behavior rather
    # than rejecting these descriptors outright.
    if weights_flat.size != filter_required or input_flat.size < input_required or expected_flat.size < output_required:
        raise UnsupportedGeneratedTestError(
            f"{generated_test.name}: generated array sizes (input={input_flat.size}, "
            f"weights={weights_flat.size}, expected_output={expected_flat.size}) are smaller than the "
            f"header dims require (input_shape={input_shape}, filter_shape={filter_shape}, "
            f"output_shape={output_shape})."
        )
    weights = weights_flat.reshape(filter_shape)
    input_data = input_flat[:input_required].reshape(input_shape)
    expected_output = expected_flat[:output_required].reshape(output_shape)

    input_offset = _extract_scalar(header_text, f"{prefix}_fc_params", "input_offset")
    filter_offset = _extract_scalar(header_text, f"{prefix}_fc_params", "filter_offset")
    output_offset = _extract_scalar(header_text, f"{prefix}_fc_params", "output_offset")
    activation_min = _extract_nested_scalar(header_text, f"{prefix}_fc_params", "activation", "min")
    activation_max = _extract_nested_scalar(header_text, f"{prefix}_fc_params", "activation", "max")

    case_id = f"{generated_test.name}_hw_generated"
    bundle_root = output_root if output_root is not None else project_root
    case_root = _case_root(bundle_root, "FullyConnectedFunctions", case_id)
    blobs_dir = case_root / "blobs"
    blobs_dir.mkdir(parents=True, exist_ok=True)

    arrays = [
        (1, "input_0", activation_dtype, input_shape, input_data, False, False),
        (2, "weights", "S8", filter_shape, weights, False, False),
        (3, "bias", bias_wire_dtype, (output_units,), biases, False, False),
        (4, "multiplier", "S32", (output_units,), multiplier, False, False),
        (5, "shift", "S32", (output_units,), shift, False, False),
        (6, "expected_output", activation_dtype, output_shape, expected_output, False, True),
    ]
    blobs: list[BlobInfo] = []
    for blob_id, role, dtype, dims, array, mutable_data, host_only in arrays:
        path = blobs_dir / f"{role}.bin"
        _write_blob(path, np.asarray(array))
        blobs.append(_blob_info(path, blob_id=blob_id, role=role, dtype=dtype, dimensions=dims, mutable_data=mutable_data, host_only=host_only))

    descriptor_path = generated_test.directory / "descriptor.yaml"
    descriptor_text = descriptor_path.read_text(encoding="utf-8")
    manifest = {
        "schema_name": "hct.case_manifest",
        "schema_version": 1,
        "case_id": case_id,
        "descriptor_name": generated_test.name,
        "descriptor_path": str(descriptor_path.relative_to(project_root)) if descriptor_path.is_relative_to(project_root) else str(descriptor_path),
        "descriptor_sha256": hashlib.sha256(descriptor_text.encode("utf-8")).hexdigest(),
        "operator": operator,
        "family": "FullyConnectedFunctions",
        "target_cpu": generated_test.cpu,
        "kernel_id": lookup_kernel_id(project_root, family="FullyConnectedFunctions", operator="FullyConnected", dtype=activation_dtype),
        "adapter_metadata_schema": 1,
        "source": "generated_test_bridge",
        "serialized_scalar_parameters": {
            "input_offset": input_offset,
            "filter_offset": filter_offset,
            "output_offset": output_offset,
            "activation_min": activation_min,
            "activation_max": activation_max,
        },
        "tensor_dtypes": {"input": activation_dtype, "weights": "S8", "bias": bias_wire_dtype, "output": activation_dtype},
        "blob_roles": [_manifest_blob_entry(blob) for blob in blobs],
        "expected_output": {"dtype": activation_dtype, "byte_length": blobs[-1].byte_length, "blob_id": blobs[-1].blob_id},
        # FullyConnected's generated harness (fully_connected.c.j2's HELIA_VALIDATE_OUTPUTS
        # call) always validates with TOLERANT_INT tolerance=1, NOT the generic dtype-based
        # exact_int default resolve_comparison() computes for resolved_comparison -- same
        # template-vs-descriptor fidelity gap documented for Softmax/Quantize (see
        # _build_softmax_case()/_build_quantize_case()): requantization rounding can differ
        # by up to 1 ULP depending on intermediate accumulation order. Match the template's
        # real validation semantics here rather than the misleading descriptor metadata.
        "correctness_comparison": {"mode": "tolerant_int", "tolerance": 1},
        # ctx->buf sized output_units * sizeof(int32_t) for both S8 (kernel_sum, computed
        # at runtime via arm_vector_sum_s8) and S16 (scratch the kernel fills itself) --
        # see run_fully_connected_once()'s header comment and
        # arm_fully_connected_{s8,per_channel_s16}_get_buffer_size{,_mve}().
        "scratch_buffer": {"bytes": output_units * 4},
        "required_target_capabilities": [
            "fully_connected_s8" if activation_dtype == "S8" else "fully_connected_s16"
        ],
        "repeated_invocation_safe": True,
        "timing": {"warmups": 2, "samples": 5, "iterations_per_sample": 4, "min_cycles": 1024, "max_iterations": 256},
    }
    return CaseBundle(root_dir=case_root, manifest_path=_write_manifest(case_root, manifest), manifest=manifest, blobs=tuple(blobs))


def _build_batch_matmul_case(
    project_root: Path,
    generated_test: GeneratedTestCase,
    *,
    output_root: Path | None = None,
) -> CaseBundle:
    """Bridge a FullyConnectedFunctions BatchMatMul generated test. Unlike FullyConnected,
    both operands share the same dtype (S8 or S16) and quantization is always a single
    per-tensor {multiplier, shift} pair (no per-channel array blob) -- see
    run_batch_matmul_once() in benchmark_server_session.c. adj_x/adj_y are never
    transmitted: arm_batch_matmul_s8/_s16 never read them (the real generated test's
    transposed-operand descriptors already pre-arrange their raw lhs/rhs data/dims into
    the final row-major layout the kernel expects). The real generated test harness
    always uses a single-invocation shape (input_lhs_dims/input_rhs_dims/output_dims.n
    == .h == 1) regardless of the descriptor name implying multiple batches -- the same
    "batch is cosmetic at the single-invocation level" pattern already established for
    FullyConnected."""
    descriptor = generated_test.descriptor
    operator = str(descriptor.get("operator", ""))
    weight_dtype = str(descriptor.get("weight_dtype", descriptor.get("resolved_tensor_dtypes", {}).get("weights", "")))
    activation_dtype = str(descriptor.get("activation_dtype", ""))

    if operator != "BatchMatMul" or weight_dtype != "S8" or activation_dtype not in ("S8", "S16"):
        raise UnsupportedGeneratedTestError(
            f"{generated_test.name}: operator={operator!r} weight_dtype={weight_dtype!r} "
            f"activation_dtype={activation_dtype!r} is not bridgeable -- perf-stream firmware only "
            f"dispatches arm_batch_matmul_s8/s16 (S8 weight + S8 or S16 activation)."
        )

    header_path = _find_header_file(generated_test.directory)
    header_text = header_path.read_text(encoding="utf-8")
    prefix = generated_test.name

    input_lhs_dims = _extract_dims(header_text, f"{prefix}_input_lhs_dims")
    input_rhs_dims = _extract_dims(header_text, f"{prefix}_input_rhs_dims")
    output_dims = _extract_dims(header_text, f"{prefix}_output_dims")
    input_lhs_shape = (input_lhs_dims["w"], input_lhs_dims["c"])
    input_rhs_shape = (input_rhs_dims["w"], input_rhs_dims["c"])
    output_shape = (output_dims["w"], output_dims["c"])

    activation_numpy_dtype = np.int16 if activation_dtype == "S16" else np.int8

    input_lhs = np.array(_extract_array(header_text, f"{prefix}_input_lhs"), dtype=activation_numpy_dtype).reshape(input_lhs_shape)
    input_rhs = np.array(_extract_array(header_text, f"{prefix}_input_rhs"), dtype=activation_numpy_dtype).reshape(input_rhs_shape)
    expected_output = np.array(_extract_array(header_text, f"{prefix}_expected_output"), dtype=activation_numpy_dtype).reshape(output_shape)

    input_offset = _extract_scalar(header_text, f"{prefix}_bmm_params", "input_offset")
    filter_offset = _extract_scalar(header_text, f"{prefix}_bmm_params", "filter_offset")
    output_offset = _extract_scalar(header_text, f"{prefix}_bmm_params", "output_offset")
    activation_min = _extract_nested_scalar(header_text, f"{prefix}_bmm_params", "activation", "min")
    activation_max = _extract_nested_scalar(header_text, f"{prefix}_bmm_params", "activation", "max")
    multiplier = _extract_scalar(header_text, f"{prefix}_quant_params", "multiplier")
    shift = _extract_scalar(header_text, f"{prefix}_quant_params", "shift")

    case_id = f"{generated_test.name}_hw_generated"
    bundle_root = output_root if output_root is not None else project_root
    case_root = _case_root(bundle_root, "FullyConnectedFunctions", case_id)
    blobs_dir = case_root / "blobs"
    blobs_dir.mkdir(parents=True, exist_ok=True)

    arrays = [
        (1, "input_0", activation_dtype, input_lhs_shape, input_lhs, False, False),
        (2, "input_1", activation_dtype, input_rhs_shape, input_rhs, False, False),
        (3, "expected_output", activation_dtype, output_shape, expected_output, False, True),
    ]
    blobs: list[BlobInfo] = []
    for blob_id, role, dtype, dims, array, mutable_data, host_only in arrays:
        path = blobs_dir / f"{role}.bin"
        _write_blob(path, np.asarray(array))
        blobs.append(_blob_info(path, blob_id=blob_id, role=role, dtype=dtype, dimensions=dims, mutable_data=mutable_data, host_only=host_only))

    descriptor_path = generated_test.directory / "descriptor.yaml"
    descriptor_text = descriptor_path.read_text(encoding="utf-8")
    manifest = {
        "schema_name": "hct.case_manifest",
        "schema_version": 1,
        "case_id": case_id,
        "descriptor_name": generated_test.name,
        "descriptor_path": str(descriptor_path.relative_to(project_root)) if descriptor_path.is_relative_to(project_root) else str(descriptor_path),
        "descriptor_sha256": hashlib.sha256(descriptor_text.encode("utf-8")).hexdigest(),
        "operator": operator,
        "family": "FullyConnectedFunctions",
        "target_cpu": generated_test.cpu,
        "kernel_id": lookup_kernel_id(project_root, family="FullyConnectedFunctions", operator="BatchMatMul", dtype=activation_dtype),
        "adapter_metadata_schema": 1,
        "source": "generated_test_bridge",
        "serialized_scalar_parameters": {
            "input_offset": input_offset,
            "filter_offset": filter_offset,
            "output_offset": output_offset,
            "activation_min": activation_min,
            "activation_max": activation_max,
            "out_mult": multiplier,
            "out_shift": shift,
        },
        "tensor_dtypes": {"input_lhs": activation_dtype, "input_rhs": activation_dtype, "output": activation_dtype},
        "blob_roles": [_manifest_blob_entry(blob) for blob in blobs],
        "expected_output": {"dtype": activation_dtype, "byte_length": blobs[-1].byte_length, "blob_id": blobs[-1].blob_id},
        # BatchMatMul's generated harness (batch_matmul.c.j2's HELIA_VALIDATE_OUTPUTS call)
        # always validates with TOLERANT_INT tolerance=1, NOT the generic dtype-based
        # exact_int default resolve_comparison() computes -- same template-vs-descriptor
        # fidelity gap already documented for FullyConnected/Softmax/Quantize.
        "correctness_comparison": {"mode": "tolerant_int", "tolerance": 1},
        # ctx->buf sized rhs_cols * sizeof(int32_t) for S8 only (kernel-sum scratch the
        # kernel fills itself at runtime); S16 needs none. See run_batch_matmul_once().
        "scratch_buffer": {"bytes": input_rhs_shape[0] * 4 if activation_dtype == "S8" else 0},
        "required_target_capabilities": [
            "batch_matmul_s8" if activation_dtype == "S8" else "batch_matmul_s16"
        ],
        "repeated_invocation_safe": True,
        "timing": {"warmups": 2, "samples": 5, "iterations_per_sample": 4, "min_cycles": 1024, "max_iterations": 256},
    }
    return CaseBundle(root_dir=case_root, manifest_path=_write_manifest(case_root, manifest), manifest=manifest, blobs=tuple(blobs))


# Dispatch table: (family, operator) -> builder. Add new bridged ops here (and a matching
# entry in assets/kernel_registry.yaml + a firmware handler) to extend hardware coverage.
_BUILDERS: dict[tuple[str, str], Callable[..., CaseBundle]] = {
    ("ConvolutionFunctions", "Convolve"): _build_convolve_case,
    ("ConvolutionFunctions", "DepthwiseConv"): _build_depthwise_conv_case,
    ("ConvolutionFunctions", "TransposeConv"): _build_transpose_conv_case,
    ("BasicMathFunctions", "Abs"): _build_abs_case,
    ("BasicMathFunctions", "ArgMax"): _build_basic_math_reduction_case,
    ("BasicMathFunctions", "ArgMin"): _build_basic_math_reduction_case,
    ("BasicMathFunctions", "Mean"): _build_basic_math_reduction_case,
    ("BasicMathFunctions", "ReduceMax"): _build_basic_math_reduction_case,
    ("BasicMathFunctions", "ReduceMin"): _build_basic_math_reduction_case,
    ("BasicMathFunctions", "Rsqrt"): _build_basic_math_lut_case,
    ("BasicMathFunctions", "Sqrt"): _build_basic_math_lut_case,
    ("BasicMathFunctions", "SquaredDifference"): _build_squared_difference_case,
    ("BasicMathFunctions", "Add"): _build_elementwise_binary_case,
    ("BasicMathFunctions", "Sub"): _build_elementwise_binary_case,
    ("BasicMathFunctions", "Mul"): _build_mul_case,
    ("BasicMathFunctions", "Maximum"): _build_min_max_case,
    ("BasicMathFunctions", "Minimum"): _build_min_max_case,
    ("PoolingFunctions", "AvgPool"): _build_pooling_case,
    ("PoolingFunctions", "MaxPool"): _build_pooling_case,
    ("ActivationFunctions", "Relu"): _build_activation_case,
    ("ActivationFunctions", "Relu6"): _build_activation_case,
    ("ActivationFunctions", "Clamp"): _build_activation_case,
    ("ActivationFunctions", "LeakyRelu"): _build_activation_case,
    ("ActivationFunctions", "Logistic"): _build_activation_case,
    ("ActivationFunctions", "Tanh"): _build_activation_case,
    ("ActivationFunctions", "HardSwishCompat"): _build_activation_case,
    ("ActivationFunctions", "HardSwishPrecise"): _build_activation_case,
    ("ActivationFunctions", "PReLU"): _build_prelu_case,
    ("ActivationFunctions", "PReLUScalar"): _build_prelu_scalar_case,
    ("QuantizationFunctions", "Quantize"): _build_quantize_case,
    ("QuantizationFunctions", "Dequantize"): _build_dequantize_case,
    ("NNSupportFunctions", "Requantize"): _build_requantize_case,
    ("ComparisonFunctions", "Comparison"): _build_comparison_case,
    ("SoftmaxFunctions", "Softmax"): _build_softmax_case,
    ("FullyConnectedFunctions", "FullyConnected"): _build_fully_connected_case,
    ("FullyConnectedFunctions", "BatchMatMul"): _build_batch_matmul_case,
}


def bridged_families() -> list[str]:
    """Distinct operator families with at least one bridged (family, operator) builder
    registered in `_BUILDERS`, in stable sorted order. Used by callers (e.g.
    `hardware_run.build_generated_test_case_bundles`) that want to bridge every family
    with real firmware dispatch support instead of a single hardcoded family."""
    return sorted({family for family, _operator in _BUILDERS})
