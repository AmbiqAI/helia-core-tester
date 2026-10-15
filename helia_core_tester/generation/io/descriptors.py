"""
Descriptor ingestion with dtype validation and kernel resolution.
"""

import copy
import functools
import json
import yaml
import os
from typing import Dict, Any, List, Tuple
from pathlib import Path

from helia_core_tester.core.cpu_targets import known_capabilities
from helia_core_tester.generation.ops.catalog import get_operator_spec, iter_operator_specs
from helia_core_tester.generation.io.dtypes import (
    derive_legacy_activation_dtype,
    derive_legacy_weight_dtype,
    descriptor_dtype_to_c_type,
    normalize_tensor_dtypes,
    resolve_comparison,
    resolve_tensor_dtypes,
)


# Allowed dtype combinations
ALLOWED_DTYPE_COMBOS = {
    ('S8', 'S8'): 's8',
    ('S16', 'S8'): 's16', 
    ('S8', 'S4'): 's4',
    ('S32', 'S8'): 's32',
}

OPERATOR_DESCRIPTOR_PROFILES = {'Abs': 'single_input_unary',
 'Add': 'dual_input_elementwise',
 'ArgMax': 'arg_reduction',
 'ArgMin': 'arg_reduction',
 'AvgPool': 'pool',
 'BatchMatMul': 'dual_input_elementwise',
 'BatchNorm': 'single_input_unary',
 'BatchToSpaceND': 'shape_transform',
 'BroadcastTo': 'shape_transform',
 'ChunkedEquivalence': 'custom',
 'Clamp': 'single_input_unary',
 'Comparison': 'dual_input_elementwise',
 'Concatenation': 'shape_transform',
 'Convolve': 'conv',
 'DepthToSpace': 'shape_transform',
 'DepthwiseConv': 'conv',
 'Dequantize': 'single_input_unary',
 'DynamicUpdateSlice': 'custom',
 'Fill': 'custom',
 'FullyConnected': 'fully_connected',
 'Gather': 'shape_transform',
 'GatherND': 'shape_transform',
 'Gelu': 'single_input_unary',
 'GRUUnidirectional': 'single_input_unary',
 'HardSwishCompat': 'single_input_unary',
 'HardSwishPrecise': 'single_input_unary',
 'LSTMUnidirectional': 'single_input_unary',
 'LeakyRelu': 'single_input_unary',
 'Logistic': 'single_input_unary',
 'MaxPool': 'pool',
 'Maximum': 'dual_input_elementwise',
 'Mean': 'single_input_unary',
 'Minimum': 'dual_input_elementwise',
 'MirrorPad': 'custom',
 'Div': 'dual_input_elementwise',
 'Mul': 'dual_input_elementwise',
 'NNActivationFloat': 'single_input_unary',
 'NNActivationS16': 'single_input_unary',
 'PReLU': 'single_input_unary',
 'Pack': 'shape_transform',
 'Pad': 'single_input_unary',
 'PReLUScalar': 'single_input_unary',
 'Quantize': 'single_input_unary',
 'ReduceMax': 'single_input_unary',
 'ReduceMin': 'single_input_unary',
 'ReduceSum': 'single_input_unary',
 'Relu': 'single_input_unary',
 'Relu6': 'single_input_unary',
 'Requantize': 'single_input_unary',
 'Reshape': 'shape_transform',
 'ResizeBilinear': 'shape_transform',
 'ResizeNearestNeighbor': 'shape_transform',
 'ReverseSequence': 'shape_transform',
 'Rsqrt': 'single_input_unary',
 'SVDF': 'single_input_unary',
 'ScatterNd': 'custom',
 'SelectV2': 'dual_input_elementwise',
 'Softmax': 'single_input_unary',
 'SpaceToBatchND': 'shape_transform',
 'SpaceToDepth': 'shape_transform',
 'Split': 'shape_transform',
 'Sqrt': 'single_input_unary',
 'SquaredDifference': 'dual_input_elementwise',
 'Squeeze': 'shape_transform',
 'StridedSlice': 'shape_transform',
 'Sub': 'dual_input_elementwise',
 'Tanh': 'single_input_unary',
 'Tile': 'shape_transform',
 'Transpose': 'shape_transform',
 'Unpack': 'shape_transform',
 'TransposeConv': 'shape_transform',
 'VariableUpdate': 'shape_transform',
 'Where': 'custom'}

PROFILE_REQUIRED_FIELDS = {
    'arg_reduction': ('input_shape',),
    'conv': ('input_shape', 'filter_shape'),
    'custom': (),
    'dual_input_elementwise': ('input_1_shape', 'input_2_shape'),
    'fully_connected': ('input_shape', 'filter_shape'),
    'pool': ('input_shape',),
    'shape_transform': ('input_shape',),
    'single_input_unary': ('input_shape',),
}

PROFILE_ONE_OF_REQUIRED_FIELDS = {
    'pool': (('pool_size',), ('filter_shape',)),
}

OPERATOR_EXTRA_REQUIRED_FIELDS = {'BroadcastTo': ('output_shape',),
 'BatchNorm': ('layout',),
 'ChunkedEquivalence': ('kernel', 'element_count', 'chunk_sizes'),
 'Clamp': ('act_min', 'act_max'),
 'Comparison': ('operation',),
 'DynamicUpdateSlice': ('operand_shape', 'update_shape', 'start_indices'),
 'Fill': ('output_shape', 'value'),
 'Gather': ('indices_shape',),
 'GatherND': ('indices_shape',),
 'MirrorPad': ('paddings',),
 'Reshape': ('target_shape',),
 'NNActivationFloat': ('activation_type',),
 'Pack': ('num_inputs', 'axis'),
 'NNActivationS16': ('activation_type',),
 'PReLUScalar': ('alpha_shape',),
 'Requantize': ('effective_scale_multiplier',
                'effective_scale_shift',
                'input_zeropoint',
                'output_zeropoint'),
 'ResizeBilinear': ('size',),
 'ResizeNearestNeighbor': ('size',),
 'ReverseSequence': ('seq_lengths', 'seq_dim', 'batch_dim'),
 'ScatterNd': ('indices', 'updates'),
 'Tile': ('multiples',),
 'Unpack': ('axis',)}

OPERATOR_FIELD_CONSTRAINTS = {'HardSwishCompat': {'activation_dtype': 'S8'}}


def _operator_descriptor_profile(operator: str) -> str:
    try:
        return OPERATOR_DESCRIPTOR_PROFILES[operator]
    except KeyError as exc:
        raise ValueError(f"Unsupported operator: {operator}") from exc


def _requires_input_shape(operator: str, desc: Dict[str, Any]) -> bool:
    return not (operator == 'LSTMUnidirectional' and desc.get('hint', {}).get('force_cmsis', False))


def _validate_profile_requirements(
    desc: Dict[str, Any],
    operator: str,
    *,
    variation: bool = False,
) -> None:
    profile = _operator_descriptor_profile(operator)
    suffix = " variation" if variation else ""

    if profile == 'fully_connected':
        if 'input_shape' not in desc or 'filter_shape' not in desc:
            raise ValueError(f"{operator}{suffix} requires input_shape and filter_shape")
    elif profile == 'conv':
        if 'input_shape' not in desc or 'filter_shape' not in desc:
            raise ValueError(f"{operator}{suffix} requires input_shape and filter_shape")
    elif profile == 'dual_input_elementwise':
        if 'input_1_shape' not in desc or 'input_2_shape' not in desc:
            raise ValueError(f"{operator}{suffix} requires input_1_shape and input_2_shape")
    elif profile == 'pool':
        if 'input_shape' not in desc or ('pool_size' not in desc and 'filter_shape' not in desc):
            raise ValueError(f"{operator}{suffix} requires input_shape and pool_size (or filter_shape)")
    else:
        for field in PROFILE_REQUIRED_FIELDS.get(profile, ()):
            if field == 'input_shape' and not _requires_input_shape(operator, desc):
                continue
            if field not in desc:
                raise ValueError(f"{operator}{suffix} requires {field}")

    for field in OPERATOR_EXTRA_REQUIRED_FIELDS.get(operator, ()):
        if field not in desc:
            raise ValueError(f"{operator}{suffix} requires {field}")

    if operator == 'PReLUScalar':
        has_scalar_field = 'scalar_input_value' in desc
        has_extras_values = bool(desc.get('hint', {}).get('extras', {}).get('input_values'))
        if not has_scalar_field and not has_extras_values:
            raise ValueError(
                f"{operator}{suffix} requires 'scalar_input_value' (single pixel) or "
                "hint.extras.input_values (one value per pixel, multi-pixel)"
            )

    if operator == "Rsqrt" and desc.get("activation_dtype") not in ("S16", "FP16", "FP32"):
        raise ValueError("Rsqrt only supports activation_dtype=S16, FP16, FP32")

    for field, expected in OPERATOR_FIELD_CONSTRAINTS.get(operator, {}).items():
        if str(desc.get(field, "")).upper() != str(expected).upper():
            raise ValueError(f"{operator}{suffix} only supports {field}={expected}")


SCHEMA_PATH = Path(__file__).resolve().parent.parent / "descriptors" / "schema.json"


class DescriptorSchemaError(RuntimeError):
    """schema.json itself cannot be used; no descriptor can be validated."""


def _descriptor_suite(desc: Dict[str, Any], source_stem: str) -> str:
    return str(desc.get("suite") or ("float" if source_stem.endswith("_float") else "default"))


def _schema_key() -> Tuple[str, int, int]:
    path = SCHEMA_PATH
    try:
        stat = path.stat()
    except OSError as exc:
        raise DescriptorSchemaError(f"Descriptor schema {path} is unusable: {exc}") from exc
    return (str(path), stat.st_mtime_ns, stat.st_size)


@functools.lru_cache(maxsize=4)
def _build_schema_validator(schema_key: Tuple[str, int, int]):
    """Draft 2020-12 validator for one version of schema.json; any problem fails closed.

    The operator and capability enums come from the catalog and cpu_targets,
    so a schema.json copy that lags them cannot reject a valid descriptor.
    """
    from jsonschema import Draft202012Validator
    from jsonschema.exceptions import SchemaError

    schema_path = schema_key[0]
    try:
        schema = json.loads(Path(schema_path).read_text())
        schema["properties"]["operator"]["enum"] = sorted(spec.operator for spec in iter_operator_specs())
        schema["properties"]["required_capabilities"]["items"]["enum"] = sorted(known_capabilities())
        Draft202012Validator.check_schema(schema)
    except (OSError, ValueError, SchemaError, KeyError, TypeError) as exc:
        raise DescriptorSchemaError(f"Descriptor schema {schema_path} is unusable: {exc!r}") from exc
    return Draft202012Validator(schema)


def _schema_validator():
    return _build_schema_validator(_schema_key())


def validate_descriptor_schema(doc: Any, source_stem: str) -> None:
    """Validate one raw descriptor document against schema.json.

    source_stem is the descriptor file's stem; the suite is defaulted from it
    first, as _annotate_descriptor_source does, so suite-gated rules see the
    effective suite. Raises ValueError listing every violation with its JSON
    path, or DescriptorSchemaError when the schema itself is unusable.
    """
    if not isinstance(doc, dict):
        raise ValueError(f"descriptor must be a mapping, got {type(doc).__name__}")
    candidate = dict(doc)
    candidate.setdefault("suite", _descriptor_suite(doc, source_stem))
    errors = sorted(
        _schema_validator().iter_errors(candidate),
        key=lambda err: (err.json_path, err.message),
    )
    if errors:
        name = doc.get("name", "<unnamed>")
        details = "; ".join(f"{err.json_path}: {err.message}" for err in errors)
        raise ValueError(f"descriptor '{name}' violates schema.json: {details}")


def _schema_cases(doc: Any) -> List[Any]:
    """The documents the schema sees: the raw doc, or each raw base+variation merge.

    A base with variations may leave out fields its variations supply, so it is
    checked only through the merges, which mirror expand_descriptor_variations.
    """
    if not isinstance(doc, dict) or 'variations' not in doc:
        return [doc]
    variations = doc['variations']
    if not isinstance(variations, list) or not variations:
        raise ValueError("'variations' must be a non-empty list of mappings")
    base = {k: v for k, v in doc.items() if k != 'variations'}
    cases = []
    for index, variation in enumerate(variations):
        if not isinstance(variation, dict):
            raise ValueError(f"variation {index} must be a mapping, got {type(variation).__name__}")
        merged = {**base, **variation}
        if 'name' in base:
            merged.setdefault('name', f"{base['name']}_var_{index}")
        cases.append(merged)
    return cases


@functools.lru_cache(maxsize=512)
def _schema_errors(data: bytes, source_stem: str, schema_key: Tuple[str, int, int]) -> Tuple[Any, ...]:
    """Schema error message (or None) per non-empty document, cached per content and schema version."""
    results = []
    for doc in _parse_yaml_docs(data):
        if doc is None:
            continue
        try:
            for case in _schema_cases(doc):
                validate_descriptor_schema(case, source_stem)
        except DescriptorSchemaError:
            raise
        except ValueError as exc:
            results.append(str(exc))
        else:
            results.append(None)
    return tuple(results)


def _annotate_descriptor_source(
    desc: Dict[str, Any],
    desc_path: str,
    descriptors_root: str,
) -> Dict[str, Any]:
    """Attach canonical grouped source/catalog metadata to a descriptor."""
    annotated = desc.copy()
    source_path = Path(desc_path).resolve()
    descriptors_root_path = Path(descriptors_root).resolve()
    source_relpath = source_path.relative_to(descriptors_root_path).as_posix()
    source_rel = Path(source_relpath)
    spec = get_operator_spec(str(annotated["operator"]))

    if spec.descriptor_relpaths and source_relpath not in spec.descriptor_relpaths:
        raise ValueError(
            f"Descriptor path mismatch for {annotated['operator']}: "
            f"expected one of {spec.descriptor_relpaths}, found {source_relpath}"
        )

    annotated["_source_family"] = source_rel.parts[0] if len(source_rel.parts) > 1 else ""
    annotated["_source_stem"] = source_rel.stem
    annotated["_source_relpath"] = source_relpath
    annotated["_descriptor_suite"] = _descriptor_suite(annotated, source_rel.stem)
    annotated["_family"] = spec.family
    annotated["_parity_kind"] = spec.parity_kind
    return annotated


def validate_dtype_combo(activation_dtype: str, weight_dtype: str) -> bool:
    """
    Validate dtype combination.
    
    Args:
        activation_dtype: Activation dtype (S8, S16)
        weight_dtype: Weight dtype (S8, S4)
        
    Returns:
        True if combination is allowed
    """
    return (activation_dtype, weight_dtype) in ALLOWED_DTYPE_COMBOS


def _validate_and_normalize_descriptor(desc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize a single descriptor dictionary.
    """
    normalized = copy.deepcopy(desc)
    required_fields = ['operator']
    for field in required_fields:
        if field not in normalized:
            raise ValueError(f"Missing required field: {field}")

    resolved_tensor_dtypes = resolve_tensor_dtypes(normalized)
    normalized["resolved_tensor_dtypes"] = resolved_tensor_dtypes
    normalized["resolved_comparison"] = resolve_comparison(normalized, resolved_tensor_dtypes)

    activation_dtype = derive_legacy_activation_dtype(normalized, resolved_tensor_dtypes)
    if activation_dtype is not None:
        normalized['activation_dtype'] = activation_dtype

    weight_dtype = derive_legacy_weight_dtype(normalized, resolved_tensor_dtypes)
    if weight_dtype is not None:
        normalized['weight_dtype'] = weight_dtype

    explicit_tensor_dtypes = normalize_tensor_dtypes(normalized.get("tensor_dtypes"))
    if explicit_tensor_dtypes:
        normalized["tensor_dtypes"] = explicit_tensor_dtypes

    if not explicit_tensor_dtypes and (
        normalized.get('activation_dtype') is None or normalized.get('weight_dtype') is None
    ):
        raise ValueError("Descriptor must provide activation_dtype/weight_dtype or tensor_dtypes")

    if not explicit_tensor_dtypes and not validate_dtype_combo(
        normalized['activation_dtype'],
        normalized['weight_dtype'],
    ):
        raise ValueError(
            f"Unsupported dtype combination: {normalized['activation_dtype']} x {normalized['weight_dtype']}"
        )

    operator = normalized['operator']

    # Operator-specific shape and field validation.
    if 'variations' not in normalized:
        _validate_profile_requirements(normalized, operator)

    for shape_key in ['input_shape', 'filter_shape', 'input_1_shape', 'input_2_shape', 'pool_size', 'indices_shape']:
        if shape_key in normalized:
            normalized[shape_key] = list(normalized[shape_key])
            # Only filter/kernel-size dimensions are required to be strictly
            # positive here; some operators (e.g. Reshape) intentionally test
            # zero-sized input/output tensors as edge cases.
            if shape_key in ('filter_shape', 'pool_size') and any(
                int(dim) <= 0 for dim in normalized[shape_key]
            ):
                raise ValueError(
                    f"Descriptor '{normalized.get('name', '<unnamed>')}' has non-positive "
                    f"dimension(s) in {shape_key}: {normalized[shape_key]}"
                )

    if operator == 'DepthwiseConv' and 'input_shape' in normalized and 'filter_shape' in normalized:
        input_channels = int(normalized['input_shape'][-1])
        filter_channels = int(normalized['filter_shape'][-2])
        depth_multiplier = int(normalized.get('depth_multiplier', 1))
        filter_multiplier = int(normalized['filter_shape'][-1])
        if filter_channels != input_channels:
            raise ValueError(
                f"Descriptor '{normalized.get('name', '<unnamed>')}': DepthwiseConv filter_shape "
                f"input-channel dimension ({filter_channels}) must match input_shape channels "
                f"({input_channels})"
            )
        if filter_multiplier != depth_multiplier:
            raise ValueError(
                f"Descriptor '{normalized.get('name', '<unnamed>')}': DepthwiseConv filter_shape "
                f"depth-multiplier dimension ({filter_multiplier}) must equal depth_multiplier "
                f"({depth_multiplier})"
            )

    if 'hint' not in normalized:
        normalized['hint'] = {}

    return normalized


_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


@functools.lru_cache(maxsize=512)
def _parse_yaml_docs(data: bytes) -> tuple:
    """Parse once per distinct content."""
    return tuple(yaml.load_all(data, Loader=_LOADER))


def load_descriptor(desc_path: str) -> List[Dict[str, Any]]:
    """
    Load and validate YAML descriptor(s) from a file.
    Supports multiple descriptors in a single YAML file (separated by ---).
    
    Args:
        desc_path: Path to YAML descriptor file
        
    Returns:
        List of validated descriptor dictionaries (one per YAML document in the file)
    """
    with open(desc_path, 'rb') as f:
        data = f.read()
    # Validator deep-copies, so cache stays pristine.
    documents = list(_parse_yaml_docs(data))
    
    # Filter out None documents (empty separators)
    documents = [doc for doc in documents if doc is not None]
    
    if not documents:
        raise ValueError(f"No valid descriptors found in {desc_path}")
    
    # Validate and normalize each descriptor
    schema_errors = _schema_errors(data, Path(desc_path).stem, _schema_key())
    descriptors = []
    for i, doc in enumerate(documents):
        try:
            if schema_errors[i] is not None:
                raise ValueError(schema_errors[i])
            validated_desc = _validate_and_normalize_descriptor(doc)
            descriptors.append(validated_desc)
        except Exception as e:
            raise ValueError(f"Error validating descriptor {i+1} in {desc_path}: {e}")
    
    return descriptors


def resolve_kernel(desc: Dict[str, Any]) -> str:
    """
    Resolve kernel symbol from descriptor.
    
    Args:
        desc: YAML descriptor dictionary
        
    Returns:
        Kernel symbol string
    """
    # Check if explicit kernel is specified
    if 'hint' in desc and 'kernel' in desc['hint']:
        return desc['hint']['kernel']
        
    # Resolve from dtype combination
    activation_dtype = desc['activation_dtype']
    weight_dtype = desc['weight_dtype']
    
    key = (activation_dtype, weight_dtype)
    if key not in ALLOWED_DTYPE_COMBOS:
        raise ValueError(f"Unsupported dtype combination: {activation_dtype} x {weight_dtype}")
        
    # Map to actual kernel symbols
    kernel_map = {
        ('S8', 'S8'): 'arm_fully_connected_wrapper_s8',
        ('S16', 'S8'): 'arm_fully_connected_s16_wrapper',
        ('S8', 'S4'): 'arm_fully_connected_s4'
    }
    
    return kernel_map[key]


def discover_descriptors(descriptors_dir: str) -> List[str]:
    """
    Discover all YAML descriptors in directory.
    
    Args:
        descriptors_dir: Directory containing YAML descriptors
        
    Returns:
        List of descriptor file paths
    """
    descriptors = []
    for root, dirs, files in os.walk(descriptors_dir):
        # Skip examples directory to avoid duplicates
        if 'examples' in root:
            continue
        for file in files:
            if file.endswith('.yaml') or file.endswith('.yml'):
                descriptors.append(os.path.join(root, file))
    return sorted(descriptors)


def expand_descriptor_variations(desc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Expand a descriptor with variations into multiple individual descriptors.
    
    Args:
        desc: Original descriptor dictionary
        
    Returns:
        List of expanded descriptor dictionaries
    """
    # If no variations, return the original descriptor
    if 'variations' not in desc:
        return [desc]
    
    expanded_descriptors = []
    base_desc = {k: v for k, v in desc.items() if k != 'variations'}
    base_name = base_desc['name']  # Preserve original base name
    
    for variation in desc['variations']:
        variation_desc = copy.deepcopy(base_desc)
        variation_desc.update(copy.deepcopy(variation))
        
        # Use the variation name directly (shorter, cleaner)
        if 'name' in variation:
            variation_desc['name'] = variation['name']
        else:
            # Generate a name based on variation index
            variation_desc['name'] = f"{base_desc['name']}_var_{len(expanded_descriptors)}"
        
        # Store base descriptor name for filtering purposes
        variation_desc['_base_name'] = base_name

        normalized_variation = _validate_and_normalize_descriptor(variation_desc)
        operator = normalized_variation['operator']
        _validate_profile_requirements(normalized_variation, operator, variation=True)
        normalized_variation['_base_name'] = base_name
        expanded_descriptors.append(normalized_variation)
    
    return expanded_descriptors


class DescriptorLoadError(Exception):
    """
    Raised by load_all_descriptors() when one or more descriptor files fail
    to load/validate and best_effort mode was not explicitly requested.

    Aggregates every failing path and its cause so a single run surfaces the
    complete set of problems instead of failing on (and hiding) just the first.
    """

    def __init__(self, failures: List[Tuple[str, Exception]]):
        self.failures = failures
        message = "Failed to load {} descriptor file(s):\n".format(len(failures)) + "\n".join(
            f"  - {path}: {exc}" for path, exc in failures
        )
        super().__init__(message)


def load_all_descriptors(descriptors_dir: str, best_effort: bool = False) -> List[Dict[str, Any]]:
    """
    Load and validate all descriptors in directory.
    Supports multiple descriptors per YAML file (separated by ---).
    Every document must carry a name (schema.json requires it).
    
    Args:
        descriptors_dir: Directory containing YAML descriptors
        best_effort: When False (default), any descriptor file that fails to
            load/validate raises DescriptorLoadError aggregating all failures,
            so generation/CI never silently proceeds with a partial corpus.
            Set True only for local, interactive descriptor authoring where a
            partial list (with warnings printed) is acceptable.
        
    Returns:
        List of validated descriptor dictionaries (expanded from variations)

    Raises:
        DescriptorSchemaError: schema.json is unusable, regardless of best_effort.
    """
    # A broken schema fails every file the same way; surface it once, even in best_effort mode.
    _schema_validator()
    descriptors = []
    failures: List[Tuple[str, Exception]] = []
    desc_paths = discover_descriptors(descriptors_dir)

    for desc_path in desc_paths:
        try:
            for desc in load_descriptor(desc_path):
                desc = _annotate_descriptor_source(desc, desc_path, descriptors_dir)
                descriptors.extend(expand_descriptor_variations(desc))
        except DescriptorSchemaError:
            raise
        except Exception as e:
            failures.append((str(desc_path), e))
            print(f"Warning: Failed to load descriptor {desc_path}: {e}")
            continue

    if failures and not best_effort:
        raise DescriptorLoadError(failures)

    return descriptors


def get_io_dtypes(desc: Dict[str, Any]) -> Dict[str, str]:
    """
    Get I/O dtype mapping from descriptor.
    
    Args:
        desc: YAML descriptor dictionary
        
    Returns:
        Dictionary mapping I/O type names to C types
    """
    resolved_tensor_dtypes = desc.get("resolved_tensor_dtypes") or resolve_tensor_dtypes(desc)
    activation_dtype = resolved_tensor_dtypes["input"]
    weight_dtype = resolved_tensor_dtypes.get("weights", desc.get("weight_dtype", "S8"))
    output_dtype = resolved_tensor_dtypes["output"]

    return {
        'ACT_T': descriptor_dtype_to_c_type(activation_dtype),
        'W_T': descriptor_dtype_to_c_type(weight_dtype),
        'OUT_T': descriptor_dtype_to_c_type(output_dtype),
    }


def descriptor_matches_op(desc: Dict[str, Any], op: str) -> bool:
    """True when `op` names the case's operator or source."""
    name = str(desc["name"])
    if name == op or name.startswith(op + "_"):
        return True
    keys = ("_base_name", "_source_stem", "_source_relpath", "operator")
    return any(desc.get(key) == op for key in keys)


def unmatched_ops(descriptors: List[Dict[str, Any]], ops: List[str]) -> List[str]:
    """Ops that match no descriptor."""
    return [op for op in ops if not any(descriptor_matches_op(desc, op) for desc in descriptors)]
