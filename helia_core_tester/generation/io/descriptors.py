"""
Descriptor ingestion with dtype validation and kernel resolution.
"""

import copy
import yaml
import os
from typing import Dict, Any, List, Tuple
from pathlib import Path

from helia_core_tester.generation.ops.catalog import get_operator_spec
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
 'Clamp': 'single_input_unary',
 'Comparison': 'dual_input_elementwise',
 'Concatenation': 'shape_transform',
 'Convolve': 'conv',
 'DepthToSpace': 'shape_transform',
 'DepthwiseConv': 'conv',
 'Dequantize': 'single_input_unary',
 'DynamicUpdateSlice': 'custom',
 'Fill': 'single_input_unary',
 'FullyConnected': 'fully_connected',
 'Gather': 'shape_transform',
 'GatherND': 'shape_transform',
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
 'Mul': 'dual_input_elementwise',
 'NNActivationFloat': 'single_input_unary',
 'NNActivationS16': 'single_input_unary',
 'PReLU': 'single_input_unary',
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
 'Clamp': ('act_min', 'act_max'),
 'Comparison': ('operation',),
 'DynamicUpdateSlice': ('operand_shape', 'update_shape', 'start_indices'),
 'Gather': ('indices_shape',),
 'GatherND': ('indices_shape',),
 'MirrorPad': ('paddings',),
 'Reshape': ('target_shape',),
 'NNActivationFloat': ('activation_type',),
 'NNActivationS16': ('activation_type',),
 'PReLUScalar': ('alpha_shape',),
 'Requantize': ('effective_scale_multiplier',
                'effective_scale_shift',
                'input_zeropoint',
                'output_zeropoint'),
 'ResizeNearestNeighbor': ('size',),
 'ReverseSequence': ('seq_lengths', 'seq_dim', 'batch_dim'),
 'ScatterNd': ('indices', 'updates'),
 'Tile': ('multiples',)}

OPERATOR_FIELD_CONSTRAINTS = {'HardSwishCompat': {'activation_dtype': 'S8'}, 'Rsqrt': {'activation_dtype': 'S16'}}


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

    for field, expected in OPERATOR_FIELD_CONSTRAINTS.get(operator, {}).items():
        if str(desc.get(field, "")).upper() != str(expected).upper():
            raise ValueError(f"{operator}{suffix} only supports {field}={expected}")


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
    annotated["_descriptor_suite"] = str(annotated.get("suite") or ("float" if source_rel.stem.endswith("_float") else "default"))
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

    if 'hint' not in normalized:
        normalized['hint'] = {}

    return normalized


def load_descriptor(desc_path: str) -> List[Dict[str, Any]]:
    """
    Load and validate YAML descriptor(s) from a file.
    Supports multiple descriptors in a single YAML file (separated by ---).
    
    Args:
        desc_path: Path to YAML descriptor file
        
    Returns:
        List of validated descriptor dictionaries (one per YAML document in the file)
    """
    with open(desc_path, 'r') as f:
        documents = list(yaml.safe_load_all(f))
    
    # Filter out None documents (empty separators)
    documents = [doc for doc in documents if doc is not None]
    
    if not documents:
        raise ValueError(f"No valid descriptors found in {desc_path}")
    
    # Validate and normalize each descriptor
    descriptors = []
    for i, doc in enumerate(documents):
        try:
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


# Operator + weight_dtype combos that expose an opt-in precomputed weight-sum ("kernel_sum")
# kernel route (ns-cmsis-nn `arm_*_with_weight_sum` entry points). Each eligible descriptor is
# automatically duplicated into a `<name>_kernel_sum` variant (hint.kernel_sum = True) in
# addition to the original default (non-weight-sum) descriptor, so both routes get covered
# without hand-duplicating every YAML test case. Extend this set as later PRs land support
# for more operator/dtype combos (e.g. depthwise conv s4/s8, conv/dwconv s16, fully connected).
KERNEL_SUM_ELIGIBLE_COMBOS = {
    ('Convolve', 'S4'),
}


def expand_kernel_sum_variant(desc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Duplicate an eligible descriptor into the default (non-weight-sum) descriptor plus a
    `_kernel_sum` sibling with hint.kernel_sum = True, so generation exercises both the
    plain and precomputed-weight-sum kernel routes for the same test shape/quantization.

    Descriptors that already set hint.kernel_sum explicitly (e.g. a hand-authored
    regression case) are left untouched and not duplicated.
    """
    operator = desc.get('operator')
    weight_dtype = str(desc.get('weight_dtype', 'S8')).upper()
    if (operator, weight_dtype) not in KERNEL_SUM_ELIGIBLE_COMBOS:
        return [desc]

    hint = desc.get('hint') or {}
    if hint.get('kernel_sum') is not None:
        return [desc]

    kernel_sum_desc = copy.deepcopy(desc)
    kernel_sum_desc['name'] = f"{desc['name']}_kernel_sum"
    kernel_sum_hint = dict(kernel_sum_desc.get('hint') or {})
    kernel_sum_hint['kernel_sum'] = True
    kernel_sum_desc['hint'] = kernel_sum_hint
    kernel_sum_desc['_base_name'] = desc.get('_base_name', desc['name'])

    return [desc, kernel_sum_desc]


def load_all_descriptors(descriptors_dir: str) -> List[Dict[str, Any]]:
    """
    Load and validate all descriptors in directory.
    Supports multiple descriptors per YAML file (separated by ---).
    Preserves original names from YAML descriptors when present.
    Falls back to numbered names (filename_1, filename_2, etc.) only when
    no name is specified in the descriptor.
    
    Args:
        descriptors_dir: Directory containing YAML descriptors
        
    Returns:
        List of validated descriptor dictionaries (expanded from variations)
    """
    descriptors = []
    desc_paths = discover_descriptors(descriptors_dir)
    
    for desc_path in desc_paths:
        try:
            # load_descriptor now returns a list (supports multiple docs per file)
            descs = load_descriptor(desc_path)
            
            # Get base filename without extension for numbering (fallback only)
            file_base = Path(desc_path).stem
            
            # If multiple descriptors from same file, preserve original names when available
            if len(descs) > 1:
                # Expand variations into individual descriptors for each descriptor
                for idx, desc in enumerate(descs, start=1):
                    # Create a copy
                    desc_copy = _annotate_descriptor_source(desc, desc_path, descriptors_dir)
                    
                    # Preserve original name if it exists, otherwise use numbered name
                    if 'name' not in desc_copy or not desc_copy['name']:
                        # No name specified, use numbered name as fallback
                        desc_copy['name'] = f"{file_base}_{idx}"
                    # If name exists, keep it as-is (preserve original meaningful names)
                    
                    expanded_descs = expand_descriptor_variations(desc_copy)
                    for expanded in expanded_descs:
                        descriptors.extend(expand_kernel_sum_variant(expanded))
            else:
                # Single descriptor - preserve original name if present
                for desc in descs:
                    desc = _annotate_descriptor_source(desc, desc_path, descriptors_dir)
                    # If no name specified, use file-based name as fallback
                    if 'name' not in desc or not desc['name']:
                        desc['name'] = file_base
                    # Otherwise, keep the original name
                    
                    expanded_descs = expand_descriptor_variations(desc)
                    for expanded in expanded_descs:
                        descriptors.extend(expand_kernel_sum_variant(expanded))
        except Exception as e:
            print(f"Warning: Failed to load descriptor {desc_path}: {e}")
            continue
            
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
