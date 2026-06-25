"""Resolved descriptor dtype and comparison helpers."""

from __future__ import annotations

from typing import Any, Dict, Mapping


TENSOR_DTYPE_KEYS = ("input", "output", "weights", "bias")
ALLOWED_TENSOR_DTYPES = ("FP32", "FP16", "S8", "S16", "S32", "S4", "BOOL")
LEGACY_ACTIVATION_DTYPES = ("S8", "S16", "S32")
LEGACY_WEIGHT_DTYPES = ("S4", "S8")

FLOAT_DTYPES = frozenset({"FP32", "FP16"})
INTEGER_DTYPES = frozenset({"S8", "S16", "S32", "S4"})

_DTYPE_TO_C_TYPE = {
    "FP32": "float",
    "FP16": "float16_t",
    "S8": "int8_t",
    "S16": "int16_t",
    "S32": "int32_t",
    "S4": "int8_t",
    "BOOL": "bool",
}

_DTYPE_TO_LITERT = {
    "FP32": "float32",
    "FP16": "float16",
    "S8": "int8",
    "S16": "int16",
    "S32": "int32",
    "S4": "int4",
    "BOOL": "bool",
}

_DEFAULT_FLOAT_COMPARISON = {
    "FP32": {"atol": 1.0e-5, "rtol": 1.0e-5},
    "FP16": {"atol": 1.0e-3, "rtol": 1.0e-3},
}


def normalize_dtype(dtype: str) -> str:
    normalized = str(dtype).upper()
    if normalized not in ALLOWED_TENSOR_DTYPES:
        raise ValueError(f"Unsupported tensor dtype: {dtype}")
    return normalized


def normalize_tensor_dtypes(tensor_dtypes: Mapping[str, Any] | None) -> Dict[str, str]:
    if tensor_dtypes is None:
        return {}
    normalized: Dict[str, str] = {}
    for role, dtype in tensor_dtypes.items():
        role_key = str(role).lower()
        if role_key not in TENSOR_DTYPE_KEYS:
            raise ValueError(
                f"Unsupported tensor_dtypes role: {role}. "
                f"Expected one of {', '.join(TENSOR_DTYPE_KEYS)}"
            )
        normalized[role_key] = normalize_dtype(str(dtype))
    return normalized


def descriptor_dtype_to_c_type(dtype: str) -> str:
    return _DTYPE_TO_C_TYPE[normalize_dtype(dtype)]


def descriptor_dtype_to_litert_dtype(dtype: str) -> str:
    return _DTYPE_TO_LITERT[normalize_dtype(dtype)]


def is_float_dtype(dtype: str) -> bool:
    return normalize_dtype(dtype) in FLOAT_DTYPES


def is_integer_dtype(dtype: str) -> bool:
    return normalize_dtype(dtype) in INTEGER_DTYPES


def _legacy_tensor_dtypes(desc: Mapping[str, Any]) -> Dict[str, str]:
    resolved: Dict[str, str] = {}
    operator = str(desc.get("operator", ""))
    activation_dtype = desc.get("activation_dtype")
    weight_dtype = desc.get("weight_dtype")

    if operator == "Quantize":
        resolved["input"] = "FP32"
        if activation_dtype is not None:
            resolved["output"] = normalize_dtype(str(activation_dtype))
    elif operator == "Dequantize":
        if activation_dtype is not None:
            resolved["input"] = normalize_dtype(str(activation_dtype))
        resolved["output"] = "FP32"
    else:
        if activation_dtype is not None:
            normalized_activation = normalize_dtype(str(activation_dtype))
            resolved["input"] = normalized_activation
            resolved["output"] = normalized_activation

    if weight_dtype is not None:
        resolved["weights"] = normalize_dtype(str(weight_dtype))

    return resolved


def resolve_tensor_dtypes(desc: Mapping[str, Any]) -> Dict[str, str]:
    resolved = _legacy_tensor_dtypes(desc)
    resolved.update(normalize_tensor_dtypes(desc.get("tensor_dtypes")))

    if "input" not in resolved:
        raise ValueError("Descriptor must resolve an input tensor dtype")
    if "output" not in resolved:
        resolved["output"] = resolved["input"]

    return resolved


def derive_legacy_activation_dtype(desc: Mapping[str, Any], resolved: Mapping[str, str]) -> str | None:
    if desc.get("activation_dtype") is not None:
        return normalize_dtype(str(desc["activation_dtype"]))
    operator = str(desc.get("operator", ""))
    if operator == "Quantize":
        return resolved.get("output")
    if operator == "Dequantize":
        return resolved.get("input")
    return resolved.get("input")


def derive_legacy_weight_dtype(desc: Mapping[str, Any], resolved: Mapping[str, str]) -> str | None:
    if desc.get("weight_dtype") is not None:
        return normalize_dtype(str(desc["weight_dtype"]))
    return resolved.get("weights")


def get_resolved_tensor_dtype(desc: Mapping[str, Any], role: str, default: str | None = None) -> str:
    resolved = desc.get("resolved_tensor_dtypes") or resolve_tensor_dtypes(desc)
    dtype = resolved.get(str(role).lower())
    if dtype is None:
        if default is None:
            raise KeyError(f"Descriptor is missing resolved dtype for role '{role}'")
        return normalize_dtype(default)
    return normalize_dtype(dtype)


def default_comparison_for_dtype(dtype: str) -> Dict[str, Any]:
    normalized = normalize_dtype(dtype)
    if normalized in FLOAT_DTYPES:
        defaults = _DEFAULT_FLOAT_COMPARISON[normalized]
        return {
            "mode": "float",
            "atol": float(defaults["atol"]),
            "rtol": float(defaults["rtol"]),
        }
    if normalized == "BOOL":
        return {"mode": "bool"}
    return {"mode": "exact_int"}


def resolve_comparison(desc: Mapping[str, Any], resolved_tensor_dtypes: Mapping[str, str] | None = None) -> Dict[str, Any]:
    resolved = dict(resolved_tensor_dtypes or desc.get("resolved_tensor_dtypes") or resolve_tensor_dtypes(desc))
    comparison = default_comparison_for_dtype(resolved["output"])
    user_config = desc.get("comparison")
    if not isinstance(user_config, Mapping):
        return comparison
    if comparison["mode"] == "exact_int":
        if user_config.get("tolerance") is not None:
            comparison = {
                "mode": "tolerant_int",
                "tolerance": int(user_config["tolerance"]),
            }
        return comparison
    if comparison["mode"] == "float":
        if user_config.get("atol") is not None:
            comparison["atol"] = float(user_config["atol"])
        if user_config.get("rtol") is not None:
            comparison["rtol"] = float(user_config["rtol"])
    return comparison


def descriptor_matches_dtype_filter(desc: Mapping[str, Any], dtype: str) -> bool:
    wanted = normalize_dtype(dtype)
    resolved = desc.get("resolved_tensor_dtypes") or resolve_tensor_dtypes(desc)
    if wanted in resolved.values():
        return True
    activation_dtype = desc.get("activation_dtype")
    return activation_dtype is not None and normalize_dtype(str(activation_dtype)) == wanted
