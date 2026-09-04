"""
Template context builder for Jinja templates.
Handles dimension conversions, quantization parameters, and array formatting.
"""

import math

import numpy as np
from pathlib import PurePosixPath
from typing import Dict, Any, List, Tuple

from helia_core_tester.generation.io.dtypes import (
    default_comparison_for_dtype,
    default_int_tolerance,
    has_int_tolerance_override,
    resolve_comparison,
)

class TemplateContextBuilder:
    """
    Builds context dictionaries for Jinja templates.
    Handles dimension conversions, quantization parameters, and array formatting.
    """

    _EXACT_INT_VALIDATION_TEMPLATES = {
        "ActivationFunctions/clamp/clamp.c.j2",
        "ActivationFunctions/nn_activation/nn_activation.c.j2",
        "BasicMathFunctions/argmax/argmax.c.j2",
        "BasicMathFunctions/argmin/argmin.c.j2",
        "BasicMathFunctions/sqrt/sqrt.c.j2",
        "GatherFunctions/gather/gather.c.j2",
        "GatherFunctions/gather_nd/gather_nd.c.j2",
        "LSTMFunctions/lstm_unidirectional/lstm_unidirectional.c.j2",
        "NNSupportFunctions/requantize/requantize.c.j2",
        "ReshapeFunctions/depth_to_space/depth_to_space.c.j2",
        "ReshapeFunctions/resize_nearest_neighbor/resize_nearest_neighbor.c.j2",
        "SVDFunctions/svdf/svdf.c.j2",
    }

    _TOLERANT_INT_VALIDATION_TEMPLATES = {
        "ActivationFunctions/hard_swish/hard_swish.c.j2",
        "ActivationFunctions/hard_swish/hard_swish_compat.c.j2",
        "ActivationFunctions/leaky_relu/leaky_relu.c.j2",
        "ActivationFunctions/logistic/logistic.c.j2",
        "ActivationFunctions/prelu/prelu.c.j2",
        "ActivationFunctions/relu/relu.c.j2",
        "ActivationFunctions/relu6/relu6.c.j2",
        "ActivationFunctions/tanh/tanh.c.j2",
        "BasicMathFunctions/abs/abs.c.j2",
        "BasicMathFunctions/add/add.c.j2",
        "BasicMathFunctions/mean/mean.c.j2",
        "BasicMathFunctions/minmax/minmax.c.j2",
        "BasicMathFunctions/mul/mul.c.j2",
        "BasicMathFunctions/reduce_max/reduce_max.c.j2",
        "BasicMathFunctions/reduce_min/reduce_min.c.j2",
        "BasicMathFunctions/squared_difference/squared_difference.c.j2",
        "BasicMathFunctions/sub/sub.c.j2",
        "ConcatenationFunctions/concatenation/concatenation.c.j2",
        "ConcatenationFunctions/split/split.c.j2",
        "ConvolutionFunctions/convolve/convolve.c.j2",
        "ConvolutionFunctions/depthwise_conv/depthwise_conv.c.j2",
        "ConvolutionFunctions/transpose_conv/transpose_conv.c.j2",
        "FullyConnectedFunctions/batch_matmul/batch_matmul.c.j2",
        "FullyConnectedFunctions/fully_connected/fully_connected.c.j2",
        "PadFunctions/pad/pad.c.j2",
        "PoolingFunctions/avg_pool/avg_pool.c.j2",
        "PoolingFunctions/max_pool/max_pool.c.j2",
        "QuantizationFunctions/quantize/quantize.c.j2",
        "ReshapeFunctions/batch_to_space_nd/batch_to_space_nd.c.j2",
        "ReshapeFunctions/reshape/reshape.c.j2",
        "ReshapeFunctions/space_to_batch_nd/space_to_batch_nd.c.j2",
        "ReshapeFunctions/space_to_depth/space_to_depth.c.j2",
        "SoftmaxFunctions/softmax/softmax.c.j2",
        "StridedSliceFunctions/strided_slice/strided_slice.c.j2",
        "TesterExtensions/squeeze/squeeze.c.j2",
        "TransposeFunctions/transpose/transpose.c.j2",
    }

    _FLOAT_VALIDATION_TEMPLATES = {
        "QuantizationFunctions/dequantize/dequantize.c.j2",
    }

    _BOOL_VALIDATION_TEMPLATES = {
        "ComparisonFunctions/comparison/comparison.c.j2",
    }

    _REPORT_LIMIT_OVERRIDES = {
        "LSTMFunctions/lstm_unidirectional/lstm_unidirectional.c.j2": 8,
        "SVDFunctions/svdf/svdf.c.j2": 8,
    }
    
    @staticmethod
    def _quant_param_scalar(quant_params: Dict[str, Any] | None, key: str, default: float | int) -> float | int:
        """Extract a scalar quantization value from LiteRT metadata."""
        if not quant_params:
            return default

        value = quant_params.get(key, default)
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return default
            return value.reshape(-1)[0].item()
        if isinstance(value, (list, tuple)):
            return default if len(value) == 0 else value[0]
        return value

    _VALIDATION_HELPERS_BY_MODE = {
        "exact_int": ["exact_int"],
        "tolerant_int": ["tolerant_int"],
        "float": ["float"],
        "bool": ["bool"],
        "none": [],
    }

    _VALIDATION_LABEL_OVERRIDES = {
        "ActivationFunctions/nn_activation/nn_activation.c.j2": "NN activation",
        "ActivationFunctions/prelu/prelu.c.j2": "PReLU",
        "ConvolutionFunctions/transpose_conv/transpose_conv.c.j2": "TransposeConv",
        "ReshapeFunctions/depth_to_space/depth_to_space.c.j2": "DepthToSpace",
        "ReshapeFunctions/resize_nearest_neighbor/resize_nearest_neighbor.c.j2": "ResizeNearestNeighbor",
        "ReshapeFunctions/space_to_batch_nd/space_to_batch_nd.c.j2": "SpaceToBatchND",
        "ReshapeFunctions/batch_to_space_nd/batch_to_space_nd.c.j2": "BatchToSpaceND",
        "ReshapeFunctions/space_to_depth/space_to_depth.c.j2": "SpaceToDepth",
        "StridedSliceFunctions/strided_slice/strided_slice.c.j2": "StridedSlice",
        "TesterExtensions/squeeze/squeeze.c.j2": "Squeeze",
        "SVDFunctions/svdf/svdf.c.j2": "SVDF",
        "LSTMFunctions/lstm_unidirectional/lstm_unidirectional.c.j2": "LSTM",
    }

    @staticmethod
    def _normalize_template_path(template_path: str) -> str:
        return str(PurePosixPath(template_path))

    @classmethod
    def infer_validation_mode(cls, template_path: str, context: Dict[str, Any]) -> str:
        normalized_path = cls._normalize_template_path(template_path)
        explicit = context.get("validation_mode")
        if explicit:
            return str(explicit).strip().lower()
        # Output dtype wins over the template-path allowlists: the allowlists
        # exist to pick the right INT comparison for shared int templates, and
        # must never route a float-typed output into an integer comparison
        # (the (long long) cast truncates |v| < 1 to 0 on both sides, making
        # the comparison vacuous — issue #54). Float descriptors reusing an
        # int template get a real float comparison; int descriptors are
        # unaffected and fall through to the allowlists as before.
        # data_dtype is the fallback because several generators (LSTM, SVDF)
        # historically set only that key; an op whose output dtype differs
        # from its data dtype (e.g. Quantize) must set output_dtype explicitly.
        output_dtype = str(
            context.get("output_dtype") or context.get("data_dtype") or ""
        ).strip().lower()
        if output_dtype == "bool":
            return "bool"
        if normalized_path in cls._BOOL_VALIDATION_TEMPLATES:
            return "bool"
        if "float" in output_dtype:
            return "float"
        if normalized_path in cls._FLOAT_VALIDATION_TEMPLATES:
            return "float"
        if normalized_path in cls._EXACT_INT_VALIDATION_TEMPLATES:
            return "exact_int"
        if normalized_path in cls._TOLERANT_INT_VALIDATION_TEMPLATES:
            return "tolerant_int"
        return "exact_int"

    @classmethod
    def infer_validation_label(
        cls,
        template_path: str,
        context: Dict[str, Any],
        desc: Dict[str, Any] | None = None,
    ) -> str:
        normalized_path = cls._normalize_template_path(template_path)
        override = cls._VALIDATION_LABEL_OVERRIDES.get(normalized_path)
        if override:
            return override

        operator = ""
        if desc:
            operator = str(desc.get("operator", "")).strip()
        if operator:
            return operator.replace("_", " ").title()

        filename = PurePosixPath(normalized_path).name
        stem = filename[:-5] if filename.endswith(".c.j2") else PurePosixPath(normalized_path).stem
        return stem.replace("_", " ").title()

    @classmethod
    def infer_validation_report_limit(cls, template_path: str, context: Dict[str, Any]) -> int:
        explicit = context.get("validation_report_limit")
        if explicit is not None:
            return int(explicit)
        normalized_path = cls._normalize_template_path(template_path)
        return int(cls._REPORT_LIMIT_OVERRIDES.get(normalized_path, 20))

    @classmethod
    def infer_validation_tolerance(
        cls, template_path: str, context: Dict[str, Any], mode: str, desc: Dict[str, Any] | None = None
    ) -> int:
        explicit = context.get("validation_tolerance")
        if explicit is not None:
            return int(explicit)
        comparison_tolerance = context.get("comparison_tolerance")
        if comparison_tolerance is not None:
            return int(comparison_tolerance)
        if mode != "tolerant_int":
            return 1

        # Reads dtypes.py's single-source-of-truth per-operator tolerance
        # table so FVP and hardware always validate a case identically.
        operator = str((desc or {}).get("operator") or context.get("operator") or "")
        output_dtype = str(context.get("output_dtype", "")).strip()
        dtype_token = "S16" if output_dtype == "int16_t" else "S8"
        if operator and has_int_tolerance_override(operator, dtype_token):
            return default_int_tolerance(operator, dtype_token)

        # KNOWN GAP: no operator-level override exists -- see dtypes.py's
        # _OPERATOR_TOLERANCE_OVERRIDES for unaudited operators.
        return 1

    @classmethod
    def infer_validation_helpers(cls, context: Dict[str, Any], mode: str) -> List[str]:
        explicit = context.get("validation_helpers")
        if explicit is not None:
            return list(explicit)
        return list(cls._VALIDATION_HELPERS_BY_MODE.get(mode, []))

    @classmethod
    def infer_float_comparison_defaults(
        cls,
        context: Dict[str, Any],
        desc: Dict[str, Any] | None = None,
    ) -> Dict[str, float]:
        comparison: Dict[str, Any] | None = None
        if desc is not None:
            resolved_comparison = desc.get("resolved_comparison")
            if isinstance(resolved_comparison, dict):
                comparison = resolved_comparison
            else:
                comparison = resolve_comparison(desc)

        if comparison and comparison.get("mode") == "float":
            return {
                "atol": float(comparison["atol"]),
                "rtol": float(comparison["rtol"]),
            }

        output_dtype = str(context.get("output_dtype", "")).strip().lower()
        if output_dtype == "float16_t":
            comparison = default_comparison_for_dtype("FP16")
        elif "float" in output_dtype:
            comparison = default_comparison_for_dtype("FP32")
        else:
            return {}

        return {
            "atol": float(comparison["atol"]),
            "rtol": float(comparison["rtol"]),
        }

    @classmethod
    def build_validation_context(
        cls,
        template_path: str,
        context: Dict[str, Any],
        desc: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        resolved = dict(context)
        comparison: Dict[str, Any] | None = None
        if desc is not None:
            raw_comparison = desc.get("resolved_comparison")
            comparison = dict(raw_comparison) if isinstance(raw_comparison, dict) else resolve_comparison(desc)
        comparison_mode = str((comparison or {}).get("mode", ""))
        if comparison_mode in {"exact_int", "tolerant_int", "float", "bool", "none"}:
            mode = comparison_mode
            resolved["validation_mode"] = mode
        else:
            mode = cls.infer_validation_mode(template_path, resolved)
        # Invariant (issue #54): a float-typed output must never be validated
        # by an integer comparison — the (long long) cast makes it vacuous.
        # This also rejects coercion via an explicit validation_mode override.
        output_dtype = str(
            resolved.get("output_dtype") or resolved.get("data_dtype") or ""
        ).strip().lower()
        if "float" in output_dtype and mode in ("exact_int", "tolerant_int", "bool", "none"):
            raise ValueError(
                f"Validation-mode coercion: template '{template_path}' resolved "
                f"validation mode '{mode}' for float output dtype "
                f"'{output_dtype}'. Float outputs require a float comparison; "
                f"'none' recreates the #54 end state (no comparison at all). "
                f"Status-only fault templates should simply not invoke output "
                f"validation rather than coercing the mode."
            )
        resolved["validation_mode"] = mode
        resolved["validation_mode_token"] = mode.upper()
        resolved.setdefault(
            "validation_label",
            cls.infer_validation_label(template_path, resolved, desc),
        )
        resolved.setdefault(
            "validation_report_limit",
            cls.infer_validation_report_limit(template_path, resolved),
        )
        if comparison_mode == "tolerant_int":
            resolved["validation_tolerance"] = int(comparison.get("tolerance", 0))
        else:
            resolved.setdefault(
                "validation_tolerance",
                cls.infer_validation_tolerance(template_path, resolved, mode, desc),
            )
        float_defaults = cls.infer_float_comparison_defaults(resolved, desc)
        resolved.setdefault(
            "validation_atol",
            float(resolved.get("comparison_atol", float_defaults.get("atol", 0.0))),
        )
        resolved.setdefault(
            "validation_rtol",
            float(resolved.get("comparison_rtol", float_defaults.get("rtol", 0.0))),
        )
        resolved.setdefault("comparison_tolerance", resolved["validation_tolerance"])
        resolved.setdefault("comparison_atol", resolved["validation_atol"])
        resolved.setdefault("comparison_rtol", resolved["validation_rtol"])
        resolved.setdefault(
            "validation_helpers",
            cls.infer_validation_helpers(resolved, mode),
        )
        return resolved
    
    @staticmethod
    def nhwc_to_cmsis_dims(shape: Tuple[int, ...]) -> Dict[str, int]:
        """
        Convert NHWC shape to CMSIS-NN dims format.
        
        Args:
            shape: Shape tuple in NHWC format
            
        Returns:
            Dictionary with n, h, w, c keys
        """
        if len(shape) == 4:
            return {
                'n': int(shape[0]),
                'h': int(shape[1]),
                'w': int(shape[2]),
                'c': int(shape[3])
            }
        elif len(shape) == 3:
            # HWC format
            return {
                'n': 1,
                'h': int(shape[0]),
                'w': int(shape[1]),
                'c': int(shape[2])
            }
        elif len(shape) == 2:
            # HW format
            return {
                'n': 1,
                'h': int(shape[0]),
                'w': int(shape[1]),
                'c': 1
            }
        elif len(shape) == 1:
            # C format
            return {
                'n': 1,
                'h': 1,
                'w': 1,
                'c': int(shape[0])
            }
        else:
            raise ValueError(f"Unsupported shape length: {len(shape)}")

    @staticmethod
    def normalize_reduction_axes(input_rank: int, axes: List[int]) -> List[int]:
        """
        Normalize reduction axes to unique, in-range positive indices.
        """
        normalized: List[int] = []
        for axis in axes:
            norm_axis = int(axis)
            if norm_axis < 0:
                norm_axis += input_rank
            if 0 <= norm_axis < input_rank and norm_axis not in normalized:
                normalized.append(norm_axis)
        return normalized

    @staticmethod
    def build_reduce_axis_dims(input_rank: int, axes: List[int]) -> Dict[str, int]:
        """
        Build CMSIS axis mask dims (n,h,w,c) for reduction ops.
        """
        if input_rank != 4:
            raise ValueError(f"Reduction ops require 4D NHWC input, got rank {input_rank}")

        axis_mask = [0, 0, 0, 0]
        for axis in TemplateContextBuilder.normalize_reduction_axes(input_rank, axes):
            axis_mask[axis] = 1
        return {'n': axis_mask[0], 'h': axis_mask[1], 'w': axis_mask[2], 'c': axis_mask[3]}

    @staticmethod
    def build_reduce_output_dims(input_shape: Tuple[int, ...], axes: List[int], keepdims: bool = False) -> Dict[str, int]:
        """
        Build CMSIS output dims for reduction ops using 4D reduction semantics.
        Reduced axes always become size 1 in CMSIS output dims.
        """
        if len(input_shape) != 4:
            raise ValueError(f"Reduction ops require 4D NHWC input, got shape {input_shape}")

        out = [int(input_shape[0]), int(input_shape[1]), int(input_shape[2]), int(input_shape[3])]
        for axis in TemplateContextBuilder.normalize_reduction_axes(4, axes):
            out[axis] = 1

        # keepdims is accepted for API clarity; CMSIS reduction output dims are always 4D.
        _ = keepdims
        return {'n': out[0], 'h': out[1], 'w': out[2], 'c': out[3]}
    
    @staticmethod
    def format_array_as_c_literal(arr: np.ndarray, indent: int = 4, max_per_line: int = 16) -> str:
        """
        Format numpy array as C array literal.
        """
        if arr is None or arr.size == 0:
            return ""

        flat = arr.flatten()

        lines = []
        current_line = []

        for i, val in enumerate(flat):
            val_str = TemplateContextBuilder.format_scalar_as_c_literal(val, arr.dtype)

            current_line.append(val_str)

            if len(current_line) >= max_per_line:
                lines.append(" " * indent + ", ".join(current_line) + ",")
                current_line = []

        if current_line:
            lines.append(" " * indent + ", ".join(current_line))

        return "\n".join(lines)

    @staticmethod
    def format_scalar_as_c_literal(value: Any, dtype: Any) -> str:
        """
        Format a scalar value for generated C arrays.
        """
        np_dtype = np.dtype(dtype)
        if np.issubdtype(np_dtype, np.bool_):
            return "true" if bool(value) else "false"
        if np.issubdtype(np_dtype, np.integer):
            return str(int(value))
        if np_dtype == np.float16:
            # Issue #64: was `.6f` (6 decimal places -- only ~6-7 significant
            # figures, up to ~4.8e-7 of rounding error for values near 1.0),
            # which put a hard floor of ~1e-6 on achievable atol for any
            # saturating output (tanh/sigmoid/normalized). format_float_literal
            # already gives full float32 round-trip precision and guarantees
            # valid C float syntax (decimal point/exponent always present).
            return f"(float16_t){TemplateContextBuilder.format_float_literal(value)}"
        if np.issubdtype(np_dtype, np.floating):
            return TemplateContextBuilder.format_float_literal(value)
        return str(value)

    @staticmethod
    def format_float_literal(value: Any, suffix: str = "f") -> str:
        """
        Format a standalone floating-point scalar as a valid C literal.
        """
        numeric = float(value)

        # NAN/INFINITY are float-typed C99 macros expanding to compiler builtins, so
        # they stay valid data under -Ofast/-ffinite-math-only: that flag licenses the
        # optimizer to assume operands are finite, but it does not stop a builtin from
        # materializing the bit pattern. Emitting them unsuffixed is deliberate --
        # `NANf` is not a token, and the decimal path below would otherwise produce
        # `nan.0f` or split-crash on `inf` (no exponent to unpack).
        if math.isnan(numeric):
            return "-NAN" if math.copysign(1.0, numeric) < 0 else "NAN"
        if math.isinf(numeric):
            return "-INFINITY" if numeric < 0 else "INFINITY"

        abs_numeric = abs(numeric)
        use_scientific = abs_numeric >= 1.0e6 or (abs_numeric != 0.0 and abs_numeric < 1.0e-4)

        if use_scientific:
            mantissa, exponent = f"{numeric:.9e}".split("e")
            mantissa = mantissa.rstrip("0").rstrip(".")
            if "." not in mantissa:
                mantissa += ".0"
            literal = f"{mantissa}e{exponent}"
        else:
            literal = f"{numeric:.9f}".rstrip("0").rstrip(".")
            if "." not in literal:
                literal += ".0"

        return f"{literal}{suffix}"

    
    @staticmethod
    def build_conv_params(
        desc: Dict[str, Any],
        input_shape: Tuple[int, ...],
        kernel_shape: Tuple[int, int],
        output_shape: Tuple[int, ...],
        input_quant: Dict[str, Any],
        output_quant: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build convolution parameters for CMSIS-NN.

        Notes:
        - input_shape/output_shape are expected NHWC (TFLite).
        - kernel_shape is (kernel_h, kernel_w).
        - For padding="same", compute padding from (in, out, stride, dilation, kernel) the same way TFLite does
        (symmetric "pad_before" used as CMSIS top/left padding).
        """
        strides = desc.get('strides', [1, 1])
        if isinstance(strides, (int, float)):
            stride_h = stride_w = int(strides)
        else:
            stride_h = int(strides[0])
            stride_w = int(strides[1])

        padding = desc.get('padding', 'valid')
        padding = 'valid' if padding is None else str(padding).lower()

        dilation = desc.get('dilation', [1, 1])
        if isinstance(dilation, (int, float)):
            dil_h = dil_w = int(dilation)
        else:
            dil_h = int(dilation[0])
            dil_w = int(dilation[1])

        kh, kw = int(kernel_shape[0]), int(kernel_shape[1])
        eff_kh = (kh - 1) * dil_h + 1
        eff_kw = (kw - 1) * dil_w + 1

        # NHWC -> H,W extraction
        if len(input_shape) == 4:
            in_h, in_w = int(input_shape[1]), int(input_shape[2])
        elif len(input_shape) == 3:
            in_h, in_w = int(input_shape[0]), int(input_shape[1])
        else:
            raise ValueError(f"Unsupported Conv2D input_shape rank: {len(input_shape)}")

        if len(output_shape) == 4:
            out_h, out_w = int(output_shape[1]), int(output_shape[2])
        elif len(output_shape) == 3:
            out_h, out_w = int(output_shape[0]), int(output_shape[1])
        else:
            raise ValueError(f"Unsupported Conv2D output_shape rank: {len(output_shape)}")

        if padding == 'same':
            pad_total_h = max((out_h - 1) * stride_h + eff_kh - in_h, 0)
            pad_total_w = max((out_w - 1) * stride_w + eff_kw - in_w, 0)
            pad_h = pad_total_h // 2
            pad_w = pad_total_w // 2
        else:  # valid
            pad_h = 0
            pad_w = 0

        # Activation clamp defaults depend on activation dtype
        resolved_tensor_dtypes = desc.get("resolved_tensor_dtypes") or {}
        tensor_dtypes = desc.get("tensor_dtypes") or {}
        act_dtype = str(
            resolved_tensor_dtypes.get(
                "input",
                tensor_dtypes.get("input", desc.get('activation_dtype', 'S8')),
            )
        ).upper()
        if act_dtype in {'FP32', 'FP16'}:
            activation_min = float(desc.get('activation_min', -1.0e30))
            activation_max = float(desc.get('activation_max', 1.0e30))
            in_zp = 0
            out_zp = 0
        elif act_dtype == 'S16':
            activation_min = int(desc.get('activation_min', -32768))
            activation_max = int(desc.get('activation_max', 32767))
            in_zp = int(TemplateContextBuilder._quant_param_scalar(input_quant, 'zero_point', 0))
            out_zp = int(TemplateContextBuilder._quant_param_scalar(output_quant, 'zero_point', 0))
        else:
            activation_min = int(desc.get('activation_min', -128))
            activation_max = int(desc.get('activation_max', 127))
            in_zp = int(TemplateContextBuilder._quant_param_scalar(input_quant, 'zero_point', 0))
            out_zp = int(TemplateContextBuilder._quant_param_scalar(output_quant, 'zero_point', 0))

        return {
            # CMSIS-NN convention: input_offset = -input_zero_point
            'input_offset': int(-in_zp),
            'output_offset': int(out_zp),
            'stride_h': int(stride_h),
            'stride_w': int(stride_w),
            'dilation_h': int(dil_h),
            'dilation_w': int(dil_w),
            'pad_h': int(pad_h),
            'pad_w': int(pad_w),
            'activation_min': int(activation_min) if act_dtype not in {'FP32', 'FP16'} else activation_min,
            'activation_max': int(activation_max) if act_dtype not in {'FP32', 'FP16'} else activation_max,
        }
    
    @staticmethod
    def build_dw_conv_params(
        desc: Dict[str, Any],
        input_shape: Tuple[int, ...],
        kernel_shape: Tuple[int, int],
        output_shape: Tuple[int, ...],
        input_quant: Dict[str, Any],
        output_quant: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build depthwise convolution parameters for CMSIS-NN.
        
        Notes:
        - input_shape/output_shape are expected NHWC (TFLite).
        - kernel_shape is (kernel_h, kernel_w).
        - For padding="same", compute padding from (in, out, stride, dilation, kernel) the same way TFLite does
        (symmetric "pad_before" used as CMSIS top/left padding).
        - ch_mult is the depth multiplier (output_channels / input_channels).
        """
        strides = desc.get('strides', [1, 1])
        if isinstance(strides, (int, float)):
            stride_h = stride_w = int(strides)
        else:
            stride_h = int(strides[0])
            stride_w = int(strides[1])
        
        padding = desc.get('padding', 'valid')
        padding = 'valid' if padding is None else str(padding).lower()
        
        dilation = desc.get('dilation', [1, 1])
        if isinstance(dilation, (int, float)):
            dil_h = dil_w = int(dilation)
        else:
            dil_h = int(dilation[0])
            dil_w = int(dilation[1])
        
        kh, kw = int(kernel_shape[0]), int(kernel_shape[1])
        eff_kh = (kh - 1) * dil_h + 1
        eff_kw = (kw - 1) * dil_w + 1
        
        # NHWC -> H,W extraction
        if len(input_shape) == 4:
            in_h, in_w = int(input_shape[1]), int(input_shape[2])
            in_c = int(input_shape[3])
        elif len(input_shape) == 3:
            in_h, in_w = int(input_shape[0]), int(input_shape[1])
            in_c = int(input_shape[2])
        else:
            raise ValueError(f"Unsupported DepthwiseConv2D input_shape rank: {len(input_shape)}")
        
        if len(output_shape) == 4:
            out_h, out_w = int(output_shape[1]), int(output_shape[2])
            out_c = int(output_shape[3])
        elif len(output_shape) == 3:
            out_h, out_w = int(output_shape[0]), int(output_shape[1])
            out_c = int(output_shape[2])
        else:
            raise ValueError(f"Unsupported DepthwiseConv2D output_shape rank: {len(output_shape)}")
        
        if padding == 'same':
            pad_total_h = max((out_h - 1) * stride_h + eff_kh - in_h, 0)
            pad_total_w = max((out_w - 1) * stride_w + eff_kw - in_w, 0)
            pad_h = pad_total_h // 2
            pad_w = pad_total_w // 2
        else:  # valid
            pad_h = 0
            pad_w = 0
        
        # Activation clamp defaults depend on activation dtype
        resolved_tensor_dtypes = desc.get("resolved_tensor_dtypes") or {}
        tensor_dtypes = desc.get("tensor_dtypes") or {}
        act_dtype = str(
            resolved_tensor_dtypes.get(
                "input",
                tensor_dtypes.get("input", desc.get('activation_dtype', 'S8')),
            )
        ).upper()
        if act_dtype in {'FP32', 'FP16'}:
            activation_min = float(desc.get('activation_min', -1.0e30))
            activation_max = float(desc.get('activation_max', 1.0e30))
            in_zp = 0
            out_zp = 0
        elif act_dtype == 'S16':
            activation_min = int(desc.get('activation_min', -32768))
            activation_max = int(desc.get('activation_max', 32767))
            in_zp = int(TemplateContextBuilder._quant_param_scalar(input_quant, 'zero_point', 0))
            out_zp = int(TemplateContextBuilder._quant_param_scalar(output_quant, 'zero_point', 0))
        else:
            activation_min = int(desc.get('activation_min', -128))
            activation_max = int(desc.get('activation_max', 127))
            in_zp = int(TemplateContextBuilder._quant_param_scalar(input_quant, 'zero_point', 0))
            out_zp = int(TemplateContextBuilder._quant_param_scalar(output_quant, 'zero_point', 0))
        
        # Calculate channel multiplier: output_channels / input_channels
        ch_mult = out_c // in_c if in_c > 0 else 1
        depth_multiplier = desc.get('depth_multiplier', ch_mult)
        ch_mult = int(depth_multiplier)
        
        return {
            # CMSIS-NN convention: input_offset = -input_zero_point
            'input_offset': int(-in_zp),
            'output_offset': int(out_zp),
            'ch_mult': int(ch_mult),
            'stride_h': int(stride_h),
            'stride_w': int(stride_w),
            'dilation_h': int(dil_h),
            'dilation_w': int(dil_w),
            'pad_h': int(pad_h),
            'pad_w': int(pad_w),
            'activation_min': int(activation_min) if act_dtype not in {'FP32', 'FP16'} else activation_min,
            'activation_max': int(activation_max) if act_dtype not in {'FP32', 'FP16'} else activation_max,
        }

    
    @staticmethod
    def build_quant_params(quant_params: Dict[str, Any], per_channel: bool = False) -> Dict[str, Any]:
        """
        Build quantization parameters for CMSIS-NN.
        
        Args:
            quant_params: Quantization parameters dictionary
            per_channel: Whether quantization is per-channel
            
        Returns:
            Dictionary with multiplier and shift arrays
        """
        # Import here to avoid circular dependency
        from .tflite_utils import calculate_multiplier_shift, calculate_per_channel_multiplier_shift
        
        if per_channel and isinstance(quant_params.get('scale'), np.ndarray):
            scales = quant_params['scale']
            multipliers, shifts = calculate_per_channel_multiplier_shift(scales)
            
            return {
                'multiplier_array': TemplateContextBuilder.format_array_as_c_literal(multipliers),
                'shift_array': TemplateContextBuilder.format_array_as_c_literal(shifts),
                'per_channel': True
            }
        else:
            scale = quant_params.get('scale', 1.0)
            # Handle numpy scalars (numpy.float32, numpy.float64, etc.)
            if isinstance(scale, np.ndarray):
                scale = float(scale.item() if scale.size == 1 else scale[0])
            elif isinstance(scale, (list, tuple)):
                scale = float(scale[0] if len(scale) > 0 else 1.0)
            else:
                # Convert numpy scalars to Python float
                scale = float(scale)
            
            multiplier, shift = calculate_multiplier_shift(scale)
            
            return {
                'multiplier': multiplier,
                'shift': shift,
                'per_channel': False
            }
    
    @staticmethod
    def calculate_depthwise_buffer_size_max(input_dims: Dict[str, int], 
                                            filter_dims: Dict[str, int],
                                            output_dims: Dict[str, int],
                                            output_dtype: str = 'S8') -> int:
        """
        Calculate a conservative upper bound for depthwise convolution buffer size.
        
        This matches CMSIS-NN's depthwise buffer size calculation:
        - MVE: 4 * CH_IN_BLOCK_MVE * filter_w * filter_h (where CH_IN_BLOCK_MVE = 16)
        - DSP: input_c * filter_w * filter_h * sizeof(int16_t)
        
        For depthwise conv, we use input_dims->c (input channels), not filter_dims->c.
        """
        input_c = input_dims['c']
        filter_w = filter_dims['w']
        filter_h = filter_dims['h']
        
        if output_dtype == 'S16':
            # S16 depthwise buffer size
            # From arm_depthwise_conv_get_buffer_sizes_s16.c:
            # MVE: 4 * input_dims->c * filter_dims->w * filter_dims->h * sizeof(int16_t) + 8
            # DSP: input_dims->c * filter_dims->w * filter_dims->h * sizeof(int16_t)
            buffer_size_mve = 4 * input_c * filter_w * filter_h * 2 + 8  # sizeof(int16_t) = 2, +8 for worst case
            buffer_size_dsp = input_c * filter_w * filter_h * 2  # sizeof(int16_t) = 2
            result = max(buffer_size_mve, buffer_size_dsp)
        else:
            # S8 depthwise buffer size (default)
            # MVE: 4 * CH_IN_BLOCK_MVE * filter_w * filter_h where CH_IN_BLOCK_MVE = 124
            # From arm_depthwise_conv_s8_opt_get_buffer_size_mve:
            # return (4 * CH_IN_BLOCK_MVE * filter_dims->w * filter_dims->h) * (int32_t)sizeof(int8_t);
            CH_IN_BLOCK_MVE = 124  # From arm_nnsupportfunctions.h
            buffer_size_mve = 4 * CH_IN_BLOCK_MVE * filter_w * filter_h  # sizeof(int8_t) = 1
            # DSP: input_c * filter_w * filter_h * sizeof(int16_t)
            # From arm_depthwise_conv_s8_opt_get_buffer_size_dsp:
            # return (input_dims->c * filter_dims->w * filter_dims->h) * sizeof(int16_t);
            buffer_size_dsp = input_c * filter_w * filter_h * 2  # sizeof(int16_t) = 2
            result = max(buffer_size_mve, buffer_size_dsp)
        
        return result
    
    @staticmethod
    def calculate_buffer_size_max(input_dims: Dict[str, int], 
                                  filter_dims: Dict[str, int],
                                  output_dims: Dict[str, int],
                                  output_dtype: str = 'S8') -> int:
        """
        Calculate a conservative upper bound for convolution buffer size.
        
        This uses the maximum of MVE and DSP buffer size calculations to ensure
        we have enough space regardless of which implementation is used.
        
        For S8 (int8_t):
          MVE: 4 * ceil((input_c * filter_w * filter_h) / 16) * 16
          DSP: 2 * ceil((filter_w * filter_h * input_c) / 4) * 4 * sizeof(int16_t)
        
        For S16 (int16_t):
          MVE: 4 * ceil((input_c * filter_w * filter_h) / 8) * 8 * sizeof(int16_t)
          DSP: 2 * input_c * filter_w * filter_h * sizeof(int16_t)

        For FP32/FP16 (float32_t/float16_t):
          CMSIS-NN's real arm_convolve_f32/f16_get_buffer_size() falls back, for the
          general (non-1x1, non-1xn) case, to the patch-gemm tile formula:
              ARM_NN_CONV_NHWC_PATCH_GEMM_{F32,F16}_MAX_TILE_ROWS(8) * filter_h * filter_w
              * input_c * sizeof(element)
          This is strictly >= every other float Convolve buffer-size specialization
          (1x1, 1xN, depthwise NT_T, packed-weight direct k3/k5), so it is used
          unconditionally here as the conservative upper bound, matching the S8/S16
          "max of known formulas" philosophy above. Previously this function silently
          fell through to the S8 (int8) formula for FP32/FP16 callers, undersizing the
          scratch buffer by 2x-4x (elem_size=1 assumed instead of 2 or 4) and causing
          real hardware ARM_CMSIS_NN_ARG_ERROR rejections for float Convolve cases.
        """
        input_c = input_dims['c']
        filter_w = filter_dims['w']
        filter_h = filter_dims['h']

        if output_dtype in ('FP32', 'FP16'):
            elem_size = 4 if output_dtype == 'FP32' else 2
            max_tile_rows = 8
            return max_tile_rows * filter_h * filter_w * input_c * elem_size
        elif output_dtype == 'S16':
            # S16 buffer size calculation
            # MVE: 4 * ceil((input_c * filter_w * filter_h) / 8) * 8 * sizeof(int16_t)
            col_length_mve = input_c * filter_w * filter_h
            col_length_mve = (col_length_mve + 7) // 8
            buffer_size_mve = 4 * col_length_mve * 8 * 2  # sizeof(int16_t) = 2
            
            # DSP: 2 * input_c * filter_w * filter_h * sizeof(int16_t)
            buffer_size_dsp = 2 * input_c * filter_w * filter_h * 2  # sizeof(int16_t) = 2
            
            # Return the maximum to be safe
            return max(buffer_size_mve, buffer_size_dsp)
        else:
            # S8 buffer size calculation (default)
            # MVE buffer size calculation
            col_length_mve = input_c * filter_w * filter_h
            col_length_mve = (col_length_mve + 15) // 16
            buffer_size_mve = 4 * col_length_mve * 16  # sizeof(int8_t) = 1
            
            # DSP buffer size calculation
            rhs_cols = filter_w * filter_h * input_c
            remainder = rhs_cols % 4
            aligned_rhs_cols = rhs_cols + (4 - remainder) if remainder != 0 else rhs_cols
            buffer_size_dsp = 2 * aligned_rhs_cols * 2  # sizeof(int16_t) = 2
            
            result = max(buffer_size_mve, buffer_size_dsp)
            
            # Return the maximum to be safe
            return result
    
    @staticmethod
    def build_transpose_conv_params(
        desc: Dict[str, Any],
        input_shape: Tuple[int, ...],
        kernel_shape: Tuple[int, int],
        output_shape: Tuple[int, ...],
        input_quant: Dict[str, Any],
        output_quant: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build transpose convolution parameters for CMSIS-NN.
        
        Notes:
        - input_shape/output_shape are expected NHWC (TFLite).
        - kernel_shape is (kernel_h, kernel_w).
        - TransposeConv uses padding_offsets which are different from regular conv padding.
        """
        strides = desc.get('strides', [1, 1])
        if isinstance(strides, (int, float)):
            stride_h = stride_w = int(strides)
        else:
            stride_h = int(strides[0])
            stride_w = int(strides[1])
        
        padding = desc.get('padding', 'valid')
        padding = 'valid' if padding is None else str(padding).lower()
        
        dilation = desc.get('dilation', [1, 1])
        if isinstance(dilation, (int, float)):
            dil_h = dil_w = int(dilation)
        else:
            dil_h = int(dilation[0])
            dil_w = int(dilation[1])
        
        kh, kw = int(kernel_shape[0]), int(kernel_shape[1])
        
        # NHWC -> H,W extraction
        if len(input_shape) == 4:
            in_h, in_w = int(input_shape[1]), int(input_shape[2])
        elif len(input_shape) == 3:
            in_h, in_w = int(input_shape[0]), int(input_shape[1])
        else:
            raise ValueError(f"Unsupported TransposeConv input_shape rank: {len(input_shape)}")
        
        if len(output_shape) == 4:
            out_h, out_w = int(output_shape[1]), int(output_shape[2])
        elif len(output_shape) == 3:
            out_h, out_w = int(output_shape[0]), int(output_shape[1])
        else:
            raise ValueError(f"Unsupported TransposeConv output_shape rank: {len(output_shape)}")
        
        # TransposeConv padding calculation
        # For 'same' padding in transpose conv, we need to calculate padding differently
        if padding == 'same':
            # Calculate total padding needed
            pad_total_h = max((in_h - 1) * stride_h + kh - out_h, 0)
            pad_total_w = max((in_w - 1) * stride_w + kw - out_w, 0)
            pad_h = pad_total_h // 2
            pad_w = pad_total_w // 2
            # Padding offsets are the remainder
            pad_offset_h = pad_total_h % 2
            pad_offset_w = pad_total_w % 2
        else:  # valid
            pad_h = 0
            pad_w = 0
            pad_offset_h = 0
            pad_offset_w = 0
        
        # Activation clamp defaults depend on activation dtype
        resolved_tensor_dtypes = desc.get("resolved_tensor_dtypes") or {}
        tensor_dtypes = desc.get("tensor_dtypes") or {}
        act_dtype = str(
            resolved_tensor_dtypes.get(
                "input",
                tensor_dtypes.get("input", desc.get('activation_dtype', 'S8')),
            )
        ).upper()
        if act_dtype in {'FP32', 'FP16'}:
            activation_min = float(desc.get('activation_min', -1.0e30))
            activation_max = float(desc.get('activation_max', 1.0e30))
            in_zp = 0
            out_zp = 0
        elif act_dtype == 'S16':
            activation_min = int(desc.get('activation_min', -32768))
            activation_max = int(desc.get('activation_max', 32767))
            in_zp = int(TemplateContextBuilder._quant_param_scalar(input_quant, 'zero_point', 0))
            out_zp = int(TemplateContextBuilder._quant_param_scalar(output_quant, 'zero_point', 0))
        else:
            activation_min = int(desc.get('activation_min', -128))
            activation_max = int(desc.get('activation_max', 127))
            in_zp = int(TemplateContextBuilder._quant_param_scalar(input_quant, 'zero_point', 0))
            out_zp = int(TemplateContextBuilder._quant_param_scalar(output_quant, 'zero_point', 0))
        
        return {
            'input_offset': int(-in_zp),
            'output_offset': int(out_zp),
            'stride_h': int(stride_h),
            'stride_w': int(stride_w),
            'dilation_h': int(dil_h),
            'dilation_w': int(dil_w),
            'pad_h': int(pad_h),
            'pad_w': int(pad_w),
            'pad_offset_h': int(pad_offset_h),
            'pad_offset_w': int(pad_offset_w),
            'activation_min': int(activation_min) if act_dtype not in {'FP32', 'FP16'} else activation_min,
            'activation_max': int(activation_max) if act_dtype not in {'FP32', 'FP16'} else activation_max,
        }
    
    @staticmethod
    def calculate_transpose_conv_buffer_size_max(input_dims: Dict[str, int], 
                                                 filter_dims: Dict[str, int],
                                                 output_dims: Dict[str, int],
                                                 output_dtype: str = 'S8',
                                                 stride_h: int = 1,
                                                 stride_w: int = 1,
                                                 reverse_tcol_threshold: int = 16) -> int:
        """
        Calculate a conservative upper bound for transpose convolution buffer size.
        
        TransposeConv requires two buffers:
        1. ctx buffer (from arm_transpose_conv_s8_get_buffer_size)
        2. output_ctx buffer (from arm_transpose_conv_s8_get_reverse_conv_buffer_size)
        
        Returns the maximum of both buffers.
        """
        input_c = input_dims['c']
        filter_w = filter_dims['w']
        filter_h = filter_dims['h']
        output_c = output_dims['c']

        reverse_conv_possible = (stride_w <= 2) and (stride_h <= 2)
        reverse_conv_efficient = (input_c > reverse_tcol_threshold)

        # Rolling-buffer sizing (ns-cmsis-nn issue #261 / PR #262): this is the
        # formula arm_transpose_conv_s8_get_buffer_size() (and the _mve variant)
        # uses directly, and also the lower bound it enforces even when the
        # reverse-conv route is taken (other direct callers of
        # arm_transpose_conv_s8 still need the rolling-buffer sizing). Computed
        # once here so the reverse and non-reverse branches below can never
        # diverge on this formula.
        buf_x = ((input_dims['w'] - 1) * stride_w + max(filter_w, stride_w)) * output_c
        buf_x_mve = ((input_dims['w'] - 1) * stride_w + max(filter_w, stride_h)) * output_c
        buf_y = max(filter_h, stride_h)
        rolling_ctx_size = max(buf_x, buf_x_mve) * buf_y * 4  # int32 scratch

        if output_dtype == 'S16':
            # S16 buffer size (conservative estimate)
            buffer_size_mve = 4 * 8 * filter_w * filter_h * 2  # sizeof(int16_t) = 2
            buffer_size_dsp = 2 * input_c * filter_w * filter_h * 2
            ctx_size = max(buffer_size_mve, buffer_size_dsp)
            output_ctx_size = output_dims['w'] * output_dims['h'] * output_c * 4
        else:
            if reverse_conv_possible and reverse_conv_efficient:
                reverse_conv_input_dims = {
                    'n': input_dims['n'],
                    'h': input_dims['h'] * stride_h,
                    'w': input_dims['w'] * stride_w,
                    'c': input_c,
                }
                reverse_conv_ctx_size = TemplateContextBuilder.calculate_buffer_size_max(
                    reverse_conv_input_dims,
                    filter_dims,
                    output_dims,
                    output_dtype='S8',
                )
                # ns-cmsis-nn issue #261 / PR #262: even when the reverse-conv route is
                # taken, arm_transpose_conv_s8_get_buffer_size() (and the _mve variant)
                # now returns MAX(reverse-conv size, rolling-buffer size), because other
                # direct callers of arm_transpose_conv_s8 still need the rolling-buffer
                # sizing. Mirror that here so this harness bound stays an upper bound.
                ctx_size = max(reverse_conv_ctx_size, rolling_ctx_size)
                output_ctx_size = input_c * filter_w * filter_h * filter_dims['n']
            else:
                ctx_size = rolling_ctx_size
                output_ctx_size = 0
        
        # Return maximum of ctx and output_ctx
        return max(ctx_size, output_ctx_size)
    
    @staticmethod
    def build_fc_params(
        desc: Dict[str, Any],
        input_quant: Dict[str, Any],
        weight_quant: Dict[str, Any],
        output_quant: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build fully connected parameters for CMSIS-NN.
        
        Notes:
        - Fully connected uses cmsis_nn_fc_params
        - Activation clamp defaults depend on activation dtype
        """
        # Activation clamp defaults depend on activation dtype
        resolved_tensor_dtypes = desc.get("resolved_tensor_dtypes") or {}
        tensor_dtypes = desc.get("tensor_dtypes") or {}
        act_dtype = str(
            resolved_tensor_dtypes.get(
                "input",
                tensor_dtypes.get("input", desc.get('activation_dtype', 'S8')),
            )
        ).upper()
        if act_dtype in {'FP32', 'FP16'}:
            activation_min = float(desc.get('activation_min', -1.0e30))
            activation_max = float(desc.get('activation_max', 1.0e30))
        elif act_dtype == 'S16':
            activation_min = int(desc.get('activation_min', -32768))
            activation_max = int(desc.get('activation_max', 32767))
        else:
            activation_min = int(desc.get('activation_min', -128))
            activation_max = int(desc.get('activation_max', 127))
        
        in_zp = input_quant.get('zero_point', 0)
        if isinstance(in_zp, (list, np.ndarray)):
            in_zp = int(in_zp[0])
        else:
            in_zp = int(in_zp)
        
        weight_zp = weight_quant.get('zero_point', 0)
        if isinstance(weight_zp, (list, np.ndarray)):
            weight_zp = int(weight_zp[0])
        else:
            weight_zp = int(weight_zp)
        
        out_zp = output_quant.get('zero_point', 0)
        if isinstance(out_zp, (list, np.ndarray)):
            out_zp = int(out_zp[0])
        else:
            out_zp = int(out_zp)
        
        # For S16 fully connected, CMSIS-NN requires offsets to be 0
        # (S16 quantization typically uses symmetric quantization with zero_point=0)
        if act_dtype in {'FP32', 'FP16'}:
            input_offset = 0
            filter_offset = 0
            output_offset = 0
        elif act_dtype == 'S16':
            input_offset = 0
            filter_offset = 0
            output_offset = 0
        else:
            # For S8, use zero points as offsets
            input_offset = int(-in_zp)
            filter_offset = int(-weight_zp)
            output_offset = int(out_zp)
        
        return {
            'input_offset': input_offset,
            'filter_offset': filter_offset,
            'output_offset': output_offset,
            'activation_min': activation_min,
            'activation_max': activation_max,
        }
    
    @staticmethod
    def calculate_fc_buffer_size_max(filter_dims: Dict[str, int],
                                     output_dtype: str = 'S8') -> int:
        """
        Calculate a conservative upper bound for fully connected buffer size.
        
        For S8, buffer size depends on filter_dims (col_dim * row_dim).
        For S16, buffer size is typically 0 or minimal.
        """
        # For fully connected, buffer size is typically small or 0
        # Use a conservative estimate based on filter dimensions
        col_dim = filter_dims.get('n', 1)  # input features
        row_dim = filter_dims.get('c', 1)  # output features
        
        if output_dtype == 'S16':
            # S16 typically doesn't need a buffer, but include small estimate
            return max(0, col_dim * 2)  # Conservative estimate
        else:
            # S8 buffer size: typically col_dim * sizeof(int32_t) for weight sum
            # But we use a conservative estimate
            return max(col_dim * 4, row_dim * 4)  # Conservative estimate
    
    @staticmethod
    def build_pool_params(
        desc: Dict[str, Any],
        input_shape: Tuple[int, ...],
        kernel_shape: Tuple[int, int],
        output_shape: Tuple[int, ...],
        output_quant: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build pooling parameters for CMSIS-NN.
        
        Notes:
        - input_shape/output_shape are expected NHWC (TFLite).
        - kernel_shape is (pool_h, pool_w).
        - Pooling uses cmsis_nn_pool_params (no input_offset/output_offset).
        """
        strides = desc.get('strides', [1, 1])
        if isinstance(strides, (int, float)):
            stride_h = stride_w = int(strides)
        else:
            stride_h = int(strides[0])
            stride_w = int(strides[1])
        
        padding = desc.get('padding', 'valid')
        padding = 'valid' if padding is None else str(padding).lower()
        
        pool_h, pool_w = int(kernel_shape[0]), int(kernel_shape[1])
        
        # NHWC -> H,W extraction
        if len(input_shape) == 4:
            in_h, in_w = int(input_shape[1]), int(input_shape[2])
        elif len(input_shape) == 3:
            in_h, in_w = int(input_shape[0]), int(input_shape[1])
        else:
            raise ValueError(f"Unsupported Pooling input_shape rank: {len(input_shape)}")
        
        if len(output_shape) == 4:
            out_h, out_w = int(output_shape[1]), int(output_shape[2])
        elif len(output_shape) == 3:
            out_h, out_w = int(output_shape[0]), int(output_shape[1])
        else:
            raise ValueError(f"Unsupported Pooling output_shape rank: {len(output_shape)}")
        
        # Pooling padding calculation
        if padding == 'same':
            pad_total_h = max((out_h - 1) * stride_h + pool_h - in_h, 0)
            pad_total_w = max((out_w - 1) * stride_w + pool_w - in_w, 0)
            pad_h = pad_total_h // 2
            pad_w = pad_total_w // 2
        else:  # valid
            pad_h = 0
            pad_w = 0
        
        # Activation clamp defaults depend on activation dtype
        resolved_tensor_dtypes = desc.get("resolved_tensor_dtypes") or {}
        tensor_dtypes = desc.get("tensor_dtypes") or {}
        act_dtype = str(
            resolved_tensor_dtypes.get(
                "input",
                tensor_dtypes.get("input", desc.get('activation_dtype', 'S8')),
            )
        ).upper()
        if act_dtype in {'FP32', 'FP16'}:
            activation_min = float(desc.get('activation_min', -1.0e30))
            activation_max = float(desc.get('activation_max', 1.0e30))
        elif act_dtype == 'S16':
            activation_min = int(desc.get('activation_min', -32768))
            activation_max = int(desc.get('activation_max', 32767))
        else:
            activation_min = int(desc.get('activation_min', -128))
            activation_max = int(desc.get('activation_max', 127))

        return {
            'stride_h': int(stride_h),
            'stride_w': int(stride_w),
            'pad_h': int(pad_h),
            'pad_w': int(pad_w),
            'activation_min': activation_min,
            'activation_max': activation_max,
        }
    
    @staticmethod
    def calculate_pooling_buffer_size_max(input_dims: Dict[str, int],
                                          output_dims: Dict[str, int],
                                          pooling_type: str = 'AVERAGE',
                                          output_dtype: str = 'S8') -> int:
        """
        Calculate maximum buffer size for pooling operations.
        
        Args:
            input_dims: Input dimensions dict with n, h, w, c
            output_dims: Output dimensions dict with n, h, w, c
            pooling_type: 'AVERAGE' or 'MAX'
            output_dtype: 'S8' or 'S16'
            
        Returns:
            Maximum buffer size in bytes
        """
        # Max pooling doesn't need a buffer
        if pooling_type == 'MAX':
            return 0
        
        # Average pooling buffer size calculation
        # MVE: 0
        # DSP: input_channels * sizeof(int32_t) = input_channels * 4
        # Default: 0
        # We return the maximum (DSP) to be safe
        input_c = input_dims.get('c', 0)
        buffer_size_dsp = input_c * 4  # ch_src * sizeof(int32_t)
        
        return buffer_size_dsp
    
    @staticmethod
    def get_dtype_c_type(dtype: str) -> str:
        """
        Convert dtype string to C type.
        
        Args:
            dtype: Dtype string (S8, S16, etc.)
            
        Returns:
            C type string
        """
        dtype_map = {
            'S8': 'int8_t',
            'S16': 'int16_t',
            'S32': 'int32_t',
            'U8': 'uint8_t',
            'U16': 'uint16_t',
            'U32': 'uint32_t',
        }
        return dtype_map.get(dtype, 'int8_t')
