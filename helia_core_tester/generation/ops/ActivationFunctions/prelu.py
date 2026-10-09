"""
PReLU operation implementation.
"""

from typing import Dict, Any, Iterable
import numpy as np
from pathlib import Path
from helia_core_tester.generation.ops._shared.base import OperationBase
from helia_core_tester.generation.harness import ArrayLiteral, Declaration
from helia_core_tester.generation.harness.simple import dims_count, tensor_case_pool


def prelu_argument_pool(context):
    """Every value a PReLU case can pass to a public PReLU kernel."""
    n = context["name"]
    alpha = Declaration(f"{n}_alpha", context["alpha_dtype"], ArrayLiteral(context["alpha_array"]),
                        array=True, comment="Alpha")
    values = {"alpha": f"{n}_alpha", "input_offset": context["input_offset"], "alpha_offset": context["alpha_offset"],
              "output_offset": context["output_offset"],
              "output_multiplier_identity": context["output_mult_identity"],
              "output_shift_identity": context["output_shift_identity"],
              "output_multiplier_alpha": context["output_mult_alpha"], "output_shift_alpha": context["output_shift_alpha"]}
    return tensor_case_pool(context, values, dims=("input_dims", "alpha_dims", "output_dims"), extra_header=(alpha,),
                            output_count=dims_count(context["output_dims"]))


class OpPReLU(OperationBase):
    """
    PReLU (Parametric ReLU) operation.
    """

    SIGN_SPAN_OPERANDS = ("input", "alpha")
    
    def _prepare_alpha_values(
        self,
        alpha_shape: tuple,
        alpha_values: Iterable[float] | None = None
    ) -> np.ndarray:
        """Prepare alpha values for PReLU layer."""
        if not alpha_shape:
            raise ValueError("alpha_shape must include at least one dimension for PReLU.")
        num_values = int(np.prod(alpha_shape))
        
        if alpha_values is None:
            # Default: linear spacing from 0.05 to 0.25
            data = np.linspace(0.05, 0.25, num=num_values, dtype=np.float32)
        else:
            data = np.asarray(alpha_values, dtype=np.float32)
            # If single scalar, expand to match input shape
            if data.size == 1:
                data = np.full(num_values, data[0], dtype=np.float32)
            elif data.size != num_values:
                raise ValueError(
                    f"alpha_values has {data.size} entries, but expected {num_values} "
                    f"to match alpha shape {alpha_shape}."
                )
        return data.reshape(alpha_shape)
    
    # The (scale, zero point) the one-op LiteRT builder gave input, alpha and output;
    # kept so the goldens do not move.
    FIXED_QUANT = {"S8": (0.125, 0), "S16": (1.0 / 32768.0, 0)}

    def uses_reference(self) -> bool:
        return not self._is_arg_error_case() and self.desc.get("activation_dtype", "S8") == "S8"

    def _expected_status(self) -> str:
        return self.desc.get("expected_status", "ARM_CMSIS_NN_SUCCESS")

    def _is_arg_error_case(self) -> bool:
        return self._expected_status() != "ARM_CMSIS_NN_SUCCESS"

    def _resolved_alpha_shape(self) -> tuple:
        input_shape = tuple(self.desc["input_shape"])
        alpha_shape = self.desc.get("alpha_shape")
        if alpha_shape is not None:
            return tuple(alpha_shape)
        return input_shape[1:]

    def _validate_broadcast_support(self, input_shape: tuple, alpha_shape: tuple) -> None:
        """
        Reject scalar-input + multi-element-alpha broadcasts at load/convert time.

        LiteRT's PRELU op preparation cannot handle a scalar (single-element)
        input broadcasting against a multi-element alpha (fails with
        "HaveSameShapes input/output" during graph preparation). Rather than
        letting that surface as an opaque LiteRT prepare failure, fail fast
        with an actionable message pointing at the supported direct-kernel
        path (operator: PReLUScalar) for this broadcast shape.
        """
        if int(np.prod(input_shape)) == 1 and int(np.prod(alpha_shape)) > 1:
            raise ValueError(
                "PReLU with a scalar (single-element) input and a multi-element alpha "
                "broadcast is not supported via LiteRT (known PRELU prepare failure: "
                "'HaveSameShapes input/output'). Use operator: PReLUScalar instead, which "
                "implements this broadcast directly against arm_prelu_scalar_s8 without "
                "requiring LiteRT model preparation."
            )

    def _select_cmsis_prelu_kernel(self) -> Dict[str, str]:
        """
        Select appropriate CMSIS-NN kernel function for PReLU operation.
        
        Returns:
            Dictionary with kernel_fn, input_c_type, output_c_type
        """
        activation_dtype = self.desc.get('activation_dtype', 'S8')
        
        if activation_dtype == 'S8':
            return {
                'kernel_fn': 'arm_prelu_s8',
                'input_c_type': 'int8_t',
                'output_c_type': 'int8_t',
                'float_kernel': False,
            }
        elif activation_dtype == 'S16':
            return {
                'kernel_fn': 'arm_prelu_s16',
                'input_c_type': 'int16_t',
                'output_c_type': 'int16_t',
                'float_kernel': False,
            }
        elif activation_dtype == 'FP32':
            return {
                'kernel_fn': 'arm_prelu_f32',
                'input_c_type': 'float',
                'output_c_type': 'float',
                'float_kernel': True,
            }
        elif activation_dtype == 'FP16':
            return {
                'kernel_fn': 'arm_prelu_f16',
                'input_c_type': 'float16_t',
                'output_c_type': 'float16_t',
                'float_kernel': True,
            }
        else:
            raise NotImplementedError(f"Unsupported PReLU dtype: {activation_dtype} (only S8/S16 supported)")
    
    def _generate_arg_error_c_files(self, output_dir: Path) -> None:
        """
        Generate a CMSIS-direct harness for a deliberately-mismatched-shape
        PReLU test case, expecting arm_prelu_s8 to return ARM_CMSIS_NN_ARG_ERROR
        (input_dims != output_dims is rejected up front by the kernel, before
        any golden is needed).
        """
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder
        from helia_core_tester.generation.utils.tflite_utils import calculate_multiplier_shift

        name = self.desc['name']
        kernel_info = self._select_cmsis_prelu_kernel()

        input_shape = tuple(self.desc['input_shape'])
        alpha_shape = self._resolved_alpha_shape()
        output_shape = tuple(self.desc.get('output_shape', input_shape))

        builder = TemplateContextBuilder()
        input_dims = builder.nhwc_to_cmsis_dims(input_shape)
        alpha_dims = builder.nhwc_to_cmsis_dims(alpha_shape)
        output_dims = builder.nhwc_to_cmsis_dims(output_shape)

        mult_identity, shift_identity = calculate_multiplier_shift(1.0)
        mult_alpha, shift_alpha = calculate_multiplier_shift(1.0)

        if kernel_info["input_c_type"] == "int16_t":
            np_dtype = np.int16
            qmin, qmax = -32768, 32768
        else:
            np_dtype = np.int8
            qmin, qmax = -128, 128

        rng = self._seeded_rng()
        input_q = rng.integers(qmin, qmax, size=input_shape, dtype=np.int32).astype(np_dtype)
        alpha_q = rng.integers(qmin, qmax, size=alpha_shape, dtype=np.int32).astype(np_dtype)

        # Kernel is expected to reject before producing real output; a single
        # placeholder element is sufficient (mirrors Transpose's ARG_ERROR path).
        expected_output = np.zeros((1,), dtype=np_dtype)

        context = {
            'name': name,
            'input_dims': input_dims,
            'alpha_dims': alpha_dims,
            'output_dims': output_dims,
            'input_offset': 0,
            'alpha_offset': 0,
            'output_offset': 0,
            'output_mult_alpha': int(mult_alpha),
            'output_shift_alpha': int(shift_alpha),
            'output_mult_identity': int(mult_identity),
            'output_shift_identity': int(shift_identity),
            'input_data_array': builder.format_array_as_c_literal(input_q),
            'alpha_array': builder.format_array_as_c_literal(alpha_q),
            'expected_output_array': builder.format_array_as_c_literal(expected_output),
            'input_dtype': kernel_info["input_c_type"],
            'output_dtype': kernel_info["output_c_type"],
            'alpha_dtype': kernel_info["input_c_type"],
            'kernel_fn': kernel_info["kernel_fn"],
            'expected_status': self._expected_status(),
        }

        self.render_harness_case(
            output_dir, stem="prelu", context=context, pool=prelu_argument_pool(context),
            validation_key="ActivationFunctions/prelu/prelu.c.j2", label="PReLU", operator="PReLU",
        )

    @staticmethod
    def _reference_prelu_s16(
        *,
        input_q: np.ndarray,
        alpha_q: np.ndarray,
        input_dims: Dict[str, int],
        alpha_dims: Dict[str, int],
        input_offset: int,
        alpha_offset: int,
        output_offset: int,
        mult_identity: int,
        shift_identity: int,
        mult_alpha: int,
        shift_alpha: int,
    ) -> np.ndarray:
        """
        Compute the golden output for arm_prelu_s16 using exact CMSIS-NN fixed-point math.

        Mirrors arm_elementwise_prelu_s16: for each element, the identity path is taken
        when (input + input_offset) >= 0, otherwise the alpha path is used. Alpha is
        broadcast across the input following the NHWC PReLU broadcast rules.
        """
        from helia_core_tester.generation.utils.tflite_utils import requantize_np

        in_shape = (input_dims['n'], input_dims['h'], input_dims['w'], input_dims['c'])
        a_shape = (alpha_dims['n'], alpha_dims['h'], alpha_dims['w'], alpha_dims['c'])

        inp = input_q.reshape(in_shape).astype(np.int64)
        alp = alpha_q.reshape(a_shape).astype(np.int64)
        alp = np.broadcast_to(alp, in_shape)

        input_value = inp + int(input_offset)
        alpha_value = alp + int(alpha_offset)

        identity = requantize_np(input_value, int(mult_identity), int(shift_identity))
        alpha_path = requantize_np(input_value * alpha_value, int(mult_alpha), int(shift_alpha))

        out = np.where(input_value >= 0, identity, alpha_path).astype(np.int64) + int(output_offset)
        out = np.clip(out, -32768, 32767).astype(np.int16)
        return out

    def _descriptor_alpha_values(self):
        """Alpha values as authored in the descriptor, or None for the default ramp."""
        alpha_values = None
        if "alpha" in self.desc:
            alpha_scalar = self.desc["alpha"]
            if isinstance(alpha_scalar, (int, float)):
                alpha_values = [float(alpha_scalar)]
            elif isinstance(alpha_scalar, list):
                alpha_values = alpha_scalar
        if alpha_values is None and "hint" in self.desc:
            extras = self.desc.get("hint", {}).get("extras", {})
            if "alpha_values" in extras:
                alpha_list = extras["alpha_values"]
                if isinstance(alpha_list, list) and len(alpha_list) > 0:
                    if isinstance(alpha_list[0], list):
                        alpha_values = [item for sublist in alpha_list for item in sublist]
                    else:
                        alpha_values = alpha_list
        return alpha_values

    def _generate_float_c_files(self, output_dir: Path, kernel_info: Dict[str, str]) -> None:
        """
        Generate C and H files for the float PReLU kernels.

        arm_prelu_f32/f16 take (input_dims, input, alpha_dims, alpha,
        output_dims, output) with no quantization parameters; alpha and the
        golden output are derived from the descriptor with numpy (PReLU is
        exact in the working precision, so no interpreter is needed).
        """
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

        name = self.desc['name']
        float_dtype = np.float16 if kernel_info["input_c_type"] == "float16_t" else np.float32

        input_shape = tuple(self.desc["input_shape"])
        alpha_shape = self._resolved_alpha_shape()
        alpha = self._prepare_alpha_values(alpha_shape, self._descriptor_alpha_values()).astype(float_dtype)

        builder = TemplateContextBuilder()
        input_dims = builder.nhwc_to_cmsis_dims(input_shape)
        output_dims = builder.nhwc_to_cmsis_dims(input_shape)
        if len(alpha_shape) == 1:
            alpha_dims = {'n': 1, 'h': 1, 'w': 1, 'c': int(alpha_shape[0])}
        else:
            alpha_dims = builder.nhwc_to_cmsis_dims(alpha_shape)

        rng_state = self.rng.__getstate__()
        self.rng = np.random.default_rng(self.seed)
        input_q = self._maybe_apply_input_mode(
            self.rng.uniform(-1.0, 1.0, size=input_shape).astype(float_dtype)
        )
        self.rng.__setstate__(rng_state)

        alpha_bc = alpha.reshape(
            (alpha_dims['n'], alpha_dims['h'], alpha_dims['w'], alpha_dims['c'])
        )
        output_data = np.where(input_q >= 0, input_q, input_q * alpha_bc).astype(float_dtype)

        context = {
            'name': name,
            'input_dims': input_dims,
            'alpha_dims': alpha_dims,
            'output_dims': output_dims,
            'input_offset': 0,
            'alpha_offset': 0,
            'output_offset': 0,
            'output_mult_alpha': 0,
            'output_shift_alpha': 0,
            'output_mult_identity': 0,
            'output_shift_identity': 0,
            'input_data_array': builder.format_array_as_c_literal(input_q),
            'alpha_array': builder.format_array_as_c_literal(alpha),
            'expected_output_array': builder.format_array_as_c_literal(output_data),
            'input_dtype': kernel_info["input_c_type"],
            'output_dtype': kernel_info["output_c_type"],
            'alpha_dtype': kernel_info["input_c_type"],
            'kernel_fn': kernel_info["kernel_fn"],
            'float_kernel': True,
            'validation_mode': 'float',
        }
        self.render_harness_case(
            output_dir, stem="prelu", context=context, pool=prelu_argument_pool(context),
            validation_key="ActivationFunctions/prelu/prelu.c.j2", label="PReLU", operator="PReLU", sidecar=True,
        )

    def generate_c_files(self, output_dir: Path) -> None:
        """
        Generate C and H files from templates for PReLU operation.
        """
        if self._is_arg_error_case():
            self._generate_arg_error_c_files(output_dir)
            return

        float_kernel_info = self._select_cmsis_prelu_kernel()
        if float_kernel_info.get('float_kernel'):
            self._generate_float_c_files(output_dir, float_kernel_info)
            return

        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder
        from helia_core_tester.generation.reference import bindings as b

        name = self.desc['name']
        kernel_info = self._select_cmsis_prelu_kernel()
        input_shape = tuple(int(d) for d in self.desc["input_shape"])
        output_shape = input_shape
        alpha_shape = tuple(int(d) for d in self._resolved_alpha_shape())
        self._validate_broadcast_support(input_shape, alpha_shape)
        alpha_weights = self._prepare_alpha_values(alpha_shape, self._descriptor_alpha_values())

        scale, zp = self.FIXED_QUANT[self.desc.get("activation_dtype", "S8")]
        input_scale = output_scale = alpha_scale = scale
        input_zp = output_zp = alpha_zp = zp

        builder = TemplateContextBuilder()
        
        # Convert shapes to CMSIS dims
        input_dims = builder.nhwc_to_cmsis_dims(input_shape)
        output_dims = builder.nhwc_to_cmsis_dims(output_shape)
        alpha_shape = alpha_weights.shape
        
        # For PReLU, alpha dimensions should match the input's non-singleton dimensions
        if len(alpha_shape) == 1 and len(input_shape) >= 2:
            # Alpha is 1D, input is 2D+: match alpha to input's layout
            # If input was converted as width-based (w=shape[1], c=1), alpha should match
            if input_dims['w'] > 1 and input_dims['c'] == 1:
                # Input uses width dimension, so alpha should too
                alpha_dims = {
                    'n': 1,
                    'h': 1,
                    'w': int(alpha_shape[0]),
                    'c': 1
                }
            elif input_dims['c'] > 1 and input_dims['w'] == 1:
                # Input uses channel dimension, so alpha should too
                alpha_dims = {
                    'n': 1,
                    'h': 1,
                    'w': 1,
                    'c': int(alpha_shape[0])
                }
            else:
                # Default: match input dimensions
                alpha_dims = builder.nhwc_to_cmsis_dims(alpha_shape)
        else:
            # Use standard conversion
            alpha_dims = builder.nhwc_to_cmsis_dims(alpha_shape)
        
        prepared = b.struct_to_dict(
            b.get_bindings().prelu_prepare(input_scale, input_zp, alpha_scale, alpha_zp, output_scale, output_zp)
        )
        mult_identity, shift_identity = prepared["multiplier_1"], prepared["shift_1"]
        mult_alpha, shift_alpha = prepared["multiplier_2"], prepared["shift_2"]

        # Quantize alpha weights
        # Check if alpha_weights are already quantized (int8/int16) or float
        if kernel_info["input_c_type"] == "int16_t":
            np_alpha_dtype = np.int16
            alpha_qmin, alpha_qmax = -32768, 32767
            alpha_c_type = "int16_t"
        else:
            np_alpha_dtype = np.int8
            alpha_qmin, alpha_qmax = -128, 127
            alpha_c_type = "int8_t"

        alpha_q = np.round(alpha_weights / float(alpha_scale) + float(alpha_zp)).astype(np.int32)
        alpha_q = np.clip(alpha_q, alpha_qmin, alpha_qmax).astype(np_alpha_dtype)

        # Generate input data and quantize
        rng_state = self.rng.__getstate__()
        self.rng = np.random.default_rng(self.seed)

        extras = self.desc.get("hint", {}).get("extras", {})
        if "input_values" in extras:
            values = np.asarray(extras["input_values"], dtype=np.float32).flatten()
            num = int(np.prod(input_shape))
            if values.size == 1:
                input_data = np.full(num, values[0], dtype=np.float32).reshape(input_shape)
            elif values.size == num:
                input_data = values.reshape(input_shape)
            else:
                raise ValueError(
                    f"input_values has {values.size} entries, expected {num} to match input shape {input_shape}."
                )
        else:
            input_data = self.rng.uniform(-1.0, 1.0, size=input_shape).astype(np.float32)

        self.rng.__setstate__(rng_state)
        
        # Quantize inputs
        if kernel_info["input_c_type"] == "int8_t":
            np_in_dtype = np.int8
            qmin, qmax = -128, 127
        elif kernel_info["input_c_type"] == "int16_t":
            np_in_dtype = np.int16
            qmin, qmax = -32768, 32767
        else:
            raise ValueError(f"Unsupported input_c_type: {kernel_info['input_c_type']}")
        
        input_q = np.round(input_data / float(input_scale) + float(input_zp)).astype(np.int32)
        input_q = np.clip(input_q, qmin, qmax).astype(np_in_dtype)
        # alpha is the case's fixed slope constant, not drawn data, so it can only
        # be waived, never steered.
        # A descriptor that pins input_values chose those exact operands, so
        # the input is check-only for the same reason the values exist.
        input_q, _ = self._enforce_int_operand_sign_span(
            (("input", input_q, input_zp), ("alpha", alpha_q, alpha_zp)),
            steerable=() if "input_values" in extras else ("input",),
        )
        
        if kernel_info["input_c_type"] == "int16_t":
            # The LiteRT reference interpreter does not support int16 PReLU, so the
            # golden output is computed here using the exact arm_prelu_s16 fixed-point
            # math (see arm_elementwise_prelu_s16 in CMSIS-NN).
            output_data = self._reference_prelu_s16(
                input_q=input_q,
                alpha_q=alpha_q,
                input_dims=input_dims,
                alpha_dims=alpha_dims,
                input_offset=-int(input_zp),
                alpha_offset=-int(alpha_zp),
                output_offset=int(output_zp),
                mult_identity=int(mult_identity),
                shift_identity=int(shift_identity),
                mult_alpha=int(mult_alpha),
                shift_alpha=int(shift_alpha),
            )
        else:
            from helia_core_tester.generation.reference.case import ReferenceCall

            q = {"scale": scale, "zero_point": zp}
            call = ReferenceCall(
                "prelu_s8", prepared,
                {"input": np.ascontiguousarray(input_q), "alpha": np.ascontiguousarray(alpha_q)},
                output_shape, "int8", quant={"input": q, "alpha": q, "output": q},
            )
            output_data = self.reference_golden(call)

        # Format arrays
        input_array_str = builder.format_array_as_c_literal(input_q)
        alpha_array_str = builder.format_array_as_c_literal(alpha_q)
        expected_output_array_str = builder.format_array_as_c_literal(output_data)
        
        # Build template context
        context = {
            'name': name,
            'input_dims': input_dims,
            'alpha_dims': alpha_dims,
            'output_dims': output_dims,
            'input_offset': -int(input_zp),  # Negated for CMSIS-NN
            'alpha_offset': -int(alpha_zp),  # Negated for CMSIS-NN
            'output_offset': int(output_zp),  # Not negated
            'output_mult_alpha': int(mult_alpha),
            'output_shift_alpha': int(shift_alpha),
            'output_mult_identity': int(mult_identity),
            'output_shift_identity': int(shift_identity),
            'input_data_array': input_array_str,
            'alpha_array': alpha_array_str,
            'expected_output_array': expected_output_array_str,
            'input_dtype': kernel_info["input_c_type"],
            'output_dtype': kernel_info["output_c_type"],
            'alpha_dtype': alpha_c_type,
            'kernel_fn': kernel_info["kernel_fn"],
        }
        
        self.render_harness_case(
            output_dir, stem="prelu", context=context, pool=prelu_argument_pool(context),
            validation_key="ActivationFunctions/prelu/prelu.c.j2", label="PReLU", operator="PReLU",
        )
        
