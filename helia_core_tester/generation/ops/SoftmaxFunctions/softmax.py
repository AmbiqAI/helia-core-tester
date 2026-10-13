"""
Softmax operation implementation for Helia-Core Tester.

Goldens come from the C reference: TFLite's Softmax<int8> (gemmlowp fixed point) for s8 input,
SoftmaxInt16 with TFLM's LUTs for s16, and exp in binary64 rounded once for f32/f16.
"""

from typing import Any, Dict, Tuple
import numpy as np
from pathlib import Path
from helia_core_tester.generation.harness import ArgumentPool, ArrayLiteral, Declaration
from helia_core_tester.generation.harness.simple import dims_count, tensor_case_pool
from helia_core_tester.generation.ops._shared.base import OperationBase
from helia_core_tester.generation.ops.SoftmaxFunctions.softmax_luts import EXP_LUT, ONE_BY_ONE_LUT


def softmax_argument_pool(context: Dict[str, Any]) -> ArgumentPool:
    """The s8 kernels return void and take diff_min; the s16 kernel takes the LUT pair instead."""
    n = context["name"]
    luts: tuple = ()
    values: Dict[str, Any] = {"num_rows": context["num_rows"], "row_size": context["row_size"]}
    if not context["float_kernel"]:
        values.update(mult=context["mult"], shift=context["shift"], diff_min=context["diff_min"])
    if context["uses_lut"]:
        luts = (Declaration(f"{n}_exp_lut", "int16_t", ArrayLiteral(EXP_LUT), array=True, extent="513"),
                Declaration(f"{n}_one_by_one_lut", "int16_t", ArrayLiteral(ONE_BY_ONE_LUT), array=True, extent="513"),
                Declaration(f"{n}_softmax_params", "cmsis_nn_softmax_lut_s16",
                            {"exp_lut": f"{n}_exp_lut", "one_by_one_lut": f"{n}_one_by_one_lut"}))
        values["softmax_params"] = f"&{n}_softmax_params"
    return tensor_case_pool(context, values, extra_header=luts, output_count=dims_count(context["output_dims"]))


class OpSoftmax(OperationBase):
    """
    Softmax operation.
    """

    def uses_reference(self) -> bool:
        return True

    def _select_cmsis_softmax_kernel(self) -> Dict[str, str]:
        activation_dtype = self.tensor_dtype("input")
        kernels = {
            'S8': ('arm_softmax_s8', 'int8_t'),
            'S16': ('arm_softmax_s16', 'int16_t'),
            'FP32': ('arm_softmax_f32', 'float'),
            'FP16': ('arm_softmax_f16', 'float16_t'),
        }
        if activation_dtype not in kernels:
            raise NotImplementedError(f"Unsupported Softmax dtype: {activation_dtype}")
        fn, c_type = kernels[activation_dtype]
        return {'kernel_fn': fn, 'input_c_type': c_type, 'output_c_type': c_type}

    def _hint(self) -> Dict[str, Any]:
        hint = self.desc.get("hint", {})
        return hint if isinstance(hint, dict) else {}

    def _s8_to_s16(self) -> bool:
        """arm_softmax_s8_s16: only the CMSIS-direct cases with an S16 output hint select it."""
        hint = self._hint()
        extras = hint.get("extras", {}) or {}
        output_dtype = str(hint.get("output_dtype", extras.get("output_dtype", ""))).upper()
        return bool(hint.get("force_cmsis", False)) and output_dtype == "S16"

    def _quantized_reference(self, kind: str, input_shape: Tuple[int, ...]):
        """Input draw (every code), quantization, prepared params and the golden."""
        from helia_core_tester.generation.reference import policy
        from helia_core_tester.generation.reference import quant as ref_quant
        from helia_core_tester.generation.reference.bindings import get_bindings
        from helia_core_tester.generation.reference.call import ReferenceCall

        s8_to_s16 = kind == "s8" and self._s8_to_s16()
        hint = self._hint()
        if "input_scale" in hint:
            in_quant = policy.TensorQuant(float(np.float32(hint["input_scale"])), 0, kind)
        else:
            # The range the converter used to calibrate over.
            in_quant = self.activation_quant("input", (-1.0, 1.0), kind)
        # The op fixes the output quantization.
        if kind == "s16":
            out_kind, out_scale, out_zp = "s16", 1.0 / 32768, 0
        elif s8_to_s16:
            out_kind, out_scale, out_zp = "s16", 1.0 / 65536, -32768
        else:
            out_kind, out_scale, out_zp = "s8", 1.0 / 256, -128

        rng_state = self.rng.__getstate__()
        self.rng = np.random.default_rng(self.seed)
        np_dtype = np.int8 if kind == "s8" else np.int16
        info = np.iinfo(np_dtype)
        input_q = self.rng.integers(info.min, info.max + 1, size=input_shape, dtype=np_dtype)
        self.rng.__setstate__(rng_state)

        params = get_bindings().prepare("softmax_prepare", {
            "input_dtype": ref_quant.hct_dtype(kind.upper()), "output_dtype": ref_quant.hct_dtype(out_kind.upper()),
            "beta": 1.0, "input_scale": in_quant.scale, "input_zero_point": in_quant.zero_point,
            "output_scale": out_scale, "output_zero_point": out_zp})
        if "diff_min" in hint:
            params = {**params, "diff_min": int(hint["diff_min"])}
        entry = "softmax_s8_s16" if s8_to_s16 else f"softmax_{kind}"
        output = self.reference_golden(ReferenceCall(
            entry, params, {"input": np.ascontiguousarray(input_q)}, {"output": input_shape},
            quant={"input": in_quant.to_json(), "output": {"scale": out_scale, "zero_point": out_zp}, "beta": 1.0},
        ))
        return input_q, params, output, s8_to_s16

    def generate_c_files(self, output_dir: Path) -> None:
        from helia_core_tester.generation.reference.call import ReferenceCall
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

        name = self.desc['name']
        kernel_info = self._select_cmsis_softmax_kernel()
        float_kernel = kernel_info["input_c_type"] in {"float", "float16_t"}
        if self._hint().get("force_cmsis", False) and kernel_info["input_c_type"] != "int8_t":
            raise ValueError("CMSIS-only softmax currently supports int8 input only.")
        input_shape = tuple(int(d) for d in self.desc["input_shape"])
        if not input_shape or any(d < 1 for d in input_shape):
            raise ValueError(f"{name}: invalid input_shape {input_shape}")

        builder = TemplateContextBuilder()
        row_size = int(input_shape[-1])
        num_rows = int(np.prod(input_shape[:-1])) if len(input_shape) >= 2 else 1

        nonfinite_context: Dict[str, Any] = {}
        if float_kernel:
            float_dtype = np.float16 if kernel_info["input_c_type"] == "float16_t" else np.float32
            input_q = self._sample_uniform(input_shape, dtype=float_dtype)
            output_data = self.reference_golden(ReferenceCall(
                "softmax_f16" if float_dtype == np.float16 else "softmax_f32", {"unused": 0},
                {"input": np.ascontiguousarray(input_q)}, {"output": input_shape},
            ))
            output_data, nonfinite_context = self.apply_nonfinite_policy(
                output_data, reference=self.reference_probe, inputs=[input_q]
            )
            mult = shift = diff_min = 0
            kernel_fn, output_c_type, uses_lut = kernel_info["kernel_fn"], kernel_info["output_c_type"], False
        else:
            kind = "s8" if kernel_info["input_c_type"] == "int8_t" else "s16"
            input_q, params, output_data, s8_to_s16 = self._quantized_reference(kind, input_shape)
            mult, shift, diff_min = params["input_multiplier"], params["input_left_shift"], params["diff_min"]
            if s8_to_s16:
                kernel_fn, output_c_type, uses_lut = "arm_softmax_s8_s16", "int16_t", False
            else:
                kernel_fn, output_c_type, uses_lut = kernel_info["kernel_fn"], kernel_info["output_c_type"], kind == "s16"

        context = {
            'name': name,
            'input_dims': builder.nhwc_to_cmsis_dims(input_shape),
            'output_dims': builder.nhwc_to_cmsis_dims(input_shape),
            'num_rows': num_rows,
            'row_size': row_size,
            'mult': int(mult),
            'shift': int(shift),
            'diff_min': int(diff_min),
            'input_data_array': builder.format_array_as_c_literal(input_q),
            'expected_output_array': builder.format_array_as_c_literal(output_data),
            'input_dtype': kernel_info["input_c_type"],
            'output_dtype': output_c_type,
            'kernel_fn': kernel_fn,
            'uses_lut': uses_lut,
            'float_kernel': float_kernel,
        }
        context.update(nonfinite_context)

        self.render_harness_case(
            output_dir, stem="softmax", context=context, pool=softmax_argument_pool(context),
            validation_key="SoftmaxFunctions/softmax/softmax.c.j2", label="Softmax", operator="Softmax",
        )
