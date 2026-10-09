"""
Requantize operation implementation.
"""

from typing import Dict
import numpy as np
from pathlib import Path
from helia_core_tester.generation.harness import ArgumentPool, ArrayLiteral, Declaration
from helia_core_tester.generation.harness.simple import tensor_case_pool
from helia_core_tester.generation.ops._shared.quantization_base import QuantizationFamilyBase


def requantize_argument_pool(context: Dict) -> ArgumentPool:
    shape = Declaration(f"{context['name']}_input_shape", "int32_t", ArrayLiteral(context["input_shape_array"]), array=True)
    values = {key: context[key] for key in ("effective_scale_multiplier", "effective_scale_shift", "input_zeropoint",
                                            "output_zeropoint")}
    return tensor_case_pool(context, {"size": context["input_size"], **values}, dims=(), extra_header=(shape,),
                            output_count=str(context["input_size"]))


class OpRequantize(QuantizationFamilyBase):
    """
    Requantize operation (int8->int8, int16->int16).
    """

    def needs_tflite(self) -> bool:
        # The golden is computed directly; nothing reads a .tflite.
        return False

    def _select_cmsis_requantize_kernel(self) -> Dict[str, str]:
        activation_dtype = self.desc.get("activation_dtype", "S8")
        if activation_dtype == "S8":
            return {
                "kernel_fn": "arm_requantize_s8_s8",
                "input_c_type": "int8_t",
                "output_c_type": "int8_t",
            }
        if activation_dtype == "S16":
            return {
                "kernel_fn": "arm_requantize_s16_s16",
                "input_c_type": "int16_t",
                "output_c_type": "int16_t",
            }
        raise NotImplementedError(f"Unsupported Requantize dtype: {activation_dtype}")

    def generate_c_files(self, output_dir: Path) -> None:
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

        name = self.desc["name"]
        kernel_info = self._select_cmsis_requantize_kernel()
        builder = TemplateContextBuilder()

        input_shape = tuple(self.desc["input_shape"])
        size = int(np.prod(input_shape))

        multiplier = int(self.desc.get("effective_scale_multiplier", 1073741824))
        shift = int(self.desc.get("effective_scale_shift", 0))
        input_zp = int(self.desc.get("input_zeropoint", 0))
        output_zp = int(self.desc.get("output_zeropoint", 0))

        rng = self._seeded_rng()

        if kernel_info["input_c_type"] == "int8_t":
            np_in_dtype = np.int8
            qmin, qmax = -128, 127
            input_q = rng.integers(qmin, qmax + 1, size=input_shape, dtype=np_in_dtype)
            out_dtype = np.int8
        elif kernel_info["input_c_type"] == "int16_t":
            np_in_dtype = np.int16
            qmin, qmax = -32768, 32767
            input_q = rng.integers(qmin, qmax + 1, size=input_shape, dtype=np_in_dtype)
            out_dtype = np.int16
        else:
            raise ValueError(f"Unsupported input_c_type: {kernel_info['input_c_type']}")

        centered = input_q.astype(np.int32) - int(input_zp)
        requant = self._requantize_np(centered, multiplier, shift)
        requant = requant + int(output_zp)
        requant = np.clip(requant, qmin, qmax).astype(out_dtype)

        context = {
            "name": name,
            "input_size": size,
            "input_shape_array": builder.format_array_as_c_literal(np.array(input_shape, dtype=np.int32)),
            "input_data_array": builder.format_array_as_c_literal(input_q),
            "expected_output_array": builder.format_array_as_c_literal(requant),
            "input_dtype": kernel_info["input_c_type"],
            "output_dtype": kernel_info["output_c_type"],
            "kernel_fn": kernel_info["kernel_fn"],
            "effective_scale_multiplier": multiplier,
            "effective_scale_shift": shift,
            "input_zeropoint": input_zp,
            "output_zeropoint": output_zp,
        }

        self.render_harness_case(
            output_dir, stem="requantize", context=context, pool=requantize_argument_pool(context),
            validation_key="NNSupportFunctions/requantize/requantize.c.j2", label="Requantize", operator="Requantize",
            sidecar=True,
        )
