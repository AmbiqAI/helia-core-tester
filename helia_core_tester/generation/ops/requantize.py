"""
Requantize operation implementation.
"""

from typing import Dict
import numpy as np
from pathlib import Path
from helia_core_tester.generation.ops.base import OperationBase


class OpRequantize(OperationBase):
    """
    Requantize operation (int8->int8, int16->int16).
    """

    def allow_no_tflite(self) -> bool:
        return True

    def needs_keras_model(self) -> bool:
        return False

    def build_keras_model(self):
        raise NotImplementedError("Requantize uses CMSIS-NN kernel directly; no model required.")

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        raise NotImplementedError("Requantize does not produce a TFLite model.")

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

    @staticmethod
    def _requantize_np(values: np.ndarray, multiplier: int, shift: int) -> np.ndarray:
        left_shift = shift if shift > 0 else 0
        right_shift = -shift if shift < 0 else 0
        prod = values.astype(np.int64) * (1 << left_shift)
        # arm_nn_doubling_high_mult_no_sat
        mult = (1 << 30) + (prod * int(multiplier))
        res = (mult >> 31).astype(np.int64)
        if right_shift == 0:
            return res.astype(np.int32)
        remainder_mask = (1 << right_shift) - 1
        remainder = res & remainder_mask
        result = res >> right_shift
        threshold = remainder_mask >> 1
        threshold = threshold + (result < 0)
        result = result + (remainder > threshold)
        return result.astype(np.int32)

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

        rng_state = self.rng.__getstate__()
        self.rng = np.random.default_rng(self.seed)

        if kernel_info["input_c_type"] == "int8_t":
            np_in_dtype = np.int8
            qmin, qmax = -128, 127
            input_q = self.rng.integers(qmin, qmax + 1, size=input_shape, dtype=np_in_dtype)
            out_dtype = np.int8
        elif kernel_info["input_c_type"] == "int16_t":
            np_in_dtype = np.int16
            qmin, qmax = -32768, 32767
            input_q = self.rng.integers(qmin, qmax + 1, size=input_shape, dtype=np_in_dtype)
            out_dtype = np.int16
        else:
            raise ValueError(f"Unsupported input_c_type: {kernel_info['input_c_type']}")

        self.rng.__setstate__(rng_state)

        centered = input_q.astype(np.int32) - int(input_zp)
        requant = self._requantize_np(centered, multiplier, shift)
        requant = requant + int(output_zp)
        requant = np.clip(requant, qmin, qmax).astype(out_dtype)

        context = {
            "name": name,
            "prefix": name,
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

        includes_api_dir = output_dir / "includes"
        includes_api_dir.mkdir(parents=True, exist_ok=True)

        h_content = self.render_template("requantize/requantize.h.j2", context)
        h_path = includes_api_dir / f"{name}_requantize.h"
        with open(h_path, "w") as f:
            f.write(h_content)

        c_content = self.render_template("requantize/requantize.c.j2", context)
        c_path = output_dir / f"{name}_requantize.c"
        with open(c_path, "w") as f:
            f.write(c_content)

        cmake_context = {
            "name": name,
            "operator": self.desc.get("operator", "Requantize"),
            "operator_name": "requantize",
        }
        cmake_content = self.render_template("common/CMakeLists.txt.j2", cmake_context)
        cmake_path = output_dir / "CMakeLists.txt"
        with open(cmake_path, "w") as f:
            f.write(cmake_content)

