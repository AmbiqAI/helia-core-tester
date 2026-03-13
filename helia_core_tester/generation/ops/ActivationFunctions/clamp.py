"""
Clamp operation implementation (arm_clamp_s8/s16).
"""

from typing import Dict, Any
import numpy as np
from pathlib import Path
from helia_core_tester.generation.ops._shared.base import OperationBase


class OpClamp(OperationBase):
    """
    Clamp operation.
    """

    def build_keras_model(self):
        raise NotImplementedError("Clamp does not use a Keras model.")

    def needs_keras_model(self) -> bool:
        return False

    def allow_no_tflite(self) -> bool:
        return True

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        raise NotImplementedError("Clamp does not generate TFLite models.")

    def _select_cmsis_clamp_kernel(self) -> Dict[str, str]:
        activation_dtype = self.desc.get('activation_dtype', 'S8')
        if activation_dtype == 'S8':
            return {
                'kernel_fn': 'arm_clamp_s8',
                'input_c_type': 'int8_t',
                'output_c_type': 'int8_t',
                'qmin': -128,
                'qmax': 127,
            }
        if activation_dtype == 'S16':
            return {
                'kernel_fn': 'arm_clamp_s16',
                'input_c_type': 'int16_t',
                'output_c_type': 'int16_t',
                'qmin': -32768,
                'qmax': 32767,
            }
        raise NotImplementedError(f"Unsupported Clamp dtype: {activation_dtype}")

    def generate_c_files(self, output_dir: Path) -> None:
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

        name = self.desc['name']
        kernel_info = self._select_cmsis_clamp_kernel()

        input_shape = tuple(self.desc['input_shape'])
        output_shape = input_shape

        act_min = int(self.desc.get('act_min', kernel_info['qmin']))
        act_max = int(self.desc.get('act_max', kernel_info['qmax']))

        act_min = max(kernel_info['qmin'], min(kernel_info['qmax'], act_min))
        act_max = max(kernel_info['qmin'], min(kernel_info['qmax'], act_max))
        if act_min > act_max:
            raise ValueError(f"Clamp act_min ({act_min}) must be <= act_max ({act_max})")

        builder = TemplateContextBuilder()
        input_dims = builder.nhwc_to_cmsis_dims(input_shape)
        output_dims = builder.nhwc_to_cmsis_dims(output_shape)

        rng_state = self.rng.__getstate__()
        self.rng = np.random.default_rng(self.seed)

        if kernel_info["input_c_type"] == "int8_t":
            np_in_dtype = np.int8
            qmin, qmax = -128, 127
        else:
            np_in_dtype = np.int16
            qmin, qmax = -32768, 32767

        input_q = self.rng.integers(qmin, qmax + 1, size=input_shape, dtype=np_in_dtype)
        self.rng.__setstate__(rng_state)

        expected_output = np.clip(input_q, act_min, act_max).astype(np_in_dtype)

        input_array_str = builder.format_array_as_c_literal(input_q)
        expected_output_array_str = builder.format_array_as_c_literal(expected_output)
        output_size = int(np.prod(output_shape))

        context = {
            'name': name,
            'prefix': name,
            'input_dims': input_dims,
            'output_dims': output_dims,
            'act_min': int(act_min),
            'act_max': int(act_max),
            'output_size': int(output_size),
            'input_data_array': input_array_str,
            'expected_output_array': expected_output_array_str,
            'input_dtype': kernel_info["input_c_type"],
            'output_dtype': kernel_info["output_c_type"],
            'kernel_fn': kernel_info["kernel_fn"],
        }

        includes_api_dir = output_dir / "includes"
        includes_api_dir.mkdir(parents=True, exist_ok=True)

        h_content = self.render_template("ActivationFunctions/clamp/clamp.h.j2", context)
        (includes_api_dir / f"{name}_clamp.h").write_text(h_content)
        c_content = self.render_template("ActivationFunctions/clamp/clamp.c.j2", context)
        (output_dir / f"{name}_clamp.c").write_text(c_content)

        cmake_context = {
            'name': name,
            'operator': self.desc.get('operator', 'Clamp'),
            'operator_name': 'clamp',
        }
        cmake_content = self.render_template("common/CMakeLists.txt.j2", cmake_context)
        (output_dir / "CMakeLists.txt").write_text(cmake_content)
