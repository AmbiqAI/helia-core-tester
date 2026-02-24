"""
NN activation S16 operation implementation (arm_nn_activation_s16).
"""

from typing import Dict, Any, List
import numpy as np
from pathlib import Path
from helia_core_tester.generation.ops.base import OperationBase


class OpNNActivationS16(OperationBase):
    """
    NN activation (sigmoid/tanh) for int16.
    """

    def build_keras_model(self):
        raise NotImplementedError("NNActivationS16 does not use a Keras model.")

    def needs_keras_model(self) -> bool:
        return False

    def allow_no_tflite(self) -> bool:
        return True

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        raise NotImplementedError("NNActivationS16 does not generate TFLite models.")

    def _load_sigmoid_table(self) -> List[int]:
        table_path = Path(__file__).resolve().parents[5] / "Source" / "NNSupportFunctions" / "arm_nntables.c"
        text = table_path.read_text()
        start = text.find("const uint16_t sigmoid_table_uint16[256] = {")
        if start == -1:
            raise RuntimeError("sigmoid_table_uint16 not found in arm_nntables.c")
        start = text.find("{", start) + 1
        end = text.find("};", start)
        nums = text[start:end].replace("\n", " ").split(",")
        values = [int(n.strip()) for n in nums if n.strip()]
        if len(values) != 256:
            raise RuntimeError(f"Expected 256 sigmoid table entries, got {len(values)}")
        return values

    def _simulate_activation(self, input_data: np.ndarray, left_shift: int, act_type: str) -> np.ndarray:
        table = self._load_sigmoid_table()
        act_type = act_type.upper()

        if act_type == "SIGMOID":
            abs_input_shift = 9
            max_saturation = 0x7FFF << 10
        else:
            abs_input_shift = 8
            max_saturation = 0xFFFF << 8

        input_multiplier = 3 if left_shift < 0 else (3 << left_shift)
        abs_left_shift = -left_shift if left_shift < 0 else 0
        rounding = (1 << (abs_left_shift - 1)) if abs_left_shift > 0 else 0

        out = np.empty_like(input_data, dtype=np.int16)
        flat_in = input_data.flatten()
        flat_out = out.flatten()

        for i, val in enumerate(flat_in):
            input_data_i = (int(val) * input_multiplier + rounding) >> abs_left_shift
            abs_input_data = input_data_i if input_data_i >= 0 else -input_data_i
            uh = abs_input_data >> abs_input_shift

            if uh >= 255:
                result = max_saturation
            else:
                ua = table[uh]
                ub = table[uh + 1]
                if act_type == "SIGMOID":
                    ut = abs_input_data & 0x1FF
                else:
                    ut = abs_input_data & 0x0FF
                result = (ua << abs_input_shift) + ut * (ub - ua)

            if act_type == "SIGMOID":
                if input_data_i >= 0:
                    result = result + (1 << 9)
                else:
                    result = (1 << 25) - result + (1 << 9) - 1
                result >>= 10
            else:
                if input_data_i >= 0:
                    result = (result - (1 << 23)) + (1 << 7)
                else:
                    result = ((-result + (1 << 23)) + (1 << 7) - 1)
                result >>= 8

            flat_out[i] = np.int16(result)

        return out

    def generate_c_files(self, output_dir: Path) -> None:
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

        name = self.desc['name']
        input_shape = tuple(self.desc['input_shape'])
        output_shape = input_shape
        left_shift = int(self.desc.get('left_shift', 0))
        act_type = str(self.desc.get('activation_type', 'TANH')).upper()
        if act_type not in ("SIGMOID", "TANH"):
            raise ValueError(f"Unsupported activation_type: {act_type}")

        builder = TemplateContextBuilder()
        input_dims = builder.nhwc_to_cmsis_dims(input_shape)
        output_dims = builder.nhwc_to_cmsis_dims(output_shape)

        rng_state = self.rng.__getstate__()
        self.rng = np.random.default_rng(self.seed)
        input_q = self.rng.integers(-32768, 32768, size=input_shape, dtype=np.int16)
        self.rng.__setstate__(rng_state)

        expected_output = self._simulate_activation(input_q, left_shift, act_type)

        input_array_str = builder.format_array_as_c_literal(input_q)
        expected_output_array_str = builder.format_array_as_c_literal(expected_output)
        output_size = int(np.prod(output_shape))

        context = {
            'name': name,
            'prefix': name,
            'input_dims': input_dims,
            'output_dims': output_dims,
            'left_shift': int(left_shift),
            'activation_type': f"ARM_{act_type}",
            'output_size': int(output_size),
            'input_data_array': input_array_str,
            'expected_output_array': expected_output_array_str,
            'input_dtype': 'int16_t',
            'output_dtype': 'int16_t',
            'kernel_fn': 'arm_nn_activation_s16',
        }

        includes_api_dir = output_dir / "includes"
        includes_api_dir.mkdir(parents=True, exist_ok=True)

        h_content = self.render_template("nn_activation/nn_activation.h.j2", context)
        (includes_api_dir / f"{name}_nn_activation.h").write_text(h_content)
        c_content = self.render_template("nn_activation/nn_activation.c.j2", context)
        (output_dir / f"{name}_nn_activation.c").write_text(c_content)

        cmake_context = {
            'name': name,
            'operator': self.desc.get('operator', 'NNActivationS16'),
            'operator_name': 'nn_activation',
        }
        cmake_content = self.render_template("common/CMakeLists.txt.j2", cmake_context)
        (output_dir / "CMakeLists.txt").write_text(cmake_content)
