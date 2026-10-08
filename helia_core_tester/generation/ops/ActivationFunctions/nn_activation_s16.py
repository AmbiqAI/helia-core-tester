"""
NN activation S16 operation implementation (arm_nn_activation_s16).
"""

from typing import Dict, Any, List
import numpy as np
from pathlib import Path
from helia_core_tester.generation.ops._shared.base import OperationBase
from helia_core_tester.generation.harness.simple import tensor_case_pool
from helia_core_tester.generation.utils.temp_sizer_probe import require_cmsis_nn_root


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
        # The same checkout the firmware compiles (CMSIS_NN_ROOT / --cmsis-nn-root);
        # the old parents[N] guess pointed one level short of the nested layout.
        table_path = require_cmsis_nn_root("sigmoid_table_uint16 from arm_nntables.c") / "Source" / "NNSupportFunctions" / "arm_nntables.c"
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
        # .flatten() returns a copy; use .reshape(-1) to get a view so writes
        # to flat_out propagate back into `out` instead of being discarded.
        flat_out = out.reshape(-1)

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

        self.render_harness_case(
            output_dir, stem="nn_activation", context=context, pool=tensor_case_pool(context, {"size": context["output_size"], "left_shift": context["left_shift"], "type": context["activation_type"]}),
            validation_key="ActivationFunctions/nn_activation/nn_activation.c.j2", label="NN activation", operator="NNActivationS16",
        )
