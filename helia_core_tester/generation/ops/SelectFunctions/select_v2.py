"""SelectV2 operation implementation."""
from pathlib import Path

from typing import Dict
import numpy as np
from pathlib import Path as _Path
from helia_core_tester.generation.ops._shared.base import OperationBase


class OpSelectV2(OperationBase):
    """SelectV2 operation."""

    def needs_tflite(self) -> bool:
        # The golden is computed in numpy; nothing reads a .tflite.
        return False

    def _select_kernel(self) -> Dict[str, str]:
        activation_dtype = self.desc.get("activation_dtype", "S8")
        if activation_dtype == "S16":
            return {"kernel_fn": "arm_select_v2_s16", "c_type": "int16_t", "np_dtype": "int16", "qmin": -32768, "qmax": 32767}
        return {"kernel_fn": "arm_select_v2_s8", "c_type": "int8_t", "np_dtype": "int8", "qmin": -128, "qmax": 127}

    def _compute_broadcast_strides(self, tensor_shape, output_shape):
        """Compute broadcast strides for a tensor relative to output shape."""
        rank = len(output_shape)
        # Pad tensor_shape to match rank
        padded = [1] * (rank - len(tensor_shape)) + list(tensor_shape)
        strides = [0] * rank
        stride = 1
        for i in range(rank - 1, -1, -1):
            if padded[i] == output_shape[i]:
                strides[i] = stride
            elif padded[i] == 1:
                strides[i] = 0  # broadcast dimension
            else:
                raise ValueError(f"Shapes not broadcastable: {tensor_shape} vs {output_shape}")
            stride *= padded[i]
        return strides

    def generate_c_files(self, output_dir: _Path) -> None:
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

        name = self.desc["name"]
        ki = self._select_kernel()
        output_shape = list(self.desc["input_shape"])
        condition_shape = list(self.desc.get("condition_shape", output_shape))
        x_shape = list(self.desc.get("x_shape", output_shape))
        y_shape = list(self.desc.get("y_shape", output_shape))
        rank = len(output_shape)

        rng = self._seeded_rng()
        np_dtype = np.int16 if ki["np_dtype"] == "int16" else np.int8

        condition = rng.integers(0, 2, size=condition_shape, dtype=np.bool_)
        x_data = rng.integers(ki["qmin"], ki["qmax"] + 1, size=x_shape, dtype=np_dtype)
        y_data = rng.integers(ki["qmin"], ki["qmax"] + 1, size=y_shape, dtype=np_dtype)

        output_data = np.where(condition, x_data, y_data)
        if list(output_data.shape) != output_shape:
            raise ValueError(f"{name}: operands broadcast to {list(output_data.shape)}, expected {output_shape}")

        cond_strides = self._compute_broadcast_strides(condition_shape, output_shape)
        x_strides = self._compute_broadcast_strides(x_shape, output_shape)
        y_strides = self._compute_broadcast_strides(y_shape, output_shape)

        builder = TemplateContextBuilder()
        context = {
            "name": name,
            "rank": rank,
            "output_shape": output_shape,
            "condition_shape": condition_shape,
            "x_shape": x_shape,
            "y_shape": y_shape,
            "cond_strides": cond_strides,
            "x_strides": x_strides,
            "y_strides": y_strides,
            "condition_size": int(np.prod(condition_shape)),
            "x_size": int(np.prod(x_shape)),
            "y_size": int(np.prod(y_shape)),
            "output_size": int(np.prod(output_shape)),
            "condition_array": builder.format_array_as_c_literal(condition),
            "x_data_array": builder.format_array_as_c_literal(x_data),
            "y_data_array": builder.format_array_as_c_literal(y_data),
            "expected_output_array": builder.format_array_as_c_literal(output_data),
            "c_type": ki["c_type"],
            "condition_c_type": "bool",
            "kernel_fn": ki["kernel_fn"],
        }

        self.render_harness_case(
            Path(output_dir), stem="select_v2", context=context, pool=select_v2_argument_pool(context),
            validation_key="SelectFunctions/select_v2/select_v2.c.j2", label="SelectV2", operator="SelectV2",
        )


from helia_core_tester.generation.harness.simple import shaped_case_pool  # noqa: E402


def select_v2_argument_pool(context):
    n = context["name"]
    return shaped_case_pool(
        context, shapes=(("output_shape", "output_shape"), ("cond_strides", "cond_strides"), ("x_strides", "x_strides"),
                         ("y_strides", "y_strides")),
        params_type="cmsis_nn_select_v2_params",
        params={"rank": context["rank"], "output_shape": f"{n}_output_shape", "cond_strides": f"{n}_cond_strides",
                "x_strides": f"{n}_x_strides", "y_strides": f"{n}_y_strides"},
        inputs=(("condition", "condition", "condition_array"), ("x", "x", "x_data_array"), ("y", "y", "y_data_array")),
        condition_ctype=context["condition_c_type"], output_count=str(context["output_size"]))
