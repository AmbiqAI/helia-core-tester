"""MirrorPad operation implementation."""
from pathlib import Path

from typing import Dict
import numpy as np
from pathlib import Path as _Path
from helia_core_tester.generation.ops._shared.base import OperationBase


class OpMirrorPad(OperationBase):
    """MirrorPad operation."""

    def _select_kernel(self) -> Dict[str, str]:
        activation_dtype = self.desc.get("activation_dtype", "S8")
        if activation_dtype == "S16":
            return {"kernel_fn": "arm_mirror_pad_s16", "c_type": "int16_t", "np_dtype": "int16", "qmin": -32768, "qmax": 32767}
        return {"kernel_fn": "arm_mirror_pad_s8", "c_type": "int8_t", "np_dtype": "int8", "qmin": -128, "qmax": 127}

    def generate_c_files(self, output_dir: _Path) -> None:
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

        name = self.desc["name"]
        ki = self._select_kernel()
        input_shape = list(self.desc["input_shape"])
        paddings = self.desc["paddings"]
        mode_str = self.desc.get("mode", "reflect")
        mode_int = 0 if mode_str == "reflect" else 1

        pad_before = [p[0] for p in paddings]
        output_shape = [input_shape[i] + paddings[i][0] + paddings[i][1] for i in range(len(input_shape))]
        rank = len(input_shape)

        rng = self._seeded_rng()
        np_dtype = np.int16 if ki["np_dtype"] == "int16" else np.int8
        input_data = rng.integers(ki["qmin"], ki["qmax"] + 1, size=input_shape, dtype=np_dtype)

        # TFLite REFLECT excludes the edge element (numpy "reflect"); SYMMETRIC repeats it.
        output_data = np.pad(input_data, paddings, mode="reflect" if mode_str == "reflect" else "symmetric")

        builder = TemplateContextBuilder()
        context = {
            "name": name,
            "rank": rank,
            "input_shape": input_shape,
            "output_shape": output_shape,
            "pad_before": pad_before,
            "mode": mode_int,
            "input_size": int(np.prod(input_shape)),
            "output_size": int(np.prod(output_shape)),
            "input_data_array": builder.format_array_as_c_literal(input_data),
            "expected_output_array": builder.format_array_as_c_literal(output_data),
            "c_type": ki["c_type"],
            "kernel_fn": ki["kernel_fn"],
        }

        self.render_harness_case(
            Path(output_dir), stem="mirror_pad", context=context, pool=mirror_pad_argument_pool(context),
            validation_key="PadFunctions/mirror_pad/mirror_pad.c.j2", label="MirrorPad", operator="MirrorPad",
        )


from helia_core_tester.generation.harness.simple import shaped_case_pool  # noqa: E402


def mirror_pad_argument_pool(context):
    n = context["name"]
    return shaped_case_pool(
        context, shapes=(("input_shape", "input_shape"), ("output_shape", "output_shape"), ("pad_before", "pad_before")),
        params_type="cmsis_nn_mirror_pad_params",
        params={"rank": context["rank"], "input_shape": f"{n}_input_shape", "output_shape": f"{n}_output_shape",
                "pad_before": f"{n}_pad_before", "mode": context["mode"]},
        inputs=(("input", "input", "input_data_array"),), output_count=str(context["output_size"]))
