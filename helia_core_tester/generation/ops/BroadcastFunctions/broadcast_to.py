"""BroadcastTo operation implementation."""

from typing import Dict
import numpy as np
from pathlib import Path
from helia_core_tester.generation.ops._shared.base import OperationBase
from helia_core_tester.generation.harness import ArgumentPool, ArrayLiteral, Declaration, FaultEdit, HarnessInput
from helia_core_tester.generation.harness.faults import with_fault


def broadcast_to_argument_pool(context):
    """BroadcastTo reads its shapes from a params struct; an argument-error case hands the kernel NULL
    for the input, the params or the output."""
    n, ctype = context["name"], context["c_type"]
    shapes = lambda key: "{ " + ", ".join(str(int(v)) for v in context[key]) + " }"
    header = [
        Declaration(f"{n}_input_shape", "int32_t", shapes("input_shape"), array=True),
        Declaration(f"{n}_output_shape", "int32_t", shapes("output_shape"), array=True),
        Declaration(f"{n}_params", "cmsis_nn_broadcast_to_params",
                    {"rank": str(context["rank"]), "input_shape": f"{n}_input_shape",
                     "output_shape": f"{n}_output_shape"}),
        Declaration(f"{n}_input", ctype, ArrayLiteral(context["input_data_array"]), array=True),
        Declaration(f"{n}_expected_output", ctype, ArrayLiteral(context["expected_output_array"]), array=True),
    ]
    pool = ArgumentPool(
        name=n, values={"params": context["params_arg"]}, header=header, benchmark=False, scratch_buffer=False,
        output_count=str(context["output_size"]), output_ctype=ctype,
        inputs=(HarnessInput("input_data", "input", f"{n}_input", ctype),),
    )
    nulls = {}
    if context["input_arg"] == "NULL":
        nulls["input_data"] = "NULL"
    if context["output_arg"] == "NULL":
        nulls["output_data"] = "NULL"
    return with_fault(pool, FaultEdit(kind="arg_error", values=nulls)) if nulls else pool


class OpBroadcastTo(OperationBase):
    """BroadcastTo operation."""

    _SUCCESS = "ARM_CMSIS_NN_SUCCESS"
    _ARG_ERROR = "ARM_CMSIS_NN_ARG_ERROR"
    _ARG_ERROR_CASES = {"input", "params", "output"}

    def _expected_status(self) -> str:
        expected_status = str(self.desc.get("expected_status", self._SUCCESS))
        if expected_status not in {self._SUCCESS, self._ARG_ERROR}:
            raise ValueError(f"Unsupported BroadcastTo expected_status: {expected_status}")
        return expected_status

    def _extras(self) -> dict:
        hint = self.desc.get("hint", {})
        extras = hint.get("extras", {}) if isinstance(hint, dict) else {}
        return extras if isinstance(extras, dict) else {}

    def _arg_error_case(self) -> str | None:
        case = self._extras().get("arg_error_case")
        if case is None:
            return None
        case = str(case)
        if case not in self._ARG_ERROR_CASES:
            raise ValueError(f"Unsupported BroadcastTo arg_error_case: {case}")
        return case

    def _params_rank(self, default_rank: int) -> int:
        return int(self._extras().get("params_rank", default_rank))

    def _select_kernel(self) -> Dict[str, str]:
        activation_dtype = self.desc.get('activation_dtype', 'S8')
        if activation_dtype == 'S16':
            return {'kernel_fn': 'arm_broadcast_to_s16', 'c_type': 'int16_t', 'np_dtype': 'int16', 'qmin': -32768, 'qmax': 32767}
        return {'kernel_fn': 'arm_broadcast_to_s8', 'c_type': 'int8_t', 'np_dtype': 'int8', 'qmin': -128, 'qmax': 127}

    def generate_c_files(self, output_dir: Path) -> None:
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

        name = self.desc['name']
        ki = self._select_kernel()
        input_shape = list(self.desc['input_shape'])
        output_shape = list(self.desc['output_shape'])
        data_rank = len(output_shape)
        expected_status = self._expected_status()
        params_rank = self._params_rank(data_rank)
        arg_error_case = self._arg_error_case()

        rng = self._seeded_rng()
        np_dtype = np.int16 if ki['np_dtype'] == 'int16' else np.int8
        input_data = rng.integers(ki['qmin'], ki['qmax'] + 1, size=input_shape, dtype=np_dtype)

        if expected_status == self._SUCCESS:
            try:
                output_data = np.broadcast_to(input_data, tuple(output_shape)).astype(np_dtype)
            except ValueError as exc:
                raise ValueError(f"{name}: {input_shape} does not broadcast to {output_shape}") from exc
        else:
            output_data = np.zeros(output_shape, dtype=np_dtype)

        builder = TemplateContextBuilder()
        context = {
            'name': name,
            'rank': params_rank,
            'input_shape': input_shape,
            'output_shape': output_shape,
            'input_size': int(np.prod(input_shape)),
            'output_size': int(np.prod(output_shape)),
            'input_data_array': builder.format_array_as_c_literal(input_data),
            'expected_output_array': builder.format_array_as_c_literal(output_data),
            'c_type': ki['c_type'],
            'kernel_fn': ki['kernel_fn'],
            'expected_status': expected_status,
            'input_arg': 'NULL' if arg_error_case == 'input' else f'{name}_input',
            'params_arg': 'NULL' if arg_error_case == 'params' else f'&{name}_params',
            'output_arg': 'NULL' if arg_error_case == 'output' else f'{name}_output',
        }

        self.render_harness_case(
            output_dir, stem="broadcast_to", context=context, pool=broadcast_to_argument_pool(context),
            validation_key="BroadcastFunctions/broadcast_to/broadcast_to.c.j2", label="BroadcastTo", operator="BroadcastTo",
        )
