"""BroadcastTo operation implementation."""

from typing import Dict
import numpy as np
from pathlib import Path
from helia_core_tester.generation.ops._shared.base import OperationBase


class OpBroadcastTo(OperationBase):
    """BroadcastTo operation."""

    _SUCCESS = "ARM_CMSIS_NN_SUCCESS"
    _ARG_ERROR = "ARM_CMSIS_NN_ARG_ERROR"
    _ARG_ERROR_CASES = {"input", "params", "output"}

    def needs_keras_model(self) -> bool:
        return False

    def allow_no_tflite(self) -> bool:
        return self._expected_status() != self._SUCCESS

    def build_keras_model(self):
        raise NotImplementedError("BroadcastTo uses LiteRT-only model generation.")

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

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        if self._expected_status() != self._SUCCESS:
            raise RuntimeError("BroadcastTo expected error; skip LiteRT generation.")

        from helia_core_tester.generation.utils.litert_builder import (
            build_shape_transform_op, TensorSpec,
        )
        import ai_edge_litert.schema_py_generated as litert

        activation_dtype = self.desc.get('activation_dtype', 'S8')
        dtype = 'int16' if activation_dtype == 'S16' else 'int8'

        input_shape = tuple(self.desc['input_shape'])
        output_shape = tuple(self.desc['output_shape'])

        shape_tensor = TensorSpec(
            name="shape",
            shape=(len(output_shape),),
            tensor_type=litert.TensorType.INT32,
            is_input=False,
            data=np.array(output_shape, dtype=np.int32),
        )

        model_bytes = build_shape_transform_op(
            op_name="BROADCAST_TO",
            input_shape=input_shape,
            output_shape=output_shape,
            dtype=dtype,
            extra_input_tensors=[shape_tensor],
        )
        self._write_tflite_bytes(out_path, model_bytes)

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
            # Use TFLite TILE op (INT32) as interpreter proxy for BROADCAST_TO.
            # BROADCAST_TO is not registered in this runtime; TILE with multiples achieves the same result.
            from ai_edge_litert.interpreter import Interpreter
            from helia_core_tester.generation.utils.litert_builder import LiteRtSingleOpBuilder, TensorSpec
            import ai_edge_litert.schema_py_generated as litert

            multiples = [output_shape[i] // input_shape[i] for i in range(data_rank)]
            builder = LiteRtSingleOpBuilder(op_name="TILE")
            inp_idx = builder.add_tensor(TensorSpec(
                name="input", shape=tuple(input_shape), tensor_type=litert.TensorType.INT32, is_input=True,
            ))
            mult_idx = builder.add_tensor(TensorSpec(
                name="multiples", shape=(data_rank,), tensor_type=litert.TensorType.INT32,
                is_input=False, data=np.array(multiples, dtype=np.int32),
            ))
            out_idx = builder.add_tensor(TensorSpec(
                name="output", shape=tuple(output_shape), tensor_type=litert.TensorType.INT32, is_output=True,
            ))
            builder.add_operator("TILE", inputs=[inp_idx, mult_idx], outputs=[out_idx],
                options=None, options_type=litert.BuiltinOptions.NONE)
            interp = Interpreter(model_content=bytes(builder.build()))
            interp.allocate_tensors()
            inp_details = interp.get_input_details()
            out_details = interp.get_output_details()
            interp.set_tensor(inp_details[0]["index"], input_data.astype(np.int32))
            interp.invoke()
            output_data = interp.get_tensor(out_details[0]["index"]).astype(np_dtype)
        else:
            output_data = np.zeros(output_shape, dtype=np_dtype)

        builder = TemplateContextBuilder()
        context = {
            'name': name,
            'prefix': name,
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

        includes_dir = output_dir / "includes"
        includes_dir.mkdir(parents=True, exist_ok=True)

        h_content = self.render_template("BroadcastFunctions/broadcast_to/broadcast_to.h.j2", context)
        (includes_dir / f"{name}_broadcast_to.h").write_text(h_content)

        c_content = self.render_template("BroadcastFunctions/broadcast_to/broadcast_to.c.j2", context)
        (output_dir / f"{name}_broadcast_to.c").write_text(c_content)

        cmake_content = self.render_template("common/CMakeLists.txt.j2", {
            'name': name, 'operator': 'BroadcastTo', 'operator_name': 'broadcast_to'
        })
        (output_dir / "CMakeLists.txt").write_text(cmake_content)
