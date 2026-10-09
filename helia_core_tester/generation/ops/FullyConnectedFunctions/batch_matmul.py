"""BatchMatMul operation implementation."""

from typing import Dict, Any
import numpy as np
from pathlib import Path
from helia_core_tester.generation.ops._shared.base import OperationBase
from helia_core_tester.generation.harness import ArgumentPool, ArrayLiteral, Declaration, HarnessInput
from helia_core_tester.generation.harness.faults import common_fault, struct_copy, with_fault

BMM_VALIDATION_KEY = "FullyConnectedFunctions/batch_matmul/batch_matmul.c.j2"


def bmm_argument_pool(context: Dict[str, Any]) -> ArgumentPool:
    """Every value a BatchMatMul case can pass to a public batch-matmul kernel or its scratch query."""
    n = context["name"]
    float_kernel = bool(context.get("float_kernel"))
    bmm = context["bmm_params"]

    def dims(d: Dict[str, Any]) -> Dict[str, Any]:
        return {"n": d["n"], "h": d["h"], "w": d["w"], "c": d["c"]}

    params_init: Dict[str, Any] = {"adj_x": str(bmm["adj_x"]).lower(), "adj_y": str(bmm["adj_y"]).lower()}
    if float_kernel:
        params_init["activation"] = {"min": context["bmm_activation_min_literal"],
                                     "max": context["bmm_activation_max_literal"]}
        params_init["rhs_format"] = "ARM_NN_WEIGHT_FORMAT_STANDARD"
    else:
        fc = bmm["fc_params"]
        params_init["fc_params"] = {"input_offset": fc["input_offset"], "filter_offset": fc["filter_offset"],
                                    "output_offset": fc["output_offset"],
                                    "activation": {"min": fc["activation_min"], "max": fc["activation_max"]}}
    header = [
        Declaration(f"{n}_input_lhs_dims", "cmsis_nn_dims", dims(context["input_lhs_dims"]), comment="Input LHS dimensions"),
        Declaration(f"{n}_input_rhs_dims", "cmsis_nn_dims", dims(context["input_rhs_dims"]), comment="Input RHS dimensions"),
        Declaration(f"{n}_output_dims", "cmsis_nn_dims", dims(context["output_dims"]), comment="Output dimensions"),
        Declaration(f"{n}_bmm_params", context.get("bmm_params_type") or "cmsis_nn_bmm_params", params_init,
                    comment="Batch matmul parameters"),
    ]
    values = {"ctx": f"&{n}_ctx", "bmm_params": f"&{n}_bmm_params", "input_lhs_dims": f"&{n}_input_lhs_dims",
              "input_rhs_dims": f"&{n}_input_rhs_dims", "output_dims": f"&{n}_output_dims"}
    source = []
    if not float_kernel:
        quant = context["quant_params"]
        header.append(Declaration(f"{n}_quant_params", "cmsis_nn_per_tensor_quant_params",
                                  {"multiplier": str(quant["multiplier"]), "shift": str(quant["shift"])},
                                  comment="Quantization parameters (per-tensor)"))
        values["quant_params"] = f"&{n}_quant_params"
        rhs = context["input_rhs_dims"]
        # The int sizer is the fully-connected one: RHS [batch, K, N] read as filter dims n=K, c=N.
        source.append(Declaration(f"{n}_filter_dims_for_buffer", "cmsis_nn_dims",
                                  {"n": rhs["c"], "h": 1, "w": 1, "c": rhs["w"]},
                                  comment="RHS dimensions as the scratch query reads them"))
        values["filter_dims"] = f"&{n}_filter_dims_for_buffer"
    header += [
        Declaration(f"{n}_input_lhs", context["input_dtype"], ArrayLiteral(context["input_lhs_array"]), array=True,
                    comment="Input LHS data (for testing)"),
        Declaration(f"{n}_input_rhs", context["input_dtype"], ArrayLiteral(context["input_rhs_array"]), array=True,
                    comment="Input RHS data (for testing)"),
        Declaration(f"{n}_expected_output", context["output_dtype"], ArrayLiteral(context["expected_output_array"]),
                    array=True, comment="Expected output (golden)"),
    ]
    output = context["output_dims"]
    return ArgumentPool(
        name=n, values=values, header=header, source=source, output_param="output",
        inputs=(HarnessInput("input_lhs", "input_lhs", f"{n}_input_lhs"),
                HarnessInput("input_rhs", "input_rhs", f"{n}_input_rhs")),
        output_count=f"({output['n']} * {output['h']} * {output['w']} * {output['c']})", benchmark=False,
    )


def bmm_fault(pool: ArgumentPool, kind: str, context: Dict[str, Any]) -> ArgumentPool:
    """The pool of a BatchMatMul fault case: the passing pool with the faulted argument edited."""
    n = context["name"]
    edit = common_fault(pool, kind)
    if edit is None and kind == "negative_dim":
        edit = struct_copy(pool, kind, "input_rhs_dims", "cmsis_nn_dims", f"{n}_input_rhs_dims", {"w": -1})
    elif edit is None and kind == "packed_rhs_adjoint":
        edit = struct_copy(pool, kind, "bmm_params", context.get("bmm_params_type") or "cmsis_nn_bmm_params",
                           f"{n}_bmm_params", {"rhs_format": "ARM_NN_WEIGHT_FORMAT_NT_N_PACKED"})
    if edit is None:
        raise ValueError(f"{n}: no BatchMatMul fault edit for {kind!r}")
    return with_fault(pool, edit)


class OpBatchMatMul(OperationBase):
    """BatchMatMul operation."""

    FAULT_KINDS = (
        "null_ctx_buf",
        "small_ctx_size",
        "negative_dim",
        "null_input",
        "null_output",
        "packed_rhs_adjoint",
    )

    def _render_batch_matmul(self, output_dir: Path, context: Dict[str, Any]) -> None:
        pool = bmm_argument_pool(context)
        fault = self.fault_kind()
        if fault:
            self._check_fault_reachable(fault, context)
            context.update(self.fault_context())
            pool = bmm_fault(pool, fault, context)
        self.render_harness_files(output_dir, stem="batch_matmul", context=context, pool=pool,
                                  validation_key=BMM_VALIDATION_KEY, label="Batch matmul")

    def _check_fault_reachable(self, kind: str, context: Dict[str, Any]) -> None:
        """Reject fault kinds the selected batch-matmul kernel does not diagnose."""
        kernel_fn = context["kernel_fn"]
        if context.get("float_kernel"):
            if kind not in ("null_input", "null_output", "packed_rhs_adjoint"):
                raise self.fault_unreachable(kind, f"{kernel_fn} has no such guard")
            params = context["bmm_params"]
            if kind == "packed_rhs_adjoint" and not (params["adj_x"] or params["adj_y"]):
                raise self.fault_unreachable(
                    kind, f"{kernel_fn} only rejects a packed RHS when adj_x or adj_y is set"
                )
            return
        if kind not in ("null_ctx_buf", "small_ctx_size", "negative_dim"):
            raise self.fault_unreachable(kind, f"{kernel_fn} does not check {kind}")
        if kernel_fn != "arm_batch_matmul_s8":
            raise self.fault_unreachable(kind, f"{kernel_fn} validates no arguments")
        if "mve" not in self.required_capabilities():
            raise self.fault_unreachable(
                kind, f"{kernel_fn} only validates arguments under ARM_MATH_MVEI; add required_capabilities: [mve]"
            )

    def _int_kind(self):
        return {"S8": "s8", "S16": "s16"}.get(
            str(self.tensor_dtype("input", default=self.desc.get("activation_dtype", "S8"))).upper())

    def uses_reference(self) -> bool:
        return True

    def _operand_shapes(self, max_rank: int = 5):
        name = self.desc['name']
        lhs_shape = tuple(int(d) for d in self.desc["input_1_shape"])
        rhs_shape = tuple(int(d) for d in self.desc["input_2_shape"])
        if not (3 <= len(lhs_shape) <= max_rank and 3 <= len(rhs_shape) <= max_rank):
            raise ValueError(f"{name}: BatchMatMul cases are rank 3 to {max_rank}, got {lhs_shape} x {rhs_shape}")
        if min(lhs_shape + rhs_shape) < 1:
            raise ValueError(f"{name}: BatchMatMul dims must be positive, got {lhs_shape} x {rhs_shape}")
        return lhs_shape, rhs_shape

    def _canonical_operands(self, lhs_data, rhs_data, adj_x: bool, adj_y: bool):
        """LHS [..., M, K], RHS [..., N, K] and the [..., M, N] output shape, from the descriptor's operands."""
        name = self.desc['name']
        lhs_c = np.swapaxes(lhs_data, -1, -2) if adj_x else lhs_data
        rhs_c = rhs_data if adj_y else np.swapaxes(rhs_data, -1, -2)
        if lhs_c.shape[-1] != rhs_c.shape[-1]:
            raise ValueError(f"{name}: depth {lhs_c.shape[-1]} vs {rhs_c.shape[-1]} (adj_x={adj_x}, adj_y={adj_y})")
        try:
            batch = tuple(int(d) for d in np.broadcast_shapes(lhs_c.shape[:-2], rhs_c.shape[:-2]))
        except ValueError as exc:
            raise ValueError(f"{name}: batches {lhs_c.shape[:-2]} and {rhs_c.shape[:-2]} do not broadcast") from exc
        return np.ascontiguousarray(lhs_c), np.ascontiguousarray(rhs_c), batch + (lhs_c.shape[-2], rhs_c.shape[-2])

    def _float_golden(self, lhs_data, rhs_data, adj_x: bool, adj_y: bool, float_dtype):
        """TFLM's f32 BatchMatMul on the canonical operands; f16 operands are rounded to half
        first and the golden cast back once."""
        from helia_core_tester.generation.reference import weighted
        from helia_core_tester.generation.reference.case import ReferenceCall
        from helia_core_tester.generation.reference.run import run_reference

        fmin, fmax = weighted.float_bounds(self.desc)

        def call_for(lhs, rhs):
            lhs_c, rhs_c, output_shape = self._canonical_operands(
                np.asarray(lhs, dtype=np.float32), np.asarray(rhs, dtype=np.float32), adj_x, adj_y)
            return ReferenceCall("bmm_f32", {"act": {"min": 0, "max": 0, "fmin": fmin, "fmax": fmax}},
                                 {"lhs": lhs_c, "rhs": rhs_c}, output_shape, "float32")

        def reference(operands):
            return run_reference(call_for(operands[0], operands[1])).astype(float_dtype)

        call = call_for(lhs_data, rhs_data)
        return self.reference_golden(call).astype(float_dtype), call.output_shape, reference

    def _generate_int_c_files(self, output_dir: Path, kind: str) -> None:
        """Integer BMM: policy quantization over the [-1, 1] draws, the output quantized over
        the float product's range, and the TFLM reference golden on the canonical operands
        (LHS [..., M, K], RHS [..., N, K]) the CMSIS harness also stores."""
        from helia_core_tester.generation.reference import params as ref_params
        from helia_core_tester.generation.reference import policy
        from helia_core_tester.generation.reference.case import ReferenceCall
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

        name = self.desc['name']
        np_dtype = np.int8 if kind == "s8" else np.int16
        kernel_info = self._select_cmsis_matmul_kernel(np_dtype, np_dtype, np_dtype)
        lhs_shape, rhs_shape = self._operand_shapes(max_rank=3)
        adj_x = bool(self.desc.get('adj_x', False))
        adj_y = bool(self.desc.get('adj_y', False))

        rng_state = self.rng.__getstate__()
        self.rng = np.random.default_rng(self.seed)
        lhs_data = self._maybe_apply_input_mode(self.rng.uniform(-1.0, 1.0, size=lhs_shape).astype(np.float32))
        rhs_data = self.rng.uniform(-1.0, 1.0, size=rhs_shape).astype(np.float32)
        self.rng.__setstate__(rng_state)

        lhs_c, rhs_c, output_shape = self._canonical_operands(lhs_data, rhs_data, adj_x, adj_y)

        draw_range = np.array([-1.0, 1.0], dtype=np.float32)
        lhs_quant = self.activation_quant("input_1", draw_range, kind)
        rhs_quant = self.activation_quant("input_2", draw_range, kind)
        float_out = np.matmul(lhs_c.astype(np.float64), np.transpose(rhs_c, (0, 2, 1)).astype(np.float64))
        out_quant = self.activation_quant("output", float_out, kind)
        lhs_q = np.ascontiguousarray(policy.quantize(lhs_c, lhs_quant))
        rhs_q = np.ascontiguousarray(policy.quantize(rhs_c, rhs_quant))

        builder = TemplateContextBuilder()
        as_dict = lambda q: {"scale": [q.scale], "zero_point": [q.zero_point]}  # noqa: E731
        fc_params = builder.build_fc_params(self.desc, as_dict(lhs_quant), as_dict(rhs_quant), as_dict(out_quant))
        multiplier, shift = ref_params.quantize_multiplier(lhs_quant.scale * rhs_quant.scale / out_quant.scale)
        call = ReferenceCall(
            f"bmm_{kind}",
            {"lhs_offset": fc_params["input_offset"], "rhs_offset": fc_params["filter_offset"],
             "output_offset": fc_params["output_offset"], "output_multiplier": multiplier, "output_shift": shift,
             "act": {"min": int(fc_params["activation_min"]), "max": int(fc_params["activation_max"])}},
            {"lhs": lhs_q, "rhs": rhs_q}, output_shape, np.dtype(np_dtype).name,
            quant={"lhs": lhs_quant.to_json(), "rhs": rhs_quant.to_json(), "output": out_quant.to_json()},
        )
        output_data = self.reference_golden(call)

        input_lhs_dims = builder.nhwc_to_cmsis_dims(lhs_q.shape)
        input_rhs_dims = builder.nhwc_to_cmsis_dims(rhs_q.shape)
        output_dims = builder.nhwc_to_cmsis_dims(output_shape)
        # CMSIS-NN's adj_y is the inverse of TFLite's: true means RHS is stored as [N, K].
        bmm_params = {'adj_x': adj_x, 'adj_y': not adj_y, 'fc_params': fc_params}
        filter_dims_for_buffer = {'n': int(rhs_q.shape[2]), 'h': 1, 'w': 1, 'c': int(rhs_q.shape[1])}
        context = {
            'name': name,
            'input_lhs_dims': input_lhs_dims,
            'input_rhs_dims': input_rhs_dims,
            'output_dims': output_dims,
            'bmm_params': bmm_params,
            'quant_params': {'multiplier': int(multiplier), 'shift': int(shift), 'per_channel': False},
            'input_lhs_array': builder.format_array_as_c_literal(lhs_q),
            'input_rhs_array': builder.format_array_as_c_literal(rhs_q),
            'expected_output_array': builder.format_array_as_c_literal(output_data),
            'input_dtype': kernel_info["input_lhs_c_type"],
            'input_rhs_dtype': kernel_info["input_rhs_c_type"],
            'output_dtype': kernel_info["output_c_type"],
            'kernel_fn': kernel_info["kernel_fn"],
            'kernel_get_buffer_size_fn': kernel_info["kernel_get_buffer_size_fn"],
            'buffer_size_max': builder.calculate_fc_buffer_size_max(
                filter_dims_for_buffer, output_dtype=self.desc.get('activation_dtype', 'S8')),
        }
        self._render_batch_matmul(output_dir, context)
        cmake_context = {'name': name, 'operator': self.desc.get('operator', 'BatchMatMul'), 'operator_name': 'batch_matmul'}
        (output_dir / "CMakeLists.txt").write_text(self.render_template("common/CMakeLists.txt.j2", cmake_context))

    def _numpy_dtype_to_c_type(self, np_dtype: np.dtype) -> str:
        """Map numpy dtype to C type string."""
        if np_dtype == np.int8:
            return 'int8_t'
        elif np_dtype == np.int16:
            return 'int16_t'
        elif np_dtype == np.float32:
            return 'float'
        elif np_dtype == np.float16:
            return 'float16_t'
        else:
            raise ValueError(f"Unsupported numpy dtype: {np_dtype}")
    
    def _select_cmsis_matmul_kernel(self, input_lhs_dtype: np.dtype, input_rhs_dtype: np.dtype, output_dtype: np.dtype) -> Dict[str, str]:
        """
        Select appropriate CMSIS-NN kernel function for BatchMatMul based on actual tensor dtypes.
        
        Args:
            input_lhs_dtype: numpy dtype of LHS tensor from TFLite
            input_rhs_dtype: numpy dtype of RHS tensor from TFLite
            output_dtype: numpy dtype of output tensor from TFLite
        
        Returns:
            Dictionary with kernel_fn, kernel_get_buffer_size_fn, 
            input_lhs_c_type, input_rhs_c_type, output_c_type
        """
        input_lhs_c_type = self._numpy_dtype_to_c_type(input_lhs_dtype)
        input_rhs_c_type = self._numpy_dtype_to_c_type(input_rhs_dtype)
        output_c_type = self._numpy_dtype_to_c_type(output_dtype)
        
        # Select kernel based on LHS dtype (output should match LHS)
        if input_lhs_c_type == 'float':
            return {
                'kernel_fn': 'arm_batch_matmul_f32',
                'kernel_get_buffer_size_fn': 'arm_batch_matmul_f32_get_buffer_size',
                'input_lhs_c_type': input_lhs_c_type,
                'input_rhs_c_type': input_rhs_c_type,
                'output_c_type': output_c_type
            }
        if input_lhs_c_type == 'float16_t':
            return {
                'kernel_fn': 'arm_batch_matmul_f16',
                'kernel_get_buffer_size_fn': 'arm_batch_matmul_f16_get_buffer_size',
                'input_lhs_c_type': input_lhs_c_type,
                'input_rhs_c_type': input_rhs_c_type,
                'output_c_type': output_c_type
            }
        if input_lhs_c_type == 'int8_t':
            return {
                'kernel_fn': 'arm_batch_matmul_s8',
                'kernel_get_buffer_size_fn': 'arm_fully_connected_s8_get_buffer_size',
                'input_lhs_c_type': input_lhs_c_type,
                'input_rhs_c_type': input_rhs_c_type,
                'output_c_type': output_c_type
            }
        elif input_lhs_c_type == 'int16_t':
            return {
                'kernel_fn': 'arm_batch_matmul_s16',
                'kernel_get_buffer_size_fn': 'arm_fully_connected_s16_get_buffer_size',
                'input_lhs_c_type': input_lhs_c_type,
                'input_rhs_c_type': input_rhs_c_type,
                'output_c_type': output_c_type
            }
        else:
            raise NotImplementedError(f"Unsupported LHS dtype: {input_lhs_c_type}")
    
    def generate_c_files(self, output_dir) -> None:
        """
        Generate C and H files from templates for BatchMatMul.
        """
        from pathlib import Path
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder
        
        name = self.desc['name']
        if self._int_kind() is not None:
            self._generate_int_c_files(Path(output_dir), self._int_kind())
            return
        execution_dtype = self.tensor_dtype("input", default=self.desc.get("activation_dtype", "S8"))
        input_lhs_dtype = input_rhs_dtype = output_dtype = np.float16 if execution_dtype == "FP16" else np.float32
        input_lhs_shape, input_rhs_shape = self._operand_shapes()

        kernel_info = self._select_cmsis_matmul_kernel(input_lhs_dtype, input_rhs_dtype, output_dtype)
        float_dtype = np.float16 if kernel_info["input_lhs_c_type"] == "float16_t" else np.float32
        builder = TemplateContextBuilder()
        adj_x = bool(self.desc.get('adj_x', False))
        adj_y = bool(self.desc.get('adj_y', False))

        # CMSIS float BMM dims use .c as matrix rows and .w as the inner dimension.
        # The CMSIS adj_y flag is inverted relative to the descriptor flag:
        # adj_y=false consumes RHS as [K, N] with strided column access, while
        # adj_y=true consumes RHS already stored as [N, K].
        if len(input_lhs_shape) == 3 and not adj_x:
            input_lhs_dims = builder.nhwc_to_cmsis_dims((input_lhs_shape[0], input_lhs_shape[2], input_lhs_shape[1]))
        else:
            input_lhs_dims = builder.nhwc_to_cmsis_dims(input_lhs_shape)
        if len(input_rhs_shape) == 3 and adj_y:
            input_rhs_dims = builder.nhwc_to_cmsis_dims((input_rhs_shape[0], input_rhs_shape[2], input_rhs_shape[1]))
        else:
            input_rhs_dims = builder.nhwc_to_cmsis_dims(input_rhs_shape)
        bmm_params = {
            'adj_x': adj_x,
            'adj_y': not adj_y,
            'activation_min': float(self.desc.get("activation_min", -1.0e30)),
            'activation_max': float(self.desc.get("activation_max", 1.0e30)),
        }

        rng_state = self.rng.__getstate__()
        self.rng = np.random.default_rng(self.seed)
        # Only the LHS is swept: the sweep is defined on the input tensor, and taking it
        # through _maybe_apply_input_mode rather than _sample_uniform keeps the draw on
        # self.rng so the RHS that follows it in the stream is unmoved.
        input_lhs_data = self._maybe_apply_input_mode(
            self.rng.uniform(-1.0, 1.0, size=input_lhs_shape).astype(float_dtype))
        input_rhs_data = self.rng.uniform(-1.0, 1.0, size=input_rhs_shape).astype(float_dtype)
        self.rng.__setstate__(rng_state)

        output_data, output_shape, float_reference = self._float_golden(
            input_lhs_data, input_rhs_data, adj_x, adj_y, float_dtype)
        output_dims = builder.nhwc_to_cmsis_dims(output_shape)
        output_data, nonfinite_context = self.apply_nonfinite_policy(
            output_data, reference=float_reference, inputs=[input_lhs_data, input_rhs_data]
        )

        input_lhs_array_str = builder.format_array_as_c_literal(input_lhs_data)
        input_rhs_array_str = builder.format_array_as_c_literal(input_rhs_data)
        expected_output_array_str = builder.format_array_as_c_literal(output_data)
        element_size = np.dtype(float_dtype).itemsize
        buffer_size_max = max(
            1024,
            int(
                (np.prod(input_lhs_shape) + np.prod(input_rhs_shape) + np.prod(output_shape)) * element_size
            ),
        )

        context = {
            'name': name,
            'input_lhs_dims': input_lhs_dims,
            'input_rhs_dims': input_rhs_dims,
            'output_dims': output_dims,
            'bmm_params': bmm_params,
            'input_lhs_array': input_lhs_array_str,
            'input_rhs_array': input_rhs_array_str,
            'expected_output_array': expected_output_array_str,
            'input_dtype': kernel_info["input_lhs_c_type"],
            'input_rhs_dtype': kernel_info["input_rhs_c_type"],
            'output_dtype': kernel_info["output_c_type"],
            'kernel_fn': kernel_info["kernel_fn"],
            'kernel_get_buffer_size_fn': kernel_info["kernel_get_buffer_size_fn"],
            'buffer_size_max': buffer_size_max,
            'float_kernel': True,
            'bmm_params_type': (
                'cmsis_nn_bmm_params_f16'
                if kernel_info["input_lhs_c_type"] == "float16_t"
                else 'cmsis_nn_bmm_params_f32'
            ),
            'bmm_activation_min_literal': builder.format_float_literal(bmm_params['activation_min']),
            'bmm_activation_max_literal': builder.format_float_literal(bmm_params['activation_max']),
            'validation_mode': 'float',
        }
        context.update(nonfinite_context)
        self._render_batch_matmul(output_dir, context)
        
        cmake_context = {
            'name': name,
            'operator': self.desc.get('operator', 'BatchMatMul'),
            'operator_name': 'batch_matmul'
        }
        cmake_content = self.render_template("common/CMakeLists.txt.j2", cmake_context)
        cmake_path = output_dir / "CMakeLists.txt"
        with open(cmake_path, 'w') as f:
            f.write(cmake_content)
        return
