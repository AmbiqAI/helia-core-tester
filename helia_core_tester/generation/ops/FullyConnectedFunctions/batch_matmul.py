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

    def uses_reference(self) -> bool:
        return True

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
        Generate C and H files for BatchMatMul, with the golden from the C reference (TFLite's
        BatchMatMul; float exact then rounded once).
        """
        from helia_core_tester.generation.ops._shared.conv_reference import FLOAT_UNBOUNDED
        from helia_core_tester.generation.reference import policy
        from helia_core_tester.generation.reference.bindings import get_bindings
        from helia_core_tester.generation.reference.call import ReferenceCall
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

        name = self.desc['name']
        input_lhs_shape = tuple(int(d) for d in self.desc['input_1_shape'])
        input_rhs_shape = tuple(int(d) for d in self.desc['input_2_shape'])
        adj_x = bool(self.desc.get('adj_x', False))
        adj_y = bool(self.desc.get('adj_y', False))
        if len(input_lhs_shape) < 2 or len(input_rhs_shape) < 2:
            raise ValueError(f"{name}: BatchMatMul operands need at least two dims")
        rows = input_lhs_shape[-1] if adj_x else input_lhs_shape[-2]
        depth = input_lhs_shape[-2] if adj_x else input_lhs_shape[-1]
        cols = input_rhs_shape[-2] if adj_y else input_rhs_shape[-1]
        if (input_rhs_shape[-1] if adj_y else input_rhs_shape[-2]) != depth:
            raise ValueError(f"{name}: inner dimensions differ for adj_x={adj_x}, adj_y={adj_y}")
        batch = np.broadcast_shapes(input_lhs_shape[:-2], input_rhs_shape[:-2])
        output_shape = tuple(int(d) for d in batch) + (rows, cols)

        execution_dtype = str(self.tensor_dtype("input", default=self.desc.get("activation_dtype", "S8"))).upper()
        np_dtype = {"S8": np.int8, "S16": np.int16, "FP32": np.float32, "FP16": np.float16}.get(execution_dtype)
        if np_dtype is None:
            raise NotImplementedError(f"Unsupported BatchMatMul dtype: {execution_dtype}")
        kernel_info = self._select_cmsis_matmul_kernel(np_dtype, np_dtype, np_dtype)
        float_kernel = kernel_info["input_lhs_c_type"] in {"float", "float16_t"}
        float_dtype = np.float16 if kernel_info["input_lhs_c_type"] == "float16_t" else np.float32

        builder = TemplateContextBuilder()
        # For float BMM, CMSIS dims use .c as matrix rows and .w as the inner dimension.
        # The CMSIS adj_y flag is inverted relative to the descriptor flag: adj_y=false
        # consumes RHS as [K, N] with strided column access, adj_y=true as [N, K].
        if float_kernel:
            if len(input_lhs_shape) == 3 and not adj_x:
                input_lhs_dims = builder.nhwc_to_cmsis_dims((input_lhs_shape[0], input_lhs_shape[2], input_lhs_shape[1]))
            else:
                input_lhs_dims = builder.nhwc_to_cmsis_dims(input_lhs_shape)
            if len(input_rhs_shape) == 3 and adj_y:
                input_rhs_dims = builder.nhwc_to_cmsis_dims((input_rhs_shape[0], input_rhs_shape[2], input_rhs_shape[1]))
            else:
                input_rhs_dims = builder.nhwc_to_cmsis_dims(input_rhs_shape)
        else:
            if len(input_lhs_shape) == 3 and adj_x:
                input_lhs_dims = builder.nhwc_to_cmsis_dims((input_lhs_shape[0], input_lhs_shape[2], input_lhs_shape[1]))
            else:
                input_lhs_dims = builder.nhwc_to_cmsis_dims(input_lhs_shape)
            if len(input_rhs_shape) == 3 and not adj_y:
                input_rhs_dims = builder.nhwc_to_cmsis_dims((input_rhs_shape[0], input_rhs_shape[2], input_rhs_shape[1]))
            else:
                input_rhs_dims = builder.nhwc_to_cmsis_dims(input_rhs_shape)
        output_dims = builder.nhwc_to_cmsis_dims(output_shape)

        rng_state = self.rng.__getstate__()
        self.rng = np.random.default_rng(self.seed)
        # Only the LHS is swept: the sweep is defined on the input tensor, and taking it
        # through _maybe_apply_input_mode rather than _sample_uniform keeps the draw on
        # self.rng so the RHS that follows it in the stream is unmoved.
        input_lhs_data = self._maybe_apply_input_mode(
            self.rng.uniform(-1.0, 1.0, size=input_lhs_shape).astype(float_dtype if float_kernel else np.float32)
        )
        input_rhs_data = self.rng.uniform(-1.0, 1.0, size=input_rhs_shape).astype(float_dtype if float_kernel else np.float32)
        self.rng.__setstate__(rng_state)

        if float_kernel:
            lo = float(np.float32(self.desc.get("activation_min", -FLOAT_UNBOUNDED)))
            hi = float(np.float32(self.desc.get("activation_max", FLOAT_UNBOUNDED)))
            output_data = self.reference_golden(ReferenceCall(
                "batch_matmul_f16" if float_dtype == np.float16 else "batch_matmul_f32",
                {"adj_x": int(adj_x), "adj_y": int(adj_y), "activation_min": lo, "activation_max": hi},
                {"lhs": np.ascontiguousarray(input_lhs_data), "rhs": np.ascontiguousarray(input_rhs_data)},
                {"output": output_shape}))
            output_data, nonfinite_context = self.apply_nonfinite_policy(
                output_data, reference=self.reference_probe, inputs=[input_lhs_data, input_rhs_data]
            )
            bmm_params = {'adj_x': adj_x, 'adj_y': not adj_y, 'activation_min': lo, 'activation_max': hi}
            element_size = np.dtype(float_dtype).itemsize
            buffer_size_max = max(
                1024,
                int((np.prod(input_lhs_shape) + np.prod(input_rhs_shape) + np.prod(output_shape)) * element_size),
            )
            context = {
                'name': name,
                'input_lhs_dims': input_lhs_dims,
                'input_rhs_dims': input_rhs_dims,
                'output_dims': output_dims,
                'bmm_params': bmm_params,
                'input_lhs_array': builder.format_array_as_c_literal(input_lhs_data),
                'input_rhs_array': builder.format_array_as_c_literal(input_rhs_data),
                'expected_output_array': builder.format_array_as_c_literal(output_data),
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
                'bmm_activation_min_literal': builder.format_float_literal(lo),
                'bmm_activation_max_literal': builder.format_float_literal(hi),
                'validation_mode': 'float',
            }
            context.update(nonfinite_context)
            self._render_batch_matmul(output_dir, context)
            self._write_cmake(output_dir)
            return

        kind = "s16" if np_dtype == np.int16 else "s8"
        # Both operands are activations, quantized over the [-1, 1] the converter calibrated on.
        lhs_quant = self.activation_quant("input", (-1.0, 1.0), kind)
        rhs_quant = self.activation_quant("input_2", (-1.0, 1.0), kind)
        input_lhs_q = policy.quantize(input_lhs_data, lhs_quant)
        input_rhs_q = policy.quantize(input_rhs_data, rhs_quant)
        lib = get_bindings()
        real = lib.run("batch_matmul_f32", {"adj_x": int(adj_x), "adj_y": int(adj_y),
                                            "activation_min": float("-inf"), "activation_max": float("inf")},
                       {"lhs": policy.dequantize(input_lhs_q, lhs_quant).astype(np.float32),
                        "rhs": policy.dequantize(input_rhs_q, rhs_quant).astype(np.float32)},
                       {"output": output_shape})["output"]
        out_quant = self.activation_quant("output", policy.data_range(real), kind)
        r = lib.run("per_channel_quant", {"input_scale": lhs_quant.scale, "output_scale": out_quant.scale},
                    {"filter_scale": np.array([rhs_quant.scale], np.float32)}, {"multiplier": (1,), "shift": (1,)})
        multiplier, shift = int(r["multiplier"][0]), int(r["shift"][0])
        qmin, qmax = policy.dtype_range(kind)
        output_data = self.reference_golden(ReferenceCall(
            f"batch_matmul_{kind}",
            {"adj_x": int(adj_x), "adj_y": int(adj_y), "lhs_offset": -lhs_quant.zero_point,
             "rhs_offset": -rhs_quant.zero_point, "output_offset": out_quant.zero_point, "multiplier": multiplier,
             "shift": shift, "activation_min": qmin, "activation_max": qmax},
            {"lhs": np.ascontiguousarray(input_lhs_q), "rhs": np.ascontiguousarray(input_rhs_q)},
            {"output": output_shape},
            quant={"lhs": lhs_quant.to_json(), "rhs": rhs_quant.to_json(), "output": out_quant.to_json()}))

        # LHS is the "input" of the CMSIS FC-style params, RHS the "filter".
        fc_params = {'input_offset': -lhs_quant.zero_point, 'filter_offset': -rhs_quant.zero_point,
                     'output_offset': out_quant.zero_point, 'activation_min': qmin, 'activation_max': qmax}
        bmm_params = {'adj_x': adj_x, 'adj_y': not adj_y, 'fc_params': fc_params}
        quant_params_dict = {'multiplier': multiplier, 'shift': shift, 'per_channel': False}

        # CMSIS-NN takes RHS as [batch, N, K] unless adj_y, and LHS as [batch, K, M] when adj_x.
        input_rhs_q_for_cmsis = input_rhs_q
        if not adj_y and len(input_rhs_shape) == 3:
            input_rhs_q_for_cmsis = np.transpose(input_rhs_q, (0, 2, 1))
        input_lhs_q_for_cmsis = input_lhs_q
        if adj_x and len(input_lhs_shape) == 3:
            input_lhs_q_for_cmsis = np.transpose(input_lhs_q, (0, 2, 1))

        if len(input_rhs_shape) != 3:
            raise ValueError(f"Unsupported RHS shape for buffer size: {input_rhs_shape}")
        filter_dims_for_buffer = {'n': int(input_rhs_shape[1]), 'h': 1, 'w': 1, 'c': int(input_rhs_shape[2])}
        buffer_size_max = builder.calculate_fc_buffer_size_max(
            filter_dims_for_buffer, output_dtype=self.desc.get('activation_dtype', 'S8'))

        context = {
            'name': name,
            'input_lhs_dims': input_lhs_dims,
            'input_rhs_dims': input_rhs_dims,
            'output_dims': output_dims,
            'bmm_params': bmm_params,
            'quant_params': quant_params_dict,
            'input_lhs_array': builder.format_array_as_c_literal(input_lhs_q_for_cmsis),
            'input_rhs_array': builder.format_array_as_c_literal(input_rhs_q_for_cmsis),
            'expected_output_array': builder.format_array_as_c_literal(output_data),
            'input_dtype': kernel_info["input_lhs_c_type"],
            'input_rhs_dtype': kernel_info["input_rhs_c_type"],
            'output_dtype': kernel_info["output_c_type"],
            'kernel_fn': kernel_info["kernel_fn"],
            'kernel_get_buffer_size_fn': kernel_info["kernel_get_buffer_size_fn"],
            'buffer_size_max': buffer_size_max,
        }
        self._render_batch_matmul(output_dir, context)
        self._write_cmake(output_dir)

    def _write_cmake(self, output_dir: Path) -> None:
        cmake_context = {
            'name': self.desc['name'],
            'operator': self.desc.get('operator', 'BatchMatMul'),
            'operator_name': 'batch_matmul'
        }
        (Path(output_dir) / "CMakeLists.txt").write_text(self.render_template("common/CMakeLists.txt.j2", cmake_context))
