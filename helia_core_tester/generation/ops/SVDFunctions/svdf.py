"""
SVDF operation implementation.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import numpy as np
from helia_core_tester.generation.ops._shared.base import OperationBase
from helia_core_tester.generation.harness import ArgumentPool, HarnessInput, fragment

SVDF_VARIANTS = ("svdf", "svdf_fault", "svdf_f32", "svdf_f32_fault")


def svdf_argument_pool(context: Dict[str, Any], variant: str) -> ArgumentPool:
    """SVDF keeps its body as a fragment (scratch contexts sized by the kernel's own sizers, the
    in-place state, the sizer probes); the kernel call goes through `_run`, which receives the
    contexts, params and state the fragment built and binds the rest from the header."""
    if variant not in SVDF_VARIANTS:
        raise ValueError(f"{context['name']}: SVDF variant {variant!r} is not one of {SVDF_VARIANTS}")
    n, float_kernel = context["name"], variant.startswith("svdf_f32")
    path = f"SVDFunctions/svdf/{variant}.fragment.j2"
    header = f"SVDFunctions/svdf/{'svdf_f32' if float_kernel else 'svdf'}.fragment.j2"
    data = context["data_dtype"] if float_kernel else "int8_t"
    state = context["data_dtype"] if float_kernel else context["state_dtype"]
    params_type = context["svdf_params_type"] if float_kernel else "cmsis_nn_svdf_params"
    takes_ctx = float_kernel or bool(context["has_ctx"])
    run_params = [("ctx", "const cmsis_nn_context *")] if takes_ctx else []
    run_params += [("input_ctx", "const cmsis_nn_context *"), ("output_ctx", "const cmsis_nn_context *"),
                   ("svdf_params", f"const {params_type} *")]
    if not float_kernel:
        run_params += [("input_quant_params", "const cmsis_nn_per_tensor_quant_params *"),
                       ("output_quant_params", "const cmsis_nn_per_tensor_quant_params *")]
    run_params.append(("state_data", f"{state} *"))
    values = {local: local for local, _ in run_params}
    values.update({
        "input_dims": f"&{n}_input_dims", "state_dims": f"&{n}_state_dims",
        "weights_feature_dims": f"&{n}_weights_feature_dims", "weights_feature_data": f"{n}_weights_feature",
        "weights_time_dims": f"&{n}_weights_time_dims", "weights_time_data": f"{n}_weights_time",
        "bias_dims": f"&{n}_bias_dims", "bias_data": f"{n}_bias" if context["use_bias"] else "NULL",
        "output_dims": f"&{n}_output_dims",
    })
    return ArgumentPool(
        name=n, values=values, inputs=(HarnessInput("input_data", "input", f"{n}_input", data),),
        output_ctype=data, run_params=tuple(run_params), owns_ctx=takes_ctx,
        header_text=fragment(header, "header"), file_scope=fragment(path, "file_scope"),
        test_body=fragment(path, "test_body"), includes=() if float_kernel else ("<string.h>",),
        benchmark=False, scratch_buffer=False,
    )


class OpSVDF(OperationBase):
    """
    SVDF operation.
    """

    FAULT_KINDS = (
        "null_ctx_buf",
        "null_input_ctx_buf",
        "null_output_ctx_buf",
        "small_input_ctx_size",
        "small_output_ctx_size",
        "zero_rank",
        "null_input",
        "null_output",
    )

    def _check_fault_reachable(self, kind: str, kernel_fn: str, float_kernel: bool) -> None:
        """Reject fault kinds the selected SVDF kernel does not diagnose."""
        if float_kernel:
            if kind == "null_ctx_buf":
                raise self.fault_unreachable(kind, f"{kernel_fn} ignores ctx")
            return
        if kind not in ("null_ctx_buf", "null_input_ctx_buf", "null_output_ctx_buf"):
            raise self.fault_unreachable(kind, f"{kernel_fn} does not check {kind}")
        if kind == "null_ctx_buf":
            if kernel_fn != "arm_svdf_s8":
                raise self.fault_unreachable(kind, f"{kernel_fn} takes no kernel-sum context")
            if "mve" not in self.required_capabilities():
                raise self.fault_unreachable(
                    kind, "arm_svdf_s8 only checks ctx->buf under ARM_MATH_MVEI; add required_capabilities: [mve]"
                )

    @staticmethod
    def _requantize_np(values: np.ndarray, multiplier: int, shift: int) -> np.ndarray:
        left_shift = shift if shift > 0 else 0
        right_shift = -shift if shift < 0 else 0
        prod = values.astype(np.int64) * (1 << left_shift)
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

    def _generate_svdf_expected(
        self,
        input_data: np.ndarray,
        state_init: np.ndarray,
        weights_feature: np.ndarray,
        weights_time: np.ndarray,
        bias: Optional[np.ndarray],
        params: Dict[str, int],
        state_dtype: np.dtype,
    ) -> np.ndarray:
        input_batches = int(params["input_batches"])
        input_height = int(params["input_height"])
        feature_batches = int(params["feature_batches"])
        time_batches = int(params["time_batches"])
        rank = int(params["rank"])
        unit_count = feature_batches // rank

        in_mult = int(params["input_multiplier"])
        in_shift = int(params["input_shift"])
        out_mult = int(params["output_multiplier"])
        out_shift = int(params["output_shift"])
        in_act_min = int(params["input_activation_min"])
        in_act_max = int(params["input_activation_max"])
        out_act_min = int(params["output_activation_min"])
        out_act_max = int(params["output_activation_max"])
        input_offset = int(params["input_offset"])
        output_offset = int(params["output_offset"])

        state = state_init.reshape(input_batches, feature_batches, time_batches).copy()
        if time_batches > 1:
            state[:, :, : time_batches - 1] = state[:, :, 1:]

        # Update last time step with input * weights_feature
        lhs_offset = -input_offset
        for b in range(input_batches):
            lhs = input_data[b].astype(np.int32) + lhs_offset
            for f in range(feature_batches):
                acc = int(np.sum(lhs * weights_feature[f].astype(np.int32)))
                acc_q = self._requantize_np(np.array([acc], dtype=np.int32), in_mult, in_shift)[0]
                acc_q = int(np.clip(acc_q, in_act_min, in_act_max))
                if state_dtype == np.int8:
                    state[b, f, time_batches - 1] = np.int8(acc_q)
                else:
                    state[b, f, time_batches - 1] = np.int16(acc_q)

        # Time weights * state
        buffer_a = np.zeros((input_batches, feature_batches), dtype=np.int32)
        for b in range(input_batches):
            for f in range(feature_batches):
                buffer_a[b, f] = int(np.sum(weights_time[f].astype(np.int32) * state[b, f].astype(np.int32)))

        # Reduce over rank and add bias if provided
        if bias is not None:
            if unit_count == feature_batches:
                buffer_b = buffer_a + bias.reshape(1, feature_batches)
            else:
                buffer_b = np.zeros((input_batches, unit_count), dtype=np.int32)
                for b in range(input_batches):
                    for u in range(unit_count):
                        acc = int(bias[u])
                        for r in range(rank):
                            acc += int(buffer_a[b, u * rank + r])
                        buffer_b[b, u] = acc
        else:
            buffer_b = np.zeros((input_batches, unit_count), dtype=np.int32)
            for b in range(input_batches):
                for u in range(unit_count):
                    acc = 0
                    for r in range(rank):
                        acc += int(buffer_a[b, u * rank + r])
                    buffer_b[b, u] = acc

        out = self._requantize_np(buffer_b.flatten(), out_mult, out_shift)
        out = out + output_offset
        out = np.clip(out, out_act_min, out_act_max).astype(np.int8)
        return out

    def _generate_svdf_expected_f32(
        self,
        input_sequence: np.ndarray,
        state_init: np.ndarray,
        weights_feature: np.ndarray,
        weights_time: np.ndarray,
        bias: Optional[np.ndarray],
        params: Dict[str, Any],
    ) -> np.ndarray:
        input_batches = int(params["input_batches"])
        input_height = int(params["input_height"])
        feature_batches = int(params["feature_batches"])
        time_batches = int(params["time_batches"])
        rank = int(params["rank"])
        unit_count = feature_batches // rank
        input_act_min = float(params["input_activation_min"])
        input_act_max = float(params["input_activation_max"])
        output_act_min = float(params["output_activation_min"])
        output_act_max = float(params["output_activation_max"])

        state = np.asarray(state_init, dtype=np.float32).reshape(input_batches, feature_batches, time_batches).copy()
        output = np.zeros((input_batches, unit_count), dtype=np.float32)

        for step_input in np.asarray(input_sequence, dtype=np.float32):
            if time_batches > 1:
                state[:, :, : time_batches - 1] = state[:, :, 1:]

            projected = np.einsum("bi,fi->bf", step_input.reshape(input_batches, input_height), weights_feature)
            projected = np.clip(projected, input_act_min, input_act_max)
            state[:, :, time_batches - 1] = projected

            buffer_a = np.einsum("bft,ft->bf", state, weights_time)
            reduced = buffer_a.reshape(input_batches, unit_count, rank).sum(axis=2)
            if bias is not None:
                reduced = reduced + bias.reshape(1, unit_count)
            output = np.clip(reduced, output_act_min, output_act_max).astype(np.float32)

        return output.flatten()

    # TFLM requires the state zero point to be 0 and the bias scale to be state * weights_time.
    _STATE_SCALE = {"S8": 1.0 / 64, "S16": 2.0 ** -10}

    def uses_reference(self) -> bool:
        return self.desc.get("hint", {}).get("force_cmsis", False) and str(
            self.desc.get("activation_dtype", "S8")).upper() == "S8"

    def _int_reference_case(self, state_dtype_str: str, *, input_batches: int, input_height: int,
                            feature_batches: int, time_batches: int, rank: int, use_bias: bool) -> Dict[str, Any]:
        """Quantize, draw and run one integer SVDF step on the TFLM reference port.

        Scales: input over [-1, 1]; feature weights keeping the state O(1); state 1/64 (s8)
        or 2^-10 (s16); time weights keeping each filter O(1); output covering the rank sum.
        effective_scale_1/2 are formed in float32 and widened, as TFLM's SVDF prepare does.
        Explicit hint multipliers/offsets (legacy) are passed through untouched."""
        from helia_core_tester.generation.reference import params as ref_params
        from helia_core_tester.generation.reference.case import ReferenceCall

        if feature_batches % rank:
            raise ValueError(f"{self.desc.get('name')}: rank {rank} does not divide feature_batches {feature_batches}")
        hint = self.desc.get("hint", {})
        f32 = np.float32
        state_t = np.int8 if state_dtype_str == "S8" else np.int16
        state_info = np.iinfo(state_t)
        in_s = f32(1.0 / 128)
        wf_s = f32(1.0 / (127 * np.sqrt(input_height)))
        st_s = f32(self._STATE_SCALE[state_dtype_str])
        wt_s = f32(1.0 / (state_info.max * np.sqrt(time_batches)))
        out_s = f32(np.sqrt(rank) / 64)
        mult1, shift1 = ref_params.quantize_multiplier(float(f32(in_s * wf_s / st_s)))
        mult2, shift2 = ref_params.quantize_multiplier(float(f32(st_s * wt_s / out_s)))
        params = {
            "batch": input_batches, "input_size": input_height, "num_filters": feature_batches,
            "memory_size": time_batches, "rank": rank,
            "input_zero_point": int(hint.get("input_offset", 0)), "output_zero_point": int(hint.get("output_offset", 0)),
            "scale1_multiplier": int(hint.get("input_multiplier", mult1)), "scale1_shift": int(hint.get("input_shift", shift1)),
            "scale2_multiplier": int(hint.get("output_multiplier", mult2)), "scale2_shift": int(hint.get("output_shift", shift2)),
        }

        rng_state = self.rng.__getstate__()
        self.rng = np.random.default_rng(self.seed)
        x = self.rng.integers(-128, 128, size=(input_batches, input_height)).astype(np.int8)
        wf = self.rng.integers(-127, 128, size=(feature_batches, input_height)).astype(np.int8)
        wt = self.rng.integers(-state_info.max, state_info.max + 1, size=(feature_batches, time_batches)).astype(state_t)
        half = state_info.max // 2
        state = self.rng.integers(-half, half + 1, size=(input_batches, feature_batches, time_batches)).astype(state_t)
        bias = None
        if use_bias:
            reach = max(1, int(0.25 / float(st_s * wt_s)))
            bias = self.rng.integers(-reach, reach + 1, size=(feature_batches // rank,)).astype(np.int32)
        self.rng.__setstate__(rng_state)

        call = ReferenceCall(
            "svdf_s8" if state_dtype_str == "S8" else "svdf_s16state", params,
            {"input": x, "weights_feature": wf, "weights_time": wt, "bias": bias, "state": state},
            (input_batches, feature_batches // rank), "int8",
            quant={"input": float(in_s), "weights_feature": float(wf_s), "state": float(st_s),
                   "weights_time": float(wt_s), "output": float(out_s)},
        )
        return {"params": params, "input": x, "weights_feature": wf, "weights_time": wt, "state": state, "bias": bias,
                "output": self.reference_golden(call).reshape(-1)}

    def _ctx_sizer_context(self, sizer_prefix: str) -> Dict[str, Any]:
        # Issue #71: the harness sizes input_ctx/output_ctx with the kernel's own
        # sizers. The expected values are measured on host and carried by the
        # descriptor, never re-derived here.
        return {
            "input_ctx_sizer_fn": f"{sizer_prefix}_input_ctx_get_buffer_size",
            "output_ctx_sizer_fn": f"{sizer_prefix}_output_ctx_get_buffer_size",
            "expected_input_ctx_size": self.desc.get("expected_input_ctx_size"),
            "expected_output_ctx_size": self.desc.get("expected_output_ctx_size"),
            "ctx_sizer_sentinels": bool(self.desc.get("ctx_sizer_sentinels", False)),
        }

    def generate_c_files(self, output_dir: Path) -> None:
        """
        Generate C and H files for SVDF operation.
        """
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

        name = self.desc["name"]
        force_cmsis = self.desc.get("hint", {}).get("force_cmsis", False)
        if not force_cmsis:
            # TFLite-based SVDF not supported for C test generation
            return

        activation_dtype = str(self.desc.get("activation_dtype", "S8")).upper()
        if activation_dtype in {"FP32", "FP16"}:
            float_dtype = np.float16 if activation_dtype == "FP16" else np.float32
            data_dtype = "float16_t" if activation_dtype == "FP16" else "float"
            kernel_fn = "arm_svdf_f16" if activation_dtype == "FP16" else "arm_svdf_f32"
            svdf_params_type = "cmsis_nn_svdf_params_f16" if activation_dtype == "FP16" else "cmsis_nn_svdf_params_f32"
            hint = self.desc.get("hint", {})
            input_batches = int(hint.get("input_batches", 1))
            input_height = int(hint.get("input_height", 4))
            feature_batches = int(hint.get("feature_batches", 4))
            time_batches = int(hint.get("time_batches", 3))
            rank = int(hint.get("rank", 1))
            sequence_steps = int(hint.get("sequence_steps", 2))
            unit_count = feature_batches // rank
            use_bias = bool(hint.get("use_bias", True))
            input_activation_min = float(hint.get("input_activation_min", -1.0e30))
            input_activation_max = float(hint.get("input_activation_max", 1.0e30))
            output_activation_min = float(hint.get("output_activation_min", -1.0e30))
            output_activation_max = float(hint.get("output_activation_max", 1.0e30))

            rng_state = self.rng.__getstate__()
            self.rng = np.random.default_rng(self.seed)
            input_sequence = self.rng.uniform(-1.0, 1.0, size=(sequence_steps, input_batches, input_height)).astype(float_dtype)
            # The state, weight and bias draws below continue this same stream, so the
            # sweep is applied to the tensor already drawn rather than resampled through
            # _sample_uniform, which restarts the stream and would move all of them.
            input_sequence = self._maybe_apply_input_mode(input_sequence)
            state_init = self.rng.uniform(-0.5, 0.5, size=(input_batches, feature_batches, time_batches)).astype(float_dtype)
            weights_feature = self.rng.uniform(-1.0, 1.0, size=(feature_batches, input_height)).astype(float_dtype)
            weights_time = self.rng.uniform(-1.0, 1.0, size=(feature_batches, time_batches)).astype(float_dtype)
            bias = self.rng.uniform(-0.5, 0.5, size=(unit_count,)).astype(float_dtype) if use_bias else None
            self.rng.__setstate__(rng_state)

            params = {
                "input_batches": input_batches,
                "input_height": input_height,
                "feature_batches": feature_batches,
                "time_batches": time_batches,
                "rank": rank,
                "input_activation_min": input_activation_min,
                "input_activation_max": input_activation_max,
                "output_activation_min": output_activation_min,
                "output_activation_max": output_activation_max,
            }
            def float_reference(operands):
                return self._generate_svdf_expected_f32(
                    input_sequence=operands[0],
                    state_init=state_init,
                    weights_feature=weights_feature,
                    weights_time=weights_time,
                    bias=bias,
                    params=params,
                )

            expected_output = float_reference([input_sequence])
            expected_output, nonfinite_context = self.apply_nonfinite_policy(
                expected_output, reference=float_reference, inputs=[input_sequence]
            )

            builder = TemplateContextBuilder()
            context = {
                "name": name,
                "kernel_fn": kernel_fn,
                "data_dtype": data_dtype,
                "output_dtype": data_dtype,
                "svdf_params_type": svdf_params_type,
                "input_batches": input_batches,
                "input_height": input_height,
                "feature_batches": feature_batches,
                "time_batches": time_batches,
                "rank": rank,
                "unit_count": unit_count,
                "sequence_steps": sequence_steps,
                "input_size": int(input_batches * input_height),
                "state_size": int(input_batches * feature_batches * time_batches),
                "scratch_input_size": int(input_batches * feature_batches),
                "scratch_output_size": int(input_batches * unit_count),
                "output_size": int(input_batches * unit_count),
                "use_bias": use_bias,
                "input_data_array": builder.format_array_as_c_literal(input_sequence),
                "weights_feature_array": builder.format_array_as_c_literal(weights_feature),
                "weights_time_array": builder.format_array_as_c_literal(weights_time),
                "state_init_array": builder.format_array_as_c_literal(state_init),
                "output_ref_array": builder.format_array_as_c_literal(expected_output.astype(float_dtype)),
                "bias_array": builder.format_array_as_c_literal(bias) if bias is not None else "",
                "input_activation_min_literal": builder.format_float_literal(input_activation_min),
                "input_activation_max_literal": builder.format_float_literal(input_activation_max),
                "output_activation_min_literal": builder.format_float_literal(output_activation_min),
                "output_activation_max_literal": builder.format_float_literal(output_activation_max),
            }
            context.update(self._ctx_sizer_context(kernel_fn))
            context.update(nonfinite_context)
            fault = self.fault_kind()
            variant = "svdf_f32"
            if fault:
                self._check_fault_reachable(fault, kernel_fn, True)
                context.update(self.fault_context())
                variant = "svdf_f32_fault"
            self.render_harness_case(
                output_dir, stem="svdf", context=context, pool=svdf_argument_pool(context, variant),
                validation_key=f"SVDFunctions/svdf/{variant}.c.j2", label="SVDF", operator="SVDF", sidecar=True,
            )
            return

        hint = self.desc.get("hint", {})
        input_batches = int(hint.get("input_batches", 2))
        input_height = int(hint.get("input_height", 4))
        feature_batches = int(hint.get("feature_batches", 4))
        time_batches = int(hint.get("time_batches", 3))
        rank = int(hint.get("rank", 2))
        unit_count = feature_batches // rank
        use_bias = bool(hint.get("use_bias", True))

        state_dtype_str = str(hint.get("state_dtype", "S8")).upper()
        if state_dtype_str not in ("S8", "S16"):
            raise ValueError(f"Unsupported state_dtype: {state_dtype_str}")

        state_dtype = np.int8 if state_dtype_str == "S8" else np.int16
        kernel_fn = "arm_svdf_s8" if state_dtype_str == "S8" else "arm_svdf_state_s16_s8"
        has_ctx = state_dtype_str == "S8"

        case = self._int_reference_case(
            state_dtype_str, input_batches=input_batches, input_height=input_height,
            feature_batches=feature_batches, time_batches=time_batches, rank=rank, use_bias=use_bias,
        )
        params = case["params"]
        input_data, weights_feature, weights_time = case["input"], case["weights_feature"], case["weights_time"]
        state_init, bias, expected_output = case["state"], case["bias"], case["output"]
        input_offset, output_offset = params["input_zero_point"], params["output_zero_point"]
        input_multiplier, input_shift = params["scale1_multiplier"], params["scale1_shift"]
        output_multiplier, output_shift = params["scale2_multiplier"], params["scale2_shift"]
        state_info = np.iinfo(state_dtype)
        input_activation_min, input_activation_max = int(state_info.min), int(state_info.max)
        output_activation_min, output_activation_max = -128, 127

        builder = TemplateContextBuilder()
        context = {
            "name": name,
            "kernel_fn": kernel_fn,
            "has_ctx": has_ctx,
            "input_batches": input_batches,
            "input_height": input_height,
            "feature_batches": feature_batches,
            "time_batches": time_batches,
            "rank": rank,
            "unit_count": unit_count,
            "input_size": int(input_batches * input_height),
            "state_size": int(input_batches * feature_batches * time_batches),
            "output_size": int(input_batches * unit_count),
            "input_offset": input_offset,
            "output_offset": output_offset,
            "input_multiplier": input_multiplier,
            "input_shift": input_shift,
            "output_multiplier": output_multiplier,
            "output_shift": output_shift,
            "input_activation_min": input_activation_min,
            "input_activation_max": input_activation_max,
            "output_activation_min": output_activation_min,
            "output_activation_max": output_activation_max,
            "use_bias": use_bias,
            "input_dtype": "int8_t",
            "state_dtype": "int8_t" if state_dtype_str == "S8" else "int16_t",
            "weights_time_dtype": "int8_t" if state_dtype_str == "S8" else "int16_t",
            "bias_dtype": "int32_t",
            "output_dtype": "int8_t",
            "input_data_array": builder.format_array_as_c_literal(input_data),
            "weights_feature_array": builder.format_array_as_c_literal(weights_feature),
            "weights_time_array": builder.format_array_as_c_literal(weights_time),
            "state_init_array": builder.format_array_as_c_literal(state_init),
            "output_ref_array": builder.format_array_as_c_literal(expected_output),
            "bias_array": builder.format_array_as_c_literal(bias) if bias is not None else "",
            # svdf.c.j2 is the only template that allocates scratch buffers with
            # malloc/free; gate the shared runtime_common.j2 stdlib.h include on it.
            "needs_stdlib": True,
        }
        context.update(self._ctx_sizer_context(kernel_fn))
        fault = self.fault_kind()
        variant = "svdf"
        if fault:
            self._check_fault_reachable(fault, kernel_fn, False)
            context.update(self.fault_context())
            context["needs_stdlib"] = False
            variant = "svdf_fault"
        self.render_harness_case(
            output_dir, stem="svdf", context=context, pool=svdf_argument_pool(context, variant),
            validation_key=f"SVDFunctions/svdf/{variant}.c.j2", label="SVDF", operator="SVDF",
        )
