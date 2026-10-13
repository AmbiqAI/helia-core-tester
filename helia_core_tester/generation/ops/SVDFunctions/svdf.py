"""
SVDF operation implementation.
"""

from typing import Dict, Any
from pathlib import Path
import numpy as np
from helia_core_tester.generation.ops._shared import kernel_faithful as kf
from helia_core_tester.generation.ops._shared.base import OperationBase
from helia_core_tester.generation.harness import ArgumentPool, HarnessInput, fragment
from helia_core_tester.generation.reference.call import ReferenceCall

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

    def uses_reference(self) -> bool:
        return not self.status_only()

    def status_only(self) -> bool:
        # A fault case asserts the kernel's ARG_ERROR status, never an output.
        return str(self.desc.get("expected_status", "ARM_CMSIS_NN_SUCCESS")).strip() != "ARM_CMSIS_NN_SUCCESS"

    def _svdf_outputs(self, call: ReferenceCall) -> Dict[str, np.ndarray]:
        """The reference outputs of `call`; recorded as the golden unless the case is status-only.

        A status-only case still emits a golden array in its header (shared with the
        passing variant), so it is filled from an unrecorded run to keep the harness unchanged."""
        return call.run() if self.status_only() else self.reference_golden(call)

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

            if input_activation_min > -1.0e30 or input_activation_max < 1.0e30:
                # svdf_f32/f16 clamp only the output; a projection clamp would go unmodelled.
                raise ValueError(f"{name}: SVDF float input_activation_min/max are not supported by the reference")
            call = ReferenceCall(
                "svdf_f16" if activation_dtype == "FP16" else "svdf_f32",
                {"rank": rank, "activation_min": output_activation_min, "activation_max": output_activation_max},
                {"input": input_sequence, "weights_feature": weights_feature, "weights_time": weights_time,
                 "bias": bias if bias is not None else np.zeros((0,), dtype=float_dtype), "state": state_init},
                {"output": (sequence_steps, input_batches, unit_count),
                 "state_out": (input_batches, feature_batches, time_batches)},
            )
            bound = (0.0, 0.0)
            if (activation_dtype == "FP16" and not self.status_only()
                    and kf.mve_float16_route(self)):
                # arm_svdf_f16 on Helium: float16 lane dots, the float32 rank sum, maxNum/minNum clamp.
                call = kf.rnn_f16_mve_call(call)
                faithful = True
            else:
                faithful = False
                if not self.status_only():
                    # The exact SVDF rounded once. Each output nests three sums the build may order freely: the
                    # input_height-term feature projection (rounded into the state), the time_batches-term time
                    # reduction and the rank-term sum with the bias, so linear_bound over all of them.
                    terms = input_height + time_batches + rank + 2
                    bound = kf.linear_bound(kf.abs_magnitude(call, "output")[-1], terms, float_dtype)
            # The harness runs sequence_steps kernel calls and compares the last step's output.
            outputs = self._svdf_outputs(call)
            if faithful:
                kf.check_svdf_within_bound(name, outputs["output"], call)
            expected_output = outputs["output"][-1].reshape(-1)

            def final_step(operands):
                probe = np.ascontiguousarray(np.asarray(operands[0]).reshape(input_sequence.shape), dtype=float_dtype)
                return call.with_inputs(input=probe).run()["output"][-1].reshape(-1)

            expected_output, nonfinite_context = self.apply_nonfinite_policy(
                expected_output, reference=final_step, inputs=[input_sequence]
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
            if faithful or bound != (0.0, 0.0):
                context.update(kf.comparison_context(self, float_dtype, *bound))
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
        weights_time_dtype = np.int8 if state_dtype_str == "S8" else np.int16
        kernel_fn = "arm_svdf_s8" if state_dtype_str == "S8" else "arm_svdf_state_s16_s8"
        has_ctx = state_dtype_str == "S8"

        input_offset = int(hint.get("input_offset", 0))
        output_offset = int(hint.get("output_offset", 0))
        input_multiplier = int(hint.get("input_multiplier", 2147483647))
        input_shift = int(hint.get("input_shift", 0))
        output_multiplier = int(hint.get("output_multiplier", 2147483647))
        output_shift = int(hint.get("output_shift", 0))

        if state_dtype_str == "S16":
            input_activation_min = int(hint.get("input_activation_min", -32768))
            input_activation_max = int(hint.get("input_activation_max", 32767))
        else:
            input_activation_min = int(hint.get("input_activation_min", -128))
            input_activation_max = int(hint.get("input_activation_max", 127))

        output_activation_min = int(hint.get("output_activation_min", -128))
        output_activation_max = int(hint.get("output_activation_max", 127))

        rng_state = self.rng.__getstate__()
        self.rng = np.random.default_rng(self.seed)

        input_data = self.rng.integers(-5, 6, size=(input_batches, input_height), dtype=np.int8)
        weights_feature = self.rng.integers(-5, 6, size=(feature_batches, input_height), dtype=np.int8)
        weights_time = self.rng.integers(-5, 6, size=(feature_batches, time_batches), dtype=weights_time_dtype)
        state_init = self.rng.integers(-5, 6, size=(input_batches, feature_batches, time_batches), dtype=state_dtype)

        if use_bias:
            bias = self.rng.integers(-25, 26, size=(unit_count,), dtype=np.int32)
        else:
            bias = None

        self.rng.__setstate__(rng_state)

        call = ReferenceCall(
            "svdf_s8" if state_dtype_str == "S8" else "svdf_state_s16_s8",
            {
                "rank": rank,
                # cmsis_nn_svdf_params.input_offset is subtracted by the kernel; the entry adds it.
                "input_offset": -input_offset,
                "output_offset": output_offset,
                "input_multiplier": input_multiplier,
                "input_shift": input_shift,
                "output_multiplier": output_multiplier,
                "output_shift": output_shift,
                "input_activation_min": input_activation_min,
                "input_activation_max": input_activation_max,
                "output_activation_min": output_activation_min,
                "output_activation_max": output_activation_max,
            },
            {"input": input_data.reshape(1, input_batches, input_height), "weights_feature": weights_feature,
             "weights_time": weights_time, "bias": bias if bias is not None else np.zeros((0,), dtype=np.int32),
             "state": state_init},
            {"output": (1, input_batches, unit_count), "state_out": (input_batches, feature_batches, time_batches)},
        )
        expected_output = self._svdf_outputs(call)["output"].reshape(-1)

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
