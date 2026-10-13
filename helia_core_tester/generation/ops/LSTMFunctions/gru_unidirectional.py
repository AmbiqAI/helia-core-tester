"""GRUUnidirectional operation implementation.

CMSIS-only (force_cmsis) float test generation for arm_gru_unidirectional_f16
and arm_gru_unidirectional_f32. The golden is the kernel's arithmetic on the target
(gru_f32_cmsis for float32 on every target, gru_f16_mve for float16 on Helium),
checked against the exact gru_f32/gru_f16 entry (gates update z, reset r,
candidate n; reset-after and pre-reset forms); float16 elsewhere keeps the exact one.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from helia_core_tester.generation.ops._shared import kernel_faithful as kf
from helia_core_tester.generation.ops._shared.base import OperationBase
from helia_core_tester.generation.ops._shared.recurrent_pool import recurrent_argument_pool

GRU_FRAGMENTS = "LSTMFunctions/gru_unidirectional"


class OpGRUUnidirectional(OperationBase):
    """GRUUnidirectional operation (FP32/FP16, CMSIS-only)."""

    FAULT_KINDS = (
        "null_input",
        "null_output",
        "null_params",
        "null_buffers",
        "missing_temp1_prereset",
        "stateful_batch_gt1",
        "negative_input_size",
        "negative_hidden_size",
        "zero_batch_size",
        "negative_time_steps",
    )

    def status_only(self) -> bool:
        # An ARG_ERROR case asserts the returned status alone; null_buffers expects SUCCESS and keeps its golden.
        return self.expected_status() != "ARM_CMSIS_NN_SUCCESS"

    def uses_reference(self) -> bool:
        return not self.status_only()

    @staticmethod
    def gru_call(
        input_tensor: np.ndarray,
        gates: Dict[str, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]],
        *,
        time_major: bool,
        reset_after: bool,
    ):
        """The gru_f32/gru_f16 ReferenceCall: per-gate (w_in, w_hidden, b_in, b_hidden) stacked in z, r, n order.

        Absent biases (None) are passed as shape (0,) tensors."""
        from helia_core_tester.generation.reference.call import ReferenceCall

        dtype = np.asarray(input_tensor).dtype
        if dtype not in (np.float32, np.float16):
            raise ValueError(f"GRU reference takes float32 or float16, got {dtype}")
        order = [gates[g] for g in ("update", "reset", "candidate")]

        def stack(index: int) -> np.ndarray:
            parts = [g[index] for g in order]
            if any(p is None for p in parts):
                if not all(p is None for p in parts):
                    raise ValueError("GRU biases must be present on every gate or on none")
                return np.zeros((0,), dtype=dtype)
            return np.ascontiguousarray(np.stack([np.asarray(p, dtype=dtype) for p in parts]))

        x = np.ascontiguousarray(input_tensor, dtype=dtype)
        if x.ndim != 3:
            raise ValueError(f"GRU input must be rank 3, got shape {x.shape}")
        w = stack(0)
        hidden_size = int(w.shape[1])
        return ReferenceCall(
            f"gru_{'f16' if dtype == np.float16 else 'f32'}",
            {"time_major": int(bool(time_major)), "reset_after": int(bool(reset_after))},
            {"input": x, "input_weights": w, "recurrent_weights": stack(1), "input_bias": stack(2),
             "recurrent_bias": stack(3)},
            {"output": (x.shape[0], x.shape[1], hidden_size)},
        )

    def generate_c_files(self, output_dir) -> None:
        if not self.desc.get("hint", {}).get("force_cmsis", False):
            raise ValueError("GRUUnidirectional descriptor requires hint.force_cmsis: true.")

        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

        name = self.desc["name"]
        batch_size = int(self.desc.get("batch_size", 1))
        time_steps = int(self.desc.get("time_steps", 1))
        input_size = int(self.desc.get("feature_size", self.desc.get("input_size", 1)))
        hidden_size = int(self.desc.get("units", self.desc.get("hidden_size", 1)))
        time_major = bool(self.desc.get("time_major", False))
        reset_after = bool(self.desc.get("reset_after", True))
        # Issue #56: use_bias: false exercises the NULL-safe bias branches in
        # arm_nn_gru_step_*.c (`gate->input_bias ? gate->input_bias[h] : 0.0f`,
        # same for hidden_bias, all three gates) -- a real GRU layer config
        # (use_bias=False), previously untested. Zero-valued bias is
        # mathematically identical to omitting it; the reference call passes
        # the biases as absent and the C side passes NULL through the
        # `use_bias` template flag.
        use_bias = bool(self.desc.get("use_bias", True))

        activation_dtype = self.tensor_dtype("input", default="FP32")
        if activation_dtype == "FP32":
            float_dtype = np.float32
            data_dtype = "float32_t"
            suffix = "f32"
        elif activation_dtype == "FP16":
            float_dtype = np.float16
            data_dtype = "float16_t"
            suffix = "f16"
        else:
            raise ValueError(
                f"Unsupported GRUUnidirectional dtype: {activation_dtype} (float-only kernels)")

        kernel_fn = f"arm_gru_unidirectional_{suffix}"
        gru_params_type = f"cmsis_nn_gru_params_{suffix}"
        gru_context_type = f"cmsis_nn_gru_context_{suffix}"

        # Deterministic data, independent of any prior RNG draws in this run.
        rng_state = self.rng.__getstate__()
        self.rng = np.random.default_rng(self.seed)

        if time_major:
            input_tensor = self.rng.uniform(-1.0, 1.0, size=(time_steps, batch_size, input_size)).astype(float_dtype)
        else:
            input_tensor = self.rng.uniform(-1.0, 1.0, size=(batch_size, time_steps, input_size)).astype(float_dtype)
        # The weight draws below continue this same stream, so the sweep is applied to
        # the tensor already drawn rather than resampled through _sample_uniform, which
        # restarts the stream and would move every weight.
        input_tensor = self._maybe_apply_input_mode(input_tensor)

        def w_in():
            return self.rng.uniform(-0.5, 0.5, size=(hidden_size, input_size)).astype(float_dtype)

        def w_hidden():
            return self.rng.uniform(-0.5, 0.5, size=(hidden_size, hidden_size)).astype(float_dtype)

        def bias():
            if not use_bias:
                return np.zeros((hidden_size,), dtype=float_dtype)
            return self.rng.uniform(-0.25, 0.25, size=(hidden_size,)).astype(float_dtype)

        update_w_in, update_w_hidden = w_in(), w_hidden()
        update_b_in, update_b_hidden = bias(), bias()
        reset_w_in, reset_w_hidden = w_in(), w_hidden()
        reset_b_in, reset_b_hidden = bias(), bias()
        cand_w_in, cand_w_hidden = w_in(), w_hidden()
        cand_b_in, cand_b_hidden = bias(), bias()

        self.rng.__setstate__(rng_state)

        gates = {
            "update": (update_w_in, update_w_hidden, update_b_in if use_bias else None,
                       update_b_hidden if use_bias else None),
            "reset": (reset_w_in, reset_w_hidden, reset_b_in if use_bias else None,
                      reset_b_hidden if use_bias else None),
            "candidate": (cand_w_in, cand_w_hidden, cand_b_in if use_bias else None,
                          cand_b_hidden if use_bias else None),
        }
        call = self.gru_call(input_tensor, gates, time_major=time_major, reset_after=reset_after)
        faithful = False
        if self.status_only():
            # The kernel rejects the call before writing; the header still declares an expected
            # array, filled from the same reference but not recorded as a golden.
            output_ref = call.output()
            nonfinite_context = {}
        else:
            if float_dtype == np.float16 and kf.mve_float16_target(self.target_cpu):
                # arm_nn_gru_step_f16 on Helium: float16 lane dots, the kernel's float16 tables and the
                # pre-reset sum as GCC vectorised it.
                exact, call = call, kf.rnn_f16_mve_call(call)
                kf.check_close_to_exact(name, call, exact, atol=kf.RNN_F16_ATOL)
                faithful = True
            elif float_dtype == np.float32:
                # arm_nn_gru_step_f32 on this target: the kernel's float32 tables and operation order.
                exact, call = call, kf.gru_f32_cmsis_call(call, self.target_cpu)
                kf.check_close_to_exact(name, call, exact, atol=kf.GRU_F32_ATOL)
                faithful = True
            output_ref = self.reference_golden(call)
            output_ref, nonfinite_context = self.apply_nonfinite_policy(
                output_ref,
                reference=lambda operands: self.reference_probe([operands[0], *list(call.inputs.values())[1:]]),
                inputs=[input_tensor],
            )
        output_ref = np.asarray(output_ref).reshape(-1)

        fault = self.fault_kind()
        stream = bool(self.desc.get("hint", {}).get("stream", False))

        builder = TemplateContextBuilder()
        context = {
            "name": name,
            "data_dtype": data_dtype,
            "kernel_fn": kernel_fn,
            "gru_params_type": gru_params_type,
            "gru_context_type": gru_context_type,
            "output_dtype": data_dtype,
            "time_major_literal": "1" if time_major else "0",
            "reset_after_literal": "1" if reset_after else "0",
            "batch_size": batch_size,
            "time_steps": time_steps,
            "input_size": input_size,
            "hidden_size": hidden_size,
            "input_tensor_array": builder.format_array_as_c_literal(input_tensor),
            "output_array": builder.format_array_as_c_literal(output_ref.astype(float_dtype)),
            "update_gate_input_weights_array": builder.format_array_as_c_literal(update_w_in),
            "update_gate_hidden_weights_array": builder.format_array_as_c_literal(update_w_hidden),
            "update_gate_input_bias_array": builder.format_array_as_c_literal(update_b_in),
            "update_gate_hidden_bias_array": builder.format_array_as_c_literal(update_b_hidden),
            "reset_gate_input_weights_array": builder.format_array_as_c_literal(reset_w_in),
            "reset_gate_hidden_weights_array": builder.format_array_as_c_literal(reset_w_hidden),
            "reset_gate_input_bias_array": builder.format_array_as_c_literal(reset_b_in),
            "reset_gate_hidden_bias_array": builder.format_array_as_c_literal(reset_b_hidden),
            "candidate_gate_input_weights_array": builder.format_array_as_c_literal(cand_w_in),
            "candidate_gate_hidden_weights_array": builder.format_array_as_c_literal(cand_w_hidden),
            "candidate_gate_input_bias_array": builder.format_array_as_c_literal(cand_b_in),
            "candidate_gate_hidden_bias_array": builder.format_array_as_c_literal(cand_b_hidden),
            "hidden_state_size": batch_size * hidden_size,
            "dst_size": batch_size * time_steps * hidden_size,
            "expected_status": self.expected_status(),
            "use_bias": use_bias,
        }
        if faithful:
            context.update(kf.tolerance_context(float_dtype))
        context.update(nonfinite_context)

        # ns-cmsis-nn#377 / tester#71: generation-time feature probe for the
        # temp-buffer sizer added by ns-cmsis-nn#381. Detected -> the main
        # template emits the sizer-calling, sizer-validating variant; absent
        # -> byte-identical legacy output (safe against ns-cmsis-nn main).
        from helia_core_tester.generation.utils.temp_sizer_probe import (
            detect_temp_sizers,
            gru_temp1_expected_bytes,
        )

        elem_bytes = 2 if suffix == "f16" else 4
        context["reset_after"] = reset_after
        context["temp_sizers_available"] = detect_temp_sizers(
            [f"{kernel_fn}_temp1_get_buffer_size"],
            f"GRUUnidirectional[{name}]",
        )
        context["gru_temp1_expected_bytes"] = gru_temp1_expected_bytes(
            reset_after=reset_after, hidden_size=hidden_size, elem_bytes=elem_bytes
        )
        context["gru_temp1_expected_bytes_flipped"] = gru_temp1_expected_bytes(
            reset_after=not reset_after, hidden_size=hidden_size, elem_bytes=elem_bytes
        )

        if fault:
            context.update(self.fault_context())
            variant = "gru_unidirectional_fault"
        elif stream:
            if batch_size != 1:
                raise ValueError("GRUUnidirectional streaming descriptors require batch_size == 1.")
            chunk_lengths = self.desc.get("stream_chunk_lengths")
            if not chunk_lengths:
                half = time_steps // 2
                chunk_lengths = [half, time_steps - half]
            if sum(chunk_lengths) != time_steps:
                raise ValueError("stream_chunk_lengths must sum to time_steps.")
            chunk_offsets = []
            offset = 0
            for length in chunk_lengths:
                chunk_offsets.append(offset)
                offset += length
            context["chunk_lengths"] = chunk_lengths
            context["chunk_input_offsets"] = [c * input_size for c in chunk_offsets]
            context["chunk_output_offsets"] = [c * hidden_size for c in chunk_offsets]
            variant = "gru_unidirectional_stream"
        else:
            variant = "gru_unidirectional"

        self.render_harness_case(
            Path(output_dir), stem="gru_unidirectional", context=context,
            pool=recurrent_argument_pool(
                context, body=f"{GRU_FRAGMENTS}/{variant}.fragment.j2",
                header=f"{GRU_FRAGMENTS}/gru_unidirectional.fragment.j2", ctype=context["data_dtype"]),
            validation_key=f"{GRU_FRAGMENTS}/{variant}.c.j2", label="GRU", operator="GRUUnidirectional", sidecar=True,
        )
