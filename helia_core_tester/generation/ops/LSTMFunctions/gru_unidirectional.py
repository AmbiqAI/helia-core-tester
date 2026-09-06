"""GRUUnidirectional operation implementation.

CMSIS-only (force_cmsis) float test generation for arm_gru_unidirectional_f16
and arm_gru_unidirectional_f32. Mirrors the OpLSTMUnidirectional force_cmsis
code path: no Keras/TFLite model is built; a pure-numpy reference (matching the
arm_nn_gru_step_*.c math for both the reset-after and pre-reset formulations)
produces the golden output directly.
"""

from pathlib import Path
from typing import Any, Dict

import numpy as np
import tensorflow as tf

from helia_core_tester.generation.ops._shared.base import OperationBase


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

    def build_keras_model(self) -> tf.keras.Model:
        raise RuntimeError("GRUUnidirectional is CMSIS-only; no Keras model is built.")

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        raise RuntimeError("GRUUnidirectional CMSIS-only test; skip TFLite generation.")

    def needs_keras_model(self) -> bool:
        return False

    def allow_no_tflite(self) -> bool:
        return True

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def _generate_gru_expected(
        self,
        input_tensor: np.ndarray,
        update_w_in: np.ndarray,
        update_w_hidden: np.ndarray,
        update_b_in: np.ndarray,
        update_b_hidden: np.ndarray,
        reset_w_in: np.ndarray,
        reset_w_hidden: np.ndarray,
        reset_b_in: np.ndarray,
        reset_b_hidden: np.ndarray,
        cand_w_in: np.ndarray,
        cand_w_hidden: np.ndarray,
        cand_b_in: np.ndarray,
        cand_b_hidden: np.ndarray,
        *,
        batch_size: int,
        time_steps: int,
        input_size: int,
        hidden_size: int,
        time_major: bool,
        reset_after: bool,
    ) -> np.ndarray:
        """
        Reference GRU forward pass matching arm_nn_gru_step_f16.c /
        arm_nn_gru_step_f32.c exactly
        (same gate order, same reset-after / pre-reset branch, same
        batch-major/time-major addressing), computed in float32 for
        simplicity and cast to float16 only for the returned output.
        """
        hidden = np.zeros((batch_size, hidden_size), dtype=np.float32)
        output = np.zeros((time_steps, batch_size, hidden_size), dtype=np.float32)

        if time_major:
            sequence = np.asarray(input_tensor, dtype=np.float32).reshape(time_steps, batch_size, input_size)
        else:
            sequence = np.transpose(
                np.asarray(input_tensor, dtype=np.float32).reshape(batch_size, time_steps, input_size),
                (1, 0, 2),
            )

        for t in range(time_steps):
            x = sequence[t]  # [batch, input_size]

            z_pre = x @ update_w_in.T + hidden @ update_w_hidden.T + update_b_in + update_b_hidden
            z = self._sigmoid(z_pre)

            r_pre = x @ reset_w_in.T + hidden @ reset_w_hidden.T + reset_b_in + reset_b_hidden
            r = self._sigmoid(r_pre)

            if reset_after:
                # n = tanh( Wn.x + b_in + r * (Un.h_prev + b_hn) )
                xh = x @ cand_w_in.T + cand_b_in
                hh = hidden @ cand_w_hidden.T + cand_b_hidden
                cand_pre = xh + r * hh
            else:
                # n = tanh( Wn.x + b_in + Un.(r . h_prev) + b_hn )
                xh = x @ cand_w_in.T + cand_b_in
                hh = (r * hidden) @ cand_w_hidden.T + cand_b_hidden
                cand_pre = xh + hh

            cand = np.tanh(cand_pre)
            hidden = z * hidden + (1.0 - z) * cand
            output[t] = hidden

        if time_major:
            return output.astype(np.float32).flatten()
        return np.transpose(output, (1, 0, 2)).astype(np.float32).flatten()

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
        # same for hidden_bias, all three gates) -- a real Keras GRU config
        # (use_bias=False), previously untested. Zero-valued bias is
        # mathematically identical to omitting it, so the golden computation
        # below needs no other change; only the C side needs to actually pass
        # NULL to exercise the guarded branch (done via the `use_bias`
        # template flag, not by zeroing the pointer).
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

        def float_reference(operands):
            return self._generate_gru_expected(
                operands[0],
                update_w_in.astype(np.float32),
                update_w_hidden.astype(np.float32),
                update_b_in.astype(np.float32),
                update_b_hidden.astype(np.float32),
                reset_w_in.astype(np.float32),
                reset_w_hidden.astype(np.float32),
                reset_b_in.astype(np.float32),
                reset_b_hidden.astype(np.float32),
                cand_w_in.astype(np.float32),
                cand_w_hidden.astype(np.float32),
                cand_b_in.astype(np.float32),
                cand_b_hidden.astype(np.float32),
                batch_size=batch_size,
                time_steps=time_steps,
                input_size=input_size,
                hidden_size=hidden_size,
                time_major=time_major,
                reset_after=reset_after,
            )

        output_ref = float_reference([input_tensor])
        output_ref, nonfinite_context = self.apply_nonfinite_policy(
            output_ref, reference=float_reference, inputs=[input_tensor]
        )

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
            h_tpl = "LSTMFunctions/gru_unidirectional/gru_unidirectional.h.j2"
            c_tpl = "LSTMFunctions/gru_unidirectional/gru_unidirectional_fault.c.j2"
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
            h_tpl = "LSTMFunctions/gru_unidirectional/gru_unidirectional.h.j2"
            c_tpl = "LSTMFunctions/gru_unidirectional/gru_unidirectional_stream.c.j2"
        else:
            h_tpl = "LSTMFunctions/gru_unidirectional/gru_unidirectional.h.j2"
            c_tpl = "LSTMFunctions/gru_unidirectional/gru_unidirectional.c.j2"

        self._write_op_outputs(
            Path(output_dir),
            "gru_unidirectional",
            h_tpl,
            c_tpl,
            context,
            {
                "name": name,
                "operator": self.desc.get("operator", "GRUUnidirectional"),
                "operator_name": "gru_unidirectional",
            },
        )
