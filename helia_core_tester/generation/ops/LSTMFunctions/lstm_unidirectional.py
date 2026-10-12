"""LSTMUnidirectional operation implementation."""

from pathlib import Path

import numpy as np

from helia_core_tester.generation.ops._shared import kernel_faithful as kf
from helia_core_tester.generation.ops._shared.base import OperationBase
from helia_core_tester.generation.ops._shared.recurrent_pool import recurrent_argument_pool
from helia_core_tester.generation.reference.call import ReferenceCall

LSTM_FRAGMENTS = "LSTMFunctions/lstm_unidirectional"


class OpLSTMUnidirectional(OperationBase):
    """LSTMUnidirectional operation."""

    FAULT_KINDS = ("null_input", "null_output", "null_params", "null_buffers")

    def uses_reference(self) -> bool:
        return not self.status_only()

    def status_only(self) -> bool:
        return self.fault_kind() is not None and self.expected_status() != "ARM_CMSIS_NN_SUCCESS"

    def _status_aware_golden(self, call: ReferenceCall) -> np.ndarray:
        """The reference output; recorded unless the case only asserts a status (its header still
        carries the array, never compared)."""
        return call.output() if self.status_only() else self.reference_golden(call)

    def generate_c_files(self, output_dir) -> None:
        """
        Generate C and H files for CMSIS-only LSTM tests.
        """
        if (
            self.desc.get("hint", {}).get("force_cmsis", False)
            and str(self.desc.get("activation_dtype", "S8")).upper() in {"FP32", "FP16"}
        ):
            from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

            name = self.desc['name']
            activation_dtype = str(self.desc.get("activation_dtype", "S8")).upper()
            float_dtype = np.float16 if activation_dtype == "FP16" else np.float32
            data_dtype = "float16_t" if activation_dtype == "FP16" else "float"
            kernel_fn = "arm_lstm_unidirectional_f16" if activation_dtype == "FP16" else "arm_lstm_unidirectional_f32"
            lstm_params_type = "cmsis_nn_lstm_params_f16" if activation_dtype == "FP16" else "cmsis_nn_lstm_params_f32"
            lstm_context_type = "cmsis_nn_lstm_context_f16" if activation_dtype == "FP16" else "cmsis_nn_lstm_context_f32"
            batch_size = int(self.desc.get("batch_size", 1))
            time_steps = int(self.desc.get("time_steps", 1))
            input_size = int(self.desc.get("feature_size", self.desc.get("input_size", 1)))
            hidden_size = int(self.desc.get("units", self.desc.get("hidden_size", 1)))
            time_major = bool(self.desc.get("time_major", False))
            cell_clip = float(self.desc.get("cell_clip", 0.0))
            # Issue #56: use_bias: false exercises the NULL-safe bias branch
            # in arm_nn_lstm_step_*.c (`gate->bias ? gate->bias[h_idx] :
            # 0.0f`, all four gates), a real LSTM layer config, previously
            # untested. Mirrors the GRU use_bias handling.
            use_bias = bool(self.desc.get("use_bias", True))

            rng_state = self.rng.__getstate__()
            self.rng = np.random.default_rng(self.seed)
            if time_major:
                input_tensor = self.rng.uniform(-1.0, 1.0, size=(time_steps, batch_size, input_size)).astype(float_dtype)
            else:
                input_tensor = self.rng.uniform(-1.0, 1.0, size=(batch_size, time_steps, input_size)).astype(float_dtype)
            # The weight draws below continue this same stream, so the sweep is applied
            # to the tensor already drawn rather than resampled through _sample_uniform,
            # which restarts the stream and would move every weight.
            input_tensor = self._maybe_apply_input_mode(input_tensor)
            forget_w_in = self.rng.uniform(-0.5, 0.5, size=(hidden_size, input_size)).astype(float_dtype)
            input_w_in = self.rng.uniform(-0.5, 0.5, size=(hidden_size, input_size)).astype(float_dtype)
            cell_w_in = self.rng.uniform(-0.5, 0.5, size=(hidden_size, input_size)).astype(float_dtype)
            output_w_in = self.rng.uniform(-0.5, 0.5, size=(hidden_size, input_size)).astype(float_dtype)
            forget_w_hidden = self.rng.uniform(-0.5, 0.5, size=(hidden_size, hidden_size)).astype(float_dtype)
            input_w_hidden = self.rng.uniform(-0.5, 0.5, size=(hidden_size, hidden_size)).astype(float_dtype)
            cell_w_hidden = self.rng.uniform(-0.5, 0.5, size=(hidden_size, hidden_size)).astype(float_dtype)
            output_w_hidden = self.rng.uniform(-0.5, 0.5, size=(hidden_size, hidden_size)).astype(float_dtype)
            if use_bias:
                forget_bias = self.rng.uniform(-0.25, 0.25, size=(hidden_size,)).astype(float_dtype)
                input_bias = self.rng.uniform(-0.25, 0.25, size=(hidden_size,)).astype(float_dtype)
                cell_bias = self.rng.uniform(-0.25, 0.25, size=(hidden_size,)).astype(float_dtype)
                output_bias = self.rng.uniform(-0.25, 0.25, size=(hidden_size,)).astype(float_dtype)
            else:
                # Zero-valued bias is mathematically identical to omitting
                # it, so the golden below needs no other change; the C side
                # separately passes NULL (not a zero array) to actually
                # exercise the guarded branch -- see the `use_bias` template
                # flag below.
                forget_bias = np.zeros((hidden_size,), dtype=float_dtype)
                input_bias = np.zeros((hidden_size,), dtype=float_dtype)
                cell_bias = np.zeros((hidden_size,), dtype=float_dtype)
                output_bias = np.zeros((hidden_size,), dtype=float_dtype)
            self.rng.__setstate__(rng_state)

            entry = "lstm_f16" if activation_dtype == "FP16" else "lstm_f32"
            gate_order = ((forget_w_in, forget_w_hidden, forget_bias), (input_w_in, input_w_hidden, input_bias),
                          (cell_w_in, cell_w_hidden, cell_bias), (output_w_in, output_w_hidden, output_bias))
            stacked = {
                "input_weights": np.stack([g[0] for g in gate_order]).astype(float_dtype),
                "recurrent_weights": np.stack([g[1] for g in gate_order]).astype(float_dtype),
                "bias": (np.stack([g[2] for g in gate_order]).astype(float_dtype) if use_bias
                         else np.zeros((0,), dtype=float_dtype)),
            }
            shape_io = (time_steps, batch_size) if time_major else (batch_size, time_steps)
            operands = {"input": np.ascontiguousarray(input_tensor.reshape(*shape_io, input_size)), **stacked}
            faithful = activation_dtype == "FP32"
            if faithful:
                # The kernel's own activation tables and operation order on this target.
                call = ReferenceCall("lstm_f32_cmsis", {"time_major": int(time_major), "cell_clip": cell_clip,
                                                        "target": kf.float_target(self.target_cpu)},
                                     operands, {"output": (*shape_io, hidden_size)})
                kf.check_close_to_exact(name, call, ReferenceCall(
                    entry, {"time_major": int(time_major), "cell_clip": cell_clip}, operands,
                    {"output": (*shape_io, hidden_size)}), atol=1e-4)
            else:
                call = ReferenceCall(entry, {"time_major": int(time_major), "cell_clip": cell_clip}, operands,
                                     {"output": (*shape_io, hidden_size)})
            output_ref = self._status_aware_golden(call).reshape(-1)
            output_ref, nonfinite_context = self.apply_nonfinite_policy(
                output_ref,
                reference=lambda ops: call.with_inputs(input=np.ascontiguousarray(
                    np.asarray(ops[0], dtype=float_dtype).reshape(*shape_io, input_size))).output().reshape(-1),
                inputs=[input_tensor],
            )

            builder = TemplateContextBuilder()
            context = {
                "name": name,
                "data_dtype": data_dtype,
                "output_dtype": data_dtype,
                "kernel_fn": kernel_fn,
                "lstm_params_type": lstm_params_type,
                "lstm_context_type": lstm_context_type,
                "time_major_literal": "1" if time_major else "0",
                "batch_size": batch_size,
                "time_steps": time_steps,
                "input_size": input_size,
                "hidden_size": hidden_size,
                "cell_clip_literal": builder.format_float_literal(cell_clip),
                "input_tensor_array": builder.format_array_as_c_literal(input_tensor),
                "output_array": builder.format_array_as_c_literal(output_ref.astype(float_dtype)),
                "forget_gate_input_weights_array": builder.format_array_as_c_literal(forget_w_in),
                "input_gate_input_weights_array": builder.format_array_as_c_literal(input_w_in),
                "cell_gate_input_weights_array": builder.format_array_as_c_literal(cell_w_in),
                "output_gate_input_weights_array": builder.format_array_as_c_literal(output_w_in),
                "forget_gate_hidden_weights_array": builder.format_array_as_c_literal(forget_w_hidden),
                "input_gate_hidden_weights_array": builder.format_array_as_c_literal(input_w_hidden),
                "cell_gate_hidden_weights_array": builder.format_array_as_c_literal(cell_w_hidden),
                "output_gate_hidden_weights_array": builder.format_array_as_c_literal(output_w_hidden),
                "forget_gate_bias_array": builder.format_array_as_c_literal(forget_bias),
                "input_gate_bias_array": builder.format_array_as_c_literal(input_bias),
                "cell_gate_bias_array": builder.format_array_as_c_literal(cell_bias),
                "output_gate_bias_array": builder.format_array_as_c_literal(output_bias),
                "cell_state_size": batch_size * hidden_size,
                "dst_size": batch_size * time_steps * hidden_size,
                "use_bias": use_bias,
            }
            if faithful:
                context.update(kf.tolerance_context(np.float32))
            context.update(nonfinite_context)

            # ns-cmsis-nn#377 / tester#71: generation-time feature probe for
            # the float LSTM temp-buffer sizers added by ns-cmsis-nn#381
            # (they report 0 -- the float steps never touch temp1/temp2).
            # Detected -> the main template calls and validates the sizers
            # and passes NULL temp buffers; absent -> byte-identical legacy
            # output (safe against ns-cmsis-nn main).
            from helia_core_tester.generation.utils.temp_sizer_probe import detect_temp_sizers

            context["temp_sizers_available"] = detect_temp_sizers(
                [
                    f"{kernel_fn}_temp1_get_buffer_size",
                    f"{kernel_fn}_temp2_get_buffer_size",
                ],
                f"LSTMUnidirectional[{name}]",
            )

            # Issue #56: port of the GRU fault: mechanism
            # (gru_unidirectional.py) -- LSTM previously had no
            # argument-validation coverage at all.
            fault = self.fault_kind()
            # Follow-up to #56: port of the GRU stream: mechanism -- LSTM
            # previously had zero hidden_state/cell_state streaming coverage.
            # See arm_lstm_unidirectional_f32.c: cell_state is caller-owned
            # state too when hidden_state != NULL (not just hidden_state, as
            # for GRU), so the stream template seeds and preserves both.
            stream = bool(self.desc.get("hint", {}).get("stream", False))
            if fault:
                context.update(self.fault_context())
                variant = "lstm_unidirectional_fault"
            elif stream:
                if batch_size != 1:
                    raise ValueError("LSTMUnidirectional streaming descriptors require batch_size == 1.")
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
                variant = "lstm_unidirectional_stream"
            else:
                variant = "lstm_unidirectional_f32"

            self.render_harness_case(
                Path(output_dir), stem="lstm_unidirectional", context=context,
                pool=recurrent_argument_pool(
                    context, body=f"{LSTM_FRAGMENTS}/{variant}.fragment.j2",
                    header=f"{LSTM_FRAGMENTS}/lstm_unidirectional_f32.fragment.j2", ctype=context["data_dtype"]),
                validation_key=f"{LSTM_FRAGMENTS}/{variant}.c.j2", label="LSTM", operator="LSTMUnidirectional",
                sidecar=True,
            )
            return

        if not self.desc.get("hint", {}).get("force_cmsis", False):
            return

        from helia_core_tester.generation.ops._shared.lstm_reference import lstm_int_case
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

        name = self.desc['name']
        dataset = self.desc.get("dataset")
        if not dataset:
            raise ValueError("LSTM CMSIS test requires 'dataset' in descriptor.")

        activation_dtype = self.desc.get("activation_dtype", "S8").upper()
        batch_size = int(self.desc.get("batch_size", 1))
        time_steps = int(self.desc.get("time_steps", 1))
        input_size = int(self.desc.get("feature_size", self.desc.get("input_size", 1)))
        hidden_size = int(self.desc.get("units", self.desc.get("hidden_size", 1)))
        time_major = bool(self.desc.get("time_major", False))
        extras = self.desc.get("hint", {}).get("extras", {})
        kind = "s16" if activation_dtype == "S16" else "s8"

        case = lstm_int_case(np.random.default_rng(self.seed), kind=kind, batch=batch_size, steps=time_steps,
                             input_size=input_size, hidden=hidden_size, time_major=time_major, extras=extras,
                             cell_scale_power=self.desc.get("cell_scale_power"))
        output = self.reference_golden(ReferenceCall(f"lstm_{kind}", case.params, case.inputs,
                                                     {"output": case.output_shape}, quant=case.context["scales"]))
        data_tensors = dict(case.context["tensors"], output=output)
        context = {
            "name": name,
            "dataset": dataset,
            "macro_prefix": dataset.upper() + "_",
            "data_prefix": dataset.lower() + "_",
            "dtype": kind,
            "output_dtype": "int16_t" if kind == "s16" else "int8_t",
            "time_major": int(time_major),
            "batch_size": batch_size,
            "time_steps": time_steps,
            "input_size": input_size,
            "hidden_size": hidden_size,
            "input_zero_point": case.context["input_zero_point"],
            "output_zero_point": case.context["output_zero_point"],
            "cell_scale_power": case.context["cell_scale_power"],
            "cell_clip": case.context["cell_clip"],
            "mult_shift": case.context["mult_shift"],
            "tensors": data_tensors,
        }

        class _Data:
            tensors = data_tensors

        data = _Data()

        builder = TemplateContextBuilder()

        def fmt(arr, *, dtype=None):
            if dtype is not None:
                return builder.format_array_as_c_literal(np.asarray(arr, dtype=dtype))
            return builder.format_array_as_c_literal(np.asarray(arr))

        input_dtype = "int16_t" if activation_dtype == "S16" else "int8_t"
        output_dtype = "int16_t" if activation_dtype == "S16" else "int8_t"
        bias_dtype = "int64_t" if activation_dtype == "S16" else "int32_t"
        weight_dtype = "int8_t"

        bias_cast = np.int64 if activation_dtype == "S16" else None
        context.update({
            "time_major_literal": "true" if time_major else "false",
            "input_dtype": input_dtype,
            "output_dtype": output_dtype,
            "bias_dtype": bias_dtype,
            "weight_dtype": weight_dtype,
            "input_tensor_array": fmt(data.tensors["input_tensor"]),
            "output_array": fmt(data.tensors["output"]),
            "input_gate_input_weights_array": fmt(data.tensors["input_gate_input_weights"]),
            "forget_gate_input_weights_array": fmt(data.tensors["forget_gate_input_weights"]),
            "cell_gate_input_weights_array": fmt(data.tensors["cell_gate_input_weights"]),
            "output_gate_input_weights_array": fmt(data.tensors["output_gate_input_weights"]),
            "input_gate_hidden_weights_array": fmt(data.tensors["input_gate_hidden_weights"]),
            "forget_gate_hidden_weights_array": fmt(data.tensors["forget_gate_hidden_weights"]),
            "cell_gate_hidden_weights_array": fmt(data.tensors["cell_gate_hidden_weights"]),
            "output_gate_hidden_weights_array": fmt(data.tensors["output_gate_hidden_weights"]),
            "input_gate_bias_array": fmt(data.tensors["input_gate_bias"], dtype=bias_cast),
            "forget_gate_bias_array": fmt(data.tensors["forget_gate_bias"], dtype=bias_cast),
            "cell_gate_bias_array": fmt(data.tensors["cell_gate_bias"], dtype=bias_cast),
            "output_gate_bias_array": fmt(data.tensors["output_gate_bias"], dtype=bias_cast),
        })

        # ns-cmsis-nn#377 / tester#71: generation-time feature probe for the
        # integer LSTM temp-buffer sizers added by ns-cmsis-nn#381. Detected
        # -> the template sizes temp1/temp2 from the sizer contract
        # ((time_major ? batch_size : 1) * hidden_size * sizeof(int16_t))
        # and validates the sizers on-target; absent -> byte-identical legacy
        # output (safe against ns-cmsis-nn main).
        from helia_core_tester.generation.utils.temp_sizer_probe import (
            detect_temp_sizers,
            lstm_int_temp_expected_bytes,
        )

        dtype_suffix = "s16" if activation_dtype == "S16" else "s8"
        context["temp_sizers_available"] = detect_temp_sizers(
            [
                f"arm_lstm_unidirectional_{dtype_suffix}_temp1_get_buffer_size",
                f"arm_lstm_unidirectional_{dtype_suffix}_temp2_get_buffer_size",
            ],
            f"LSTMUnidirectional[{name}]",
        )
        context["lstm_temp_expected_bytes"] = lstm_int_temp_expected_bytes(
            time_major=time_major, batch_size=batch_size, hidden_size=hidden_size
        )
        context["lstm_temp_expected_bytes_flipped"] = lstm_int_temp_expected_bytes(
            time_major=not time_major, batch_size=batch_size, hidden_size=hidden_size
        )

        context.setdefault("kernel_fn", "arm_lstm_unidirectional_s16" if activation_dtype == "S16" else "arm_lstm_unidirectional_s8")
        self.render_harness_case(
            Path(output_dir), stem="lstm_unidirectional", context=context,
            pool=recurrent_argument_pool(
                context, body=f"{LSTM_FRAGMENTS}/lstm_unidirectional.fragment.j2",
                header=f"{LSTM_FRAGMENTS}/lstm_unidirectional.fragment.j2", ctype=context["output_dtype"]),
            validation_key=f"{LSTM_FRAGMENTS}/lstm_unidirectional.c.j2", label="LSTM", operator="LSTMUnidirectional",
        )
