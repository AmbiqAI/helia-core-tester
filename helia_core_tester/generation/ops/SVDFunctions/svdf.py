"""
SVDF operation implementation.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import numpy as np
import tensorflow as tf
from helia_core_tester.generation.ops._shared.base import OperationBase


class OpSVDF(OperationBase):
    """
    SVDF operation.
    """
    
    def build_keras_model(self) -> tf.keras.Model:
        """Build Keras model for SVDF operation."""
        input_shape = self.desc['input_shape']
        inputs = tf.keras.Input(shape=input_shape[1:], dtype=tf.float32, name='input')
        x = tf.keras.layers.Dense(units=64)(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=x)
        return model

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        """Convert Keras model to TFLite with quantization."""
        # Create converter
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Apply quantization based on activation_dtype
        activation_dtype = self.desc.get('activation_dtype', 'S8')
        
        if activation_dtype == 'S8':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.int8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        elif activation_dtype == 'S16':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.int16]
            # For int16 quantization, keep input/output as float32
            # For int16 quantization, keep input/output as float32
        
        # Generate representative dataset
        def representative_data_gen():
            for _ in range(100):
                if 'input_shape' in self.desc:
                    inputs = self.rng.uniform(-1.0, 1.0, size=self.desc['input_shape']).astype(np.float32)
                    yield [inputs]
                elif 'input_1_shape' in self.desc and 'input_2_shape' in self.desc:
                    inputs1 = self.rng.uniform(-1.0, 1.0, size=self.desc['input_1_shape']).astype(np.float32)
                    inputs2 = self.rng.uniform(-1.0, 1.0, size=self.desc['input_2_shape']).astype(np.float32)
                    yield [inputs1, inputs2]
        
        converter.representative_dataset = representative_data_gen
        
        # Convert and save
        tflite_model = converter.convert()
        with open(out_path, 'wb') as f:
            f.write(tflite_model)

    def needs_keras_model(self) -> bool:
        if self.desc.get("hint", {}).get("force_cmsis", False):
            return False
        return True

    def allow_no_tflite(self) -> bool:
        if self.desc.get("hint", {}).get("force_cmsis", False):
            return True
        return False

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

        if str(self.desc.get("activation_dtype", "S8")).upper() == "FP32":
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
            input_sequence = self.rng.uniform(-1.0, 1.0, size=(sequence_steps, input_batches, input_height)).astype(np.float32)
            state_init = self.rng.uniform(-0.5, 0.5, size=(input_batches, feature_batches, time_batches)).astype(np.float32)
            weights_feature = self.rng.uniform(-1.0, 1.0, size=(feature_batches, input_height)).astype(np.float32)
            weights_time = self.rng.uniform(-1.0, 1.0, size=(feature_batches, time_batches)).astype(np.float32)
            bias = self.rng.uniform(-0.5, 0.5, size=(unit_count,)).astype(np.float32) if use_bias else None
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
            expected_output = self._generate_svdf_expected_f32(
                input_sequence=input_sequence,
                state_init=state_init,
                weights_feature=weights_feature,
                weights_time=weights_time,
                bias=bias,
                params=params,
            )

            builder = TemplateContextBuilder()
            context = {
                "name": name,
                "prefix": name,
                "kernel_fn": "arm_svdf_f32",
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
                "output_ref_array": builder.format_array_as_c_literal(expected_output.astype(np.float32)),
                "bias_array": builder.format_array_as_c_literal(bias) if bias is not None else "",
                "input_activation_min_literal": builder.format_float_literal(input_activation_min),
                "input_activation_max_literal": builder.format_float_literal(input_activation_max),
                "output_activation_min_literal": builder.format_float_literal(output_activation_min),
                "output_activation_max_literal": builder.format_float_literal(output_activation_max),
            }
            self._write_op_outputs(
                output_dir,
                "svdf",
                "SVDFunctions/svdf/svdf_f32.h.j2",
                "SVDFunctions/svdf/svdf_f32.c.j2",
                context,
                {
                    "name": name,
                    "operator": self.desc.get("operator", "SVDF"),
                    "operator_name": "svdf",
                },
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
            bias_size = feature_batches if unit_count == feature_batches else unit_count
            bias = self.rng.integers(-25, 26, size=(bias_size,), dtype=np.int32)
        else:
            bias = None

        self.rng.__setstate__(rng_state)

        params = {
            "input_batches": input_batches,
            "input_height": input_height,
            "feature_batches": feature_batches,
            "time_batches": time_batches,
            "rank": rank,
            "input_multiplier": input_multiplier,
            "input_shift": input_shift,
            "output_multiplier": output_multiplier,
            "output_shift": output_shift,
            "input_activation_min": input_activation_min,
            "input_activation_max": input_activation_max,
            "output_activation_min": output_activation_min,
            "output_activation_max": output_activation_max,
            "input_offset": input_offset,
            "output_offset": output_offset,
        }

        expected_output = self._generate_svdf_expected(
            input_data=input_data,
            state_init=state_init,
            weights_feature=weights_feature,
            weights_time=weights_time,
            bias=bias,
            params=params,
            state_dtype=state_dtype,
        )

        builder = TemplateContextBuilder()
        context = {
            "name": name,
            "prefix": name,
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
        }

        includes_api_dir = output_dir / "includes"
        includes_api_dir.mkdir(parents=True, exist_ok=True)

        h_content = self.render_template("SVDFunctions/svdf/svdf.h.j2", context)
        h_path = includes_api_dir / f"{name}_svdf.h"
        h_path.write_text(h_content)

        c_content = self.render_template("SVDFunctions/svdf/svdf.c.j2", context)
        c_path = output_dir / f"{name}_svdf.c"
        c_path.write_text(c_content)

        cmake_context = {
            "name": name,
            "operator": self.desc.get("operator", "SVDF"),
            "operator_name": "svdf",
        }
        cmake_content = self.render_template("common/CMakeLists.txt.j2", cmake_context)
        cmake_path = output_dir / "CMakeLists.txt"
        cmake_path.write_text(cmake_content)
