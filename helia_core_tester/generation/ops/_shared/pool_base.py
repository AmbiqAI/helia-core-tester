"""Shared implementation for CMSIS-NN avg/max pool operators."""

from typing import Dict
from pathlib import Path
import numpy as np
import tensorflow as tf
from helia_core_tester.generation.ops._shared.base import OperationBase


class PoolFamilyBase(OperationBase):
    """Shared implementation for `AvgPool` and `MaxPool` operators."""

    POOL_KIND = "AVERAGE"
    OPERATOR_NAME = "AvgPool"
    TEMPLATE_DIR = "PoolingFunctions/avg_pool"
    TEMPLATE_SUFFIX = "avg_pool"
    
    def build_keras_model(self) -> tf.keras.Model:
        """Build Keras model for Pooling operation."""
        input_shape = self.desc['input_shape']
        
        # Build model with float32 inputs (will be quantized later)
        inputs = tf.keras.Input(shape=input_shape[1:], dtype=tf.float32, name='input')
        
        # Normalize padding to lowercase
        padding = self.desc.get('padding', 'valid')
        if isinstance(padding, str):
            padding = padding.lower()
        
        if self.POOL_KIND == 'AVERAGE':
            x = tf.keras.layers.AveragePooling2D(
                pool_size=self.desc.get('pool_size', [2, 2]),
                strides=self.desc.get('strides', [2, 2]),
                padding=padding
            )(inputs)
        elif self.POOL_KIND == 'MAX':
            x = tf.keras.layers.MaxPooling2D(
                pool_size=self.desc.get('pool_size', [2, 2]),
                strides=self.desc.get('strides', [2, 2]),
                padding=padding
            )(inputs)
        else:
            raise ValueError(f"Unsupported pooling kind: {self.POOL_KIND}")
            
        model = tf.keras.Model(inputs=inputs, outputs=x)
        return model

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        """Convert Keras model to TFLite with quantization."""
        super().convert_to_tflite(model, out_path, rep_seed)
    
    def _select_cmsis_pooling_kernel(self) -> Dict[str, str]:
        """
        Select appropriate CMSIS-NN pooling kernel function.
        
        Returns:
            Dictionary with kernel function name, C types, and buffer size function
        """
        activation_dtype = self.tensor_dtype("input").upper()
        
        if self.POOL_KIND == 'MAX':
            if activation_dtype == 'S8':
                return {
                    'kernel_fn': 'arm_max_pool_s8',
                    'kernel_get_buffer_size_fn': None,  # Max pooling doesn't need buffer
                    'input_c_type': 'int8_t',
                    'output_c_type': 'int8_t',
                }
            elif activation_dtype == 'S16':
                return {
                    'kernel_fn': 'arm_max_pool_s16',
                    'kernel_get_buffer_size_fn': None,
                    'input_c_type': 'int16_t',
                    'output_c_type': 'int16_t',
                }
            elif activation_dtype == 'FP32':
                return {
                    'kernel_fn': 'arm_max_pool_f32',
                    'kernel_get_buffer_size_fn': None,
                    'input_c_type': 'float',
                    'output_c_type': 'float',
                }
            else:
                raise NotImplementedError(f"Unsupported MaxPool dtype: {activation_dtype}")
        elif self.POOL_KIND == 'AVERAGE':
            if activation_dtype == 'S8':
                return {
                    'kernel_fn': 'arm_avgpool_s8',
                    'kernel_get_buffer_size_fn': 'arm_avgpool_s8_get_buffer_size',
                    'input_c_type': 'int8_t',
                    'output_c_type': 'int8_t',
                }
            elif activation_dtype == 'S16':
                return {
                    'kernel_fn': 'arm_avgpool_s16',
                    'kernel_get_buffer_size_fn': 'arm_avgpool_s16_get_buffer_size',
                    'input_c_type': 'int16_t',
                    'output_c_type': 'int16_t',
                }
            elif activation_dtype == 'FP32':
                return {
                    'kernel_fn': 'arm_avg_pool_f32',
                    'kernel_get_buffer_size_fn': None,
                    'input_c_type': 'float',
                    'output_c_type': 'float',
                }
            else:
                raise NotImplementedError(f"Unsupported AvgPool dtype: {activation_dtype}")
        else:
            raise ValueError(f"Unsupported pooling kind: {self.POOL_KIND}")
    
    def generate_c_files(self, output_dir: Path) -> None:
        """
        Generate C and H files from templates for Pooling operation.
        """
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder
        
        name = self.desc['name']
        tflite_path = output_dir / f"{name}.tflite"
        if not tflite_path.exists():
            raise FileNotFoundError(f"TFLite file not found: {tflite_path}")
        
        kernel_info = self._select_cmsis_pooling_kernel()
        pooling_type = self.POOL_KIND
        
        op_tensors = self.load_primary_operator_tensors(str(tflite_path))
        
        # Extract shapes from LiteRT
        input_shape = op_tensors['inputs'][0]['shape']
        output_shape = op_tensors['outputs'][0]['shape']
        
        # Ensure shapes are tuples
        if input_shape is not None:
            input_shape = tuple(input_shape)
        if output_shape is not None:
            output_shape = tuple(output_shape)
        
        # Extract quantization from LiteRT
        quant_params = {
            'input': op_tensors['inputs'][0]['quantization'],
            'output': op_tensors['outputs'][0]['quantization']
        }
        
        builder = TemplateContextBuilder()
        
        # Convert shapes to CMSIS dims
        input_dims = builder.nhwc_to_cmsis_dims(input_shape)
        output_dims = builder.nhwc_to_cmsis_dims(output_shape)
        
        # Filter dims for pooling: pool_size from descriptor
        pool_size = self.desc.get('pool_size', [2, 2])
        if isinstance(pool_size, (int, float)):
            pool_h = pool_w = int(pool_size)
        else:
            pool_h = int(pool_size[0])
            pool_w = int(pool_size[1])
        
        filter_dims = {
            'n': 1,
            'h': pool_h,
            'w': pool_w,
            'c': 1
        }
        
        # Build pool parameters
        pool_params = builder.build_pool_params(
            self.desc,
            input_shape,
            (pool_h, pool_w),
            output_shape,
            quant_params['output']
        )
        
        input_data = self.generate_input_data()
        
        if kernel_info["input_c_type"] == "float":
            input_q = input_data.astype(np.float32)
        else:
            input_scale = float(self._quant_param_scalar(quant_params['input'], 'scale', 1.0))
            input_zp = int(self._quant_param_scalar(quant_params['input'], 'zero_point', 0))

        if kernel_info["input_c_type"] == "int8_t":
            qmin, qmax = -128, 127
            np_in_dtype = np.int8
        elif kernel_info["input_c_type"] == "int16_t":
            qmin, qmax = -32768, 32767
            np_in_dtype = np.int16
        elif kernel_info["input_c_type"] != "float":
            raise ValueError(f"Unsupported input_c_type: {kernel_info['input_c_type']}")
        
        if kernel_info["input_c_type"] != "float":
            input_q = np.round(input_data / float(input_scale) + float(input_zp)).astype(np.int32)
            input_q = np.clip(input_q, qmin, qmax).astype(np_in_dtype)
        
        output_data = self.run_inference(str(tflite_path), input_q)
        
        # Format input and output arrays
        input_data_array_str = builder.format_array_as_c_literal(input_q)
        expected_output_array_str = builder.format_array_as_c_literal(output_data)
        
        # Calculate buffer size max
        activation_dtype = self.tensor_dtype("input")
        buffer_size_max = builder.calculate_pooling_buffer_size_max(
            input_dims,
            output_dims,
            pooling_type=pooling_type,
            output_dtype=activation_dtype
        )
        
        # Build template context
        context = {
            'name': name,
            'prefix': name,
            'input_dims': input_dims,
            'filter_dims': filter_dims,
            'output_dims': output_dims,
            'pool_params': pool_params,
            'input_data_array': input_data_array_str,
            'expected_output_array': expected_output_array_str,
            'input_dtype': kernel_info["input_c_type"],
            'output_dtype': kernel_info["output_c_type"],
            'kernel_fn': kernel_info["kernel_fn"],
            'kernel_get_buffer_size_fn': kernel_info["kernel_get_buffer_size_fn"],
            'buffer_size_max': buffer_size_max,
            'pooling_type': pooling_type,
            'pool_params_type': 'cmsis_nn_pool_params_f32' if kernel_info["input_c_type"] == "float" else 'cmsis_nn_pool_params',
            'float_kernel': kernel_info["input_c_type"] == "float",
        }
        if kernel_info["input_c_type"] == "float":
            context["pool_activation_min_literal"] = builder.format_float_literal(pool_params["activation_min"])
            context["pool_activation_max_literal"] = builder.format_float_literal(pool_params["activation_max"])
        
        cmake_context = {
            'name': name,
            'operator': self.desc.get('operator', self.OPERATOR_NAME),
            'operator_name': self.TEMPLATE_SUFFIX,
        }
        self._write_op_outputs(
            output_dir,
            self.TEMPLATE_SUFFIX,
            f"{self.TEMPLATE_DIR}/{self.TEMPLATE_SUFFIX}.h.j2",
            f"{self.TEMPLATE_DIR}/{self.TEMPLATE_SUFFIX}.c.j2",
            context,
            cmake_context,
        )
        
