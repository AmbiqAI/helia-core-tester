# Adding a New Operator to CMSIS-NN Tests

This guide explains how to add support for a new operator to the CMSIS-NN test framework. The process involves creating a YAML descriptor, implementing the operator class in the TFLite generator, and ensuring it integrates with the test pipeline.

## Overview

The test framework follows this workflow:

1. **YAML Descriptor** → Defines the operator configuration and test parameters
2. **TFLite Generator** → Creates TensorFlow Lite models from descriptors
3. **TFLite to C Conversion** → Converts models to C inference modules (helia-aot)
4. **Test Runner Generation** → Creates Unity test runners
5. **Build & Run** → Compiles and executes tests on FVP

## Step 1: Create a YAML Descriptor

Create a new YAML file in the `descriptors/` directory. The filename should follow the pattern: `<operator_name>_<dtype>.yaml`

### Descriptor Structure

A descriptor is a YAML file that defines:
- Operator type and name
- Input/output shapes
- Data types (activation and weight)
- Activation function
- Optional hints for code generation

### Example: Simple Single-Input Operator

```yaml
operator: Relu
name: relu_int8
activation_dtype: S8
weight_dtype: S8
activation: NONE
hint:
  call_style: per_tensor
input_shape: [1, 32, 32, 3]
```

### Example: Two-Input Operator

```yaml
operator: Add
name: add_int8
activation_dtype: S8
weight_dtype: S8
activation: NONE
hint:
  call_style: per_tensor
input_1_shape: [1, 32, 32, 3]
input_2_shape: [1, 32, 32, 3]
```

### Example: Convolution Operator

```yaml
operator: Conv2D
name: conv2d_int8
activation_dtype: S8
weight_dtype: S8
activation: NONE
hint:
  call_style: per_tensor
input_shape: [1, 32, 32, 3]
filter_shape: [3, 3, 3, 1]
```

### Required Fields

- `operator`: Operator name (must match enum in schema.json)
- `name`: Unique test name (alphanumeric + underscores, must start with letter)
- `activation_dtype`: `S8` or `S16`
- `weight_dtype`: `S8` or `S4`

### Shape Fields (choose based on operator type)

- Single input: `input_shape: [batch, height, width, channels]`
- Two inputs: `input_1_shape` and `input_2_shape`
- Convolution: `input_shape` and `filter_shape: [height, width, in_channels, out_channels]`

### Optional Fields

- `activation`: `NONE`, `RELU`, `RELU6`, `TANH`, or `SIGMOID`
- `hint`: Object with operator-specific hints:
  - `call_style`: `per_tensor` or `per_channel`
  - `kernel`: Kernel name hint
  - `transpose_weights`: Boolean
  - `extras`: Additional operator-specific parameters

### Valid Operator Names

Check `cmsis_nn_tools/tflite_generator/tester/descriptors/schema.json` for the complete list. Currently supported:
- `FullyConnected`, `Conv2D`, `DepthwiseConv2D`, `MatMul`
- `Add`, `Mul`, `Maximum`, `Minimum`
- `Relu`, `Relu6`, `LeakyRelu`, `Softmax`
- `Quantize`, `Dequantize`
- `Pooling`, `Transpose`, `StridedSlice`, `Pad`
- `LSTM`, `SVDF`, `Mean`, `ReduceMax`, `TransposeConv`

## Step 2: Update Schema (if adding new operator type)

If you're adding a **completely new operator type** (not just a new test case), update the schema:

1. Edit `cmsis_nn_tools/tflite_generator/tester/descriptors/schema.json`
2. Add the operator name to the `operator` enum (line 7)
3. Add a conditional requirement block in `allOf` (lines 72-241) specifying required fields

Example for a new operator:
```json
{
  "if": {
    "properties": {"operator": {"const": "YourNewOperator"}}
  },
  "then": {
    "required": ["input_shape"]  // or whatever fields are needed
  }
}
```

## Step 3: Implement Operator Class

Create a new Python file in `cmsis_nn_tools/tflite_generator/tester/ops/` following the pattern of existing operators.

### Operator Class Template

```python
"""
YourOperator operation implementation.
"""

from typing import Dict, Any
import numpy as np
import tensorflow as tf
from .base import OperationBase


class OpYourOperator(OperationBase):
    """
    YourOperator operation.
    """
    
    def build_keras_model(self) -> tf.keras.Model:
        """Build Keras model for YourOperator operation."""
        # Extract shapes from descriptor
        input_shape = self.desc['input_shape']
        
        # Build model with float32 inputs (will be quantized later)
        input_layer = tf.keras.Input(
            shape=input_shape[1:],  # Remove batch dimension
            dtype=tf.float32,
            name='input'
        )
        
        # Add your operator layer
        x = tf.keras.layers.YourLayer()(input_layer)
        
        # Create and return model
        model = tf.keras.Model(inputs=input_layer, outputs=x)
        return model

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        """Convert Keras model to TFLite with quantization."""
        import tensorflow as tf
        import numpy as np
        
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
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
            ]
            converter.inference_input_type = tf.int16
            converter.inference_output_type = tf.int16
        
        # Generate representative dataset for quantization
        def representative_data_gen():
            rng = np.random.default_rng(rep_seed)
            for _ in range(100):
                # Generate random input data matching descriptor shape
                if 'input_shape' in self.desc:
                    inputs = rng.uniform(-1.0, 1.0, size=self.desc['input_shape']).astype(np.float32)
                    yield [inputs]
                elif 'input_1_shape' in self.desc and 'input_2_shape' in self.desc:
                    inputs1 = rng.uniform(-1.0, 1.0, size=self.desc['input_1_shape']).astype(np.float32)
                    inputs2 = rng.uniform(-1.0, 1.0, size=self.desc['input_2_shape']).astype(np.float32)
                    yield [inputs1, inputs2]
        
        converter.representative_dataset = representative_data_gen
        
        # Convert and save
        tflite_model = converter.convert()
        with open(out_path, 'wb') as f:
            f.write(tflite_model)
```

### Key Points

1. **Inherit from `OperationBase`**: All operators must inherit from `OperationBase`
2. **Implement `build_keras_model()`**: Create the Keras model using TensorFlow layers
3. **Implement `convert_to_tflite()`**: Handle quantization based on `activation_dtype` and `weight_dtype`
4. **Use deterministic seeds**: Use `rep_seed` for reproducible representative datasets
5. **Handle shape extraction**: Support both single-input (`input_shape`) and two-input (`input_1_shape`, `input_2_shape`) patterns

### Examples to Reference

- **Simple activation**: `tester/ops/relu.py` - Single input, no weights
- **Two-input elementwise**: `tester/ops/add.py` - Two inputs, elementwise operation
- **Convolution**: `tester/ops/conv2d.py` - Single input with filter weights
- **Complex operation**: `tester/ops/lstm.py` - Multi-step operation with state

## Step 4: Register Operator in TFLite Generator

Add your operator to the operator mapping in `cmsis_nn_tools/tflite_generator/test_ops.py`:

1. Import your operator class at the top:
```python
from tester.ops.your_operator import OpYourOperator
```

2. Add to `OP_MAP` dictionary:
```python
OP_MAP = {
    # ... existing operators ...
    'YourOperator': OpYourOperator,
}
```

The key must match the `operator` field in your YAML descriptor.

## Step 5: Test Your Implementation

### Test TFLite Generation

Generate TFLite models for your new operator:

```bash
# Generate all tests
python3 cmsis_nn_tools/cli.py

# Generate only your operator
python3 cmsis_nn_tools/cli.py --op YourOperator

# Generate with specific dtype
python3 cmsis_nn_tools/cli.py --op YourOperator --dtype S8
```

This will:
1. Load your YAML descriptor
2. Create a TFLite model using your operator class
3. Save it to `GeneratedTests/<name>/<name>.tflite`

### Verify Generated Model

Check that the model was created:
```bash
ls GeneratedTests/<your_test_name>/
```

### Run Full Pipeline

Run the complete test pipeline:
```bash
python3 cmsis_nn_tools/cli.py --op YourOperator --cpu cortex-m55
```

This will:
1. Generate TFLite model
2. Convert to C module (helia-aot)
3. Generate test runner
4. Build for FVP
5. Run on FVP and generate report

## Step 6: Create Multiple Test Variations

Consider creating multiple descriptor files to test different scenarios:

- Different data types: `your_operator_int8.yaml`, `your_operator_int16.yaml`
- Different shapes: `your_operator_small.yaml`, `your_operator_large.yaml`
- Different activations: `your_operator_relu.yaml`, `your_operator_none.yaml`
- Edge cases: `your_operator_broadcast.yaml`, `your_operator_1x1.yaml`

## Common Patterns

### Pattern 1: Single Input, No Weights (Activations)

```python
def build_keras_model(self) -> tf.keras.Model:
    input_shape = self.desc['input_shape']
    input_layer = tf.keras.Input(shape=input_shape[1:], dtype=tf.float32)
    x = tf.keras.layers.Activation('relu')(input_layer)  # or your activation
    return tf.keras.Model(inputs=input_layer, outputs=x)
```

### Pattern 2: Two Inputs, Elementwise

```python
def build_keras_model(self) -> tf.keras.Model:
    input_1_shape = self.desc['input_1_shape']
    input_2_shape = self.desc['input_2_shape']
    
    input1 = tf.keras.Input(shape=input_1_shape[1:], dtype=tf.float32, name='input1')
    input2 = tf.keras.Input(shape=input_2_shape[1:], dtype=tf.float32, name='input2')
    
    x = tf.keras.layers.Add()([input1, input2])  # or Multiply, Maximum, etc.
    return tf.keras.Model(inputs=[input1, input2], outputs=x)
```

### Pattern 3: Convolution with Weights

```python
def build_keras_model(self) -> tf.keras.Model:
    input_shape = self.desc['input_shape']
    filter_shape = self.desc['filter_shape']
    
    input_layer = tf.keras.Input(shape=input_shape[1:], dtype=tf.float32)
    
    # filter_shape: [height, width, in_channels, out_channels]
    conv = tf.keras.layers.Conv2D(
        filters=filter_shape[3],
        kernel_size=(filter_shape[0], filter_shape[1]),
        strides=(1, 1),
        padding='same'
    )(input_layer)
    
    return tf.keras.Model(inputs=input_layer, outputs=conv)
```

## Troubleshooting

### TFLite Generation Fails

- Check that your operator name matches in YAML and `OP_MAP`
- Verify all required fields are in the descriptor
- Check that shapes are valid (positive integers, correct dimensions)

### Conversion Fails

- Ensure quantization settings match the operator's capabilities
- Check that representative dataset generator matches input shapes
- Verify TensorFlow version supports your operator

### Build Fails

- Check that helia-aot supports your operator
- Verify platform configuration is correct
- Review CMake build errors for missing dependencies

### Test Fails on FVP

- Verify the operator is correctly implemented in CMSIS-NN
- Check that quantization parameters are correct
- Review FVP output for specific error messages

## File Locations Summary

- **Descriptors**: `descriptors/*.yaml`
- **Schema**: `cmsis_nn_tools/tflite_generator/tester/descriptors/schema.json`
- **Operator Classes**: `cmsis_nn_tools/tflite_generator/tester/ops/*.py`
- **Operator Registry**: `cmsis_nn_tools/tflite_generator/test_ops.py`
- **Generated Models**: `GeneratedTests/<name>/<name>.tflite`
- **Generated C Code**: `GeneratedTests/<name>/includes-api/`

## Next Steps

After adding a new operator:

1. **Test thoroughly**: Create multiple test cases covering edge cases
2. **Document**: Add operator to relevant documentation
3. **Validate**: Ensure tests pass on both cortex-m4 and cortex-m55
4. **Review reports**: Check generated HTML/JSON reports for correctness

## Example: Complete Workflow

Here's a complete example adding a hypothetical `Tanh` operator:

1. **Create descriptor** (`descriptors/tanh_int8.yaml`):
```yaml
operator: Tanh
name: tanh_int8
activation_dtype: S8
weight_dtype: S8
activation: NONE
hint:
  call_style: per_tensor
input_shape: [1, 32, 32, 3]
```

2. **Create operator class** (`tester/ops/tanh.py`):
```python
class OpTanh(OperationBase):
    def build_keras_model(self) -> tf.keras.Model:
        input_shape = self.desc['input_shape']
        input_layer = tf.keras.Input(shape=input_shape[1:], dtype=tf.float32)
        x = tf.keras.layers.Activation('tanh')(input_layer)
        return tf.keras.Model(inputs=input_layer, outputs=x)
    
    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        # ... (standard quantization code) ...
```

3. **Register in test_ops.py**:
```python
from tester.ops.tanh import OpTanh
OP_MAP = {
    # ...
    'Tanh': OpTanh,
}
```

4. **Update schema.json** (add `Tanh` to enum and add requirement block)

5. **Test**:
```bash
python3 cmsis_nn_tools/cli.py --op Tanh
```

That's it! Your new operator is now integrated into the test framework.

