"""Preserve explicitly batched signatures for batch-aware generators."""

import tensorflow as tf


def converter_for_batched_model(model, input_shapes):
    if all(shape[0] == 1 for shape in input_shapes):
        return tf.lite.TFLiteConverter.from_keras_model(model)

    # Keras export makes even fixed leading dimensions dynamic. Freeze the
    # explicit signature so variables remain constant weights in LiteRT.
    from tensorflow.python.framework.convert_to_constants import (
        convert_variables_to_constants_v2,
    )

    signature = [
        tf.TensorSpec(shape, dtype=tensor.dtype, name=tensor.name)
        for shape, tensor in zip(input_shapes, model.inputs, strict=True)
    ]

    @tf.function(input_signature=signature)
    def serve(*inputs):
        return model(inputs[0] if len(inputs) == 1 else list(inputs), training=False)

    frozen = convert_variables_to_constants_v2(serve.get_concrete_function())
    return tf.lite.TFLiteConverter.from_concrete_functions([frozen], model)
