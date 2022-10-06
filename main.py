import numpy as np
import tf2onnx
import tensorflow as tf

from onnx2code.generator import Generator
from onnx2code.model_check import model_check
from onnx2code.service import ModelService

# Test model
inputs = tf.keras.Input([3])
outputs = tf.keras.layers.Lambda(lambda x: x + np.array([1, 2, 3]))(inputs)
model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model_proto, _ = tf2onnx.convert.from_keras(model)

# generate
output = Generator(model_proto).generate()
print(output)

# model service?
with ModelService(output) as service:
    for i in range(10):
        outputs = service.inference(
            inputs=[np.array([0, 1, 2], dtype=np.float32) + i]
        )
        print(outputs)

# quick test
correct = model_check(output, n_inputs=1)
print(f"Model check: {correct}")
