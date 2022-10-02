import tf2onnx
import tensorflow as tf

from onnx2code.generator import Generator
from onnx2code.model_check import model_check

# Identity test model
inputs = tf.keras.Input([1])
outputs = tf.keras.layers.Lambda(lambda x: x)(inputs)
model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model_proto, _ = tf2onnx.convert.from_keras(model)

# quick test
output = Generator(model_proto).generate()
print(output)
correct = model_check(output, n_inputs=1)
print(f"Model check: {correct}")
