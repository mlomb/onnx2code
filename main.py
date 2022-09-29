import tf2onnx
import tensorflow as tf
from onnx2code.generator import Generator

inputs = tf.keras.Input([1])
outputs = tf.keras.layers.Lambda(lambda x: x)(inputs)
model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

model_proto, _ = tf2onnx.convert.from_keras(model)

Generator(model_proto).generate()
