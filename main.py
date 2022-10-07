import numpy as np
import tf2onnx
import tensorflow as tf

from onnx2code.generator import Generator
from onnx2code.service import ModelService
from onnx2code.checker import check_model

# Test model
inputs = tf.keras.Input([3])
outputs = tf.keras.layers.Lambda(lambda x: x + np.array([1, 2, 3]))(inputs)
model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model_proto, _ = tf2onnx.convert.from_keras(model)

# generate
#result = Generator(model_proto).generate()
#print(result)

# model service?
# with ModelService(result) as service:
#    for i in range(10):
#        outputs = service.inference(inputs=[np.array([0, 1, 2], dtype=np.float32) + i])
#        print(outputs)

# quick test
correct = check_model(model_proto, n_inputs=1)
print(f"Model check: {correct}")
