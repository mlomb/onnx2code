import setup  # noqa # isort:skip

import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers
from measure import measure_all

from onnx2code.ops.gemm import LoopTilingParams, set_tiling_params

# Custom MNIST-like model
input = tf.keras.Input([4096 * 64])
out = tf.keras.layers.Lambda(lambda x: x)(input)

input_shape = (512, 512)

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Dense(512, activation="relu"),
    ]
)

set_tiling_params(LoopTilingParams(nc=4096, kc=256, mc=128, mr=4, nr=8))

# Measure models
data = measure_all(model, variations=["gemm-naive", "loop-tiling", "libxsmm"])

# Plot results
plt.boxplot(data.values(), labels=data.keys())
plt.ylabel("Time (ms)")
plt.title("Inference time of Identity model")

plt.show()
