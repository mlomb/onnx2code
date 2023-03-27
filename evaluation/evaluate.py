import setup  # noqa # isort:skip

import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers
from measure import measure_all

from onnx2code.ops.gemm_tiling.GEMM import LoopTilingParams, set_tiling_params

# Custom MNIST-like model
input = tf.keras.Input([4096 * 64])
out = tf.keras.layers.Lambda(lambda x: x)(input)

input_shape = (512, 512)

M = 4
K = 16
N = 64009


model = keras.Sequential(
    [
        keras.Input(shape=(M, K)),
        layers.Dense(N, activation="relu"),
    ]
)

set_tiling_params(LoopTilingParams(nc=4096, kc=256, mc=128, mr=4, nr=8, mv=4, nu=4))

# Measure models
data = measure_all(model, variations=["loop-tiling", "gemm-naive"])

# Plot results
plt.boxplot(data.values(), labels=data.keys())
plt.ylabel("Time (ms)")
plt.title("Inference time of Identity model")

plt.show()
