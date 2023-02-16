import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers
from measure import measure_all

# Custom MNIST-like model
input = tf.keras.Input([4096 * 64])
out = tf.keras.layers.Lambda(lambda x: x)(input)

input_shape = (256, 256)

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Dense(256, activation="relu"),
    ]
)

# Measure models
data = measure_all(model, variations=["gemm-naive", "loop-tiling", "libxsmm"])

# Plot results
plt.boxplot(data.values(), labels=data.keys())
plt.ylabel("Time (ms)")
plt.title("Inference time of Identity model")

plt.show()
