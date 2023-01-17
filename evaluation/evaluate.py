import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers
from measure import measure_all

# Custom MNIST-like model
input = tf.keras.Input([4096 * 64])
out = tf.keras.layers.Lambda(lambda x: x)(input)

num_classes = 10
input_shape = (28, 28, 1)

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Dense(100, activation="relu"),
        layers.Dense(100, activation="relu"),
        layers.Dense(100, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

# Measure models
data = measure_all(model)

# Plot results
plt.boxplot(data.values(), labels=data.keys())
plt.ylabel("Time (ms)")
plt.title("Inference time of Identity model")

plt.show()
