from measure import measure_all

import tensorflow as tf

import keras
from keras import layers

import matplotlib.pyplot as plt

# Custom MNIST-like model
input = tf.keras.Input([4096 * 64])
out = tf.keras.layers.Lambda(lambda x: x)(input)

num_classes = 10
input_shape = (28, 28, 1)

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
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
