import setup  # noqa # isort:skip

import keras
import matplotlib.pyplot as plt
import numpy as np
from keras import layers
from measure import measure_all

SIZES = np.arange(1, 50) ** 2
VARIATIONS = ["gemm-naive", "loop-tiling", "libxsmm"]

results: dict[str, list[float]] = {}

for x in SIZES:
    model = keras.Sequential(
        [
            keras.Input(shape=(x, x)),
            layers.Dense(x, activation="relu"),
        ]
    )

    print(x)
    result = measure_all(model, variations=VARIATIONS, runs=5)

    for var, times in result.items():
        results.setdefault(var, []).append(np.mean(times))

# plot
for var, times in results.items():
    plt.plot(SIZES, times, label=var)

plt.legend()
plt.show()
