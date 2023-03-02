import setup  # noqa # isort:skip

from itertools import product
import numpy as np

import tensorflow as tf
from measure import measure_all
from onnx2code.ops.gemm import LoopTilingParams, set_tiling_params

# Custom MNIST-like model
input = tf.keras.Input([4096 * 64])
out = tf.keras.layers.Lambda(lambda x: x)(input)

input_shape = (512, 512)

model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=input_shape),
        tf.keras.layers.Dense(512, activation="relu"),
    ]
)

# nc, kc, mc, mr, nr
nc_options = [4096]
kc_options = [256, 512]
mc_options = [256, 512]
mr_options = [4, 8, 16]
nr_options = [8, 4, 16]

for nc, kc, mc, mr, nr in product(
    nc_options, kc_options, mc_options, mr_options, nr_options
):
    set_tiling_params(LoopTilingParams(nc=nc, kc=kc, mc=mc, mr=mr, nr=nr))

    data = measure_all(model, variations=["loop-tiling"], measure_base=False)

    assert len(data) == 1
    result = data[next(iter(data.keys()))]

    print(f"nc={nc}, kc={kc}, mc={mc}, mr={mr}, nr={nr}: {np.mean(result):.5f} ms")
