import setup  # noqa # isort:skip

from itertools import product

import numpy as np
import tensorflow as tf
from measure import measure_all

from onnx2code.ops.gemm import LoopTilingParams, set_tiling_params

FLOAT_SIZE = 4
KB = 1024
L1_SIZE = 32 * KB
L2_SIZE = 256 * KB

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
kc_options = [256]
mc_options = [256]
mr_options = [4]
nr_options = [8]
mv_options = [4]
nu_options = [4]

for nc, kc, mc, mr, nr, mv, nu in product(
    nc_options, kc_options, mc_options, mr_options, nr_options, mv_options, nu_options
):
    set_tiling_params(LoopTilingParams(nc=nc, kc=kc, mc=mc, mr=mr, nr=nr, mv=mv, nu=nu))
    print(f"\n## nc={nc}, kc={kc}, mc={mc}, mr={mr}, nr={nr}\n")

    B_sliver = nr * kc * FLOAT_SIZE
    A_sliver = mr * kc * FLOAT_SIZE
    AB = mr * nr * FLOAT_SIZE
    L1_total = A_sliver + B_sliver + AB
    L1_remaining = L1_SIZE - L1_total
    print("L1:")
    print(f"\t{A_sliver=}")
    print(f"\t{B_sliver=}")
    print(f"\t{AB=}")
    print(f"\t{L1_total=}")
    print(f"\t{L1_remaining=}")

    A_panel = mc * kc * FLOAT_SIZE
    C_writeback = AB
    L2_total = A_panel + B_sliver + C_writeback
    L2_remaining = L2_SIZE - L2_total
    print("\nL2:")
    print(f"\t{A_panel=}")
    print(f"\t{B_sliver=}")
    print(f"\t{C_writeback=}")
    print(f"\t{L2_total=}")
    print(f"\t{L2_remaining=}")

    data = measure_all(model, variations=["loop-tiling"], measure_base=False)

    assert len(data) == 1
    result = data[next(iter(data.keys()))]

    print(f"result: {np.mean(result):.2f}ms")
