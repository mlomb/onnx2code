import setup  # noqa # isort:skip

import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product

import tensorflow as tf
from measure import measure_all

from onnx2code.ops.gemm_tiling.GEMM import LoopTilingParams, set_tiling_params

N = 512
input_shape = (N, N)

model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=input_shape),
        tf.keras.layers.Dense(N, activation=None, use_bias=False),
    ]
)

# nc, kc, mc, mr, nr
nc_options = [N]
kc_options = [64, 128, 256, 512]
mc_options = [64, 128, 256, 512]
mr_options = [2, 4, 8, 16, 32]
nr_options = [2, 4, 8, 16, 32]
mv_options = [2, 4, 8, 16]
nu_options = [2, 4, 8, 16]


params = [
    LoopTilingParams(nc=nc, kc=kc, mc=mc, mr=mr, nr=nr, mv=mv, nu=nu)
    for nc, kc, mc, mr, nr, mv, nu in product(
        nc_options,
        kc_options,
        mc_options,
        mr_options,
        nr_options,
        mv_options,
        nu_options,
    )
]


def is_valid_configuration(params: LoopTilingParams) -> bool:
    # dont blame me
    try:
        assert params.nr % params.nu == 0
        assert params.mr % params.mv == 0

        assert params.nc % params.nr == 0
        assert params.mc % params.mr == 0

        assert params.kc <= N

        return True
    except AssertionError:
        return False


params = [p for p in params if is_valid_configuration(p)]
results = pd.DataFrame(columns=["nc", "kc", "mc", "mr", "nr", "mv", "nu", "time"])

for p in tqdm(params, desc="Tiling params"):
    set_tiling_params(p)

    data = measure_all(
        model,
        variations=["loop-tiling"],
        measure_base=False,
        runs=300,
        tqdm_leave=False,
    )

    assert len(data) == 1
    result = data[next(iter(data.keys()))]
    # print(f"result: {np.mean(result):.2f}ms")

    entry = {
        "nc": int(p.nc),
        "kc": int(p.kc),
        "mc": int(p.mc),
        "mr": int(p.mr),
        "nr": int(p.nr),
        "mv": int(p.mv),
        "nu": int(p.nu),
        "time": np.mean(result),
    }
    results = pd.concat(
        [
            results,
            pd.DataFrame.from_records([entry]),
        ]
    )
    results.to_csv("results.csv", index=False)
