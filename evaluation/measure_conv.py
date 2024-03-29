import setup  # noqa # isort:skip

import keras
import numpy as np
import pandas as pd
from keras import layers
from measure import measure_all
from tqdm import tqdm

from onnx2code.ops.gemm_tiling.GEMM import LoopTilingParams, set_tiling_params

# should be set to the best
set_tiling_params(LoopTilingParams(nc=4096, kc=256, mc=64, mr=4, nr=32, mv=2, nu=4))

SIZES = 2 ** np.arange(8, 10)
VARIATIONS = ["conv-naive", "im2col"]


results = pd.DataFrame(columns=["MNK", "runtime", "time_mean", "time_std"])

for x in tqdm(range(256, 1280, 32)):
    model = keras.Sequential(
        [
            keras.Input(shape=(x, x, 1)),
            layers.Conv2D(
                filters=4,
                padding="valid",
                kernel_size=4,
                use_bias=False,
            ),
        ]
    )

    result = measure_all(model, variations=VARIATIONS, runs=100, tqdm_leave=False)

    for var, times in result.items():
        entry = {
            "MNK": x,
            "runtime": var,
            "time_mean": np.mean(times),
            "time_std": np.std(times),
        }
        results = pd.concat(
            [
                results,
                pd.DataFrame.from_records([entry]),
            ]
        )
    results.to_csv("results_conv.csv", index=False)
