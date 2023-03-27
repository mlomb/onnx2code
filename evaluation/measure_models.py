import setup  # noqa # isort:skip

import numpy as np
import pandas as pd
import onnx
from measure import measure_all
from tqdm import tqdm

from onnx2code.ops.gemm_tiling.GEMM import LoopTilingParams, set_tiling_params

# should be set to the best
set_tiling_params(LoopTilingParams(nc=4096, kc=256, mc=64, mr=4, nr=32, mv=2, nu=4))

VARIATIONS = ["conv-naive", "im2col"]

results = pd.DataFrame(columns=["model", "runtime", "time_mean", "time_std"])

models = [
    "../data/vision/classification/mnist/model/mnist-12.onnx",
    "../data/vision/super_resolution/sub_pixel_cnn_2016/model/super-resolution-10.onnx",
    "../data/vision/classification/squeezenet/model/squeezenet1.1-7.onnx",
    "../data/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx",
    "../data/vision/classification/inception_and_googlenet/inception_v2/model/inception-v2-7.onnx",
    "../data/vision/classification/resnet/model/resnet50-caffe2-v1-9.onnx",
]

for model in tqdm(models):
    result = measure_all(
        None,
        variations=VARIATIONS,
        runs=15,
        tqdm_leave=False,
        onnx_model=onnx.load(model),
    )

    for var, times in result.items():
        entry = {
            "model": model,
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
    results.to_csv("results_models.csv", index=False)
