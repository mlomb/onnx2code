import os
import sys

# Silence TF
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# Do not use GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Make onnxruntime only use 1 CPU thread
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

sys.path.append("../")

from time import perf_counter_ns  # noqa: E402

import numpy as np  # noqa: E402
import onnx  # noqa: E402
import onnxruntime  # noqa: E402
import tensorflow as tf  # noqa: E402
import tf2onnx  # noqa: E402

from onnx2code.generator import Generator  # noqa: E402
from onnx2code.result import ModelResult  # noqa: E402
from onnx2code.service import ModelService  # noqa: E402
from onnx2code.tensor import TensorsMap  # noqa: E402

# Make tensorflow only use 1 CPU thread
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.run_functions_eagerly(False)  # this line does not work 🤡
tf.compat.v1.disable_eager_execution()


def measure_tf(tf_model: tf.keras.Model, inputs: TensorsMap, runs: int) -> list[int]:
    times = []

    for _ in range(runs):
        start = perf_counter_ns()
        tf_model.predict(inputs, verbose=0)
        end = perf_counter_ns()
        times.append(end - start)

    return times


def measure_onnxruntime(
    model_proto: onnx.ModelProto, inputs: TensorsMap, runs: int
) -> list[int]:
    times = []
    ort_sess = onnxruntime.InferenceSession(model_proto.SerializeToString())

    for _ in range(runs):
        start = perf_counter_ns()
        ort_sess.run(None, inputs)
        end = perf_counter_ns()
        times.append(end - start)

    return times


def measure_onnx2code(
    model_result: ModelResult, inputs: TensorsMap, runs: int
) -> list[int]:
    times = []

    with ModelService(model_result) as service:
        for _ in range(runs):
            start = perf_counter_ns()
            service.inference(inputs)
            end = perf_counter_ns()
            times.append(end - start)

    return times


def measure_all(tf_model: tf.keras.Model, runs: int = 300) -> dict[str, list[float]]:
    """
    Measure the inference time of the given model in tf, onnxruntime and onnx2code.

    Time in milliseconds.
    """
    model_proto, _ = tf2onnx.convert.from_keras(tf_model)
    model_result = Generator(model_proto).generate()

    inputs = {
        name: np.random.random_sample(shape).astype(np.float32) * 2 - 1
        for name, shape in model_result.input_shapes.items()
    }

    warmup_runs = 100
    total = runs + warmup_runs

    def postprocess(times_in_ns: list[int]) -> list[float]:
        return [t / 1_000_000 for t in times_in_ns[warmup_runs:]]

    return {
        "tensorflow": postprocess(measure_tf(tf_model, inputs, total)),
        "onnxruntime": postprocess(measure_onnxruntime(model_proto, inputs, total)),
        "onnx2code": postprocess(measure_onnx2code(model_result, inputs, total)),
    }
