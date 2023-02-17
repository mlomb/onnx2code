import setup  # noqa # isort:skip

from time import perf_counter_ns

import numpy as np
import onnx
import onnxruntime
import tensorflow as tf
import tf2onnx
from tqdm import tqdm

from onnx2code.generator import Generator
from onnx2code.result import ModelResult
from onnx2code.service import ModelService, TensorsMap


def measure_tf(tf_model: tf.keras.Model, inputs: TensorsMap, runs: int) -> list[int]:
    times = []

    # ⚠️ Make sure to use graph execution and NOT eager execution
    graph_model = tf.function(tf_model)

    for _ in tqdm(range(runs), desc="tensorflow"):
        start = perf_counter_ns()
        graph_model(inputs)
        end = perf_counter_ns()
        times.append(end - start)

    return times


def measure_onnxruntime(
    model_proto: onnx.ModelProto, inputs: TensorsMap, runs: int
) -> list[int]:
    times = []
    ort_sess = onnxruntime.InferenceSession(model_proto.SerializeToString())

    for _ in tqdm(range(runs), desc="onnxruntime"):
        start = perf_counter_ns()
        ort_sess.run(None, inputs)
        end = perf_counter_ns()
        times.append(end - start)

    return times


def measure_onnx2code(
    model_result: ModelResult, inputs: TensorsMap, runs: int, variation_name: str = ""
) -> list[int]:
    times = []

    with ModelService(model_result) as service:
        for _ in tqdm(
            range(runs),
            desc="onnx2code" if not variation_name else f"onnx2code-{variation_name}",
        ):
            start = perf_counter_ns()
            service.inference(inputs)
            end = perf_counter_ns()
            times.append(end - start)

    return times


def measure_all(
    tf_model: tf.keras.Model, runs: int = 300, variations: list[str] = []
) -> dict[str, list[float]]:
    """
    Measure the inference time of the given model in tf, onnxruntime and onnx2code.

    Time in milliseconds.
    """
    model_proto, _ = tf2onnx.convert.from_keras(tf_model)

    warmup_runs = int(min(100, max(5, runs * 0.1)))
    total = runs + warmup_runs

    def postprocess(times_in_ns: list[int]) -> list[float]:
        return [t / 1_000_000 for t in times_in_ns[warmup_runs:]]

    results: dict[str, list[float]] = {}

    for variation in variations:
        model_variation = Generator(model_proto, variations=[variation]).generate()

        inputs = {
            name: np.random.random_sample(shape).astype(np.float32) * 2 - 1
            for name, shape in model_variation.input_shapes.items()
        }

        results[f"onnx2code-{variation}"] = postprocess(
            measure_onnx2code(model_variation, inputs, total, variation)
        )

    return results | {
        "tensorflow": postprocess(measure_tf(tf_model, inputs, total)),
        "onnxruntime": postprocess(measure_onnxruntime(model_proto, inputs, total)),
    }
