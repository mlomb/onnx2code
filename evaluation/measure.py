import os
import sys

# Silence TF
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Make onnxruntime only use 1 CPU thread
os.environ["OMP_NUM_THREADS"] = "1"

sys.path.append("../")

import time  # noqa: E402
import tensorflow as tf  # noqa: E402
import onnx  # noqa: E402
import onnxruntime  # noqa: E402
import tf2onnx  # noqa: E402
import numpy as np  # noqa: E402
import numpy.typing as npt  # noqa: E402

from onnx2code.generator import Generator  # noqa: E402
from onnx2code.result import ModelResult  # noqa: E402
from onnx2code.service import ModelService  # noqa: E402

# Make tensorflow only use 1 CPU thread
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# TODO: warmup
# TODO: average runs


Inputs = dict[str, npt.NDArray[np.float32]]


def measure_tf(tf_model: tf.keras.Model, inputs: Inputs) -> float:
    start = time.time()
    tf_model.predict(inputs, verbose=0)
    end = time.time()
    return end - start


def measure_onnxruntime(model_proto: onnx.ModelProto, inputs: Inputs) -> float:
    ort_sess = onnxruntime.InferenceSession(model_proto.SerializeToString())

    start = time.time()
    ort_sess.run(None, inputs)
    end = time.time()
    return end - start


def measure_onnx2code(model_result: ModelResult, inputs: Inputs) -> float:
    with ModelService(model_result) as service:
        start = time.time()
        service.inference(inputs)
        end = time.time()
        return end - start


def measure_all(tf_model: tf.keras.Model) -> None:
    model_proto, _ = tf2onnx.convert.from_keras(tf_model)
    model_result = Generator(model_proto).generate()

    inputs = {
        name: np.random.random_sample(shape).astype(np.float32) * 2 - 1
        for name, shape in model_result.input_shapes.items()
    }

    print("tensorflow:", measure_tf(tf_model, inputs))
    print("onnxruntime:", measure_onnxruntime(model_proto, inputs))
    print("onnx2code:", measure_onnx2code(model_result, inputs))
