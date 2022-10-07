import numpy as np

from .result import ModelResult
from .service import ModelService


def model_check(result: ModelResult, n_inputs: int = 2) -> bool:
    """
    Checks if the generated output matches the reference (ONNX Runtime)

    :param n_inputs: random inputs will be generated
    """

    with ModelService(result) as service:
        for _ in range(n_inputs):
            # TODO: generar inputs
            service.inference(inputs=[np.array([1, 2, 3], dtype=np.float32)])
            # invocar a onnxruntime y comparar!

    return False
