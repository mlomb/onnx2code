import numpy as np
import numpy.typing as npt

from .output import Output


def model_check(
    output: Output, inputs: None | npt.NDArray[np.float32] = None, n_inputs: int = 2
) -> bool:
    """
    Checks if the generated output matches the reference (ONNX Runtime)

    If no inputs are provided, <n_inputs> random inputs will be generated
    """

    return False
