from .output import Output
from .service import ModelService


def model_check(output: Output, n_inputs: int = 2) -> bool:
    """
    Checks if the generated output matches the reference (ONNX Runtime)

    <n_inputs> random inputs will be generated
    """

    with ModelService(output) as service:
        for _ in range(n_inputs):
            # TODO: generar inputs
            service.inference(inputs=[])
            # invocar a onnxruntime y comparar!

    return False
