import numpy as np
import onnx
import onnxruntime

from .generator import Generator
from .service import ModelService


def check_model(
    model_proto: onnx.ModelProto, variations: list[str] = [], n_inputs: int = 10
) -> None:
    """
    Checks if the generated output matches the reference (ONNX Runtime)

    :param n_inputs: random inputs will be generated
    """

    result = Generator(model_proto, variations).generate()

    ort_sess = onnxruntime.InferenceSession(model_proto.SerializeToString())

    with ModelService(result) as service:
        for _ in range(n_inputs):

            inputs = {
                name: np.random.random_sample(shape).astype(np.float32) * 2 - 1
                for name, shape in result.input_shapes.items()
            }

            out1 = service.inference(inputs)
            out2 = ort_sess.run(None, inputs)

            assert len(out1) == len(out2)
            for o1, o2 in zip(out1, out2):
                assert np.allclose(o1, o2), "output mismatch"
