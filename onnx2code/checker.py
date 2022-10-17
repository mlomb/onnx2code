import os
import shutil
from pathlib import Path

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
                name: np.random.uniform(-1.0, 1.0, shape).astype(np.float32)
                for name, shape in result.input_shapes.items()
            }

            out1 = service.inference(inputs)
            out2 = ort_sess.run(None, inputs)

            assert len(out1) == len(out2)
            for o1, o2 in zip(out1, out2):
                correct = np.allclose(o1, o2)

                if not correct and os.getenv("ONNX2CODE_DEBUG", "0") == "1":
                    temp_dir = Path(__file__).parent.parent / "tmp/"
                    inputs_np = np.concatenate(
                        [inp.reshape(-1) for inp in inputs.values()]
                    )
                    inputs_np.tofile(temp_dir / "sample_inputs.bin")
                    o2.reshape(-1).tofile(temp_dir / "sample_outputs.bin")
                    shutil.copyfile(
                        Path(__file__).parent / "debugger.c",
                        temp_dir / "debugger.c",
                    )

                assert np.allclose(o1, o2, atol=1e-5), "output mismatch"
