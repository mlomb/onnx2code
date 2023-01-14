import os
import shutil
from pathlib import Path

import numpy as np
import onnx
import onnxruntime

from .generator import Generator
from .result import ModelResult
from .service import ModelService


def check_model_result(
    model_proto: onnx.ModelProto, result: ModelResult, n_inputs: int = 1
) -> None:
    """
    Checks if the generated output matches the reference runtime (ONNX Runtime)

    :param n_inputs: random inputs will be generated
    """
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

            output_matches = True

            for o1, o2 in zip(out1, out2):
                output_matches = output_matches and np.allclose(o1, o2, atol=1e-5)

            if not output_matches and os.getenv("ONNX2CODE_DEBUG", "0") == "1":
                temp_dir = Path(__file__).parent.parent / "tmp/"
                inputs_np = np.concatenate([inp.reshape(-1) for inp in inputs.values()])
                outputs_np = np.concatenate([o.reshape(-1) for o in out2])
                inputs_np.tofile(temp_dir / "sample_inputs.bin")
                outputs_np.tofile(temp_dir / "sample_outputs.bin")
                shutil.copyfile(
                    Path(__file__).parent / "debugger.c",
                    temp_dir / "debugger.c",
                )

            if not output_matches:
                raise RuntimeError("output mismatch")


def check_model(
    model_proto: onnx.ModelProto, variations: list[str] = [], n_inputs: int = 1
) -> None:
    """
    Generates code for the given model and checks if the generated output matches the reference runtime (ONNX Runtime)

    :param n_inputs: random inputs will be generated
    """
    result = Generator(model_proto, variations).generate()

    check_model_result(model_proto, result, n_inputs)
