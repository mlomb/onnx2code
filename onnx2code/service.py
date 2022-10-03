import tempfile
from pathlib import Path
from subprocess import call
from typing import Any

import numpy as np
import numpy.typing as npt

from .output import Output


class ModelService:
    """
    Allows using a model generated by onnx2code in a convenient way

    Used for testing and evaluation
    """

    def __init__(self, output: Output):
        self.output = output

    def __enter__(self) -> "ModelService":
        """
        Compiles the model and starts a subprocess
        """
        self.temp_dir = tempfile.TemporaryDirectory()

        self._compile()
        self._boot()

        return self

    def _compile(self) -> None:
        # temp_dir = Path(self.temp_dir.name)

        temp_dir = Path("tmp/")
        temp_dir.mkdir(exist_ok=True)

        cpp_file = temp_dir / "model.cpp"
        hpp_file = temp_dir / "model.hpp"
        asm_file = temp_dir / "model.asm"
        weights_file = temp_dir / "weights.bin"
        svc_file = Path(__file__).parent / "service.cpp"

        asm_object = temp_dir / "model-asm.o"
        self.service_executable = temp_dir / "service"
        self.output.weights.tofile(weights_file)

        for file, content in [
            (cpp_file, self.output.source_cpp),
            (hpp_file, self.output.source_hpp),
            (asm_file, self.output.source_asm),
        ]:
            with open(file, "w") as f:
                f.write(content)

        compile_asm_cmd = [
            "nasm",
            "-f",
            "elf64",
            str(asm_file),
            "-o",
            str(asm_object),
            "-g",
        ]
        compile_svc_cmd = [
            "gcc",
            "-m64",  # 64 bit env
            str(asm_object),
            str(hpp_file),
            str(cpp_file),
            str(svc_file),
            # TODO: service cpp (will not compile right now)
            "-o",
            str(self.service_executable),
            "-O0",
            "-g",
            "-fsanitize=address",
        ]

        # TODO: hacer una funcion que corra el comando y parse el output
        #       si falla poner el output en la Exception
        #       y todo el resto del output tirarlo (warning etc) asi no poluciona
        if call(compile_asm_cmd) != 0:
            raise Exception("failure compiling asm")
        if call(compile_svc_cmd) != 0:
            raise Exception("failure compiling service")

        pass

    def _boot(self) -> None:
        pass

    def inference(
        self, inputs: list[npt.NDArray[np.float32]]
    ) -> list[npt.NDArray[np.float32]]:
        # TODO: pasar por pipes los inputs y outputs
        return []

    def __exit__(self, _1: Any, _2: Any, _3: Any) -> None:
        self.temp_dir.cleanup()
        pass
