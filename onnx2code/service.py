import tempfile
from pathlib import Path
from subprocess import call
from time import sleep
from typing import Any
from multiprocessing import shared_memory
import subprocess

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
        """
        Creates the shared memory blocks and starts the service subprocess
        """
        self.shm_inputs = shared_memory.SharedMemory(
            "/onnx2code-inputs", create=True, size=5
        )
        self.shm_outputs = shared_memory.SharedMemory(
            "/onnx2code-outputs", create=True, size=5
        )
        self.process = subprocess.Popen(
            [self.service_executable], stdin=subprocess.PIPE, stdout=subprocess.PIPE
        )

    def inference(
        self, inputs: list[npt.NDArray[np.float32]]
    ) -> list[npt.NDArray[np.float32]]:
        """
        Runs the model with the given inputs
        """
        # load inputs into shared memory
        # TODO: copy from inputs to shm_inputs

        # signal service that inputs are ready
        assert self.process.stdin and self.process.stdout
        self.process.stdin.write("1".encode())
        self.process.stdin.flush()
        # wait for service to finish inference
        self.process.stdout.read(1)

        # read outputs from shared memory
        # TODO: copy shm_outputs to return value
        return []

    def __exit__(self, _1: Any, _2: Any, _3: Any) -> None:
        # exit service
        self.process.terminate()

        # release shared memory
        self.shm_inputs.unlink()
        self.shm_outputs.unlink()

        # remove compilation files
        self.temp_dir.cleanup()
