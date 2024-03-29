import os
import subprocess
import tempfile
from multiprocessing import shared_memory
from pathlib import Path
from subprocess import PIPE, run
from typing import Any

import numpy as np

from .result import ModelResult
from .tensor import TensorData
from .util import ShapesMap

TensorsMap = dict[str, TensorData]
TensorsList = list[TensorData]


def _run_compilation_command(cmd: list[str]) -> None:
    """
    Runs a given compilation command as a subprocess

    :param cmd: A list containing the command and its CLI args
    :raises SyntaxError: If the process return code is non-zero
    """
    compilation_process = run(cmd, stderr=PIPE)
    if compilation_process.returncode != 0:
        raise SyntaxError(compilation_process.stderr.decode("utf8"))


class ModelService:
    """
    Allows using a model generated by onnx2code in a convenient way

    Used for testing and evaluation
    """

    def __init__(self, result: ModelResult):
        self.result = result

    def __enter__(self) -> "ModelService":
        """
        Compiles the model and starts a subprocess
        """
        self.temp_dir = tempfile.TemporaryDirectory()

        self._compile()
        self._boot()

        return self

    def _compile(self) -> None:
        debug = os.getenv("ONNX2CODE_DEBUG", "0") == "1"

        if debug:
            # save for later inspection
            temp_dir = Path(__file__).parent.parent / "tmp/"
        else:
            temp_dir = Path(self.temp_dir.name)

        temp_dir.mkdir(exist_ok=True)

        c_file = temp_dir / "model.cpp"
        h_file = temp_dir / "model.h"
        asm_file = temp_dir / "model.asm"
        asm_object = temp_dir / "model-asm.o"
        svc_file = Path(__file__).parent / "service.c"
        self.weights_file = temp_dir / "weights.bin"
        self.service_executable = temp_dir / "service"

        self.result.weights.tofile(self.weights_file)

        for file, content in [
            (c_file, self.result.source_c),
            (h_file, self.result.source_h),
            (asm_file, self.result.source_asm),
        ]:
            with open(file, "w") as f:
                f.write(content)

        _run_compilation_command(
            [
                "nasm",
                "-f",
                "elf64",
                str(asm_file),
                "-o",
                str(asm_object),
            ]
            + (["-g", "-w+all", "-w+error"] if debug else [])
        )

        _run_compilation_command(
            [
                "g++",
                "-m64",  # 64 bit env
                str(asm_object),
                str(h_file),
                str(c_file),
                str(svc_file),
                "-o",
                str(self.service_executable),
                "-I",
                temp_dir.__str__(),
                "-lrt",  # for shm
                "-lm",  # for math
                "-march=native",
                "-mtune=native",
                "-O3",
            ]
            + (
                [
                    "-g",
                    "-fsanitize=address",
                    "-Wall",
                    "-Werror",
                    "-Wno-unused-result",
                    "-Wno-unused-but-set-variable",
                    "-Wno-unused-variable",
                ]
                if debug
                else []
            )
        )

    def _boot(self) -> None:
        """
        Creates the shared memory buffers and starts the service subprocess
        """
        self.inputs_buffer = SharedNDArrays("/o2c-inputs", self.result.input_shapes)
        self.outputs_buffer = SharedNDArrays("/o2c-outputs", self.result.output_shapes)

        self.process = subprocess.Popen(
            [self.service_executable, self.weights_file],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )

    def inference(self, inputs: TensorsMap) -> TensorsList:
        """
        Runs the model with the given inputs
        """
        assert len(inputs) == len(self.result.input_shapes)

        # load inputs into shared memory
        self.inputs_buffer.set(inputs)

        # signal service that inputs are ready
        assert self.process.stdin and self.process.stdout
        self.process.stdin.write("1".encode())
        self.process.stdin.flush()
        # wait for service to finish inference
        self.process.stdout.read(1)

        # read outputs from shared memory
        return self.outputs_buffer.get()

    def __exit__(self, _1: Any, _2: Any, _3: Any) -> None:
        # exit service
        self.process.terminate()

        # release shared memory
        self.inputs_buffer.cleanup()
        self.outputs_buffer.cleanup()

        # remove compilation files
        self.temp_dir.cleanup()


class SharedNDArrays:
    """
    List of NDArray[float32]'s backed by shared memory
    """

    def __init__(self, name: str, shapes: ShapesMap):
        self.shapes = shapes
        self.offsets = np.cumsum([0, *[np.prod(s) for s in shapes.values()]])
        self.elems = self.offsets[-1]
        self.size = self.elems * 4

        try:
            shm = shared_memory.SharedMemory(name, create=False)
            shm.unlink()
        except FileNotFoundError:
            pass

        self.shm = shared_memory.SharedMemory(name, create=True, size=self.size)
        self.buffer: TensorData = np.ndarray(
            self.elems, dtype=np.float32, buffer=self.shm.buf
        )

    def set(self, inputs: TensorsMap) -> None:
        self.buffer[:] = np.concatenate([inp.reshape(-1) for inp in inputs.values()])

    def get(self) -> TensorsList:
        return [
            self.buffer[self.offsets[i] : self.offsets[i + 1]].reshape(self.shapes[n])
            for i, n in enumerate(self.shapes)
        ]

    def cleanup(self) -> None:
        del self.buffer
        self.shm.close()
        self.shm.unlink()
