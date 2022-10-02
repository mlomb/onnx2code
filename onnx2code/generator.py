import onnx
import numpy as np

from .tensor import TensorInfo, parse_tensors
from .output import Output


class Generator:
    """
    Code generator

    Proto ref: https://github.com/onnx/onnx/blob/main/docs/IR.md
    """

    def __init__(self, model_proto: onnx.ModelProto):
        self.model_proto = model_proto
        self.tensors = {tensor.name: tensor for tensor in parse_tensors(model_proto)}

        # TODO: hacer mas lindo :)
        self.c_code_blocks: list[str] = []
        self.asm_code_blocks: list[str] = []
        self.calls: list[str] = []

    def weld_tensors(self, name_from: str, name_to: str) -> None:
        """
        Weld tensors together
        This means they should point to the same variable in runtime

        :param name_from: Name of the origin tensor
        :param name_to: Name of the destination tensor
        :raises KeyError: If the tensor names are not found
        """

        self.tensors[name_from].variable = self.tensors[name_to].variable

    def generate(self) -> Output:
        """
        Generate code
        """
        from .ops.operation import Operation

        for node in self.model_proto.graph.node:
            op = Operation.get(node.op_type, ["cpp"])(self, node)

            op.emit(self)

        source_cpp = ""
        source_hpp = ""
        source_asm = ""

        source_cpp = "\n".join(self.c_code_blocks)
        source_asm = "\n".join(self.asm_code_blocks)

        return Output(
            input_shapes={
                tensor.name: tensor.shape
                for tensor in self.tensors.values()
                if tensor.tag == "input"
            },
            ouput_shapes={
                tensor.name: tensor.shape
                for tensor in self.tensors.values()
                if tensor.tag == "output"
            },
            source_cpp=source_cpp,
            source_hpp=source_hpp,
            source_asm=source_asm,
            weights=np.array([]),
        )

    def add_c_block(self, code: str) -> None:
        """
        Add a C code block
        """
        if code not in self.c_code_blocks:
            self.c_code_blocks.append(code)

    def add_asm_block(self, code: str) -> None:
        """
        Add a ASM code block
        """
        if code not in self.asm_code_blocks:
            self.asm_code_blocks.append(code)

    def add_call(self, function: str, *args: TensorInfo) -> None:
        """
        Add a function call
        """
        self.calls.append(f"{function}();")
