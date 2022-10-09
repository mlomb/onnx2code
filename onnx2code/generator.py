from typing import Literal
import onnx
import numpy as np

from .tensor import TensorInfo, parse_tensors
from .result import ModelResult


class Generator:
    """
    Code generator

    Proto ref: https://github.com/onnx/onnx/blob/main/docs/IR.md
    """

    def __init__(self, model_proto: onnx.ModelProto):
        self.model_proto = model_proto
        self.tensors = {tensor.name: tensor for tensor in parse_tensors(model_proto)}

        # TODO: hacer mas lindo :)
        self.functions: list[str] = []
        self.c_code_blocks: list[str] = []
        self.asm_code_blocks: list[str] = []
        self.calls: list[str] = []

    def get_tensors_with_tag(self, tag: str) -> list[TensorInfo]:
        return [tensor for tensor in self.tensors.values() if tensor.tag == tag]

    def weld_tensors(self, name_from: str, name_to: str) -> None:
        """
        Weld tensors together
        This means they should point to the same variable in runtime

        :param name_from: Name of the origin tensor
        :param name_to: Name of the destination tensor
        :raises KeyError: If the tensor names are not found
        """

        self.tensors[name_from].variable = self.tensors[name_to].variable

    def generate(self) -> ModelResult:
        """
        Generate C and ASM code to run the model
        """
        # register models ↓
        from .ops.operation import Operation

        for node in self.model_proto.graph.node:
            op = Operation.get(node.op_type, ["cpp", "asm"])(
                node,
                [self.tensors[name] for name in node.input],
                [self.tensors[name] for name in node.output],
            )
            op.emit(self)

        source_cpp = ""
        source_hpp = ""
        source_asm = ""

        source_cpp += "\n".join(self.c_code_blocks)
        source_asm += "\n".join(self.asm_code_blocks)

        inputs = self.get_tensors_with_tag("input")
        outputs = self.get_tensors_with_tag("output")

        source_cpp += f"""\n\nvoid inference(const float* weights, const float* inputs, float* outputs) {{"""

        for tensor in self.tensors.values():
            if tensor.tag == "input":
                source_cpp += f"""\n\tconst float* {tensor.variable} = inputs + {0};"""
            elif tensor.tag == "output":
                source_cpp += f"""\n\tfloat* {tensor.variable} = outputs + {0};"""

        tensors_data = []
        tensors_data_offset = 0
        for tensor in self.tensors.values():
            if tensor.data is not None:
                source_cpp += f"\nconst float* {tensor.variable} = weights + {tensors_data_offset}; // {tensor.name} {tensor.shape}\n"
                tensors_data.append(tensor.data)
                tensors_data_offset += tensor.data.size

        source_cpp += "\n".join([call for call in self.calls])
        source_cpp += "}"

        return ModelResult(
            input_shapes={tensor.name: tensor.shape for tensor in inputs},
            ouput_shapes={tensor.name: tensor.shape for tensor in outputs},
            inputs_size=sum([tensor.size for tensor in inputs]),
            outputs_size=sum([tensor.size for tensor in outputs]),
            source_cpp=source_cpp,
            source_hpp=source_hpp,
            source_asm=source_asm,
            weights=np.array(tensors_data),
        )

    def add_function(
        self,
        name: str,
        inputs: list[str],
        outputs: list[str],
        lang: Literal["cpp", "asm"],
        code: str,
    ) -> None:
        """
        Add a function definition
        """
        if name in self.functions:
            return

        input_list = ", ".join(f"const float* {name}" for name in inputs)
        output_list = ", ".join(f"float* {name}" for name in outputs)
        decl = f"void {name}({input_list}, {output_list})"

        if lang == "cpp":
            self._add_c_block(f"{decl} {{\n{code}\n}}")
        elif lang == "asm":
            self._add_c_block(f"{decl};")
            self._add_asm_block(
                "\n".join([f";; {decl}", f"global {name}", f"{name}:", code])
            )

        self.functions.append(name)

    def add_call(self, function: str, *args: TensorInfo) -> None:
        """
        Add a function call
        """
        self.calls.append(f"""{function}({", ".join(t.variable for t in args)});""")

    def _add_c_block(self, code: str) -> None:
        """
        Add a C code block
        """
        if code not in self.c_code_blocks:
            self.c_code_blocks.append(code)

    def _add_asm_block(self, code: str) -> None:
        """
        Add a ASM code block
        """
        if code not in self.asm_code_blocks:
            self.asm_code_blocks.append(code)
