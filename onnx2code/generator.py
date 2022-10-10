from typing import Literal
from textwrap import dedent, indent
import onnx
import numpy as np

from .tensor import TensorInfo, parse_tensors
from .result import ModelResult
from .ops.operation import Operation


class Generator:
    """
    Code generator

    Proto ref: https://github.com/onnx/onnx/blob/main/docs/IR.md
    """

    def __init__(self, model_proto: onnx.ModelProto, variations: list[str] = []):
        self.model_proto = model_proto
        self.tensors = {tensor.name: tensor for tensor in parse_tensors(model_proto)}
        self.variations = variations + ["asm", "c"]

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
        for node in self.model_proto.graph.node:
            op = Operation.get(node.op_type, self.variations)(
                node,
                [self.tensors[name] for name in node.input],
                [self.tensors[name] for name in node.output],
            )
            impl = op.impl()
            call = op.call()

            if call is not None and impl is not None:
                self.calls.append(
                    f"""{call.name}({", ".join(t.variable for t in call.inputs + call.outputs)});"""  # noqa: E501
                )
                source = (
                    impl.source if type(impl.source) is str else "\n".join(impl.source)
                )
                self.add_function(call.name, ["A"], ["B"], impl.lang, source)

        source_c = "#include <math.h>\n"
        source_asm = ""

        source_c += "\n".join(self.c_code_blocks)
        source_asm += "\n".join(self.asm_code_blocks)

        inputs = self.get_tensors_with_tag("input")
        outputs = self.get_tensors_with_tag("output")

        source_c += """\n\nvoid inference(const float* weights, const float* inputs, float* outputs) {"""  # noqa: E501

        for tensor in self.tensors.values():
            if tensor.tag == "input":
                source_c += f"""\n\tconst float* {tensor.variable} = inputs + {0};"""
            elif tensor.tag == "output":
                source_c += f"""\n\tfloat* {tensor.variable} = outputs + {0};"""

        tensors_data = []
        tensors_data_offset = 0
        for tensor in self.tensors.values():
            if tensor.data is not None:
                source_c += f"\nconst float* {tensor.variable} = weights + {tensors_data_offset}; // {tensor.name} {tensor.shape}\n"  # noqa: E501
                tensors_data.append(tensor.data)
                tensors_data_offset += tensor.data.size

        source_c += "\n\n    "
        source_c += "\n    ".join([call for call in self.calls])
        source_c += "\n}"

        return ModelResult(
            input_shapes={tensor.name: tensor.shape for tensor in inputs},
            ouput_shapes={tensor.name: tensor.shape for tensor in outputs},
            source_c=source_c,
            source_h="extern void inference(const float* weights, const float* inputs, float* outputs);",  # noqa: E501
            source_asm=source_asm,
            weights=np.array(tensors_data),
        )

    def add_function(
        self,
        name: str,
        inputs: list[str],
        outputs: list[str],
        lang: Literal["c", "asm"],
        code: str,
    ) -> None:
        """
        Add a function definition
        """
        if name in self.functions:
            return

        code = indent(dedent(code), prefix=" " * 4)
        input_list = ", ".join(f"const float* {name}" for name in inputs)
        output_list = ", ".join(f"float* {name}" for name in outputs)
        decl = f"void {name}({input_list}, {output_list})"

        if lang == "c":
            self._add_c_block(f"{decl} {{{code}}}")
        elif lang == "asm":
            self._add_c_block(f"extern {decl};")

            register_order = ["rdi", "rsi", "rdx", "rcx", "r8", "r9"]

            comments = [decl]
            for i, var_name in enumerate(inputs + outputs):
                comments.append(f"{var_name}: {register_order[i]}")

            self._add_asm_block(
                "\n".join(
                    [
                        *[f";; {c}" for c in comments],
                        f"global {name}",
                        f"{name}:",
                        code,
                    ]
                )
            )

        self.functions.append(name)

    def add_call(self, function: str, *args: TensorInfo) -> None:
        """
        Add a function call
        """
        # self.calls.append(f"""{function}({", ".join(t.variable for t in args)});""")

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
