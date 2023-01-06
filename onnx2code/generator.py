import os
from collections import defaultdict
from pathlib import Path
from textwrap import indent

import numpy as np
import onnx
import onnxsim.onnx_simplifier as onnx_simplifier

from .ops.operation import OpCall, Operation, OpImpl
from .result import ModelResult
from .tensor import TensorData, parse_tensors
from .util import get_fixed_input_shapes, shape_str

REGISTER_ORDER = ["rdi", "rsi", "rdx", "rcx", "r8", "r9"]
INFERENCE_SIGNATURE = (
    "void inference(const float* weights, const float* inputs, float* outputs)"
)


class Generator:
    """
    Code generator

    Proto ref: https://github.com/onnx/onnx/blob/main/docs/IR.md
    """

    def __init__(self, _model_proto: onnx.ModelProto, variations: list[str] = []):
        model_proto, check = onnx_simplifier.simplify(
            model=_model_proto,
            overwrite_input_shapes=get_fixed_input_shapes(_model_proto),
        )
        assert check, "ONNX model could not be simplified"

        # save model for later inspection
        if os.getenv("ONNX2CODE_DEBUG", "0") == "1":
            tmp = Path(__file__).parent.parent / "tmp"
            tmp.mkdir(exist_ok=True)
            onnx.save_model(model_proto, tmp / "model.onnx")

        self.model_proto = model_proto
        self.tensors = {tensor.name: tensor for tensor in parse_tensors(model_proto)}
        self.variations = variations + ["asm", "c"]

        self.impls: dict[OpImpl, OpCall] = {}
        self.calls: list[OpCall] = []

    def weld_tensors(self, name_from: str, name_to: str) -> None:
        """
        Weld tensors together
        This means they should point to the same variable in runtime

        :param name_from: Name of the origin tensor
        :param name_to: Name of the destination tensor
        :raises KeyError: If the tensor names are not found
        """
        self.tensors[name_to].variable = self.tensors[name_from].variable

        if self.tensors[name_to].tag != "output":
            self.tensors[name_to].tag = "welded"

    def generate(self) -> ModelResult:
        """
        Generate C and ASM code to run the model
        """
        for node in self.model_proto.graph.node:
            if node.op_type in ["Reshape", "Squeeze", "Unsqueeze"]:
                """
                Reshape/Squeeze/Unsqueeze operator ⚠️ SPECIAL CASE ⚠️

                https://github.com/onnx/onnx/blob/main/docs/Operators.md#reshape
                https://github.com/onnx/onnx/blob/main/docs/Operators.md#squeeze
                https://github.com/onnx/onnx/blob/main/docs/Operators.md#unsqueeze
                """
                assert len(node.output) == 1, "expected one output"

                # Since it just reshapes the tensor, we don't need to do anything in runtime
                # But we must must be weld the input and output tensors (variables/data)
                self.weld_tensors(node.input[0], node.output[0])

                continue

            if node.op_type in ["BatchNormalization"]:
                """
                No operation
                """
                continue

            op = Operation.get(node.op_type, self.variations)(
                node,
                [self.tensors[name] for name in node.input],
                [self.tensors[name] for name in node.output],
            )
            impl = op.impl()
            call = op.call()

            if call is not None and impl is not None:
                self.impls[impl] = call
                self.calls.append(call)

        inputs = [tensor for tensor in self.tensors.values() if tensor.tag == "input"]
        outputs = [tensor for tensor in self.tensors.values() if tensor.tag == "output"]

        return ModelResult(
            input_shapes={tensor.name: tensor.shape for tensor in inputs},
            output_shapes={tensor.name: tensor.shape for tensor in outputs},
            source_c=self._gen_c_source(),
            source_h=f"extern {INFERENCE_SIGNATURE};",
            source_asm=self._gen_asm_source(),
            weights=self._gen_weights(),
        )

    def _gen_weights(self) -> TensorData:
        return np.concatenate(
            [
                tensor.data.reshape(-1)
                for tensor in self.tensors.values()
                if tensor.tag == "weight"
                and tensor.data is not None
                and tensor.data.dtype == np.float32
            ]
            # concatenate needs at least one array
            + [np.array([], dtype=np.float32)],
        )

    def _gen_c_source(self) -> str:
        source = "#include <math.h>" + "\n" * 2

        for impl, call in self.impls.items():
            if impl.lang == "asm":
                source += f"extern {call.signature()};"

        for impl, call in self.impls.items():
            if impl.lang == "c":
                source += call.signature() + " {\n"
                source += indent(impl.full_source().strip(), prefix=" " * 4)
                source += "\n}"

        inference_source = ""
        offsets: defaultdict[str, int] = defaultdict(int)
        # build tensor variables
        for tensor in self.tensors.values():
            if tensor.tag in ["input", "output", "weight"]:
                if (
                    tensor.tag == "weight"
                    and tensor.data is not None
                    and tensor.data.dtype != np.float32
                ):
                    continue

                decl = "const " if tensor.tag != "output" else ""
                decl += f"float* {tensor.variable} = "
                decl += f"{tensor.tag}s"
                decl += f" + {offsets[tensor.tag]};"

                offsets[tensor.tag] += tensor.size

            elif tensor.tag == "intermediate":
                # IF an intermediate tensor is welded with the output
                # we want to preserve the output tensor instead of the intermediate one
                # so we skip the definition of the intermediate in favor of the output
                skip = False
                for other in self.tensors.values():
                    if other.tag == "output" and other.variable == tensor.variable:
                        # already defined as output
                        skip = True
                        break
                if skip:
                    continue

                decl = f"float {tensor.variable}[{tensor.size}];"
            else:
                # welded
                continue

            inference_source += (
                f"\n{decl : <34} // ({shape_str(tensor.shape)}) {tensor.name}"
            )

        # make op calls
        inference_source += "\n"
        for call in self.calls:
            inference_source += f"\n{call.invocation()};"

        source += "\n" * 2
        source += INFERENCE_SIGNATURE + " {"
        source += indent(inference_source, prefix=" " * 4)
        source += "\n}"

        return source

    def _gen_asm_source(self) -> str:
        source = ""

        for impl, call in self.impls.items():
            if impl.lang == "asm":
                comments = [call.signature()] + [
                    f"{p}: {REGISTER_ORDER[i]}" for i, p in enumerate(call.params)
                ]
                source = "\n".join(
                    [
                        *[f";; {c}" for c in comments],
                        f"global {call.name}",
                        f"{call.name}:",
                        indent(impl.full_source(), prefix=" " * 4),
                    ]
                )

        return source
