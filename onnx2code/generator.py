from collections import defaultdict
from textwrap import indent
import onnx
import numpy as np

from .tensor import TensorData, parse_tensors
from .result import ModelResult
from .ops.operation import Operation, OpCall, OpImpl

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
            model=_model_proto, input_shapes=get_fixed_input_shapes(_model_proto)
        )
        assert check, "ONNX model could not be simplified"

        if True:
            onnx.save_model(model_proto, "tmp/model.onnx")

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
                self.impls[impl] = call
                self.calls.append(call)

        inputs = [tensor for tensor in self.tensors.values() if tensor.tag == "input"]
        outputs = [tensor for tensor in self.tensors.values() if tensor.tag == "output"]

        return ModelResult(
            input_shapes={tensor.name: tensor.shape for tensor in inputs},
            ouput_shapes={tensor.name: tensor.shape for tensor in outputs},
            source_c=self._gen_c_source(),
            source_h=f"extern {INFERENCE_SIGNATURE};",
            source_asm=self._gen_asm_source(),
            weights=self._gen_weights(),
        )

    def _gen_weights(self) -> TensorData:
        return np.array(
            [tensor.data for tensor in self.tensors.values() if tensor.data is not None]
        )

    def _gen_c_source(self) -> str:
        source = "#include <math.h>" + "\n" * 2

        for impl, call in self.impls.items():
            if impl.lang == "asm":
                source += f"extern {call.signature()};"

        for impl, call in self.impls.items():
            if impl.lang == "c":
                source += call.signature() + " {"
                source += indent(impl.full_source(), prefix=" " * 4)
                source += "}"

        inference_source = ""
        offsets: defaultdict[str, int] = defaultdict(int)
        # build tensor variables
        for tensor in self.tensors.values():
            if tensor.tag is None:
                continue

            decl = "const " if tensor.tag == "input" else ""
            decl += f"float* {tensor.variable} = "
            decl += f"{tensor.tag}s"
            decl += f" + {offsets[tensor.tag]};"
            offsets[tensor.tag] += tensor.size

            inference_source += "\n" + decl

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
