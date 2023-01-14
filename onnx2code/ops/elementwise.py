import numpy as np

from ..util import get_attribute
from .operation import LETTERS, OpCall, Operation, OpImpl


class Elementwise(Operation):
    """
    Elementwise operators

    For example: ReLU, Tanh, Sigmoid, etc.
    """

    node_types = {"Relu", "Tanh", "Sigmoid", "Clip", "Sum"}

    def parse(self) -> None:
        assert len(self.outputs) == 1, "expected one output"

        for input in self.inputs:
            assert (
                input.size == self.outputs[0].size
            ), "input and output tensors should have the same size"

            if self.node.op_type == "Clip":
                # Clip may have min and max as inputs
                # or as attributes (depending on ONNX opset)
                break

        self.op: str = self.node.op_type
        self.size = self.inputs[0].size

    def call(self) -> OpCall:
        return OpCall(
            sig_name=self.op,
            sig_params=[self.size],
            inputs=self.inputs,
            outputs=self.outputs,
        )


@Elementwise.variant("c")
class ElementwiseC(Elementwise):
    def impl(self) -> OpImpl:
        impl: str
        match self.op:
            case "Sum":
                impl = "+".join([f"{LETTERS[i]}[i]" for i in range(len(self.inputs))])
            case "Relu":
                impl = "A[i] > 0 ? A[i] : 0"
            case "Tanh":
                impl = "tanh(A[i])"
            case "Sigmoid":
                impl = "1.0f / (1.0f + exp(-A[i]))"
            case "Clip":
                if len(self.inputs) == 3:
                    min_data = self.inputs[1].data
                    max_data = self.inputs[2].data

                    if min_data is None or max_data is None:
                        raise ValueError("Clip: min and max should be constants")

                    # "cast" the 0-dimensional arrays to numbers
                    min = min_data + 0
                    max = max_data + 0
                else:
                    finfo = np.finfo(dtype=np.float32)
                    min = get_attribute(self.node, "min", finfo.min)
                    max = get_attribute(self.node, "max", finfo.max)

                impl = "A[i] < {} ? {} : (A[i] > {} ? {} : A[i])".format(
                    min, min, max, max
                )
            case _:
                raise NotImplementedError(f"ElementwiseC: {self.op}")

        source = f"""
        for(int i = 0; i < {self.size}; i++) {{
            OUT[i] = {impl};
        }}
        """

        return OpImpl(lang="c", source=source)
