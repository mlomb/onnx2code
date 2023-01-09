import numpy as np

from ..util import get_attribute
from .operation import OpCall, Operation, OpImpl

LETTERS = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
]


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

        self.op: str = self.node.op_type
        self.size = self.inputs[0].size

    def call(self) -> OpCall:
        return OpCall(
            name=self.op,
            sig_params=[self.size],
            params=LETTERS[: len(self.inputs)] + ["OUT"],
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
                finfo = np.finfo(dtype=np.float32)
                min = get_attribute(self.node, "min", finfo.min)
                max = get_attribute(self.node, "max", finfo.max)
                impl = "A[i] < {} ? {} : A[i] > {} ? {} : A[i]".format(
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
