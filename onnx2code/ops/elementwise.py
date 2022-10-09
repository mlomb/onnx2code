import numpy as np

from .operation import Operation
from ..generator import Generator
from ..util import get_attribute


class Elementwise(Operation):
    """
    Elementwise operators

    For example: ReLU, Tanh, Sigmoid, etc.
    """

    node_types = {"Relu", "Tanh", "Sigmoid", "Clip"}

    def asserts(self) -> None:
        assert len(self.inputs) == 1, "expected one input"
        assert len(self.outputs) == 1, "expected one output"
        assert (
            self.inputs[0].size == self.outputs[0].size
        ), "input and output tensors should have the same size"


@Elementwise.variant("c")
class ElementwiseC(Elementwise):
    def emit(self, gen: Generator) -> None:
        size = self.inputs[0].size

        op = self.node.op_type

        if op == "Relu":
            impl = "A[i] > 0 ? A[i] : 0"
        elif op == "Tanh":
            impl = "tanh(A[i])"
        elif op == "Sigmoid":
            impl = "1.0f / (1.0f + exp(-A[i]))"
        elif op == "Clip":  # TODO: test!
            finfo = np.finfo(dtype=np.float32)
            min = get_attribute(self.node, "min", finfo.min)
            max = get_attribute(self.node, "max", finfo.max)
            impl = "A[i] < {} ? {} : A[i] > {} ? {} : A[i]".format(min, min, max, max)
        else:
            raise RuntimeError(f"Unsupported elementwise: {op}")

        gen.add_function(
            f"{op}_{size}",
            ["A"],
            ["B"],
            "c",
            f"""
            for(int i = 0; i < {size}; i++) {{
                B[i] = {impl};
            }}
            """,
        )

        gen.add_call(f"{op}_{size}", self.inputs[0], self.outputs[0])
