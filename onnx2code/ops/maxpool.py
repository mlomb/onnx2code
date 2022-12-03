from onnx2code.util import get_attribute

from .operation import OpCall, Operation, OpImpl


class MaxPool(Operation):
    """
    GEneral Matrix Multiplication operator

    https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
    """

    node_types = {"MaxPool"}

    def parse(self) -> None:
        assert len(self.inputs) == 1, "expected one input"
        assert len(self.outputs) == 1, "expected one output"

        self.X = self.inputs[0]

        self.pads = get_attribute(self.node, "pads", [0] * len(self.X.shape) * 2)
        self.strides = get_attribute(self.node, "strides", [1] * len(self.X.shape))

        self.kernel_shape = get_attribute(
            self.node, "kernel_shape", [1] * len(self.X.shape)
        )

    def call(self) -> OpCall:
        return OpCall(
            name=f"MaxPool_{0}",
            params=["X", "Y"],
            inputs=self.inputs,
            outputs=self.outputs,
        )


@MaxPool.variant("c")
class MaxPoolC(MaxPool):
    def impl(self) -> OpImpl:
        source = ""

        return OpImpl(lang="c", source=source)
