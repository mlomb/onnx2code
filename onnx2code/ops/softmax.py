from onnx2code.util import get_attribute

from .operation import OpCall, Operation, OpImpl


class Softmax(Operation):
    """
    Softmax operator

    https://github.com/onnx/onnx/blob/main/docs/Operators.md#Softmax
    """

    node_types = {"Softmax"}

    def parse(self) -> None:
        assert len(self.inputs) == 1, "expected one input"
        assert len(self.outputs) == 1, "expected one output"

        self.X = self.inputs[0]
        self.Y = self.outputs[0]

        self.axis = get_attribute(self.node, "axis", 1)

    def call(self) -> OpCall:
        return OpCall(
            name=f"Softmax_{self.axis}",
            params=["X", "Y"],
            inputs=self.inputs,
            outputs=self.outputs,
        )


@Softmax.variant("c")
class SoftmaxC(Softmax):
    def impl(self) -> OpImpl:
        axis = self.axis
        if axis < 0:
            axis += len(self.X.shape)

        source = f"""
        for(int x = 0; x < {self.X.shape[axis]}; x++) {{
            float sum = 0;
            for(int y = 0; y < {self.Y.shape[axis]}; y++) {{
                sum += exp(X[x * {self.X.shape[axis]} + y]);
            }}
            for(int y = 0; y < {self.X.shape[axis]}; y++) {{
                Y[x * {self.X.shape[axis]} + y] = exp(X[x * {self.X.shape[axis]} + y]) / sum;
            }}
        }}
        """

        return OpImpl(lang="c", source=source)
