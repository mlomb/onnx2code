from .operation import OpCall, Operation, OpImpl


class Conv(Operation):
    """
    Conv operator

    Only 2D convolutions are supported

    https://github.com/onnx/onnx/blob/main/docs/Operators.md#conv
    """

    node_types = {"Conv"}

    def parse(self) -> None:
        assert len(self.inputs) == 2, "expected two inputs"
        assert len(self.outputs) == 1, "expected one output"

        self.op: str = self.node.op_type
        self.b_is_scalar = self.inputs[1].size == 1

    def call(self) -> OpCall:
        return OpCall(
            name=f"{self.op}_{self.inputs[0].shape_str()}_{self.outputs[0].shape_str()}",
            params=["A", "B", "C"],
            inputs=self.inputs,
            outputs=self.outputs,
        )


@Conv.variant("c")
class ConvC(Conv):
    def impl(self) -> OpImpl:
        source = ""

        source += f"""
        ya kisieras
        """

        return OpImpl(lang="c", source=source)
