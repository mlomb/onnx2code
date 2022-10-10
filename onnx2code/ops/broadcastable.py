from .operation import OpCall, OpImpl, Operation


class Broadcastable(Operation):
    """
    Broadcastable operators like Add, Sub, etc.

    https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md
    """

    node_types = {"Add"}

    def parse(self) -> None:
        assert len(self.inputs) == 2, "expected two inputs"
        assert len(self.outputs) == 1, "expected one output"
        assert (
            self.inputs[0].size == self.outputs[0].size
        ), "input and output tensors should have the same size"

        self.op: str = self.node.op_type
        self.b_is_scalar = self.inputs[1].size == 1

    def call(self) -> OpCall:
        return OpCall(
            name=f"{self.op}_{self.inputs[0].shape_str()}",
            params=["A", "B", "C"],
            inputs=self.inputs,
            outputs=self.outputs,
        )


@Broadcastable.variant("c")
class BroadcastableC(Broadcastable):
    def impl(self) -> OpImpl:

        source = ""

        return OpImpl(lang="c", source=source)
