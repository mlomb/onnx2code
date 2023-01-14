from ..util import compute_strides, get_attribute
from .operation import OpCall, Operation, OpImpl


class Transpose(Operation):
    """
    Transpose operator

    https://github.com/onnx/onnx/blob/main/docs/Operators.md#transpose
    """

    node_types = {"Transpose"}

    def parse(self) -> None:
        assert len(self.inputs) == 1, "expected one input"
        assert len(self.outputs) == 1, "expected one output"
        assert (
            self.inputs[0].size == self.outputs[0].size
        ), "input and output tensors should have the same size"

        self.input_strides = compute_strides(self.inputs[0].shape)
        self.output_strides = compute_strides(self.outputs[0].shape)
        self.perm = get_attribute(self.node, "perm", [])

    def call(self) -> OpCall:
        return OpCall(
            sig_name="Transpose",
            sig_params=[self.inputs[0].shape, self.outputs[0].shape, self.perm],
            inputs=self.inputs,
            outputs=self.outputs,
        )


@Transpose.variant("c")
class TransposeC(Transpose):
    def impl(self) -> OpImpl:
        output_shape = self.outputs[0].shape

        for_loops = []
        out_index = []
        in_index = []

        for i in range(len(output_shape)):
            for_loops.append(
                f"""for (int d{i} = 0; d{i} < {output_shape[i]}; ++d{i})"""
            )
            out_index.append(f"d{i}*{self.output_strides[i]}")
            in_index.append(f"d{i}*{self.input_strides[self.perm[i]]}")

        source = "\n".join([loop + "{" for loop in for_loops])
        source += (
            "\n\tOUT[" + "+".join(out_index) + "] = A[" + "+".join(in_index) + "];\n"
        )
        source += "}" * len(for_loops)
        source += "\n"

        return OpImpl(lang="c", source=source)
