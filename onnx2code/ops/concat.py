from ..util import compute_strides, get_attribute
from .operation import LETTERS, OpCall, Operation, OpImpl


class Concat(Operation):
    """
    Concat operator

    https://github.com/onnx/onnx/blob/main/docs/Operators.md#concat
    """

    node_types = {"Concat"}

    def parse(self) -> None:
        assert len(self.outputs) == 1, "expected one output"

        self.axis = get_attribute(self.node, "axis", None)

        assert self.axis is not None, "axis is not set"

    def call(self) -> OpCall:
        return OpCall(
            sig_name="Concat",
            sig_params=[inp.shape for inp in self.inputs],
            inputs=self.inputs,
            outputs=self.outputs,
        )


@Concat.variant("c")
class ConcatC(Concat):
    def impl(self) -> OpImpl:
        source = ""

        output_strides = compute_strides(self.outputs[0].shape)

        def output_index(axis_offset: int) -> str:
            output_index = ""
            for i, stride in enumerate(output_strides):
                output_index += "+"
                if i == self.axis:
                    output_index += f"({axis_offset}+d{i})"
                else:
                    output_index += f"d{i}"
                output_index += f"*{stride}"
            return output_index

        axis_offset = 0
        for k, input in enumerate(self.inputs):
            index = ""
            input_strides = compute_strides(input.shape)

            for i, elems in enumerate(input.shape):
                source += f"for (int d{i} = 0; d{i} < {elems}; d{i}++) {{\n"
                index += f"+ d{i} * {input_strides[i]}"

            source += f"OUT[{output_index(axis_offset)}] = {LETTERS[k]}[{index}];\n"
            source += "}\n" * len(input.shape)

            axis_offset += input.shape[self.axis]

        return OpImpl(lang="c", source=source)
