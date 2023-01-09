from typing import Any

import numpy as np

from .operation import OpCall, Operation, OpImpl


class Broadcastable(Operation):
    """
    Broadcastable operators like Add, Sub, etc.

    https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md
    """

    node_types = {"Add", "Div", "Mul", "Sub"}

    def parse(self) -> None:
        assert len(self.inputs) == 2, "expected two inputs"
        assert len(self.outputs) == 1, "expected one output"

        self.op: str = self.node.op_type
        self.b_is_scalar = self.inputs[1].size == 1
        self.input_A = self.inputs[0]
        self.input_B = self.inputs[1]

    def call(self) -> OpCall:
        return OpCall(
            name=self.op,
            sig_params=[self.inputs[0].shape, self.outputs[0].shape],
            params=["A", "B", "C"],
            inputs=self.inputs,
            outputs=self.outputs,
        )


@Broadcastable.variant("c")
class BroadcastableC(Broadcastable):
    def impl(self) -> OpImpl:
        source = ""

        symbol = {
            "Add": "+",
            "Div": "/",
            "Mul": "*",
            "Sub": "-",
        }[self.op]

        if self.b_is_scalar:
            source += f"""
            const float D = B[0];
            for (int i = 0; i < {self.inputs[0].size}; i++) {{
                C[i] = A[i] {symbol} D;
            }}
            """
        else:
            # we use nditer to generate the for loops for the broadcastable ops
            # it is a bit of a hack, but it works and it hides the complexity of
            # broadcasting :)

            a = np.arange(start=0, stop=self.input_A.size).reshape(self.input_A.shape)
            b = np.arange(start=0, stop=self.input_B.size).reshape(self.input_B.shape)
            offset = 0

            for x, y in np.nditer([a, b], flags=["external_loop"], order="C"):
                assert x.size == y.size, "nditer size expected to match"
                size = x.size

                # ARBITRARY ASSUMPTIONS I AM MAKING:
                def is_consecutive(z: Any) -> Any:
                    return z[z.size - 1] - z[0] == z.size - 1

                x_is_consecutive = is_consecutive(x)
                y_is_consecutive = is_consecutive(y)
                assert x_is_consecutive, "nditer x expected to be consecutive"
                assert (
                    y[y.size - 1] == y[0] or y_is_consecutive
                ), "nditer y expected to be all equal or consecutive"

                source += f"""for(int i = 0; i < {size}; i++) C[{offset} + i] = A[{x[0]} + i] {symbol} B[{y[0]}{" + i" if y_is_consecutive else ""}];\n"""
                offset += size

        return OpImpl(lang="c", source=source)
