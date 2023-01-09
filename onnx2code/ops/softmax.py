from typing import Callable
from onnx2code.util import compute_strides, get_attribute

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

        self.strides = compute_strides(self.X.shape)
        self.sizes = self.X.shape.copy()
        self.axis = get_attribute(self.node, "axis", -1)
        if self.axis < 0:
            self.axis += len(self.X.shape)

    def call(self) -> OpCall:
        return OpCall(
            name=f"Softmax",
            sig_params=[],
            params=["X", "Y"],
            inputs=self.inputs,
            outputs=self.outputs,
        )


@Softmax.variant("c")
class SoftmaxC(Softmax):
    def impl(self) -> OpImpl:
        strides, sizes, axis = self.strides, self.sizes, self.axis

        labels_size = sizes[axis]
        labels_stride = strides[axis]

        del sizes[axis]
        del strides[axis]

        NL = "\n"

        def iterate(predicate: Callable[[str], str]) -> str:
            iterators = []
            offset = f"i * {labels_stride}"

            for i, size in enumerate(sizes):
                iterators.append(f"for (int d{i} = 0; d{i} < {size}; ++d{i}) {{")
                offset += f" + d{i} * {strides[i]}"

            return f"""
                {NL.join(iterators)}
                {predicate(offset)}
                {NL.join("}" for _ in iterators)}
            """

        source = iterate(
            lambda offset: f"""
            float max = -INFINITY;
            float sum = 0.0f;

            for (int i = 0; i < {labels_size}; ++i) {{
                max = fmax(max, X[{offset}]);
            }}
            for (int i = 0; i < {labels_size}; ++i) {{
                Y[{offset}] = exp(X[{offset}] - max);
                sum += Y[{offset}];
            }}
            for (int i = 0; i < {labels_size}; ++i) {{
                Y[{offset}] /= sum;
            }}
        """
        )

        return OpImpl(lang="c", source=source)
