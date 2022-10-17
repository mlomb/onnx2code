from onnx2code.util import get_attribute

from .operation import OpCall, Operation, OpImpl


class GEMM(Operation):
    """
    GEneral Matrix Multiplication operator

    https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
    """

    node_types = {"Gemm", "MatMul"}

    def parse(self) -> None:
        assert (
            len(self.inputs) == 2 or len(self.inputs) == 3
        ), "expected two or three inputs"
        assert len(self.outputs) == 1, "expected one output"

        self.hasC = len(self.inputs) == 3
        self.transA = get_attribute(self.node, "transA", 0) > 0.5
        self.transB = get_attribute(self.node, "transB", 0) > 0.5
        self.alpha = get_attribute(self.node, "alpha", 1.0)
        self.beta = get_attribute(self.node, "beta", 1.0)

        # normalize
        self.alpha = None if self.alpha == 1.0 else self.alpha
        self.beta = None if self.beta == 1.0 else self.beta

        assert not self.transA, "transA not supported"
        assert not self.transB, "transB not supported"
        assert self.alpha is None, "alpha not supported"
        assert self.beta is None, "beta not supported"

    def call(self) -> OpCall:
        return OpCall(
            name=f"GEMM_{'C' if self.hasC else ''}_{0}",
            params=["A", "B"] + (["C"] if self.hasC else []) + ["Y"],
            inputs=self.inputs,
            outputs=self.outputs,
        )


@GEMM.variant("c")
class GEMMC(GEMM):
    def impl(self) -> OpImpl:
        N = self.inputs[0].shape[0]
        M = self.inputs[0].shape[1]
        K = self.inputs[1].shape[1]

        source = f"""
        for(int row = 0; row < {N}; row++) {{
            for(int col = 0; col < {K}; col++) {{
                float sum = 0;
                for(int i = 0; i < {M}; i++) {{
                    sum += A[row * {M} + i] * B[i * {K} + col];
                }}
                Y[row * {K} + col] = sum{f' + C[row * {K} + col]' if self.hasC else ''};
            }}
        }}
        """

        return OpImpl(lang="c", source=source)