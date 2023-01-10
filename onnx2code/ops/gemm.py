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
        assert self.alpha is None, "alpha not supported"
        assert self.beta is None, "beta not supported"

        A = self.inputs[0]
        B = self.inputs[1]
        Y = self.outputs[0]

        self.N = A.shape[0]
        self.M = B.shape[1] if self.transB else B.shape[0]
        self.K = B.shape[0] if self.transB else B.shape[1]

        assert Y.shape[0] == self.N
        assert Y.shape[1] == self.K

    def call(self) -> OpCall:
        return OpCall(
            name="GEMM",
            sig_params=[
                self.hasC,
                self.N,
                self.M,
                self.K,
                self.transB,
            ],
            params=["A", "B"] + (["C"] if self.hasC else []) + ["Y"],
            inputs=self.inputs,
            outputs=self.outputs,
        )


@GEMM.variant("c")
class GEMMC(GEMM):
    def impl(self) -> OpImpl:
        N, M, K = self.N, self.M, self.K

        index_B = f"i * {K} + col" if self.transB == False else f"col * {M} + i"

        source = f"""
        for(int row = 0; row < {N}; row++) {{
            for(int col = 0; col < {K}; col++) {{
                float sum = 0;
                for(int i = 0; i < {M}; i++) {{
                    sum += A[row * {M} + i] * B[{index_B}];
                }}
                Y[row * {K} + col] = sum{f' + C[row * {K} + col]' if self.hasC else ''};
            }}
        }}
        """

        return OpImpl(lang="c", source=source)
