from .operation import OpCall, Operation, OpImpl


class Identity(Operation):
    """
    Identity operator

    https://github.com/onnx/onnx/blob/main/docs/Operators.md#identity
    """

    node_types = {"Identity"}

    def parse(self) -> None:
        assert len(self.inputs) == 1, "expected one input"
        assert len(self.outputs) == 1, "expected one output"
        assert (
            self.inputs[0].size == self.outputs[0].size
        ), "input and output tensors should have the same size"

        self.size = self.inputs[0].size

    def call(self) -> OpCall:
        return OpCall(
            name=f"Identity_{self.size}",
            params=["A", "B"],
            inputs=self.inputs,
            outputs=self.outputs,
        )


@Identity.variant("c")
class IdentityC(Identity):
    def impl(self) -> OpImpl:
        source = f"""
        for (int i = 0; i < {self.size}; i++) {{
            B[i] = A[i];
        }}
        """

        return OpImpl(lang="c", source=source)


@Identity.variant("asm")
class IdentityASM(Identity):
    def impl(self) -> OpImpl:
        source = (
            f"mov rax, {self.size}",
            ".loop:",
            "movss xmm0, [rdi]",
            "add rdi, 4",
            "movss [rsi], xmm0",
            "add rsi, 4",
            "dec rax",
            "jnz .loop",
            "ret",
        )

        return OpImpl(lang="asm", source=source)
