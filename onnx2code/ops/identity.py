from ..generator import Generator
from .operation import Operation


class Identity(Operation):
    """
    Identity operator

    https://github.com/onnx/onnx/blob/main/docs/Operators.md#identity
    """

    node_types = {"Identity"}

    def asserts(self) -> None:
        assert len(self.inputs) == 1, "expected one input"
        assert len(self.outputs) == 1, "expected one output"
        assert (
            self.inputs[0].size == self.outputs[0].size
        ), "input and output tensors should have the same size"


@Identity.variant("c")
class IdentityC(Identity):
    def emit(self, gen: Generator) -> None:
        size = self.inputs[0].size

        gen.add_function(
            f"identity_{size}",
            ["A"],
            ["B"],
            "c",
            f"""
            for (int i = 0; i < {size}; i++) {{
                B[i] = A[i];
            }}
            """,
        )

        gen.add_call(f"identity_{size}", self.inputs[0], self.outputs[0])


@Identity.variant("asm")
class IdentityASM(Identity):
    def emit(self, gen: Generator) -> None:
        size = self.inputs[0].size

        gen.add_function(
            f"identity_{size}",
            ["A"],
            ["B"],
            "asm",
            "\n".join(
                [
                    f"mov rax, {size}",
                    ".loop:",
                    "movss xmm0, [rdi]",
                    "add rdi, 4",
                    "movss [rsi], xmm0",
                    "add rsi, 4",
                    "dec rax",
                    "jnz .loop",
                    "ret",
                ]
            ),
        )

        gen.add_call(f"identity_{size}", self.inputs[0], self.outputs[0])
