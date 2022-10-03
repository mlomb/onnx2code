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


@Identity.variant("cpp")
class IdentityCPP(Identity):
    def emit(self, gen: Generator) -> None:
        gen.add_function(
            "identity",
            ["A"],
            ["B"],
            "cpp",
            f"memcpy(B, A, {self.inputs[0].size} * sizeof(float));",
        )

        gen.add_call("identity", self.inputs[0], self.outputs[0])


@Identity.variant("asm")
class IdentityASM(Identity):
    def emit(self, gen: Generator) -> None:
        gen.add_function(
            "identity",
            ["A"],
            ["B"],
            "asm",
            "mov rax, A\nmov rbx, B\nmov rcx, SIZE\nrep movsb",
        )

        gen.add_call("identity", self.inputs[0], self.outputs[0])
