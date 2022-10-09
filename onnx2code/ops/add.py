from ..generator import Generator
from .operation import Operation


class Add(Operation):
    """ """

    node_types = {"Add"}

    def asserts(self) -> None:
        assert len(self.inputs) == 2, "expected two inputs"
        assert len(self.outputs) == 1, "expected one output"


@Add.variant("c")
class AddC(Add):
    def emit(self, gen: Generator) -> None:
        gen.add_function(
            "add",
            ["A", "B"],
            ["C"],
            "c",
            f"for(int i = 0; i < {self.inputs[0].size}; i++) {{ C[i] = A[i] + B[i]; }}",
        )

        gen.add_call("add", *self.inputs, *self.outputs)
