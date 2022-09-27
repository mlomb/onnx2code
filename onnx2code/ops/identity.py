from .operation import Operation


class Identity(Operation):
    """
    Identity operator

    https://github.com/onnx/onnx/blob/main/docs/Operators.md#identity
    """

    node_types = ["Identity"]

    def asserts(self):
        assert len(self.inputs) == 1, "expected one input"
        assert len(self.outputs) == 1, "expected one output"
        assert (
            self.inputs[0]["size"] == self.outputs[0]["size"]
        ), "input and output tensors should have the same size"


@Identity.variant("cpp")
class IdentityCPP(Identity):
    def generate(self, gen):
        gen.add_c_block(
            """
        template<typename T, unsigned int SIZE>
        void Identity(const T* A, T* B) {{
            memcpy(B, A, SIZE * sizeof(T));
        }}
        """
        )

        gen.add_call(
            f"""Identity<float, {input["size"]}>""", self.inputs[0], self.outputs[0]
        )


@Identity.variant("asm")
class IdentityASM(Identity):
    def generate(self):
        print("ASM variant called")
