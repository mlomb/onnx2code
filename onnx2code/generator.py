import onnx

from .tensor import parse_tensors


class Generator:
    """
    Code generator

    Proto ref: https://github.com/onnx/onnx/blob/main/docs/IR.md
    """

    def __init__(self, model_proto: onnx.ModelProto):
        from .ops.operation import Operation

        self.model_proto = model_proto
        self.tensors = {tensor.name: tensor for tensor in parse_tensors(model_proto)}
        for node in self.model_proto.graph.node:
            Operation.get(node.op_type, ["asm"])(self, node)

    def weld_tensors(self, name_from: str, name_to: str) -> None:
        """
        Weld tensors together
        This means they should point to the same variable in runtime

        :param name_from: Name of the origin tensor
        :param name_to: Name of the destination tensor
        :raises KeyError: If the tensor names are not found
        """

        self.tensors[name_from].variable = self.tensors[name_to].variable

    def generate(self) -> str:
        """
        Generate code
        """
        print("Generating code")

        return ""
