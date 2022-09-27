import numpy as np
import onnx
import onnxsim.onnx_simplifier as onnx_simplifier
from .ops import Operation
from .tensor import parse_tensors


class Generator:
    """
    Code generator

    Proto ref: https://github.com/onnx/onnx/blob/main/docs/IR.md
    """

    def __init__(self, model_proto: onnx.ModelProto):
        self.model_proto = model_proto
        self.tensors = parse_tensors(model_proto)