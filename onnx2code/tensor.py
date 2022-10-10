import operator
from dataclasses import dataclass
from functools import reduce
import numpy as np
import onnx
from numpy.typing import NDArray

from .util import get_model_inputs, get_shape_from_value_info_proto


ShapesMap = dict[str, list[int]]
TensorData = NDArray[np.float32]
TensorsMap = dict[str, TensorData]
TensorsList = list[TensorData]


@dataclass
class TensorInfo:
    name: str
    tag: str | None
    shape: list[int]
    size: int
    data: TensorData | None
    variable: str

    def shape_str(self, sep: str = "x") -> str:
        return sep.join(map(str, self.shape))

    @staticmethod
    def from_value(
        value_info: onnx.ValueInfoProto,
        tag: str | None,
        var_index: int,
        model_proto: onnx.ModelProto,
    ) -> "TensorInfo":
        """
        Parses a ValueInfo and returns the tensor
        """
        name = value_info.name
        shape = get_shape_from_value_info_proto(value_info)
        data: NDArray[np.float32] | None = None

        for node in model_proto.graph.node:
            if node.op_type == "Constant" and node.output[0] == name:
                data = np.array(node.attribute[0].t.float_data)

        return TensorInfo(
            name=name,
            tag=tag,
            shape=shape,
            size=reduce(operator.mul, shape, 1),
            data=data,
            variable=f"T{var_index}",
        )

    @staticmethod
    def from_initializer(initializer: onnx.TensorProto, var_index: int) -> "TensorInfo":
        """
        Parses a TensorProto and returns the tensor
        """
        shape = [dim for dim in initializer.dims]
        data: NDArray[np.float32] = onnx.numpy_helper.to_array(initializer)
        assert list(data.shape) == shape, "Tensor shape and data shape should match"
        return TensorInfo(
            name=initializer.name,
            tag=None,
            shape=shape,
            size=reduce(operator.mul, shape, 1),
            data=data,
            variable=f"T{var_index}",
        )


def parse_tensors(model_proto: onnx.ModelProto) -> list[TensorInfo]:
    """
    Reads ALL tensors and store them in a manegeable format
    input, output, intermediate and constant tensors
    """
    tensors: list[TensorInfo] = []

    # input
    tensors.extend(
        TensorInfo.from_value(vi, "input", i, model_proto)
        for i, vi in enumerate(get_model_inputs(model_proto), start=0)
    )

    # output
    tensors.extend(
        TensorInfo.from_value(vi, "output", i, model_proto)
        for i, vi in enumerate(model_proto.graph.output, start=len(tensors))
    )

    # intermediate
    tensors.extend(
        TensorInfo.from_value(vi, None, i, model_proto)
        for i, vi in enumerate(model_proto.graph.value_info, start=len(tensors))
    )

    # constant
    tensors.extend(
        TensorInfo.from_initializer(initializer, i)
        for i, initializer in enumerate(
            model_proto.graph.initializer, start=len(tensors)
        )
    )

    return tensors
