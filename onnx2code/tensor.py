import onnx
import numpy as np
from dataclasses import dataclass
from .util import get_model_inputs, get_shape_from_value_info_proto


@dataclass
class Tensor:
    name: str
    tag: str | None
    shape: list[int]
    size: int
    data: np.ndarray
    variable: str


def parse_value(
    value_info: onnx.ValueInfoProto,
    tag: str or None,
    var_index: int,
    model_proto: onnx.ModelProto,
) -> Tensor:
    """
    Parses a ValueInfo and returns the tensor
    """
    name = value_info.name
    shape = get_shape_from_value_info_proto(value_info)
    data = None

    for node in model_proto.graph.node:
        if node.op_type == "Constant" and node.output[0] == name:
            data = np.array(node.attribute[0].t.float_data)

    return Tensor(
        name=name,
        tag=tag,
        shape=shape,
        size=np.prod(shape),
        data=data,
        variable=f"T{var_index}",
    )


def parse_initializer(initializer: onnx.TensorProto, var_index: int) -> Tensor:
    """
    Parses a TensorProto and returns the tensor
    """
    shape = [dim for dim in initializer.dims]
    data = onnx.numpy_helper.to_array(initializer)
    assert list(data.shape) == shape, "Tensor shape and data shape should match"
    return Tensor(
        name=initializer.name,
        tag=None,
        shape=shape,
        size=np.prod(shape),
        data=data,
        variable=f"T{var_index}",
    )


def parse_tensors(model_proto: onnx.ModelProto) -> list[Tensor]:
    """
    Reads ALL tensors and store them in a manegeable format
    input, output, intermediate and constant tensors
    """
    tensors: list[Tensor] = []

    # input
    tensors.extend(
        parse_value(vi, "input", i, model_proto)
        for i, vi in enumerate(get_model_inputs(model_proto), start=0)
    )

    # output
    tensors.extend(
        parse_value(vi, "output", i, model_proto)
        for i, vi in enumerate(model_proto.graph.output, start=len(tensors))
    )

    # intermediate
    tensors.extend(
        parse_value(vi, None, i, model_proto)
        for i, vi in enumerate(model_proto.graph.value_info, start=len(tensors))
    )

    # constant
    tensors.extend(
        parse_initializer(initializer, i)
        for i, initializer in enumerate(
            model_proto.graph.initializer, start=len(tensors)
        )
    )

    return tensors
