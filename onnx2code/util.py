from typing import Any, Optional

import numpy as np
import onnx

TensorShape = list[int]
ShapesMap = dict[str, TensorShape]


# taken from onnx_simplifier.get_inputs
def get_model_inputs(model: onnx.ModelProto) -> list[onnx.ValueInfoProto]:
    initializer_names = [x.name for x in model.graph.initializer]
    return [ipt for ipt in model.graph.input if ipt.name not in initializer_names]


# taken from onnx_simplifier.get_shape_from_value_info_proto
def get_shape_from_value_info_proto(v: onnx.ValueInfoProto) -> TensorShape:
    return [dim.dim_value for dim in v.type.tensor_type.shape.dim]


# taken from onnx_simplifier.get_value_info_all
def get_value_info_all(m: onnx.ModelProto, name: str) -> Optional[onnx.ValueInfoProto]:
    for v in m.graph.value_info:
        if v.name == name:
            return v

    for v in m.graph.input:
        if v.name == name:
            return v

    for v in m.graph.output:
        if v.name == name:
            return v

    return None


# taken from onnx_simplifier.get_shape
def get_shape(m: onnx.ModelProto, name: str) -> TensorShape:
    v = get_value_info_all(m, name)
    if v is not None:
        return get_shape_from_value_info_proto(v)
    raise RuntimeError('Cannot get shape of "{}"'.format(name))


def get_fixed_input_shapes(onnx_model: onnx.ModelProto) -> ShapesMap:
    """
    Returns a map with the input name as key and the shape of the input
    fixed to one batch.

    For example, if one of the inputs of the model is [None, 32, 32, 3],
    the resulting shape for that input will be [1, 32, 32, 3].
    """

    def fix_shape(shape: list[int]) -> list[int]:
        return [1 if (d == 0 or d is None) else d for d in shape]

    return {
        tensor.name: fix_shape(get_shape(onnx_model, tensor.name))
        for tensor in get_model_inputs(onnx_model)
    }


def get_attribute(node: onnx.NodeProto, name: str, default: Any = None) -> Any:
    """
    Returns the value of the attribute with the given name.
    If the attribute is not found, returns the default value.
    """
    for attr in node.attribute:
        if attr.name == name:
            return onnx.helper.get_attribute_value(attr)
    return default


def compute_strides(shape: list[int]) -> list[int]:
    """
    Returns the strides of the given shape.

    For example, compute_strides([1, 2, 3]) returns [6, 3, 1].
    """
    strides = []
    for i in range(len(shape)):
        after = shape[i + 1 :]
        if len(after) == 0:
            strides.append(1)
        else:
            strides.append(int(np.prod(after)))
    return strides


def shape_str(shape: list[int], sep: str = "x") -> str:
    """
    Returns a string representation of the shape with the given separator

    For example, shape_str([1, 2, 3], "x") returns "1x2x3"
    """
    size_str = ""
    for dim in shape:
        size_str += f"{int(dim)}{sep}"
    if len(sep) > 0:
        size_str = size_str[: -len(sep)]  # remove last {sep}
    return size_str
