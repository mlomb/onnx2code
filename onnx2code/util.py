from typing import Any, Literal, Optional

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


def resolve_stride(node: onnx.NodeProto) -> list[int]:
    strides: list[int] = get_attribute(node, "strides", [1] * 2)
    return strides


def compute_pad(
    in_dim: int,
    stride: int,
    kernel: int,
    pad_type: Literal[b"SAME_UPPER", b"SAME_LOWER", b"VALID", b"NOTSET"],
) -> tuple[int, int]:
    """
    https://github.com/microsoft/onnxruntime/blob/9ec1ed42a809170b87474f5822c4557101812399/onnxruntime/core/providers/common.h#L73
    """
    pad_head = 0
    pad_tail = 0

    if pad_type == b"VALID" or pad_type == b"NOTSET":
        pass
    elif pad_type == b"SAME_UPPER" or pad_type == b"SAME_LOWER":
        legacy_target_size = (in_dim + stride - 1) // stride
        pad_needed = (legacy_target_size - 1) * stride + kernel - in_dim

        if pad_type == b"SAME_LOWER":
            pad_head = (pad_needed + 1) // 2
        else:
            pad_head = pad_needed // 2

        pad_tail = pad_needed - pad_head
    else:
        raise NotImplementedError(f"Pad type {pad_type} not implemented")

    return pad_head, pad_tail


def resolve_padding(node: onnx.NodeProto, X: TensorShape, W: TensorShape) -> list[int]:
    """
    Resolve padding attribute from node
    """
    ndims = len(X) - 2  # number of spatial dimensions (excluding batch and channel)
    pads: list[int] = get_attribute(node, "pads", None)
    auto_pad = get_attribute(node, "auto_pad", b"NOTSET")
    stride = resolve_stride(node)

    if pads is not None:
        assert auto_pad == b"NOTSET", "Cannot specify both pads and auto_pad"
        return pads

    pads = [0] * ndims * 2
    for i in range(ndims):
        pad_head, pad_tail = compute_pad(X[i + 2], stride[i], W[i + 2], auto_pad)
        pads[i] = pad_head
        pads[i + ndims] = pad_tail

    return pads


def shape_str(shape: list[int], sep: str = "x") -> str:
    """
    Returns a string representation of the shape with the given separator

    For example, shape_str([1, 2, 3], "x") returns "1x2x3"
    """
    return sep.join(map(str, shape))