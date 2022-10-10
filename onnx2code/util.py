from typing import Any
import onnx


# taken from onnx_simplifier.get_inputs
def get_model_inputs(model: onnx.ModelProto) -> list[onnx.ValueInfoProto]:
    initializer_names = [x.name for x in model.graph.initializer]
    return [ipt for ipt in model.graph.input if ipt.name not in initializer_names]


# taken from onnx_simplifier.get_shape_from_value_info_proto
def get_shape_from_value_info_proto(v: onnx.ValueInfoProto) -> list[int]:
    # TODO: aca las dimensiones unknown las estamos poniendo en 1,
    #       eso no esta bien. Hay que usar onnxsimplifier con el shape del input
    #       en 1 y que infiera los sizes correctos para el resto de los tensors
    return [dim.dim_value or 1 for dim in v.type.tensor_type.shape.dim]


def get_attribute(node: onnx.NodeProto, name: str, default: Any = None) -> Any:
    """
    Returns the value of the attribute with the given name.
    If the attribute is not found, returns the default value.
    """
    for attr in node.attribute:
        if attr.name == name:
            return onnx.helper.get_attribute_value(attr)
    return default
