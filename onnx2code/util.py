from typing import Any
import onnx


def get_attribute(node: onnx.NodeProto, name: str, default: Any = None) -> Any:
    """
    Returns the value of the attribute with the given name.
    If the attribute is not found, returns the default value.
    """
    for attr in node.attribute:
        if attr.name == name:
            return onnx.helper.get_attribute_value(attr)
    return default
