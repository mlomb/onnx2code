import onnx

# taken from onnx_simplifier.get_inputs
def get_model_inputs(model: onnx.ModelProto) -> list[onnx.ValueInfoProto]:
    initializer_names = [x.name for x in model.graph.initializer]
    return [ipt for ipt in model.graph.input if ipt.name not in initializer_names]


# taken from onnx_simplifier.get_shape_from_value_info_proto
def get_shape_from_value_info_proto(v: onnx.ValueInfoProto) -> list[int]:
    return [dim.dim_value for dim in v.type.tensor_type.shape.dim]
