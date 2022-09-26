import onnx
from onnx2code.generator import Generator

model_proto = onnx.load("mnist-1.onnx")

for t in Generator(model_proto).tensors:
    print(t)
