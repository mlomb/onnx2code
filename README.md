# onnx2code

Generate plain C/ASM code for inference of ONNX models without dependencies

## Model support

The following models have been tested and work as expected.

| Model | Size |
|---|---|
| [mnist](https://github.com/onnx/models/tree/main/vision/classification/mnist) | 26 KB |
| [Super_Resolution](https://github.com/onnx/models/tree/main/vision/super_resolution/sub_pixel_cnn_2016) | 240 KB |
| [squeezenet1.1](https://github.com/onnx/models/tree/main/vision/classification/squeezenet) | 9 MB |
| [emotion_ferplus](https://github.com/onnx/models/tree/main/vision/body_analysis/emotion_ferplus) | 34 MB |
| [inception-v2](https://github.com/onnx/models/tree/main/vision/classification/inception_and_googlenet/inception_v2) | 44 MB |
| [resnet50-caffe2-v1](https://github.com/onnx/models/tree/main/vision/classification/resnet) | 98 MB |
| [VGG 16 and VGG 16-bn](https://github.com/onnx/models/tree/main/vision/classification/vgg) | 527 MB |
| [VGG 19 and VGG 19-bn](https://github.com/onnx/models/tree/main/vision/classification/vgg) | 548 MB |
| [VGG 19-caffe2](https://github.com/onnx/models/tree/main/vision/classification/vgg) | 561 MB |

* Minimum ONNX opset version: **7**
* Quantized models are not supported
* Only `float` tensors supported

## Operator support

Only `float` data type is supported.

| Operator | Attribute support |
|---|---|
| [Add](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Add), [Div](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Div), [Mul](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Mul), [Sub](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Sub) | ✅ with broadcasting |
| [Concat](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Concat) | ✅ with multiple inputs<br/>✅ axis |
| [Conv](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Conv) | ✅ bias<br/>✅ stride<br/>✅ padding (and `auto_pad`)<br/>❌ dilations<br/>❌ depthwise (group != 1) |
| [Sum](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Sum) | ✅ with multiple inputs<br/>❌ with broadcasting |
| [Relu](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Relu), [Tanh](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Tanh), [Sigmoid](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Sigmoid),  [Clip](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Clip) | ✅ |
| [Gemm](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gemm) | ✅ with bias<br/>❌ transpose A<br/>✅ tranpose B<br/>❌ alpha != 1<br/>❌ beta != 1 |
| [Identity](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Identity) | ✅ |
| [MaxPool](https://github.com/onnx/onnx/blob/main/docs/Operators.md#MaxPool), [AveragePool](https://github.com/onnx/onnx/blob/main/docs/Operators.md#AveragePool) | ✅ stride<br/>✅  padding (and `auto_pad`)<br/>❌ dilations<br/>❌ storage_order != 0<br/>❌ count_include_pad != 0 |
| [Softmax](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Softmax) | ✅ stride<br/>✅ axis |
| [Transpose](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Transpose) | ✅ perm |


## Setting up with Docker

We provide a ready to use [Docker image](https://hub.docker.com/r/mlomb/onnx2code):

```sh
docker run --rm -it -v "$PWD/mnist.onnx":/app/input.onnx:ro -v "$PWD/output":/app/output:rw mlomb/onnx2code:latest --variations=asm,c
```

The command above will generate C and ASM code for the `mnist.onnx` model in the `output` folder.

## Setting up locally

### Prerequisites

* gcc, nasm (required if checking models)
* Python 3.10
* [pipenv](https://pypi.org/project/pipenv/)

Clone and install dependencies with `pipenv install`.

### Run

To generate code from an ONNX model, run the following command inside a pipenv shell:

```sh
python -m onnx2code --variation=asm,c mnist.onnx output_folder
```
