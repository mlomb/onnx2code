# onnx2code



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
| Add | a |

## Usage

Required software:

* gcc, nasm
* Python 3.10
* [pipenv](https://pypi.org/project/pipenv/)
