from pathlib import Path
from typing import Any

import onnx
import pytest

from onnx2code.checker import check_model
from tests.zoo import download_from_zoo, zoo_manifest

# To avoid downloading big models we know are going to fail
EXCLUDED_MODELS = {
    "Too big to compile": ["ResNet101-DUC-7.onnx", "ResNet101-DUC-12.onnx"],
    "Operation GlobalAveragePool not implemented": [
        "resnet101-v1-7.onnx",
        "resnet101-v2-7.onnx",
        "resnet152-v1-7.onnx",
        "resnet152-v2-7.onnx",
        "resnet50-v1-12.onnx",
        "resnet18-v1-7.onnx",
        "resnet18-v2-7.onnx",
        "resnet34-v1-7.onnx",
        "resnet34-v2-7.onnx",
        "resnet50-v1-7.onnx",
        "resnet50-v2-7.onnx",
        "densenet-12.onnx",
        "densenet-7.onnx",
        "densenet-8.onnx",
        "densenet-9.onnx",
    ],
    "Operation LRN not implemented": [
        "rcnn-ilsvrc13-7.onnx",
        "rcnn-ilsvrc13-8.onnx",
        "rcnn-ilsvrc13-9.onnx",
        "bvlcalexnet-12.onnx",
        "bvlcalexnet-7.onnx",
        "bvlcalexnet-8.onnx",
        "bvlcalexnet-9.onnx",
        "caffenet-12.onnx",
        "caffenet-7.onnx",
        "caffenet-8.onnx",
        "caffenet-9.onnx",
        "googlenet-12.onnx",
        "googlenet-7.onnx",
        "googlenet-8.onnx",
        "googlenet-9.onnx",
        "inception-v1-12.onnx",
        "inception-v1-7.onnx",
        "inception-v1-8.onnx",
        "inception-v1-9.onnx",
        "zfnet512-12.onnx",
        "zfnet512-7.onnx",
        "zfnet512-8.onnx",
        "zfnet512-9.onnx",
    ],
    "Operation Pad not implemented": [
        "candy-8.onnx",
        "candy-9.onnx",
        "mosaic-8.onnx",
        "mosaic-9.onnx",
        "pointilism-8.onnx",
        "pointilism-9.onnx",
        "rain-princess-8.onnx",
        "rain-princess-9.onnx",
        "udnie-8.onnx",
        "udnie-9.onnx",
    ],
    "Operation Resize not implemented": [
        "FasterRCNN-10.onnx",
        "fcn-resnet101-11.onnx",
        "fcn-resnet50-11.onnx",
        "fcn-resnet50-12.onnx",
        "MaskRCNN-10.onnx",
    ],
}


def idfn(model: Any) -> str:
    return Path(model["model_path"]).name


def check_io_is_float(model: Any) -> None:
    io = model["metadata"]["io_ports"]

    for input in io["inputs"]:
        if input["type"] != "tensor(float)":
            raise NotImplementedError(f"No support for IO port type {input['type']}")


@pytest.mark.parametrize("variation", ["c"])
@pytest.mark.parametrize("model", zoo_manifest(), ids=idfn)
def test_zoo(model: Any, variation: str) -> None:
    if model["opset_version"] < 7:
        pytest.skip("Opset version < 7")

    # manual exclusion
    for reason, models in EXCLUDED_MODELS.items():
        if idfn(model) in models:
            pytest.skip(reason)

    # avoid downloading big models!
    # try to early out if IO is incompatible
    try:
        check_io_is_float(model)
    except NotImplementedError as e:
        pytest.skip(e.__str__())
    # early out if model is quantized
    if "int8" in model["model"] or "qdq" in model["model"]:
        pytest.skip("Quantized models are not supported")

    model_path = download_from_zoo(
        model["model_path"], model["metadata"]["model_bytes"]
    )
    model_proto = onnx.load(model_path.__str__())

    try:
        check_model(model_proto, [variation])
    except NotImplementedError as e:
        pytest.skip(e.__str__())
