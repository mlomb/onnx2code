import os
from pathlib import Path

import onnx
import pytest

from onnx2code.checker import check_model


def check_zoo(
    area: str, problem: str, model: str, version: str, variation: str
) -> None:
    model_proto = onnx.load(
        Path(os.path.dirname(__file__))
        / "../models"
        / area
        / problem
        / model
        / "model"
        / f"{version}.onnx"
    )

    check_model(model_proto, [variation])


@pytest.mark.parametrize("variation", ["c"])
@pytest.mark.parametrize("version", [7, 8, 12])
def test_mnist(variation: str, version: int) -> None:
    check_zoo("vision", "classification", "mnist", f"mnist-{str(version)}", variation)


@pytest.mark.parametrize("variation", ["c"])
@pytest.mark.parametrize("version", [10])
def test_sub_pixel_cnn_2016(variation: str, version: int) -> None:
    check_zoo(
        "vision",
        "super_resolution",
        "sub_pixel_cnn_2016",
        "super-resolution-10",
        variation,
    )


@pytest.mark.parametrize("variation", ["c"])
@pytest.mark.parametrize("version", [7])
def test_squeezenet(variation: str, version: int) -> None:
    check_zoo(
        "vision", "classification", "squeezenet", f"squeezenet1.1-{version}", variation
    )


# No support for depthwise conv
# @pytest.mark.parametrize("variation", ["c"])
# @pytest.mark.parametrize("version", [7, 8, 9])
# def test_shufflenet(variation: str, version: int) -> None:
#     check_zoo(
#         "vision", "classification", "shufflenet", f"shufflenet-{version}", variation
#     )


@pytest.mark.parametrize("variation", ["c"])
@pytest.mark.parametrize("version", [7, 8])
def test_emotion_ferplus(variation: str, version: int) -> None:
    check_zoo(
        "vision",
        "body_analysis",
        "emotion_ferplus",
        f"emotion-ferplus-{version}",
        variation,
    )


# No support for this kind of broadcasting
# @pytest.mark.parametrize("variation", ["c"])
# @pytest.mark.parametrize("version", [7, 8, 9])
# def test_inception_v2(variation: str, version: int) -> None:
#     check_zoo(
#         "vision",
#         "classification",
#         "inception_and_googlenet/inception_v2",
#         f"inception-v2-{version}",
#         variation,
#     )


@pytest.mark.parametrize("variation", ["c"])
@pytest.mark.parametrize("version", [7, 8, 9])
def test_resnet50_caffe2_v1(variation: str, version: int) -> None:
    check_zoo(
        "vision",
        "classification",
        "resnet",
        f"resnet50-caffe2-v1-{version}",
        variation,
    )


# TODO: support bigger models
# @pytest.mark.parametrize("variation", ["c"])
# @pytest.mark.parametrize("version", [7, 12])
# def test_resnet101_duc(variation: str, version: int) -> None:
#     check_zoo(
#         "vision",
#         "object_detection_segmentation",
#         "duc",
#         f"ResNet101-DUC-{version}",
#         variation,
#     )


@pytest.mark.parametrize("variation", ["c"])
@pytest.mark.parametrize("version", [7, 8, 9])
def test_vgg19_caffe2(variation: str, version: int) -> None:
    check_zoo(
        "vision",
        "classification",
        "vgg",
        f"vgg19-caffe2-{version}",
        variation,
    )


# No support for depthwise conv
# @pytest.mark.parametrize("variation", ["c"])
# @pytest.mark.parametrize("version", [11])
# def test_efficientnet_lite4(variation: str, version: int) -> None:
#     check_zoo(
#         "vision",
#         "classification",
#         "efficientnet-lite4",
#         f"efficientnet-lite4-{version}",
#         variation,
#     )


@pytest.mark.parametrize("variation", ["c"])
@pytest.mark.parametrize("version", [7, 12])
def test_vgg16(variation: str, version: int) -> None:
    check_zoo(
        "vision",
        "classification",
        "vgg",
        f"vgg16-{version}",
        variation,
    )
