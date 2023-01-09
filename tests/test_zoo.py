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


# @pytest.mark.parametrize("variation", ["c"])
# def test_super_resolution(variation: str) -> None:
#    check_zoo("vision", "super_resolution", "sub_pixel_cnn_2016", "super-resolution-10", variation)


@pytest.mark.parametrize("version", [7, 8])
def test_emotion_ferplus(version: int) -> None:
    check_zoo(
        "vision", "body_analysis", "emotion_ferplus", f"emotion-ferplus-{version}", "c"
    )


@pytest.mark.parametrize("version", [7])
def test_squeezenet(version: int) -> None:
    check_zoo("vision", "classification", "squeezenet", f"squeezenet1.1-{version}", "c")
