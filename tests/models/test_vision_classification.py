import os
from pathlib import Path

import onnx
import pytest

from onnx2code.checker import check_model


@pytest.mark.parametrize("variation", ["c"])
@pytest.mark.parametrize("size", [7, 8, 12])
def test_mnist(variation: str, size: int) -> None:
    model_proto = onnx.load(
        Path(os.path.dirname(__file__))
        / (
            "../../models/vision/classification/mnist/model/mnist-"
            + str(size)
            + ".onnx"
        )
    )
    check_model(model_proto, [variation])
