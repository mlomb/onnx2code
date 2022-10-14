import os
from pathlib import Path

import onnx
import pytest

from onnx2code.checker import check_model


@pytest.mark.parametrize("variation", ["c"])
@pytest.mark.skip(reason="Not all ops are implemented")
def test_mnist(variation: str) -> None:
    model_proto = onnx.load(Path(os.path.dirname(__file__)) / "../../data/mnist-7.onnx")
    check_model(model_proto, [variation])
