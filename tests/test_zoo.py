from pathlib import Path
from typing import Any

import onnx
import pytest

from onnx2code.checker import check_model
from tests.zoo import download_from_zoo, zoo_manifest


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
