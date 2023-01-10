import json
import os
from typing import Any
import urllib.request
from pathlib import Path


def download_from_zoo(path: str, expected_size: int | None = None) -> Path:
    target = Path(os.path.dirname(__file__)) / "../data" / path

    if target.is_file():
        # file already downloaded

        # check if the size is the expected one
        if expected_size is None or target.stat().st_size == expected_size:
            return target

    target.parent.mkdir(parents=True, exist_ok=True)

    urllib.request.urlretrieve(
        f"https://github.com/onnx/models/raw/main/{path}", target
    )

    return target


def zoo_manifest() -> Any:
    return json.loads(download_from_zoo("ONNX_HUB_MANIFEST.json").read_text())
