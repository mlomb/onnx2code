from typing import Any


def shapes_id(shape: list[int]) -> str:
    return f"""({",".join(map(str, shape))})"""


def pytest_make_parametrize_id(config: Any, val: Any, argname: str) -> str | None:
    if argname == "shape":
        return shapes_id(val)
    return None
