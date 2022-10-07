def shapes_id(shape: list[int]) -> str:
    return f"""({",".join(map(str, shape))})"""


def pytest_make_parametrize_id(config, val, argname):
    if argname == "shape":
        return shapes_id(val)
    return None
