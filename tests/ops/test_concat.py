import pytest
import tensorflow as tf

from ..util import check_keras


@pytest.mark.parametrize(
    "shapes",
    [
        [[1, 2, 3, 4], [1, 2, 3, 4]],
        [[2, 2, 5, 1], [2, 1, 5, 1]],
        [[2, 2, 5, 1], [2, 1, 5, 1], [2, 3, 5, 1]],
    ],
)
@pytest.mark.parametrize("axis", [0, 1, 2, 3])
@pytest.mark.parametrize("variation", ["c"])
def test_concat(shapes: list[list[int]], axis: int, variation: str) -> None:
    inputs = [tf.keras.Input(shape) for shape in shapes]

    try:
        out = tf.keras.layers.Concatenate(axis=1 + axis)(inputs)  # +1 for batch dim
        model = tf.keras.Model(inputs=inputs, outputs=[out])
    except Exception:
        pytest.skip("incompatible configuration")

    check_keras(model, [variation])
