import pytest
import tensorflow as tf

from ..util import check_keras


@pytest.mark.parametrize(
    "shape",
    [
        # 1D
        *[shape for shape in [[1, 1], [5, 1], [10, 2], [16, 3]]],
        # 2D
        *[shape for shape in [[1, 1, 1], [5, 5, 1], [10, 8, 3], [16, 8, 8]]],
    ],
)
@pytest.mark.parametrize("axis", [1, 2])
def test_softmax(shape: list[int], axis: int) -> None:
    input = tf.keras.Input(shape=shape[1:])
    output = tf.keras.layers.Softmax(axis=axis)(input)
    model = tf.keras.Model(inputs=input, outputs=output)
    check_keras(model)
