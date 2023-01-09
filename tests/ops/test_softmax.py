import pytest
import tensorflow as tf

from ..util import check_keras


@pytest.mark.parametrize(
    "shape",
    [[1], [2, 3], [4, 5, 6]],
)
@pytest.mark.parametrize("axis", [-1, 1, 2])
def test_softmax(shape: list[int], axis: int) -> None:
    input = tf.keras.Input(shape)
    try:
        output = tf.keras.layers.Softmax(axis=axis)(input)
        model = tf.keras.Model(inputs=[input], outputs=[output])
    except Exception:
        pytest.skip("incompatible configuration")

    check_keras(model)
