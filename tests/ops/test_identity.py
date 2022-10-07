import tensorflow as tf
import pytest

from ..util import check_keras


@pytest.mark.parametrize("shape", [[1], [2, 3], [4, 5, 6]])
def test_identity(shape: list[int]) -> None:
    input = tf.keras.Input(shape)
    out = tf.keras.layers.Lambda(lambda x: x)(input)
    model = tf.keras.Model(inputs=[input], outputs=[out])
    check_keras(model)
