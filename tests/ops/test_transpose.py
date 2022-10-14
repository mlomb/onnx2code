import pytest
import tensorflow as tf

from ..util import check_keras


@pytest.mark.parametrize("shape", [[2, 3], [3, 4, 5]])
def test_transpose_default(shape: list[int]) -> None:
    input = tf.keras.Input(shape=shape)
    out = tf.keras.backend.transpose(input)
    model = tf.keras.Model(inputs=[input], outputs=[out])
    check_keras(model)


@pytest.mark.parametrize(
    "perm",
    [
        # ([1,2,3]), # this gets optimized with Identity
        ([2, 3, 1]),
        ([3, 2, 1]),
        ([3, 1, 2]),
        ([1, 3, 2]),
        ([2, 1, 3]),
    ],
    ids=lambda x: ",".join(map(str, x)),
)
def test_transpose_perm(perm: list[int]) -> None:
    input = tf.keras.Input(shape=[3, 4, 5])
    out = tf.keras.layers.Permute(perm)(input)
    model = tf.keras.Model(inputs=[input], outputs=[out])
    check_keras(model)
