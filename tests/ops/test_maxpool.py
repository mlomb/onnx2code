import pytest
import tensorflow as tf

from ..util import check_keras


@pytest.mark.parametrize("shape", [[1, 1], [5, 1], [10, 1], [16, 1]])
@pytest.mark.parametrize("pool_size", [1, 2])
@pytest.mark.parametrize("strides", [1, 2])
@pytest.mark.parametrize("padding", ["valid", "same"])
def test_maxpool_1D(
    shape: list[int],
    pool_size: int,
    strides: int,
    padding: str,
) -> None:
    input = tf.keras.Input(shape)
    try:
        pool = tf.keras.layers.MaxPooling1D(
            pool_size=pool_size,
            strides=strides,
            padding=padding,
        )(input)
    except ValueError as e:
        pytest.skip("incompatible configuration: " + str(e))

    model = tf.keras.Model(inputs=[input], outputs=[pool])
    check_keras(model)


# TODO: 2D
# TODO: 3D
