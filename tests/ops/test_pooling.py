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
@pytest.mark.parametrize("pool_size", [1, 2, 3])
@pytest.mark.parametrize("strides", [1, 2, 3])
@pytest.mark.parametrize("padding", ["valid", "same"])
@pytest.mark.parametrize("op", ["max", "average"])
def test_maxpool(
    shape: list[int],
    pool_size: int,
    strides: int,
    padding: str,
    op: str,
) -> None:
    impl = {
        "max": {
            2: tf.keras.layers.MaxPooling1D,
            3: tf.keras.layers.MaxPooling2D,
        },
        "average": {
            2: tf.keras.layers.AveragePooling1D,
            3: tf.keras.layers.AveragePooling2D,
        },
    }[op][len(shape)]
    input = tf.keras.Input(shape)
    try:
        pool = impl(
            pool_size=pool_size,
            strides=strides,
            padding=padding,
        )(input)
    except ValueError as e:
        pytest.skip("incompatible configuration: " + str(e))

    model = tf.keras.Model(inputs=[input], outputs=[pool])
    check_keras(model)
