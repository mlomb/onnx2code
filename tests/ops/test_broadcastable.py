import tensorflow as tf
import pytest

from ..util import check_keras


shapes = [[1], [2, 3], [4, 5, 6]]  # scalar, 2d, 3d


@pytest.mark.parametrize(
    "shapeA,shapeB",
    [
        # same shape
        *[(s, s) for s in shapes],
        # broadcasting
        # https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md
        ([2], [1]),
        ([2, 3, 4, 5], [1]),
        ([2, 3, 4, 5], [5]),
        ([4, 5], [2, 3, 4, 5]),
        ([1, 4, 5], [2, 3, 1, 1]),
        ([3, 4, 5], [2, 1, 1, 1]),
        ([3, 4, 5], [5]),
        ([3, 4, 5], [4, 5]),
        ([3, 4, 5, 6], [5, 6]),
        ([3, 4, 5, 6], [4, 5, 6]),
        ([1, 4, 1, 6], [3, 1, 5, 6]),
    ],
)
@pytest.mark.parametrize(
    "operation",
    [
        tf.keras.layers.Add,
        tf.keras.layers.Subtract,
        tf.keras.layers.Multiply,
    ],
    ids=lambda x: str(x.__name__),
)
def test_basic_ops(
    operation: tf.keras.layers.Layer, shapeA: list[int], shapeB: list[int]
) -> None:
    inputA = tf.keras.Input(shapeA)
    inputB = tf.keras.Input(shapeB)
    result = operation()([inputA, inputB])
    model = tf.keras.Model(inputs=[inputA, inputB], outputs=[result])
    check_keras(model)
