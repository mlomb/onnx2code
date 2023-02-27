import pytest
import tensorflow as tf

from ..util import check_keras


@pytest.mark.parametrize("shape", [[1], [2, 2], [2, 3], [4, 5, 6]])
@pytest.mark.parametrize("units", [1, 2, 10, 100], ids=lambda x: f"{x}_units")
@pytest.mark.parametrize("use_bias", [False, True], ids=["no_bias", "bias"])
def test_naive(shape: list[int], units: int, use_bias: bool) -> None:
    input = tf.keras.Input(shape)
    dense = tf.keras.layers.Dense(units, use_bias=use_bias, bias_initializer="uniform")(
        input
    )
    model = tf.keras.Model(inputs=[input], outputs=[dense])
    check_keras(model, variations=["gemm-naive"])


@pytest.mark.parametrize("shape", [[1], [2, 2], [2, 3], [4, 5, 6]])
@pytest.mark.parametrize("units", [1, 2, 10, 100], ids=lambda x: f"{x}_units")
@pytest.mark.parametrize("use_bias", [False, True], ids=["no_bias", "bias"])
def test_libxsmm(shape: list[int], units: int, use_bias: bool) -> None:
    input = tf.keras.Input(shape)
    dense = tf.keras.layers.Dense(units, use_bias=use_bias, bias_initializer="uniform")(
        input
    )
    model = tf.keras.Model(inputs=[input], outputs=[dense])
    check_keras(model, variations=["libxsmm"])


@pytest.mark.parametrize("shape", [[64, 64], [19, 37]])
@pytest.mark.parametrize("units", [64], ids=lambda x: f"{x}_units")
def test_tiling(shape: list[int], units: int) -> None:
    input = tf.keras.Input(shape)
    dense = tf.keras.layers.Dense(units, use_bias=False, bias_initializer="uniform")(
        input
    )
    model = tf.keras.Model(inputs=[input], outputs=[dense])
    check_keras(model, variations=["loop-tiling"])
