import tensorflow as tf
import pytest

from ..util import check_keras


@pytest.mark.parametrize(
    "activation",
    [
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.Activation("tanh"),
        tf.keras.layers.Activation("sigmoid"),
    ],
    ids=lambda x: str(x.activation.__name__),
)
def test_activations(activation: tf.keras.layers.Activation) -> None:
    input = tf.keras.Input(shape=(4, 5, 6))
    output = activation(input)
    model = tf.keras.Model(inputs=[input], outputs=[output])
    check_keras(model)
