import tensorflow as tf
import tf2onnx

from onnx2code.checker import check_model


def check_keras(model: tf.keras.Model, variations: list[str] = []) -> None:
    model_proto, _ = tf2onnx.convert.from_keras(model)
    check_model(model_proto, variations)
