import tensorflow as tf
import tf2onnx
import onnx

from onnx2code.checker import check_model


def check_keras(model: tf.keras.Model) -> None:
    model_proto, _ = tf2onnx.convert.from_keras(model)
    if False:
        onnx.save_model(model_proto, "tmp/model.onnx")
    check_model(model_proto)
