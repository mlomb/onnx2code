# Makes sure the libraries are using only 1 CPU thread
# and are optimized for inference.

import os
import sys

# Silence TF
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# Do not use GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Make onnxruntime only use 1 CPU thread
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import tensorflow as tf

# Make tensorflow only use 1 CPU thread
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# We don't need to disable eager execution, because we are using tf.function (I hope)
# tf.compat.v1.disable_v2_behavior()
# tf.compat.v1.disable_eager_execution()
# tf.config.run_functions_eagerly(False)  # this line does not work 🤡

sys.path.append("../")
