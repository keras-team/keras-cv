import sys

import keras
from absl import flags

import keras_cv

flags.DEFINE_string("weights_path", None, "Path of weights to load")
flags.DEFINE_string(
    "output_weights_path", None, "Path of notop weights to store"
)
flags.DEFINE_string("model_name", None, "Name of the KerasCV.model")

FLAGS = flags.FLAGS
FLAGS(sys.argv)

if not FLAGS.weights_path.endswith(".h5"):
    raise ValueError("Weights path must end in .h5")

model = eval(
    f"keras_cv.models.{FLAGS.model_name}(include_rescaling=True, include_top=True, num_classes=1000, weights=FLAGS.weights_path)"
)

without_top = keras.models.Model(model.input, model.layers[-3].output)
without_top.save_weights(FLAGS.output_weights_path)

# Because the usage of keras_cv is in an eval() call, the linter is angry.
# We include this to avoid an unused import warning
keras_cv.models
