# Copyright 2023 The KerasCV Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Keras backend module.

This module adds a temporarily Keras API surface that is fully under KerasCV
control. This allows us to switch between `keras_core` and `tf.keras`, as well
as add shims to support older version of `tf.keras`.
- `config`: check which backend is being run.
- `keras`: The full `keras` API (via `keras_core` or `tf.keras`).
- `ops`: `keras_core.ops`, always tf-backed if using `tf.keras`.
"""

import types

from packaging.version import parse

from keras_cv.backend.config import multi_backend

# Keys are of the form: "module.where.attr.exists->module.where.to.alias"
# Value are of the form: ["attr1", "attr2", ...] or
#                        [("attr1_original_name", "attr1_alias_name")]
_KERAS_CORE_ALIASES = {
    "utils->saving": [
        "register_keras_serializable",
        "deserialize_keras_object",
        "serialize_keras_object",
        "get_registered_object",
    ],
    "models->saving": ["load_model"],
}


if multi_backend():
    import keras_core as keras
else:
    import tensorflow as tf

    if parse(tf.__version__) >= parse("2.12.0"):
        import keras
    else:
        from tensorflow import keras

    # Various shims for resolving conflicts between Keras versions.
    if not hasattr(keras, "saving"):
        keras.saving = types.SimpleNamespace()

    if hasattr(keras, "src"):
        keras.backend.RandomGenerator = keras.src.backend.RandomGenerator

    if not hasattr(keras.saving, "deserialize_keras_object"):
        keras.saving.deserialize_keras_object = (
            keras.utils.deserialize_keras_object
        )
    if not hasattr(keras.saving, "serialize_keras_object"):
        keras.saving.serialize_keras_object = keras.utils.serialize_keras_object
    if not hasattr(keras.saving, "register_keras_serializable"):
        keras.saving.register_keras_serializable = (
            keras.utils.register_keras_serializable
        )
    if not hasattr(keras.saving, "load_model"):
        keras.saving.load_model = keras.models.load_model

    # TF Keras doesn't have this rename.
    keras.activations.silu = keras.activations.swish

from keras_cv.backend import config  # noqa: E402
from keras_cv.backend import ops  # noqa: E402
from keras_cv.backend import tf_ops  # noqa: E402


def assert_tf_keras(src):
    if multi_backend():
        raise NotImplementedError(
            f"KerasCV component {src} does not yet support Keras Core, and can "
            "only be used in `tf.keras`."
        )


def supports_ragged():
    return not multi_backend()
