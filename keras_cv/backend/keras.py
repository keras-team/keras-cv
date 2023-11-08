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

import types

import tensorflow as tf

from keras_cv.backend import config

if config.detect_if_tensorflow_uses_keras_3():
    import keras  # noqa: F403, F401
    from keras import *  # noqa: F403, F401

    keras.backend.name_scope = keras.name_scope
elif config.multi_backend():
    import keras_core as keras  # noqa: F403, F401
    from keras_core import *  # noqa: F403, F401

    keras.backend.name_scope = keras.name_scope
else:
    from tensorflow import keras  # noqa: F403, F401
    from tensorflow.keras import *  # noqa: F403, F401

    # Shims to handle symbol renames for older `tf.keras` versions.
    if not hasattr(tf.keras, "saving"):
        saving = types.SimpleNamespace()
    else:
        from tensorflow.keras import saving
    from tensorflow.keras import utils

    if not hasattr(saving, "deserialize_keras_object"):
        saving.deserialize_keras_object = utils.deserialize_keras_object
    if not hasattr(saving, "serialize_keras_object"):
        saving.serialize_keras_object = utils.serialize_keras_object
    if not hasattr(saving, "register_keras_serializable"):
        saving.register_keras_serializable = utils.register_keras_serializable
    # TF Keras doesn't have this rename.
    keras.activations.silu = keras.activations.swish
