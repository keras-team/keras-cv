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

from tensorflow import keras  # noqa: F403, F401
from tensorflow.keras import *  # noqa: F403, F401

from keras_cv.src.backend import config  # noqa: F403, F401

_KERAS_CORE_ALIASES = {
    "utils->saving": [
        "register_keras_serializable",
        "deserialize_keras_object",
        "serialize_keras_object",
        "get_registered_object",
    ],
    "models->saving": ["load_model"],
}

if not hasattr(keras, "saving"):
    keras.saving = types.SimpleNamespace()

# add aliases
for key, value in _KERAS_CORE_ALIASES.items():
    src, _, dst = key.partition("->")
    src = src.split(".")
    dst = dst.split(".")

    src_mod, dst_mod = keras, keras

    # navigate to where we want to alias the attributes
    for mod in src:
        src_mod = getattr(src_mod, mod)
    for mod in dst:
        dst_mod = getattr(dst_mod, mod)

    # add an alias for each attribute
    for attr in value:
        if isinstance(attr, tuple):
            src_attr, dst_attr = attr
        else:
            src_attr, dst_attr = attr, attr
        attr_val = getattr(src_mod, src_attr)
        setattr(dst_mod, dst_attr, attr_val)

# TF Keras doesn't have this rename.
keras.activations.silu = keras.activations.swish
