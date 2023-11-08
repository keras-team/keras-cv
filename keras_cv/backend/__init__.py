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
"""
import types

from keras_cv.backend.config import (
    detect_if_tensorflow_uses_keras_3,
    multi_backend,
)
from packaging.version import parse

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

if (not detect_if_tensorflow_uses_keras_3()) and multi_backend():
    import keras

    if not hasattr(keras, "__version__") or parse(keras.__version__) < parse(
        "3.0"
    ):
        import keras_core as keras

    keras.backend.name_scope = keras.name_scope
else:
    from tensorflow import keras

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
"""
from keras_cv.backend import config  # noqa: E402
from keras_cv.backend import keras  # noqa: E402
from keras_cv.backend import ops  # noqa: E402
from keras_cv.backend import random  # noqa: E402
from keras_cv.backend import tf_ops  # noqa: E402
