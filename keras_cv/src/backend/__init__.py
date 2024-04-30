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

This module adds a temporary Keras API surface that is fully under KerasCV
control. The goal is to allow us to write Keras 3-like code everywhere, while
still supporting Keras 2. We do this by using the `keras_core` package with
Keras 2 to backport Keras 3 numerics APIs (`keras.ops` and `keras.random`) into
Keras 2. The sub-modules exposed are as follows:

- `config`: check which version of Keras is being run.
- `keras`: The full `keras` API with compat shims for older Keras versions.
- `ops`: `keras.ops` for Keras 3 or `keras_core.ops` for Keras 2.
- `random`: `keras.random` for Keras 3 or `keras_core.ops` for Keras 2.
"""
from keras_cv.src.backend import config  # noqa: E402

if not config.keras_3():
    import keras_cv.src.backend.keras2 as keras  # noqa: E402
else:
    import keras  # noqa: E402

    keras.backend.name_scope = keras.name_scope

from keras_cv.src.backend import ops  # noqa: E402
from keras_cv.src.backend import random  # noqa: E402
from keras_cv.src.backend import tf_ops  # noqa: E402


def assert_tf_keras(src):
    if config.keras_3():
        raise NotImplementedError(
            f"KerasCV component {src} does not yet support Keras 3, and can "
            "only be used in `tf.keras`."
        )


def supports_ragged():
    return not config.keras_3()
