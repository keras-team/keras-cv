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

import os

from tensorflow import keras

# We follow the version of keras that tensorflow is configured to use.
_USE_KERAS_3 = False

# Note that only recent versions of keras have a `version()` function.
if hasattr(keras, "version") and keras.version().startswith("3."):
    _USE_KERAS_3 = True

if not _USE_KERAS_3:
    backend = os.environ.get("KERAS_BACKEND")
    if backend and backend != "tensorflow":
        raise RuntimeError(
            "When running Keras 2, the `KERAS_BACKEND` environment variable "
            f"must either be unset or `'tensorflow'`. Received: `{backend}`. "
            "To set another backend, please install Keras 3. See "
            "https://github.com/keras-team/keras-cv#installation"
        )


def keras_3():
    """Check if Keras 3 is being used."""
    return _USE_KERAS_3


def backend():
    """Check the backend framework."""
    if not keras_3():
        return "tensorflow"

    return keras.config.backend()
