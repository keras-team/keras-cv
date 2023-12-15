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


def detect_if_tensorflow_uses_keras_3():
    # We follow the version of keras that tensorflow is configured to use.
    try:
        from tensorflow import keras

        # Note that only recent versions of keras have a `version()` function.
        if hasattr(keras, "version") and keras.version().startswith("3."):
            return True
    except:
        raise ValueError(
            "Unable to import `keras` with `tensorflow`.  Please check your "
            "Keras and Tensorflow version are compatible; Keras 3 requires "
            "TensorFlow 2.15 or later. See keras.io/getting_started for more "
            "information on installing Keras."
        )

    # No `keras.version()` means we are on an old version of keras.
    return False


_USE_KERAS_3 = detect_if_tensorflow_uses_keras_3()


def keras_3():
    """Check if Keras 3 is being used."""
    return _USE_KERAS_3


def backend():
    """Check the backend framework."""
    if not keras_3():
        return "tensorflow"

    import keras

    return keras.config.backend()
