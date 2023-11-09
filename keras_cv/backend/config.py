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
import json
import os

_MULTI_BACKEND = False

# Set Keras base dir path given KERAS_HOME env variable, if applicable.
# Otherwise either ~/.keras or /tmp.
if "KERAS_HOME" in os.environ:
    _keras_dir = os.environ.get("KERAS_HOME")
else:
    _keras_base_dir = os.path.expanduser("~")
    if not os.access(_keras_base_dir, os.W_OK):
        _keras_base_dir = "/tmp"
    _keras_dir = os.path.join(_keras_base_dir, ".keras")


def detect_if_tensorflow_uses_keras_3():
    # We follow the version of keras that tensorflow is configured to use.
    from tensorflow import keras

    # Note that only recent versions of keras have a `version()` function.
    if hasattr(keras, "version") and keras.version().startswith("3."):
        return True

    # No `keras.version()` means we are on an old version of keras.
    return False


_USE_KERAS_3 = detect_if_tensorflow_uses_keras_3()
if _USE_KERAS_3:
    _MULTI_BACKEND = True


def keras_3():
    """Check if Keras 3 is being used."""
    return _USE_KERAS_3


# Attempt to read KerasCV config file.
_config_path = os.path.expanduser(os.path.join(_keras_dir, "keras_cv.json"))
if os.path.exists(_config_path):
    try:
        with open(_config_path) as f:
            _config = json.load(f)
    except ValueError:
        _config = {}
    _MULTI_BACKEND = _config.get("multi_backend", _MULTI_BACKEND)

# Save config file, if possible.
if not os.path.exists(_keras_dir):
    try:
        os.makedirs(_keras_dir)
    except OSError:
        # Except permission denied and potential race conditions
        # in multi-threaded environments.
        pass

if not os.path.exists(_config_path):
    _config = {
        "multi_backend": _MULTI_BACKEND,
    }
    try:
        with open(_config_path, "w") as f:
            f.write(json.dumps(_config, indent=4))
    except IOError:
        # Except permission denied.
        pass

if "KERAS_BACKEND" in os.environ and os.environ["KERAS_BACKEND"]:
    _MULTI_BACKEND = True


def multi_backend():
    return _MULTI_BACKEND


def backend():
    """Check the backend framework."""
    if not multi_backend():
        return "tensorflow"
    if not keras_3():
        import keras_core

        return keras_core.config.backend()

    from tensorflow import keras

    return keras.config.backend()
