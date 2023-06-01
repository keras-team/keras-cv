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
from keras_cv.backend.config import multi_backend

if multi_backend():
    import keras_core as keras

    if not hasattr(keras.utils, "register_keras_serializable"):
        keras.utils.register_keras_serializable = (
            keras.saving.register_keras_serializable
        )
    if not hasattr(keras.utils, "deserialize_keras_object"):
        keras.utils.deserialize_keras_object = (
            keras.saving.deserialize_keras_object
        )
    if not hasattr(keras.utils, "serialize_keras_object"):
        keras.utils.serialize_keras_object = keras.saving.serialize_keras_object
    if not hasattr(keras.utils, "get_file"):
        keras.utils.get_file = keras.utils.file_utils.get_file
    if not hasattr(keras.utils, "get_registered_object"):
        keras.utils.get_registered_object = keras.saving.get_registered_object

    if not hasattr(keras.layers, "serialize"):
        keras.layers.serialize = keras.saving.serialize_keras_object
    if not hasattr(keras.layers, "deserialize"):
        keras.layers.deserialize = keras.saving.deserialize_keras_object

    if not hasattr(keras.models, "load_model"):
        keras.models.load_model = keras.saving.load_model

    if not hasattr(keras.backend, "get_uid"):
        keras.backend.get_uid = keras.utils.naming.get_uid
else:
    from tensorflow import keras


from keras_cv.backend import ops
