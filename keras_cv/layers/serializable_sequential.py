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

from keras_cv.api_export import keras_cv_export
from keras_cv.backend import keras
from keras_cv.utils.python_utils import classproperty


# TODO(tirthasheshpatel): Use `Sequential` model once the bug is resolved.
# Temporarily substitute the `Sequential` model with this because a
# bug in Keras/Keras Core prevents the weights of a sequential model to
# load in TensorFlow if they are saved in JAX/PyTorch and vice versa.
# This only happens when the `build` method is called in the `__init__`
# step.
@keras_cv_export("keras_cv.layers.SerializableSequential")
class SerializableSequential(keras.layers.Layer):
    def __init__(self, layers_list, **kwargs):
        super().__init__(**kwargs)
        self.layers_list = layers_list

    def build(self, input_shape):
        output_shape = input_shape
        for layer in self.layers_list:
            layer.build(output_shape)
            output_shape = layer.compute_output_shape(output_shape)
        self.built = True

    def call(self, x):
        for layer in self.layers_list:
            x = layer(x)
        return x

    def get_config(self):
        config = super().get_config()
        layers_list_serialized = [
            keras.saving.serialize_keras_object(layer)
            for layer in self.layers_list
        ]
        config.update({"layers_list": layers_list_serialized})

    @classproperty
    def from_config(self, config):
        config.update(
            {
                "layers_list": [
                    keras.layers.deserialize(layer)
                    for layer in config["layers_list"]
                ]
            }
        )
        return super().from_config(config)
