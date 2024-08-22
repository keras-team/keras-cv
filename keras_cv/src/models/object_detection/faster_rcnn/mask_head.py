# Copyright 2024 The KerasCV Authors
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

from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.backend import keras

@keras_cv_export(
    "keras_cv.models.mask_rcnn.RCNNHead",
    package="keras_cv.models.mask_rcnn",
)

class MaskHead(keras.layers.Layer):
    def __init__(
        self,
        num_classes,
        num_conv = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.num_conv = num_conv
        self.layers = []
        # architecture is from https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/model.py
        for _ in range(num_conv):
            conv = keras.layers.TimeDistributed(
                keras.layers.Conv2D(
                    filters=256,
                    kernel_size=3,
                    padding="same",
                )
            )
            bn = keras.layers.TimeDistributed(keras.layers.BatchNormalization())
            activation = keras.layers.Activation('relu')
            self.layers.extend([conv, bn, activation])

        self.deconv = keras.layers.TimeDistributed(
            keras.layers.Conv2DTranspose(
                256,
                kernel_size=2,
                strides=2,
                activation="relu",
                padding="valid",
            )
        )
        self.mask_output = keras.layers.TimeDistributed(
            keras.layers.Conv2D(
                num_classes + 1,
                kernel_size=1,
                strides=1,
                activation="linear" # no activation, as we're using from_logits
            )
        )

    def call(self, feature_map, training=False):
        x = feature_map
        for layer in self.layers:
            x = layer(x, training=training)
        x = self.deconv(x)
        mask = self.mask_output(x)
        return mask

    def build(self, input_shape):
        intermediate_shape = input_shape
        for idx in range(self.num_conv):
            self.layers[idx*3].build(intermediate_shape)
            intermediate_shape = tuple(intermediate_shape[:-1]) + (
                256,
            )
            self.layers[idx*3+1].build(intermediate_shape)
        self.deconv.build(intermediate_shape)
        intermediate_shape = tuple(intermediate_shape[:-3]) + (
            intermediate_shape[-3]*2, intermediate_shape[-2]*2, 256
        )
        self.mask_output.build(intermediate_shape)
        self.built = True

    def get_config(self):
        config = super().get_config()
        config["num_classes"] = self.num_classes
        config["num_conv"] = self.num_conv

        return config
