# Copyright 2022 The KerasCV Authors
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
from keras_cv.src.backend import ops


@keras_cv_export(
    "keras_cv.models.retinanet.FeaturePyramid",
    package="keras_cv.models.retinanet",
)
class FeaturePyramid(keras.layers.Layer):
    """Builds the Feature Pyramid with the feature maps from the backbone."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv_c3_1x1 = keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c4_1x1 = keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c5_1x1 = keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c3_3x3 = keras.layers.Conv2D(256, 3, 1, "same")
        self.conv_c4_3x3 = keras.layers.Conv2D(256, 3, 1, "same")
        self.conv_c5_3x3 = keras.layers.Conv2D(256, 3, 1, "same")
        self.conv_c6_3x3 = keras.layers.Conv2D(256, 3, 2, "same")
        self.conv_c7_3x3 = keras.layers.Conv2D(256, 3, 2, "same")
        self.upsample_2x = keras.layers.UpSampling2D(2)

    def call(self, inputs, training=False):
        if isinstance(inputs, dict):
            c3_output = inputs["P3"]
            c4_output = inputs["P4"]
            c5_output = inputs["P5"]
        else:
            c3_output, c4_output, c5_output = inputs
        p3_output = self.conv_c3_1x1(c3_output, training=training)
        p4_output = self.conv_c4_1x1(c4_output, training=training)
        p5_output = self.conv_c5_1x1(c5_output, training=training)
        p4_output = p4_output + self.upsample_2x(p5_output, training=training)
        p3_output = p3_output + self.upsample_2x(p4_output, training=training)
        p3_output = self.conv_c3_3x3(p3_output, training=training)
        p4_output = self.conv_c4_3x3(p4_output, training=training)
        p5_output = self.conv_c5_3x3(p5_output, training=training)
        p6_output = self.conv_c6_3x3(c5_output, training=training)
        p7_output = self.conv_c7_3x3(ops.relu(p6_output), training=training)
        return p3_output, p4_output, p5_output, p6_output, p7_output

    def build(self, input_shape):
        p3_channels = input_shape["P3"][-1]
        p4_channels = input_shape["P4"][-1]
        p5_channels = input_shape["P5"][-1]
        self.conv_c3_1x1.build((None, None, None, p3_channels))
        self.conv_c4_1x1.build((None, None, None, p4_channels))
        self.conv_c5_1x1.build((None, None, None, p5_channels))
        self.conv_c3_3x3.build((None, None, None, 256))
        self.conv_c4_3x3.build((None, None, None, 256))
        self.conv_c5_3x3.build((None, None, None, 256))
        self.conv_c6_3x3.build((None, None, None, p5_channels))
        self.conv_c7_3x3.build((None, None, None, 256))
        self.built = True

    def compute_output_shape(self, input_shape):
        p3_shape = input_shape["P3"][:-1]
        p4_shape = input_shape["P4"][:-1]
        p5_shape = input_shape["P5"][:-1]
        return (
            (tuple(p3_shape) + (256,)),
            (tuple(p4_shape) + (256,)),
            (tuple(p5_shape) + (256,)),
            (tuple(p5_shape) + (256,)),
            (tuple(p5_shape) + (256,)),
        )
