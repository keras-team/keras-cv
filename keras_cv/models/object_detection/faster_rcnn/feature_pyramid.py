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


@keras_cv_export(
    "keras_cv.models.faster_rcnn.FeaturePyramid",
    package="keras_cv.models.faster_rcnn",
)
class FeaturePyramid(keras.layers.Layer):
    """Builds the Feature Pyramid with the feature maps from the backbone."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv_c2_1x1 = keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c3_1x1 = keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c4_1x1 = keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c5_1x1 = keras.layers.Conv2D(256, 1, 1, "same")

        self.conv_c2_3x3 = keras.layers.Conv2D(256, 3, 1, "same")
        self.conv_c3_3x3 = keras.layers.Conv2D(256, 3, 1, "same")
        self.conv_c4_3x3 = keras.layers.Conv2D(256, 3, 1, "same")
        self.conv_c5_3x3 = keras.layers.Conv2D(256, 3, 1, "same")
        self.conv_c6_3x3 = keras.layers.Conv2D(256, 3, 1, "same")
        self.conv_c6_pool = keras.layers.MaxPool2D()
        self.upsample_2x = keras.layers.UpSampling2D(2)

    def call(self, inputs, training=None):
        c2_output = inputs["P2"]
        c3_output = inputs["P3"]
        c4_output = inputs["P4"]
        c5_output = inputs["P5"]

        c6_output = self.conv_c6_pool(c5_output)
        p6_output = c6_output
        p5_output = self.conv_c5_1x1(c5_output)
        p4_output = self.conv_c4_1x1(c4_output)
        p3_output = self.conv_c3_1x1(c3_output)
        p2_output = self.conv_c2_1x1(c2_output)

        p4_output = p4_output + self.upsample_2x(p5_output)
        p3_output = p3_output + self.upsample_2x(p4_output)
        p2_output = p2_output + self.upsample_2x(p3_output)

        p6_output = self.conv_c6_3x3(p6_output)
        p5_output = self.conv_c5_3x3(p5_output)
        p4_output = self.conv_c4_3x3(p4_output)
        p3_output = self.conv_c3_3x3(p3_output)
        p2_output = self.conv_c2_3x3(p2_output)

        return {
            "P2": p2_output,
            "P3": p3_output,
            "P4": p4_output,
            "P5": p5_output,
            "P6": p6_output,
        }

    def get_config(self):
        config = {}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
