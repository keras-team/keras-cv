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

import tensorflow as tf

from keras_cv import layers


class RetinaNetFPN(tf.keras.layers.Layer):
    """Implements RetinaNet' modified Feature Pyramid Network

    RetinaNet uses the FPN structure to detect object at different scales. While
    the structure is the same as FPNs, RetinaNet proposes a slight modification
    which includes exit convolutions on P2, P3, P4, and P5. This FPN further optionally
    downscales the C5 features using Conv2D layers to form P6 and P7.

    Features must be passed only for levels less than 6. This layer further supports
    levels only upto level 7.

    Args:
        pyramid_levels: a python list that specifies all the values of level `i`
            for which the feature Ci will be specified. The size of this list is
            num_pyramid_levels. Defaults to [3, 4, 5, 6, 7].
        num_channels: an integer representing the number of channels for the FPN
            operations. Defaults to 256.

    References:
        [Focal Loss for Dense Object Detection paper](https://arxiv.org/pdf/1708.02002)
        [Keras.io RetinaNet Tutorial](https://keras.io/examples/vision/retinanet/)

    Sample Usage:
    ```python
    inp = tf.keras.layers.Input((512, 512, 3))
    backbone = keras.applications.ResNet50(include_top=False, input_tensor = inp)

    # only C3, C4 and C5 are passed, P6 and P7 are extracted from C5
    layer_names = ["conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]
    backbone_outputs = [backbone.get_layer(name).output for name in layer_names]

    # features is a dict with P3, P4, P5, P6 and P7 as keys and the outputs as values
    features = keras_cv.layers.RetinaNetFPN()(backbone_outputs)
    ```
    """

    def __init__(self, pyramid_levels=[3, 4, 5, 6, 7], num_channels=256, **kwargs):
        super(RetinaNetFPN, self).__init__(name="RetinaNetFPN", **kwargs)
        self.pyramid_levels = sorted(pyramid_levels)
        self.num_pyramid_levels = len(self.pyramid_levels)
        self.num_channels = num_channels

        # the levels that we can reuse from the FPN implementation
        self.fpn_levels = self.pyramid_levels

        if 6 not in self.fpn_levels and 7 in self.fpn_levels:
            raise ValueError(
                f"Pyramid level 6 is required to compute pyramid level 7. Received "
                f"`pyramid_levels`={self.pyramid_levels}"
            )

        self.P6_top_down = None
        self.P7_top_down = None

        # handle cases for P6 and P7
        if 6 in self.pyramid_levels:
            self.P6_top_down = tf.keras.layers.Conv2D(
                self.num_channels,
                kernel_size=3,
                strides=2,
                padding="same",
                name="top_down_P6",
            )

            # overwrite fpn_levels to exclude level 6
            self.fpn_levels = self.pyramid_levels[:-1]

            if 7 in self.pyramid_levels:
                self.P7_top_down = tf.keras.layers.Conv2D(
                    self.num_channels,
                    kernel_size=3,
                    strides=2,
                    padding="same",
                    name="top_down_P7",
                )

                # overwrite fpn_levels to exclude level 7
                self.fpn_levels = self.pyramid_levels[:-2]

        # throw an error if levels 3, 4 and 5 are not present
        if (
            3 not in self.pyramid_levels
            or 4 not in self.pyramid_levels
            or 5 not in self.pyramid_levels
        ):
            raise ValueError(
                f"Pyramid levels 3, 4, and 5 required for functionality of "
                f"RetinaNetFPN. Received `pyramid_levels`={self.pyramid_levels}"
            )

        # layers 2, 3, 4 and 5 will follow an fpn structure
        self.fpn = layers.FeaturePyramid(self.fpn_levels, self.num_channels)

        # RetinaNetFPN applies a conv layer to outputs
        self.exit_ops = {}
        for i in self.fpn_levels:
            self.exit_ops[f"P{i}"] = tf.keras.layers.Conv2D(
                self.num_channels,
                kernel_size=3,
                strides=1,
                padding="same",
                name=f"P{i}_out",
            )

    def call(self, features):
        output_features = self.fpn(features)

        # apply exit convs on all fpn levels
        for level, feature in output_features.items():
            output_features[level] = self.exit_ops[level](feature)

        # generate P6 and P7 using C5
        if self.P6_top_down:
            output_features["P6"] = self.P6_top_down(features[-1])
        if self.P7_top_down:
            output_features["P7"] = self.P7_top_down(tf.nn.relu(output_features["P6"]))

        return output_features

    def get_config(self):
        config = {
            "pyramid_levels": self.pyramid_levels,
            "num_channels": self.num_channels,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
