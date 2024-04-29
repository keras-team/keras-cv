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


from tensorflow import keras

from keras_cv.src.models.backbones.csp_darknet.csp_darknet_utils import (
    DarknetConvBlock,
)
from keras_cv.src.models.backbones.csp_darknet.csp_darknet_utils import (
    DarknetConvBlockDepthwise,
)


class YoloXHead(keras.layers.Layer):
    """The YoloX prediction head.

    Arguments:
        num_classes: The number of classes to be considered for the
            classification head.
        bias_initializer: Bias Initializer for the final convolution layer for
            the classification and regression heads. Defaults to None.
        width_multiplier: A float value used to calculate the base width of the
            model this changes based on the detection model being used. Defaults
            to 1.0.
        num_level: the number of levels in the FPN output. Defaults to 3.
        activation: the activation applied after the BatchNorm layer. One of
            "silu", "relu" or "leaky_relu". Defaults to "silu".
        use_depthwise: a boolean value used to decide whether a depthwise conv
            block should be used over a regular darknet block. Defaults to
            `False`.
    """

    def __init__(
        self,
        num_classes,
        bias_initializer=None,
        width_multiplier=1.0,
        num_level=3,
        activation="silu",
        use_depthwise=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.stems = []

        self.classification_convs = []
        self.regression_convs = []

        self.classification_preds = []
        self.regression_preds = []
        self.objectness_preds = []

        ConvBlock = (
            DarknetConvBlockDepthwise if use_depthwise else DarknetConvBlock
        )

        for _ in range(num_level):
            self.stems.append(
                DarknetConvBlock(
                    filters=int(256 * width_multiplier),
                    kernel_size=1,
                    strides=1,
                    activation=activation,
                )
            )

            self.classification_convs.append(
                keras.Sequential(
                    [
                        ConvBlock(
                            filters=int(256 * width_multiplier),
                            kernel_size=3,
                            strides=1,
                            activation=activation,
                        ),
                        ConvBlock(
                            filters=int(256 * width_multiplier),
                            kernel_size=3,
                            strides=1,
                            activation=activation,
                        ),
                    ]
                )
            )

            self.regression_convs.append(
                keras.Sequential(
                    [
                        ConvBlock(
                            filters=int(256 * width_multiplier),
                            kernel_size=3,
                            strides=1,
                            activation=activation,
                        ),
                        ConvBlock(
                            filters=int(256 * width_multiplier),
                            kernel_size=3,
                            strides=1,
                            activation=activation,
                        ),
                    ]
                )
            )

            self.classification_preds.append(
                keras.layers.Conv2D(
                    filters=num_classes,
                    kernel_size=1,
                    strides=1,
                    padding="same",
                    bias_initializer=bias_initializer,
                )
            )
            self.regression_preds.append(
                keras.layers.Conv2D(
                    filters=4,
                    kernel_size=1,
                    strides=1,
                    padding="same",
                    bias_initializer=bias_initializer,
                )
            )
            self.objectness_preds.append(
                keras.layers.Conv2D(
                    filters=1,
                    kernel_size=1,
                    strides=1,
                    padding="same",
                )
            )

    def call(self, inputs, training=False):
        outputs = []

        for i, p_i in enumerate(inputs):
            stem = self.stems[i](p_i)

            classes = self.classification_convs[i](stem)
            classes = self.classification_preds[i](classes)

            boxes_feat = self.regression_convs[i](stem)
            boxes = self.regression_preds[i](boxes_feat)
            objectness = self.objectness_preds[i](boxes_feat)

            output = keras.layers.Concatenate(axis=-1)(
                [boxes, objectness, classes]
            )
            outputs.append(output)
        return outputs
