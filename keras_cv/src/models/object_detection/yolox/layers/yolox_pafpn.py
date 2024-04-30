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
    CrossStagePartial,
)
from keras_cv.src.models.backbones.csp_darknet.csp_darknet_utils import (
    DarknetConvBlock,
)
from keras_cv.src.models.backbones.csp_darknet.csp_darknet_utils import (
    DarknetConvBlockDepthwise,
)


class YoloXPAFPN(keras.layers.Layer):
    """The YoloX PAFPN.

    YoloX PAFPN is an FPN layer used in YoloX models. The YoloX PAFPN is based
    on the feature pyramid module used in Path Aggregation networks (PANet).

    Arguments:
        depth_multiplier: A float value used to calculate the base depth of the
            model this changes based on the detection model being used. Defaults
            to 1.0.
        width_multiplier: A float value used to calculate the base width of the
            model this changes based on the detection model being used. Defaults
            to 1.0.
        in_channels: A list representing the number of filters in the FPN
            output. The length of the list will be same as the number of
            outputs. Defaults to `(256, 512, 1024)`.
        use_depthwise: a boolean value used to decide whether a depthwise conv
            block should be used over a regular darknet block. Defaults to
            `False`.
        activation: the activation applied after the BatchNorm layer. One of
            `"silu"`, `"relu"` or `"leaky_relu"`. Defaults to `"silu"`.
    """

    def __init__(
        self,
        depth_multiplier=1.0,
        width_multiplier=1.0,
        in_channels=(256, 512, 1024),
        use_depthwise=False,
        activation="silu",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.in_channels = in_channels

        ConvBlock = (
            DarknetConvBlockDepthwise if use_depthwise else DarknetConvBlock
        )

        self.lateral_conv0 = DarknetConvBlock(
            filters=int(in_channels[1] * width_multiplier),
            kernel_size=1,
            strides=1,
            activation=activation,
        )
        self.C3_p4 = CrossStagePartial(
            filters=int(in_channels[1] * width_multiplier),
            num_bottlenecks=round(3 * depth_multiplier),
            residual=False,
            use_depthwise=use_depthwise,
            activation=activation,
        )

        self.reduce_conv1 = DarknetConvBlock(
            filters=int(in_channels[0] * width_multiplier),
            kernel_size=1,
            strides=1,
            activation=activation,
        )
        self.C3_p3 = CrossStagePartial(
            filters=int(in_channels[0] * width_multiplier),
            num_bottlenecks=round(3 * depth_multiplier),
            residual=False,
            use_depthwise=use_depthwise,
            activation=activation,
        )

        self.bu_conv2 = ConvBlock(
            filters=int(in_channels[0] * width_multiplier),
            kernel_size=3,
            strides=2,
            activation=activation,
        )
        self.C3_n3 = CrossStagePartial(
            filters=int(in_channels[1] * width_multiplier),
            num_bottlenecks=round(3 * depth_multiplier),
            residual=False,
            use_depthwise=use_depthwise,
            activation=activation,
        )

        self.bu_conv1 = ConvBlock(
            filters=int(in_channels[1] * width_multiplier),
            kernel_size=3,
            strides=2,
            activation=activation,
        )
        self.C3_n4 = CrossStagePartial(
            filters=int(in_channels[2] * width_multiplier),
            num_bottlenecks=round(3 * depth_multiplier),
            residual=False,
            use_depthwise=use_depthwise,
            activation=activation,
        )

        self.concat = keras.layers.Concatenate(axis=-1)
        self.upsample_2x = keras.layers.UpSampling2D(2)

    def call(self, inputs, training=False):
        c3_output, c4_output, c5_output = inputs[3], inputs[4], inputs[5]

        fpn_out0 = self.lateral_conv0(c5_output)
        f_out0 = self.upsample_2x(fpn_out0)
        f_out0 = self.concat([f_out0, c4_output])
        f_out0 = self.C3_p4(f_out0)

        fpn_out1 = self.reduce_conv1(f_out0)
        f_out1 = self.upsample_2x(fpn_out1)
        f_out1 = self.concat([f_out1, c3_output])
        pan_out2 = self.C3_p3(f_out1)

        p_out1 = self.bu_conv2(pan_out2)
        p_out1 = self.concat([p_out1, fpn_out1])
        pan_out1 = self.C3_n3(p_out1)

        p_out0 = self.bu_conv1(pan_out1)
        p_out0 = self.concat([p_out0, fpn_out0])
        pan_out0 = self.C3_n4(p_out0)

        return pan_out2, pan_out1, pan_out0
