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


# TODO(scottzhu): Register it later due to the conflict in the retina_net
# @tf.keras.utils.register_keras_serializable(package="keras_cv")
class FeaturePyramid(tf.keras.layers.Layer):
    """Implements a Feature Pyramid Network.

    This implements the paper:
      Tsung-Yi Lin, Piotr Dollar, Ross Girshick, Kaiming He, Bharath Hariharan, and
      Serge Belongie.
      Feature Pyramid Networks for Object Detection.
      (https://arxiv.org/pdf/1612.03144)

    Feature Pyramid Networks (FPNs) are basic components that are added to an
    existing CNN to extract features at different scales. For the basic FPN, the
    inputs are features `Ci` from different levels of a CNN, where the feature is
    scaled from the image size by `1/2^i` for any level `i`.

    There is an output associated with each level in the basic FPN. The output Pi
    at level `i` (corresponding to Ci) is given by performing a merge operation on
    the outputs of:

    1) a lateral operation on Ci (usually a conv2D layer with kernel = 1 and strides = 1)
    2) a top-down upsampling operation from Pi+1 (except for the top most level)

    The final output of each level  will also have a conv2D operation
    (usually with kernel = 3 and strides = 1).

    The inputs to the layer should be a dict with int keys should match the
    pyramid_levels, e.g. for `pyramid_levels` = [2,3,4,5], the expected input dict should
    be `{2:c2, 3:c3, 4:c4, 5:c5}`.

    The output of the layer will have same structures as the inputs, a dict with int keys
    and value for each of the level.

    Args:
        pyramid_levels: a sorted python int list that specifies all the values of level
            `i` for which the feature Ci will be specified.
        num_channels: an integer representing the number of channels for the FPN
            operations. Defaults to 256.
        lateral_layers: a python dict with int keys that matches to each of the pyramid
            level. The values of the dict should be `keras.Layer`, which will be called
            with feature inputs from backbone at each level. Default to None, and a
            `keras.Conv2D` layer with kernel 1x1 will be created for each pyramid level.
        output_layers: a python dict with int keys that matches to each of the pyramid
            level. The values of the dict should be `keras.Layer`, which will be called
            with feature inputs and merged result from upstream levels. Default to None,
            and a `keras.Conv2D` layer with kernel 3x3 will be created for each pyramid
            level.
        top_down_op: optional upsampling op between each layer. Default to None, and
            `keras.layers.UpSampling2D` with 2x will be used.
        merge_op: optional merge op for lateral result and upstream result. Default to
            None, and `keras.layers.Add` will be used.

    Sample Usage:
    ```python

    inp = tf.keras.layers.Input((384, 384, 3))
    backbone = tf.keras.applications.EfficientNetB0(input_tensor=inp, include_top=False)
    layer_names = ['block2b_add', 'block3b_add', 'block5c_add', 'top_activation']

    backbone_outputs = {}
    for i, layer_name in enumerate(layer_names):
        backbone_outputs[i+2] = backbone.get_layer(layer_name).output

    # output_dict is a dict with 2, 3, 4, 5 as keys
    output_dict = keras_cv.layers.FeaturePyramid([2,3,4,5])(backbone_outputs)
    ```
    """

    def __init__(
        self,
        pyramid_levels,
        num_channels=256,
        lateral_layers=None,
        output_layers=None,
        top_down_op=None,
        merge_op=None,
        **kwargs,
    ):
        super(FeaturePyramid, self).__init__(**kwargs)
        self.pyramid_levels = sorted(pyramid_levels)
        self.num_channels = num_channels

        # required for successful serialization
        self.lateral_layers_passed = lateral_layers
        self.output_layers_passed = output_layers
        self.top_down_op_passed = top_down_op
        self.merge_op_passed = merge_op

        if not lateral_layers:
            # populate self.lateral_ops with default FPN Conv2D 1X1 layers
            self.lateral_layers = {}
            for i in self.pyramid_levels:
                self.lateral_layers[i] = tf.keras.layers.Conv2D(
                    self.num_channels,
                    kernel_size=1,
                    strides=1,
                    padding="same",
                    name=f"lateral_P{i}",
                )

        else:
            if (
                not isinstance(lateral_layers, dict)
                or sorted(lateral_layers.keys()) != self.pyramid_levels
            ):
                raise ValueError(
                    f"Expect lateral_layers to be a dict with keys as "
                    f"{self.pyramid_levels}, got {lateral_layers}"
                )
            self.lateral_layers = lateral_layers

        # Output conv2d layers.
        if not output_layers:
            self.output_layers = {}
            for i in self.pyramid_levels:
                self.output_layers[i] = tf.keras.layers.Conv2D(
                    self.num_channels,
                    kernel_size=3,
                    strides=1,
                    padding="same",
                    name=f"output_P{i}",
                )
        else:
            if (
                not isinstance(output_layers, dict)
                or sorted(output_layers.keys()) != self.pyramid_levels
            ):
                raise ValueError(
                    f"Expect output_layers to be a dict with keys as "
                    f"{self.pyramid_levels}, got {output_layers}"
                )
            self.output_layers = output_layers

        # the same upsampling layer is used for all levels
        if not top_down_op:
            self.top_down_op = tf.keras.layers.UpSampling2D(size=2)
        else:
            self.top_down_op = top_down_op
        # the same merge layer is used for all levels
        if not merge_op:
            self.merge_op = tf.keras.layers.Add()
        else:
            self.merge_op = merge_op

    def call(self, features):
        # Note that this assertion might not be true for all the subclasses. It is
        # possible to have FPN that has high levels than the height of backbone outputs.
        if (
            not isinstance(features, dict)
            or sorted(features.keys()) != self.pyramid_levels
        ):
            raise ValueError(
                "Expect the input features to be a dict with int keys that"
                f"matches to the {self.pyramid_levels}, got: {features}"
            )
        return self.build_feature_pyramid(features)

    def build_feature_pyramid(self, input_features):
        # To illustrate the connection/topology, the basic flow for a FPN with level
        # 3, 4, 5 is like below:
        #
        # input_l5 -> conv2d_1x1_l5 ----V---> conv2d_3x3_l5 -> output_l5
        #                               V
        #                          upsample2d
        #                               V
        # input_l4 -> conv2d_1x1_l4 -> Add -> conv2d_3x3_l4 -> output_l4
        #                               V
        #                          upsample2d
        #                               V
        # input_l3 -> conv2d_1x1_l3 -> Add -> conv2d_3x3_l3 -> output_l3

        output_features = {}
        reversed_levels = list(sorted(input_features.keys(), reverse=True))
        top_level = reversed_levels[0]
        for level in reversed_levels:
            output = self.lateral_layers[level](input_features[level])
            if level < top_level:
                # for the top most output, it doesn't need to merge with any upper stream
                # outputs
                upstream_output = self.top_down_op(output_features[level + 1])
                output = self.merge_op([output, upstream_output])
            output_features[level] = output

        # Post apply the output layers so that we don't leak them to the down stream level
        for level in reversed_levels:
            output_features[level] = self.output_layers[level](output_features[level])

        return output_features

    # TODO(scottzhu): Add serialization with get_config/from_config(). It might have
    # some issue for sub-layers serialization.
