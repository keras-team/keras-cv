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


class FeaturePyramid(tf.keras.layers.Layer):
    """Implements a Feature Pyramid Network.

    Feature Pyramid Networks (FPNs) are basic components that are added to an
    existing CNN to extract features at different scales. For the basic FPN, the
    inputs are features `Ci` from different levels of a CNN, where the feature is
    scaled from the image size by `1/2^i` for any level `i`.

    There is an output associated with each input in the basic FPN. The output Pi
    at level `i` (corresponding to Ci) is given by performing a merge operation on
    the outputs after:
    1) a top down operation on Pj (where j = i - 1)
    2) a lateral operation on Ci

    Args:
        pyramid_levels: a python list that specifies all the values of level `i`
            for which the feature Ci will be specified. The size of this list is
            num_pyramid_levels.
        num_channels: an integer representing the number of channels for the FPN
            operations. Defaults to 256.
        lateral_ops: a python list of size num_pyramid_levels that specifies all the
            lateral operations performed by the FPN. Each operation from the list must
            match the corresponding level from pyramid_levels list. Passing None will
            populate the list with the default FPN operation (a 1X1 Conv2D layer).
            Defaults to None.
        top_down_ops: a python list of size num_pyramid_levels that specifies all the
            top down operations performed by the FPN. Each operation from the list must
            match the corresponding level from pyramid_levels list. Passing None will
            populate the list with the default FPN operation (an UpSampling2D layer).
            Defaults to None.
        merge_ops: a python list of size (num_pyramid_levels - 1) that specifies all
            the merge operations performed by the FPN. Each operation from the list must
            match the corresponding level from pyramid_levels list. The top-most layer
            is not merged and the corresponding operation is therefore not included.
            Passing None will populate the list with the default FPN operation
            (a 1X1 Conv2D layer). Defaults to None.

    References:
        [FPN paper](https://arxiv.org/pdf/1612.03144)

    Sample Usage:
    ```python
    inp = tf.keras.layers.Input((384, 384, 3))
    efnb0 = tf.keras.applications.EfficientNetB0(input_tensor=inp, include_top = False)

    layer_names = ['block2b_add', 'block3b_add', 'block5c_add', 'top_activation']
    backbone_outputs = [efnb0.get_layer(name).output for name in layer_names]

    features = FeaturePyramid([2,3,4,5])(backbone_outputs)
    P2, P3, P4, P5 = features.values()
    ```
    """

    def __init__(
        self,
        pyramid_levels,
        num_channels=256,
        lateral_ops=None,
        top_down_ops=None,
        merge_ops=None,
        **kwargs,
    ):
        super(FeaturePyramid, self).__init__(name="FPN", **kwargs)
        self.pyramid_levels = sorted(pyramid_levels)
        self.num_pyramid_levels = len(self.pyramid_levels)
        self.num_channels = num_channels

        if not lateral_ops:
            # populate self.lateral_ops with default FPN Conv2D 1X1 layers
            self.lateral_ops = []
            for i in pyramid_levels:
                self.lateral_ops.append(
                    tf.keras.layers.Conv2D(
                        self.num_channels,
                        kernel_size=1,
                        strides=1,
                        padding="same",
                        name=f"lateral_P{i}",
                    )
                )
        else:
            self.lateral_ops = lateral_ops

        # the same upsampling layer is used for all levels
        self.top_down_ops = (
            [tf.keras.layers.UpSampling2D(size=2)] * self.num_pyramid_levels
            if not top_down_ops
            else top_down_ops
        )
        # same merge_ops for all layers as well
        self.merge_ops = (
            [tf.keras.layers.Add()] * (self.num_pyramid_levels - 1)
            if not merge_ops
            else merge_ops
        )

    def _create_pyramid(self, features):
        # process first scale outside loop because no merge op required
        output_features = {}
        P_i = self.lateral_ops[-1](features[-1])
        P_top_down_i = self.top_down_ops[-1](P_i)

        # store the output in a dictionary
        output_features[f"P{self.pyramid_levels[-1]}"] = P_i

        # loop across all scales and perform FPN ops
        for i in range(self.num_pyramid_levels - 2, -1, -1):
            P_i = self.lateral_ops[i](features[i])
            P_i = self.merge_ops[i - 1]([P_i, P_top_down_i])
            P_top_down_i = self.top_down_ops[i](P_i)

            output_features[f"P{self.pyramid_levels[i]}"] = P_i

        return output_features

    def call(self, features):
        return self._create_pyramid(features)
