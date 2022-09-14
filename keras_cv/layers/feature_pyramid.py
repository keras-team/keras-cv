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


# Not register it for now due to the conflict in the retina_net
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
    2) a top-down upsampling operation on Pj (where j = i - 1 and j > 0)
    3) an output conv2D operation on Pi (usually with kernel = 3 and strides = 1)

    The inputs to the layer can be either list, or dict:
    When in dict, the keys should match the pyramid_levels, e.g. for `pyramid_levels` =
    [2,3,4,5], the expected input dict should be `{'C2':c2, 'C3':c3, 'C4':c4, 'C5':c5}`.
    when in list, it will be interpreted as bottom up input features, e.g. for
    `pyramid_levels` = [2,3,4,5], the input list is expected to be [c2, c3, c4, c5]

    The output of the layer will have same structure as the inputs (either list or dict).
    When in dict, the output keys will be 'Pi' where the `i` is the level.

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
        top_down_ops: a python list of size (num_pyramid_levels - 1) that specifies all
            the top-down operations performed by the FPN. Each operation from the list
            must match the corresponding level from pyramid_levels list. Passing None will
            populate the list with the default FPN operation (an UpSampling2D layer).
            Defaults to None.
        merge_ops: a python list of size (num_pyramid_levels - 1) that specifies all
            the merge operations performed by the FPN. Each operation from the list must
            match the corresponding level from pyramid_levels list. The top-most layer
            is not merged and the corresponding operation is therefore not included.
            Passing None will populate the list with the default FPN operation
            (an add operation). Defaults to None.
        lateral_ops: a python list of size num_pyramid_levels that specifies all the
            output operations performed by the FPN. Each operation from the list must
            match the corresponding level from pyramid_levels list. Passing None will
            populate the list with the default FPN operation (a 3X3 Conv2D layer).
            Defaults to None.

    Sample Usage:
    ```python

    inp = tf.keras.layers.Input((384, 384, 3))
    efnb0 = tf.keras.applications.EfficientNetB0(input_tensor=inp, include_top = False)
    layer_names = ['block2b_add', 'block3b_add', 'block5c_add', 'top_activation']
    backbone_outputs = [efnb0.get_layer(name).output for name in layer_names]

    # output is list with P2, P3, P4 and P5
    output = keras_cv.layers.FeaturePyramid([2,3,4,5])(backbone_outputs)

    # when input with dict
    input_dict = {}
    for i, layer_name in enumerate(layer_names):
        input_dict[f'C{i+2}'] = efnb0.get_layer(layer_name).output
    output_dict = keras_cv.layers.FeaturePyramid([2,3,4,5])(input_dict)

    # output_dict is a dict with P2, P3, P4 and P5 as keys
    ```
    """

    def __init__(
            self,
            pyramid_levels,
            num_channels=256,
            lateral_ops=None,
            top_down_ops=None,
            merge_ops=None,
            output_ops=None,
            **kwargs,
    ):
        super(FeaturePyramid, self).__init__(**kwargs)
        self.pyramid_levels = sorted(pyramid_levels)
        self.num_pyramid_levels = len(self.pyramid_levels)
        self.num_channels = num_channels

        # required for successful serialization
        self.lateral_ops_passed = lateral_ops
        self.top_down_ops_passed = lateral_ops
        self.merge_ops_passed = lateral_ops
        self.output_ops_passed = output_ops

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
            if len(lateral_ops) != self.num_pyramid_levels:
                raise ValueError(f"Expecxt {lateral_ops} to have same length as "
                                 f"pyramid_levels, which is {self.num_pyramid_levels}")
            self.lateral_ops = lateral_ops

        # the same upsampling layer is used for all levels
        if not top_down_ops:
            self.top_down_ops = [tf.keras.layers.UpSampling2D(size=2)
                                 ] * (self.num_pyramid_levels - 1)
        else:
            if len(top_down_ops) != self.num_pyramid_levels:
                raise ValueError(f"Expecxt {top_down_ops} to have same length as "
                                 f"pyramid_levels - 1, "
                                 f"which is {self.num_pyramid_levels - 1}")
            self.top_down_ops = top_down_ops
        # the same merge layer is used for all levels
        if not merge_ops:
            self.merge_ops = [tf.keras.layers.Add()] * (self.num_pyramid_levels - 1)
        else:
            if len(merge_ops) != self.num_pyramid_levels:
                raise ValueError(f"Expecxt {merge_ops} to have same length as "
                                 f"pyramid_levels -1 , "
                                 f"which is {self.num_pyramid_levels - 1}")
            self.merge_ops = merge_ops
        # Output conv2d layers.
        if not output_ops:
            self.output_ops = []
            for i in pyramid_levels:
                self.output_ops.append(
                    tf.keras.layers.Conv2D(
                        self.num_channels,
                        kernel_size=3,
                        strides=1,
                        padding="same",
                        name=f"output_P{i}",
                    )
                )
        else:
            if len(output_ops) != self.num_pyramid_levels:
                raise ValueError(f"Expecxt {output_ops} to have same length as "
                                 f"pyramid_levels, which is {self.num_pyramid_levels}")
            self.output_ops = output_ops

        # Reserve the order of all the internal fields so that they are in top-down order
        # (smaller size feature map to larger size feature map). This makes the loop logic
        # easy to write.
        self.lateral_ops = self.lateral_ops[::-1]
        self.top_down_ops = self.top_down_ops[::-1]
        self.merge_ops = self.merge_ops[::-1]
        self.output_ops = self.output_ops[::-1]

    def call(self, features):
        # input features are in bottom up order, and reverse it to make the loop logic
        # easier to understand.
        features = features[::-1]
        output_features = []
        for i in range(self.num_pyramid_levels):
            output = self.lateral_ops[i](features[i])
            if i > 0:
                # for the top most output, it doesn't need to merge with any upper stream
                # outputs
                upstream_output = self.top_down_ops[i](output_features[i-1])
                output = self.merge_ops[i]([output, upstream_output])
            output_features.append(output)

        for i in range(self.num_pyramid_levels):
            output_features[i] = self.output_ops[i](output_features[i])

        # Convert the output features to be dict with proper keys from pyramid_levels
        output_feature_dict = {}
        for i in range(self.num_pyramid_levels):
            level = self.pyramid_levels[-i]
            output_feature_dict[f"P{level}"] = output_features[i]
        return output_feature_dict

    def get_config(self):
        config = {
            "pyramid_levels": self.pyramid_levels,
            "num_channels": self.num_channels,
            "lateral_ops": self.lateral_ops_passed,
            "top_down_ops": self.top_down_ops_passed,
            "merge_ops": self.merge_ops_passed,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
