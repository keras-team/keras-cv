# Copyright 2020 The Keras CV Authors
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
from .anchor_generator import AnchorGenerator


class MultiScaleAnchorGenerator(tf.keras.layers.Layer):
    """Defines a MultiScaleAnchorGenerator that generates anchor boxes for multiple feature maps.

        # Attributes:
            image_size: A list/tuple of 2 ints, the 1st represents the image height, the 2nd image width.
            scales: A list/tuple of list/tuple of positive floats (usually less than 1.) as a fraction to shorter
                side of `image_size`. It represents the base anchor size (when aspect ratio is 1.).
                For example, if `image_size=(300, 200)`, and `scales=[[.1]]`, then the base anchor size is 20.
                If `image_size=(300, 200)` and `scales=[[.1], [.2]]`, then the base anchor sizes are 20 and 40.
            aspect_ratios: a list/tuple of list/tuple of positive floats representing the ratio of anchor width
                to anchor height. **Must** have the same length as `scales`.
                For example, if `image_size=(300, 200)`, `scales=[[.1]]`, and `aspect_ratios=[[.64]]`, the base anchor
                size is 20, then anchor height is 25 and anchor width is 16. If `image_size=(300, 200)`,
                `scales=[[.1], [.2]]`, and `aspect_ratios=[[.64], [.1]]`, the base anchor size is 20 and 40, then
                the anchor heights are 25 and 40, the anchor widths are 16 and 40.
                The anchor aspect ratio is independent to the original aspect ratio of image size.
            strides: A list/tuple of list/tuple of 2 ints or floats representing the distance between anchor
                points. For example, `stride=[(30, 40)]` means each anchor is separated by 30 pixels in height,
                and 40 pixels in width. Defaults to `None`, where anchor stride would be calculated as
                `min(image_height, image_width) / feature_map_height` and
                `min(image_height, image_width) / feature_map_width` for each feature map.
            offsets: A list/tuple of list/tuple of 2 floats between [0., 1.] representing the center of anchor
                points relative to the upper-left border of each feature map cell. Defaults to `None`, which is the
                center of each feature map cell when `strides=None`, or center of each anchor stride otherwise.
            clip_boxes: Boolean to represents whether the anchor coordinates should be clipped to the image size.
                Defaults to `True`.
            normalize_coordinates: Boolean to represents whether the anchor coordinates should be normalized to [0., 1.]
                with respect to the image size. Defaults to `True`.

    """

    def __init__(
        self,
        image_size,
        scales,
        aspect_ratios,
        strides=None,
        offsets=None,
        clip_boxes=True,
        normalize_coordinates=True,
        name=None,
        **kwargs
    ):
        self.image_size = image_size
        self.image_height = image_size[0]
        self.image_width = image_size[1]
        self.scales = scales
        self.aspect_ratios = aspect_ratios
        if strides is None:
            strides = [None] * len(scales)
        if offsets is None:
            offsets = [None] * len(scales)
        self.strides = strides
        self.offsets = offsets
        self.clip_boxes = clip_boxes
        self.normalize_coordinates = normalize_coordinates
        self.anchor_generators = []
        for (i, (scale_list, aspect_ratio_list, stride, offset)) in enumerate(
            zip(scales, aspect_ratios, strides, offsets)
        ):
            self.anchor_generators.append(
                AnchorGenerator(
                    image_size,
                    scales=scale_list,
                    aspect_ratios=aspect_ratio_list,
                    stride=stride,
                    offset=offset,
                    clip_boxes=clip_boxes,
                    normalize_coordinates=normalize_coordinates,
                    name="anchor_generator_" + str(i),
                )
            )
        super(MultiScaleAnchorGenerator, self).__init__(name=name, **kwargs)

    def call(self, feature_map_sizes):
        result = []
        for feature_map_size, anchor_generator in zip(
            feature_map_sizes, self.anchor_generators
        ):
            anchors = anchor_generator(feature_map_size)
            anchors = tf.reshape(anchors, (-1, 4))
            result.append(anchors)
        return tf.concat(result, axis=0)

    def get_config(self):
        config = {
            "image_size": self.image_size,
            "scales": self.scales,
            "aspect_ratios": self.aspect_ratios,
            "strides": self.strides,
            "offsets": self.offsets,
            "clip_boxes": self.clip_boxes,
            "normalize_coordinates": self.normalize_coordinates,
        }
        base_config = super(MultiScaleAnchorGenerator, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
