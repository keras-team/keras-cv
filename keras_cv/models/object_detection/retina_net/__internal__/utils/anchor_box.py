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

import keras_cv


class AnchorBox(tf.keras.layers.Layer):
    """Generates anchor boxes.

    Args:
        bounding_box_format:   The format of bounding boxes of input dataset. Refer
            https://github.com/keras-team/keras-cv/blob/master/keras_cv/bounding_box/converters.py
            for more details on supported bounding box formats.
        aspect_ratios: A list of float values representing the aspect ratios of
            the anchor boxes at each location on the feature map
        scales: A list of float values representing the scale of the anchor boxes
            at each location on the feature map.
        strides: A list of float value representing the strides for each feature
            map in the feature pyramid.
        areas: A list of float values representing the areas of the anchor
            boxes for each feature map in the feature pyramid.
    """

    def __init__(
        self,
        bounding_box_format,
        aspect_ratios=(0.5, 1.0, 2.0),
        scales=None,
        strides=(8, 16, 32, 64, 128),
        areas=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.bounding_box_format = bounding_box_format
        self.aspect_ratios = aspect_ratios
        self.scales = scales or [2**x for x in [0, 1 / 3, 2 / 3]]

        self._num_anchors = len(self.aspect_ratios) * len(self.scales)
        self._strides = strides
        self._areas = areas or [x**2 for x in [32.0, 64.0, 128.0, 256.0, 512.0]]
        self._anchor_dims = self._compute_dims()

    def _compute_dims(self):
        """Computes anchor box dimensions at all levels of the feature pyramid."""
        anchor_dims_all = []
        for area in self._areas:
            anchor_dims = []
            for ratio in self.aspect_ratios:
                anchor_height = tf.math.sqrt(area / ratio)
                anchor_width = area / anchor_height
                dims = tf.reshape(
                    tf.stack([anchor_width, anchor_height], axis=-1), [1, 1, 2]
                )
                for scale in self.scales:
                    anchor_dims.append(scale * dims)
            anchor_dims_all.append(tf.stack(anchor_dims, axis=-2))
        return anchor_dims_all

    def _get_anchors(self, feature_height, feature_width, level):
        """Generates anchor boxes for a given feature map size and level

        Args:
          feature_height: An integer representing the height of the feature map.
          feature_width: An integer representing the width of the feature map.
          level: An integer representing the level of the feature map in the
            feature pyramid.
        Returns:
          anchor boxes with the shape
          `(feature_height * feature_width * num_anchors, 4)`
        """
        rx = tf.range(feature_width, dtype=tf.float32) + 0.5
        ry = tf.range(feature_height, dtype=tf.float32) + 0.5
        centers = tf.stack(tf.meshgrid(rx, ry), axis=-1) * self._strides[level - 3]
        centers = tf.expand_dims(centers, axis=-2)
        centers = tf.tile(centers, [1, 1, self._num_anchors, 1])
        dims = tf.tile(
            self._anchor_dims[level - 3], [feature_height, feature_width, 1, 1]
        )
        anchors = tf.concat([centers, dims], axis=-1)
        return tf.reshape(
            anchors, [feature_height * feature_width * self._num_anchors, 4]
        )

    def call(self, images):
        """Generates anchor boxes for all the feature maps of the feature pyramid.
        Arguments:
          images: batch of images with shape [batch_size, height, width, 3].
        Returns:
          anchor boxes for all the feature maps, stacked as a single tensor
            with shape `(total_anchors, 4)`
        """
        images_shape = tf.shape(images)
        image_height = images_shape[1]
        image_width = images_shape[2]

        anchors = [
            self._get_anchors(
                tf.math.ceil(image_height / 2**i),
                tf.math.ceil(image_width / 2**i),
                i,
            )
            for i in range(3, 8)
        ]
        return keras_cv.bounding_box.convert_format(
            tf.concat(anchors, axis=0),
            source="xywh",
            target=self.bounding_box_format,
            # anchor_box generates unbatched AnchorBoxes
            images=images[0],
        )
