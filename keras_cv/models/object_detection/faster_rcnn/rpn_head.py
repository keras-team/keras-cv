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

import tree

from keras_cv.api_export import keras_cv_export
from keras_cv.backend import keras
from keras_cv.backend import ops

@keras_cv_export(
    "keras_cv.models.faster_rcnn.RPNHead",
    package="keras_cv.models.faster_rcnn",
)
class RPNHead(keras.layers.Layer):
    def __init__(
        self,
        num_anchors_per_location=3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_anchors = num_anchors_per_location

    def build(self, input_shape):
        if isinstance(input_shape, (dict, list, tuple)):
            input_shape = tree.flatten(input_shape)
            input_shape = input_shape[0]
        filters = input_shape[-1]
        self.conv = keras.layers.Conv2D(
            filters=filters,
            kernel_size=3,
            strides=1,
            padding="same",
            activation="relu",
            kernel_initializer="truncated_normal",
        )
        self.objectness_logits = keras.layers.Conv2D(
            filters=self.num_anchors * 1,
            kernel_size=1,
            strides=1,
            padding="same",
            kernel_initializer="truncated_normal",
        )
        self.anchor_deltas = keras.layers.Conv2D(
            filters=self.num_anchors * 4,
            kernel_size=1,
            strides=1,
            padding="same",
            kernel_initializer="truncated_normal",
        )

    def call(self, feature_map, training=None):
        def call_single_level(f_map):
            batch_size = f_map.get_shape().as_list()[0] or ops.shape(f_map)[0]
            # [BS, H, W, C]
            t = self.conv(f_map)
            # [BS, H, W, K]
            rpn_scores = self.objectness_logits(t)
            # [BS, H, W, K * 4]
            rpn_boxes = self.anchor_deltas(t)
            # [BS, H*W*K, 4]
            rpn_boxes = ops.reshape(rpn_boxes, [batch_size, -1, 4])
            # [BS, H*W*K, 1]
            rpn_scores = ops.reshape(rpn_scores, [batch_size, -1, 1])
            return rpn_boxes, rpn_scores

        if not isinstance(feature_map, (dict, list, tuple)):
            return call_single_level(feature_map)
        elif isinstance(feature_map, (list, tuple)):
            rpn_boxes = []
            rpn_scores = []
            for f_map in feature_map:
                rpn_box, rpn_score = call_single_level(f_map)
                rpn_boxes.append(rpn_box)
                rpn_scores.append(rpn_score)
            return rpn_boxes, rpn_scores
        else:
            rpn_boxes = {}
            rpn_scores = {}
            for lvl, f_map in feature_map.items():
                rpn_box, rpn_score = call_single_level(f_map)
                rpn_boxes[lvl] = rpn_box
                rpn_scores[lvl] = rpn_score
            return rpn_boxes, rpn_scores

    def get_config(self):
        config = {
            "num_anchors_per_location": self.num_anchors,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))