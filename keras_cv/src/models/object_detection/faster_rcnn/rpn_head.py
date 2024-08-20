# Copyright 2024 The KerasCV Authors
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

from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.backend import keras


@keras_cv_export(
    "keras_cv.models.faster_rcnn.RPNHead",
    package="keras_cv.models.faster_rcnn",
)
class RPNHead(keras.layers.Layer):
    """A Keras layer implementing the RPN architecture.

    Region Proposal Networks (RPN) was first suggested in
    [FasterRCNN](https://arxiv.org/abs/1506.01497).
    This is an end to end trainable layer which proposes regions
    for a detector (RCNN).

    Args:
        num_achors_per_location: (Optional) the number of anchors per location,
            defaults to 3.
        num_filters: (Optional) number convolution filters
        kernel_size: (Optional) kernel size of the convolution filters.
    """

    def __init__(
        self,
        num_anchors_per_location=3,
        num_filters=256,
        kernel_size=3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_anchors = num_anchors_per_location
        self.num_filters = num_filters
        self.kernel_size = kernel_size

        self.conv = keras.layers.Conv2D(
            filters=num_filters,
            kernel_size=kernel_size,
            strides=1,
            padding="same",
            activation="relu",
            kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
        )
        self.objectness_logits = keras.layers.Conv2D(
            filters=self.num_anchors * 1,
            kernel_size=1,
            strides=1,
            padding="valid",
            kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
        )
        self.anchor_deltas = keras.layers.Conv2D(
            filters=self.num_anchors * 4,
            kernel_size=1,
            strides=1,
            padding="valid",
            kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
        )

    def call(self, feature_map, training=False):
        def call_single_level(f_map):
            # [BS, H, W, C]
            t = self.conv(f_map, training=training)
            # [BS, H, W, K]
            rpn_scores = self.objectness_logits(t, training=training)
            # [BS, H, W, K * 4]
            rpn_boxes = self.anchor_deltas(t, training=training)
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
        config = super().get_config()
        config["num_anchors_per_location"] = self.num_anchors
        config["num_filters"] = self.num_filters
        config["kernel_size"] = self.kernel_size
        return config

    def build(self, input_shape):
        self.conv.build((None, None, None, self.num_filters))
        self.objectness_logits.build((None, None, None, self.num_filters))
        self.anchor_deltas.build((None, None, None, self.num_filters))
        self.built = True
