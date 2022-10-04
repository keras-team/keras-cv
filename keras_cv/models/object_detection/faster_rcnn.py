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

from keras_cv import bounding_box
from keras_cv.bounding_box.converters import _decode_deltas_to_boxes
from keras_cv.layers.object_detection.anchor_generator import AnchorGenerator
from keras_cv.layers.object_detection.roi_align import _ROIAligner
from keras_cv.layers.object_detection.roi_generator import ROIGenerator
from keras_cv.layers.object_detection.roi_sampler import _ROISampler
from keras_cv.layers.object_detection.rpn_label_encoder import _RpnLabelEncoder
from keras_cv.ops.box_matcher import ArgmaxBoxMatcher


def _resnet50_backbone(include_rescaling=False):
    inputs = tf.keras.layers.Input(shape=(None, None, 3))
    x = inputs

    if include_rescaling:
        x = tf.keras.applications.resnet.preprocess_input(x)

    backbone = tf.keras.applications.ResNet50(include_top=False, input_tensor=x)

    c2_output, c3_output, c4_output, c5_output = [
        backbone.get_layer(layer_name).output
        for layer_name in [
            "conv2_block3_out",
            "conv3_block4_out",
            "conv4_block6_out",
            "conv5_block3_out",
        ]
    ]
    return tf.keras.Model(
        inputs=inputs, outputs=[c2_output, c3_output, c4_output, c5_output]
    )


class FeaturePyramid(tf.keras.layers.Layer):
    """Builds the Feature Pyramid with the feature maps from the backbone."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv_c2_1x1 = tf.keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c3_1x1 = tf.keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c4_1x1 = tf.keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c5_1x1 = tf.keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c6_1x1 = tf.keras.layers.Conv2D(256, 1, 1, "same")

        self.conv_c2_3x3 = tf.keras.layers.Conv2D(256, 3, 1, "same")
        self.conv_c3_3x3 = tf.keras.layers.Conv2D(256, 3, 1, "same")
        self.conv_c4_3x3 = tf.keras.layers.Conv2D(256, 3, 1, "same")
        self.conv_c5_3x3 = tf.keras.layers.Conv2D(256, 3, 1, "same")
        self.conv_c6_3x3 = tf.keras.layers.Conv2D(256, 3, 1, "same")
        self.conv_c6_pool = tf.keras.layers.MaxPool2D()
        self.upsample_2x = tf.keras.layers.UpSampling2D(2)

    def call(self, inputs, training=None):
        c2_output, c3_output, c4_output, c5_output = inputs

        c6_output = self.conv_c6_pool(c5_output)
        p6_output = self.conv_c6_1x1(c6_output)
        p5_output = self.conv_c5_1x1(c5_output)
        p4_output = self.conv_c4_1x1(c4_output)
        p3_output = self.conv_c3_1x1(c3_output)
        p2_output = self.conv_c2_1x1(c2_output)

        p5_output = p5_output + self.upsample_2x(p6_output)
        p4_output = p4_output + self.upsample_2x(p5_output)
        p3_output = p3_output + self.upsample_2x(p4_output)
        p2_output = p2_output + self.upsample_2x(p3_output)

        p6_output = self.conv_c6_3x3(p6_output)
        p5_output = self.conv_c5_3x3(p5_output)
        p4_output = self.conv_c4_3x3(p4_output)
        p3_output = self.conv_c3_3x3(p3_output)
        p2_output = self.conv_c2_3x3(p2_output)

        return {2: p2_output, 3: p3_output, 4: p4_output, 5: p5_output, 6: p6_output}

    def get_config(self):
        config = {}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class RPNHead(tf.keras.layers.Layer):
    def __init__(
        self,
        num_anchors_per_location=3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_anchors = num_anchors_per_location

    def build(self, input_shape):
        if isinstance(input_shape, (dict, list, tuple)):
            input_shape = tf.nest.flatten(input_shape)
            input_shape = input_shape[0]
        filters = input_shape[-1]
        self.conv = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=3,
            strides=1,
            padding="same",
            activation="relu",
            kernel_initializer="truncated_normal",
        )
        self.objectness_logits = tf.keras.layers.Conv2D(
            filters=self.num_anchors * 1,
            kernel_size=1,
            strides=1,
            padding="same",
            kernel_initializer="truncated_normal",
        )
        self.anchor_deltas = tf.keras.layers.Conv2D(
            filters=self.num_anchors * 4,
            kernel_size=1,
            strides=1,
            padding="same",
            kernel_initializer="truncated_normal",
        )

    def call(self, feature_map):
        def call_single_level(f_map):
            batch_size = f_map.get_shape().as_list()[0]
            if batch_size is None:
                raise ValueError("Cannot handle static shape")
            # [BS, H, W, C]
            t = self.conv(f_map)
            # [BS, H, W, K]
            rpn_scores = self.objectness_logits(t)
            # [BS, H, W, K * 4]
            rpn_boxes = self.anchor_deltas(t)
            # [BS, H*W*K, 4]
            rpn_boxes = tf.reshape(rpn_boxes, [batch_size, -1, 4])
            # [BS, H*W*K, 1]
            rpn_scores = tf.reshape(rpn_scores, [batch_size, -1, 1])
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
            rpn_boxes = tf.concat(rpn_boxes, axis=1)
            rpn_scores = tf.concat(rpn_scores, axis=1)
            return rpn_boxes, rpn_scores
        else:
            rpn_boxes = {}
            rpn_scores = {}
            for lvl, f_map in feature_map.items():
                rpn_box, rpn_score = call_single_level(f_map)
                rpn_boxes[lvl] = rpn_box
                rpn_scores[lvl] = rpn_score
            rpn_boxes = tf.concat(tf.nest.flatten(rpn_boxes), axis=1)
            rpn_scores = tf.concat(tf.nest.flatten(rpn_scores), axis=1)
            return rpn_boxes, rpn_scores

    def get_config(self):
        config = {
            "num_anchors_per_location": self.num_anchors,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


# class agnostic regression
class RCNNHead(tf.keras.layers.Layer):
    def __init__(
        self,
        classes,
        conv_dims=[],
        fc_dims=[1024, 1024],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_classes = classes
        self.conv_dims = conv_dims
        self.fc_dims = fc_dims
        self.convs = []
        for conv_dim in conv_dims:
            layer = tf.keras.layers.Conv2D(
                filters=conv_dim,
                kernel_size=3,
                strides=1,
                padding="same",
                activation="relu",
            )
            self.convs.append(layer)
        self.fcs = []
        for fc_dim in fc_dims:
            layer = tf.keras.layers.Dense(units=fc_dim, activation="relu")
            self.fcs.append(layer)
        self.box_pred = tf.keras.layers.Dense(units=4)
        self.cls_score = tf.keras.layers.Dense(units=classes + 1)

    def call(self, feature_map):
        x = feature_map
        for conv in self.convs:
            x = conv(x)
        for fc in self.fcs:
            x = fc(x)
        rcnn_boxes = self.box_pred(x)
        rcnn_scores = self.cls_score(x)
        return rcnn_boxes, rcnn_scores

    def get_config(self):
        config = {
            "classes": self.num_classes,
            "conv_dims": self.conv_dims,
            "fc_dims": self.fc_dims,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


# TODO(tanzheny): add more configurations
class FasterRCNN(tf.keras.Model):
    """A Keras model implementing the FasterRCNN architecture.

    Implements the FasterRCNN architecture for object detection.  The constructor
    requires `classes`, `bounding_box_format` and a `backbone`.

    Usage:
    ```python
    retina_net = keras_cv.models.FasterRCNN(
        classes=20,
        bounding_box_format="xywh",
        backbone="resnet50",
        include_rescaling=False,
    )
    ```

    Args:
        classes: the number of classes in your dataset excluding the background
            class.  Classes should be represented by integers in the range
            [0, classes).
        bounding_box_format: The format of bounding boxes of input dataset. Refer
            [to the keras.io docs](https://keras.io/api/keras_cv/bounding_box/formats/)
            for more details on supported bounding box formats.
        backbone: Either `"resnet50"` or a custom backbone model.
        include_rescaling: Required if provided backbone is a pre-configured model.
            If set to `True`, inputs will be passed through a `Rescaling(1/255.0)`
            layer. Default to False.
    """

    def __init__(
        self,
        classes,
        bounding_box_format,
        backbone=None,
        include_rescaling=False,
        **kwargs,
    ):
        self.bounding_box_format = bounding_box_format
        super().__init__(**kwargs)
        self.anchor_generator = AnchorGenerator(
            bounding_box_format="yxyx",
            sizes={2: 16.0, 3: 32.0, 4: 64.0, 5: 128.0, 6: 256.0},
            scales=[2**x for x in [0, 1 / 3, 2 / 3]],
            aspect_ratios=[0.5, 1.0, 2.0],
            strides={i: 2**i for i in range(2, 7)},
            clip_boxes=True,
        )
        self.rpn_head = RPNHead(num_anchors_per_location=9)
        self.roi_generator = ROIGenerator(bounding_box_format="yxyx")
        self.box_matcher = ArgmaxBoxMatcher(
            thresholds=[0.0, 0.5], match_values=[-2, -1, 1]
        )
        self.roi_sampler = _ROISampler(
            bounding_box_format="yxyx",
            roi_matcher=self.box_matcher,
            background_class=classes,
            num_sampled_rois=512,
        )
        self.roi_pooler = _ROIAligner(bounding_box_format="yxyx")
        self.rcnn_head = RCNNHead(classes)
        self.backbone = backbone or _resnet50_backbone(include_rescaling)
        self.feature_pyramid = FeaturePyramid()
        self.rpn_labeler = _RpnLabelEncoder(
            anchor_format="yxyx",
            ground_truth_box_format="yxyx",
            positive_threshold=0.7,
            negative_threshold=0.3,
            samples_per_image=256,
            positive_fraction=0.5,
        )

    def call(self, images, gt_boxes=None, gt_classes=None, training=None):
        feature_map = self.backbone(images, training=training)
        feature_map = self.feature_pyramid(feature_map, training=training)
        # [BS, num_anchors, 4], [BS, num_anchors, 1]
        rpn_boxes, rpn_scores = self.rpn_head(feature_map)
        anchors = self.anchor_generator(image_shape=tf.shape(images[0]))
        if isinstance(anchors, (dict, list, tuple)):
            anchors = tf.concat(tf.nest.flatten(anchors), axis=0)
        anchors = tf.reshape(anchors, [-1, 4])
        # the decoded format is center_xywh
        decoded_rpn_boxes = _decode_deltas_to_boxes(anchors, rpn_boxes, "yxyx")
        decoded_rpn_boxes = bounding_box.convert_format(
            decoded_rpn_boxes, source="center_yxhw", target="yxyx"
        )
        rois, _ = self.roi_generator(
            decoded_rpn_boxes, tf.squeeze(rpn_scores, axis=-1), training=training
        )
        if training:
            rois = tf.stop_gradient(rois)
            (
                rois,
                rcnn_box_targets,
                rcnn_box_weights,
                rcnn_cls_targets,
                rcnn_cls_weights,
            ) = self.roi_sampler(rois, gt_boxes, gt_classes)
        feature_map = self.roi_pooler(feature_map, rois)
        # [BS, H*W*K, pool_shape*C]
        feature_map = tf.reshape(
            feature_map, tf.concat([tf.shape(rois)[:2], [-1]], axis=0)
        )
        # [BS, H*W*K, 4], [BS, H*W*K, num_classes + 1]
        rcnn_box_pred, rcnn_cls_pred = self.rcnn_head(feature_map)
        res = {}
        if training:
            res["rcnn_box_pred"] = rcnn_box_pred
            res["rcnn_cls_pred"] = rcnn_cls_pred
            res["rpn_box_pred"] = rpn_boxes
            res["rpn_cls_pred"] = rpn_scores
            res["rcnn_box_targets"] = rcnn_box_targets
            res["rcnn_box_weights"] = rcnn_box_weights
            res["rcnn_cls_targets"] = rcnn_cls_targets
            res["rcnn_cls_weights"] = rcnn_cls_weights
        else:
            # now it will be on "center_yxhw" format
            rcnn_box_pred = _decode_deltas_to_boxes(
                rois, rcnn_box_pred, anchor_format="yxyx"
            )
            rcnn_box_pred = bounding_box.convert_format(
                rcnn_box_pred, source="center_yxhw", target="yxyx"
            )
            res["rcnn_box_pred"] = rcnn_box_pred
            res["rcnn_cls_pred"] = rcnn_cls_pred

        return res
