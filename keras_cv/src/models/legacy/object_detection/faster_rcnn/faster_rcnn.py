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
from absl import logging
from tensorflow import keras

from keras_cv.src import bounding_box
from keras_cv.src import layers as cv_layers
from keras_cv.src import models
from keras_cv.src.bounding_box.converters import _decode_deltas_to_boxes
from keras_cv.src.bounding_box.utils import _clip_boxes
from keras_cv.src.layers.object_detection.anchor_generator import (
    AnchorGenerator,
)
from keras_cv.src.layers.object_detection.box_matcher import BoxMatcher
from keras_cv.src.layers.object_detection.roi_align import _ROIAligner
from keras_cv.src.layers.object_detection.roi_generator import ROIGenerator
from keras_cv.src.layers.object_detection.roi_sampler import _ROISampler
from keras_cv.src.layers.object_detection.rpn_label_encoder import (
    _RpnLabelEncoder,
)
from keras_cv.src.models.object_detection import predict_utils
from keras_cv.src.models.object_detection.__internal__ import unpack_input
from keras_cv.src.utils.train import get_feature_extractor

BOX_VARIANCE = [0.1, 0.1, 0.2, 0.2]


class FeaturePyramid(keras.layers.Layer):
    """Builds the Feature Pyramid with the feature maps from the backbone."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv_c2_1x1 = keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c3_1x1 = keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c4_1x1 = keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c5_1x1 = keras.layers.Conv2D(256, 1, 1, "same")

        self.conv_c2_3x3 = keras.layers.Conv2D(256, 3, 1, "same")
        self.conv_c3_3x3 = keras.layers.Conv2D(256, 3, 1, "same")
        self.conv_c4_3x3 = keras.layers.Conv2D(256, 3, 1, "same")
        self.conv_c5_3x3 = keras.layers.Conv2D(256, 3, 1, "same")
        self.conv_c6_3x3 = keras.layers.Conv2D(256, 3, 1, "same")
        self.conv_c6_pool = keras.layers.MaxPool2D()
        self.upsample_2x = keras.layers.UpSampling2D(2)

    def call(self, inputs, training=None):
        c2_output = inputs["P2"]
        c3_output = inputs["P3"]
        c4_output = inputs["P4"]
        c5_output = inputs["P5"]

        c6_output = self.conv_c6_pool(c5_output)
        p6_output = c6_output
        p5_output = self.conv_c5_1x1(c5_output)
        p4_output = self.conv_c4_1x1(c4_output)
        p3_output = self.conv_c3_1x1(c3_output)
        p2_output = self.conv_c2_1x1(c2_output)

        p4_output = p4_output + self.upsample_2x(p5_output)
        p3_output = p3_output + self.upsample_2x(p4_output)
        p2_output = p2_output + self.upsample_2x(p3_output)

        p6_output = self.conv_c6_3x3(p6_output)
        p5_output = self.conv_c5_3x3(p5_output)
        p4_output = self.conv_c4_3x3(p4_output)
        p3_output = self.conv_c3_3x3(p3_output)
        p2_output = self.conv_c2_3x3(p2_output)

        return {
            "P2": p2_output,
            "P3": p3_output,
            "P4": p4_output,
            "P5": p5_output,
            "P6": p6_output,
        }

    def get_config(self):
        config = {}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


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
            input_shape = tf.nest.flatten(input_shape)
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
            batch_size = f_map.get_shape().as_list()[0] or tf.shape(f_map)[0]
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


# class agnostic regression
class RCNNHead(keras.layers.Layer):
    def __init__(
        self,
        num_classes,
        conv_dims=[],
        fc_dims=[1024, 1024],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.conv_dims = conv_dims
        self.fc_dims = fc_dims
        self.convs = []
        for conv_dim in conv_dims:
            layer = keras.layers.Conv2D(
                filters=conv_dim,
                kernel_size=3,
                strides=1,
                padding="same",
                activation="relu",
            )
            self.convs.append(layer)
        self.fcs = []
        for fc_dim in fc_dims:
            layer = keras.layers.Dense(units=fc_dim, activation="relu")
            self.fcs.append(layer)
        self.box_pred = keras.layers.Dense(units=4)
        self.cls_score = keras.layers.Dense(
            units=num_classes + 1, activation="softmax"
        )

    def call(self, feature_map, training=None):
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
            "num_classes": self.num_classes,
            "conv_dims": self.conv_dims,
            "fc_dims": self.fc_dims,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


# TODO(tanzheny): add more configurations
@keras.utils.register_keras_serializable(package="keras_cv")
class FasterRCNN(keras.Model):
    """A Keras model implementing the FasterRCNN architecture.

    Implements the FasterRCNN architecture for object detection. The constructor
    requires `num_classes`, `bounding_box_format` and a `backbone`.

    References:
        - [FasterRCNN](https://arxiv.org/pdf/1506.01497.pdf)

    Example:
    ```python
    retinanet = keras_cv.models.FasterRCNN(
        num_classes=20,
        bounding_box_format="xywh",
        backbone=None,
    )
    ```

    Args:
        num_classes: the number of classes in your dataset excluding the
            background class. classes should be represented by integers in the
            range [0, num_classes).
        bounding_box_format: The format of bounding boxes of model output. Refer
            [to the keras.io docs](https://keras.io/api/keras_cv/bounding_box/formats/)
            for more details on supported bounding box formats.
        backbone: Optional `keras.Model`. Must implement the
            `pyramid_level_inputs` property with keys "P2", "P3", "P4", and "P5"
            and layer names as values. If `None`, defaults to
            `keras_cv.models.ResNet50Backbone()`.
        anchor_generator: (Optional) a `keras_cv.layers.AnchorGenerator`. It is
            used in the model to match ground truth boxes and labels with
            anchors, or with region proposals. By default it uses the sizes and
            ratios from the paper, that is optimized for image size between
            [640, 800]. The users should pass their own anchor generator if the
            input image size differs from paper. For now, only anchor generator
            with per level dict output is supported,
        label_encoder: (Optional) a keras.Layer that accepts an anchors Tensor,
            a bounding box Tensor and a bounding box class Tensor to its
            `call()` method, and returns RetinaNet training targets. It returns
            box and class targets as well as sample weights.
        rcnn_head: (Optional) a `keras.layers.Layer` that takes input feature
            map and returns a box delta prediction (in reference to rois) and
            multi-class prediction (all foreground classes + one background
            class). By default it uses the rcnn head from paper, which is 2 FC
            layer with 1024 dimension, 1 box regressor and 1 softmax classifier.
        prediction_decoder: (Optional) a `keras.layers.Layer` that takes input
            box prediction and softmaxed score prediction, and returns NMSed box
            prediction, NMSed softmaxed score prediction, NMSed class
            prediction, and NMSed valid detection.
    """  # noqa: E501

    def __init__(
        self,
        num_classes,
        bounding_box_format,
        backbone=None,
        anchor_generator=None,
        label_encoder=None,
        rcnn_head=None,
        prediction_decoder=None,
        **kwargs,
    ):
        self.bounding_box_format = bounding_box_format
        super().__init__(**kwargs)
        scales = [2**x for x in [0]]
        aspect_ratios = [0.5, 1.0, 2.0]
        self.anchor_generator = anchor_generator or AnchorGenerator(
            bounding_box_format="yxyx",
            sizes={
                "P2": 32.0,
                "P3": 64.0,
                "P4": 128.0,
                "P5": 256.0,
                "P6": 512.0,
            },
            scales=scales,
            aspect_ratios=aspect_ratios,
            strides={f"P{i}": 2**i for i in range(2, 7)},
            clip_boxes=True,
        )
        self.rpn_head = RPNHead(
            num_anchors_per_location=len(scales) * len(aspect_ratios)
        )
        self.roi_generator = ROIGenerator(
            bounding_box_format="yxyx",
            nms_score_threshold_train=float("-inf"),
            nms_score_threshold_test=float("-inf"),
        )
        self.box_matcher = BoxMatcher(
            thresholds=[0.0, 0.5], match_values=[-2, -1, 1]
        )
        self.roi_sampler = _ROISampler(
            bounding_box_format="yxyx",
            roi_matcher=self.box_matcher,
            background_class=num_classes,
            num_sampled_rois=512,
        )
        self.roi_pooler = _ROIAligner(bounding_box_format="yxyx")
        self.rcnn_head = rcnn_head or RCNNHead(num_classes)
        self.backbone = backbone or models.ResNet50Backbone()
        extractor_levels = ["P2", "P3", "P4", "P5"]
        extractor_layer_names = [
            self.backbone.pyramid_level_inputs[i] for i in extractor_levels
        ]
        self.feature_extractor = get_feature_extractor(
            self.backbone, extractor_layer_names, extractor_levels
        )
        self.feature_pyramid = FeaturePyramid()
        self.rpn_labeler = label_encoder or _RpnLabelEncoder(
            anchor_format="yxyx",
            ground_truth_box_format="yxyx",
            positive_threshold=0.7,
            negative_threshold=0.3,
            samples_per_image=256,
            positive_fraction=0.5,
            box_variance=BOX_VARIANCE,
        )
        self._prediction_decoder = (
            prediction_decoder
            or cv_layers.MultiClassNonMaxSuppression(
                bounding_box_format=bounding_box_format,
                from_logits=False,
                max_detections_per_class=10,
                max_detections=10,
            )
        )

    def _call_rpn(self, images, anchors, training=None):
        image_shape = tf.shape(images[0])
        backbone_outputs = self.feature_extractor(images, training=training)
        feature_map = self.feature_pyramid(backbone_outputs, training=training)
        # [BS, num_anchors, 4], [BS, num_anchors, 1]
        rpn_boxes, rpn_scores = self.rpn_head(feature_map, training=training)
        # the decoded format is center_xywh, convert to yxyx
        decoded_rpn_boxes = _decode_deltas_to_boxes(
            anchors=anchors,
            boxes_delta=rpn_boxes,
            anchor_format="yxyx",
            box_format="yxyx",
            variance=BOX_VARIANCE,
        )
        rois, _ = self.roi_generator(
            decoded_rpn_boxes, rpn_scores, training=training
        )
        rois = _clip_boxes(rois, "yxyx", image_shape)
        rpn_boxes = tf.concat(tf.nest.flatten(rpn_boxes), axis=1)
        rpn_scores = tf.concat(tf.nest.flatten(rpn_scores), axis=1)
        return rois, feature_map, rpn_boxes, rpn_scores

    def _call_rcnn(self, rois, feature_map, training=None):
        feature_map = self.roi_pooler(feature_map, rois)
        # [BS, H*W*K, pool_shape*C]
        feature_map = tf.reshape(
            feature_map, tf.concat([tf.shape(rois)[:2], [-1]], axis=0)
        )
        # [BS, H*W*K, 4], [BS, H*W*K, num_classes + 1]
        rcnn_box_pred, rcnn_cls_pred = self.rcnn_head(
            feature_map, training=training
        )
        return rcnn_box_pred, rcnn_cls_pred

    def call(self, images, training=None):
        image_shape = tf.shape(images[0])
        anchors = self.anchor_generator(image_shape=image_shape)
        rois, feature_map, _, _ = self._call_rpn(
            images, anchors, training=training
        )
        box_pred, cls_pred = self._call_rcnn(
            rois, feature_map, training=training
        )
        if not training:
            # box_pred is on "center_yxhw" format, convert to target format.
            box_pred = _decode_deltas_to_boxes(
                anchors=rois,
                boxes_delta=box_pred,
                anchor_format="yxyx",
                box_format=self.bounding_box_format,
                variance=[0.1, 0.1, 0.2, 0.2],
            )

        return box_pred, cls_pred

    # TODO(tanzhenyu): Support compile with metrics.
    def compile(
        self,
        box_loss=None,
        classification_loss=None,
        rpn_box_loss=None,
        rpn_classification_loss=None,
        weight_decay=0.0001,
        loss=None,
        **kwargs,
    ):
        # TODO(tanzhenyu): Add metrics support once COCOMap issue is addressed.
        # https://github.com/keras-team/keras-cv/issues/915
        if "metrics" in kwargs.keys():
            raise ValueError(
                "`FasterRCNN` does not currently support the use of "
                "`metrics` due to performance and distribution concerns. "
                "Please use the `PyCOCOCallback` to evaluate COCO metrics."
            )
        if loss is not None:
            raise ValueError(
                "`FasterRCNN` does not accept a `loss` to `compile()`. "
                "Instead, please pass `box_loss` and `classification_loss`. "
                "`loss` will be ignored during training."
            )
        box_loss = _validate_and_get_loss(box_loss, "box_loss")
        classification_loss = _validate_and_get_loss(
            classification_loss, "classification_loss"
        )
        rpn_box_loss = _validate_and_get_loss(rpn_box_loss, "rpn_box_loss")
        if rpn_classification_loss == "BinaryCrossentropy":
            rpn_classification_loss = keras.losses.BinaryCrossentropy(
                from_logits=True, reduction=keras.losses.Reduction.SUM
            )
        rpn_classification_loss = _validate_and_get_loss(
            rpn_classification_loss, "rpn_cls_loss"
        )
        if not rpn_classification_loss.from_logits:
            raise ValueError(
                "`rpn_classification_loss` must come with `from_logits`=True"
            )

        self.rpn_box_loss = rpn_box_loss
        self.rpn_cls_loss = rpn_classification_loss
        self.box_loss = box_loss
        self.cls_loss = classification_loss
        self.weight_decay = weight_decay
        losses = {
            "box": self.box_loss,
            "classification": self.cls_loss,
            "rpn_box": self.rpn_box_loss,
            "rpn_classification": self.rpn_cls_loss,
        }
        super().compile(loss=losses, **kwargs)

    def compute_loss(self, images, boxes, classes, training):
        local_batch = images.get_shape().as_list()[0]
        if tf.distribute.has_strategy():
            num_sync = tf.distribute.get_strategy().num_replicas_in_sync
        else:
            num_sync = 1
        global_batch = local_batch * num_sync
        anchors = self.anchor_generator(image_shape=tuple(images[0].shape))
        (
            rpn_box_targets,
            rpn_box_weights,
            rpn_cls_targets,
            rpn_cls_weights,
        ) = self.rpn_labeler(
            tf.concat(tf.nest.flatten(anchors), axis=0), boxes, classes
        )
        rpn_box_weights /= (
            self.rpn_labeler.samples_per_image * global_batch * 0.25
        )
        rpn_cls_weights /= self.rpn_labeler.samples_per_image * global_batch
        rois, feature_map, rpn_box_pred, rpn_cls_pred = self._call_rpn(
            images, anchors, training=training
        )
        rois = tf.stop_gradient(rois)
        (
            rois,
            box_targets,
            box_weights,
            cls_targets,
            cls_weights,
        ) = self.roi_sampler(rois, boxes, classes)
        box_weights /= self.roi_sampler.num_sampled_rois * global_batch * 0.25
        cls_weights /= self.roi_sampler.num_sampled_rois * global_batch
        box_pred, cls_pred = self._call_rcnn(
            rois, feature_map, training=training
        )
        y_true = {
            "rpn_box": rpn_box_targets,
            "rpn_classification": rpn_cls_targets,
            "box": box_targets,
            "classification": cls_targets,
        }
        y_pred = {
            "rpn_box": rpn_box_pred,
            "rpn_classification": rpn_cls_pred,
            "box": box_pred,
            "classification": cls_pred,
        }
        weights = {
            "rpn_box": rpn_box_weights,
            "rpn_classification": rpn_cls_weights,
            "box": box_weights,
            "classification": cls_weights,
        }
        return super().compute_loss(
            x=images, y=y_true, y_pred=y_pred, sample_weight=weights
        )

    def train_step(self, data):
        images, y = unpack_input(data)

        boxes = y["boxes"]
        if len(y["classes"].shape) != 2:
            raise ValueError(
                "Expected 'classes' to be a tf.Tensor of rank 2. "
                f"Got y['classes'].shape={y['classes'].shape}."
            )
        # TODO(tanzhenyu): remove this hack and perform broadcasting elsewhere
        classes = tf.expand_dims(y["classes"], axis=-1)
        with tf.GradientTape() as tape:
            total_loss = self.compute_loss(
                images, boxes, classes, training=True
            )
            reg_losses = []
            if self.weight_decay:
                for var in self.trainable_variables:
                    if "bn" not in var.name:
                        reg_losses.append(
                            self.weight_decay * tf.nn.l2_loss(var)
                        )
                l2_loss = tf.math.add_n(reg_losses)
            total_loss += l2_loss
        self.optimizer.minimize(total_loss, self.trainable_variables, tape=tape)
        return self.compute_metrics(images, {}, {}, sample_weight={})

    def test_step(self, data):
        images, y = unpack_input(data)

        boxes = y["boxes"]
        if len(y["classes"].shape) != 2:
            raise ValueError(
                "Expected 'classes' to be a tf.Tensor of rank 2. "
                f"Got y['classes'].shape={y['classes'].shape}."
            )
        classes = tf.expand_dims(y["classes"], axis=-1)
        self.compute_loss(images, boxes, classes, training=False)
        return self.compute_metrics(images, {}, {}, sample_weight={})

    def make_predict_function(self, force=False):
        return predict_utils.make_predict_function(self, force=force)

    @property
    def prediction_decoder(self):
        return self._prediction_decoder

    @prediction_decoder.setter
    def prediction_decoder(self, prediction_decoder):
        self._prediction_decoder = prediction_decoder
        self.make_predict_function(force=True)

    def decode_predictions(self, predictions, images):
        # no-op if default decoder is used.
        box_pred, scores_pred = predictions
        box_pred = bounding_box.convert_format(
            box_pred,
            source=self.bounding_box_format,
            target=self.prediction_decoder.bounding_box_format,
            images=images,
        )
        y_pred = self.prediction_decoder(box_pred, scores_pred[..., :-1])
        box_pred = bounding_box.convert_format(
            y_pred["boxes"],
            source=self.prediction_decoder.bounding_box_format,
            target=self.bounding_box_format,
            images=images,
        )
        y_pred["boxes"] = box_pred
        return y_pred

    def get_config(self):
        return {
            "num_classes": self.num_classes,
            "bounding_box_format": self.bounding_box_format,
            "backbone": self.backbone,
            "anchor_generator": self.anchor_generator,
            "label_encoder": self.rpn_labeler,
            "prediction_decoder": self._prediction_decoder,
            "feature_pyramid": self.feature_pyramid,
            "rcnn_head": self.rcnn_head,
        }


def _validate_and_get_loss(loss, loss_name):
    if isinstance(loss, str):
        loss = keras.losses.get(loss)
    if loss is None or not isinstance(loss, keras.losses.Loss):
        raise ValueError(
            f"FasterRCNN only accepts `keras.losses.Loss` for {loss_name}, "
            f"got {loss}"
        )
    if loss.reduction != keras.losses.Reduction.SUM:
        logging.info(
            f"FasterRCNN only accepts `SUM` reduction, got {loss.reduction}, "
            "automatically converted."
        )
        loss.reduction = keras.losses.Reduction.SUM
    return loss
