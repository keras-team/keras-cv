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
import tensorflow.keras.backend as K
from tensorflow import keras

import keras_cv
from keras_cv import bounding_box
from keras_cv.models.object_detection import predict_utils
from keras_cv.models.object_detection.yolox.__internal__.layers.yolox_decoder import (
    DecodePredictions,
)
from keras_cv.models.object_detection.yolox.__internal__.layers.yolox_head import (
    YoloXHead,
)
from keras_cv.models.object_detection.yolox.__internal__.layers.yolox_label_encoder import (
    YoloXLabelEncoder,
)
from keras_cv.models.object_detection.yolox.__internal__.layers.yolox_pafpn import (
    YoloXPAFPN,
)

DEPTH_MULTIPLIERS = {
    "tiny": 0.33,
    "s": 0.33,
    "m": 0.67,
    "l": 1.00,
    "x": 1.33,
}
WIDTH_MULTIPLIERS = {
    "tiny": 0.375,
    "s": 0.50,
    "m": 0.75,
    "l": 1.00,
    "x": 1.25,
}

# TODO(quantumalaviya): Register
# @keras.utils.register_keras_serializable(package="keras_cv")
class YoloX(tf.keras.Model):
    """Instantiates the YoloX architecture using the given phi value.

    Arguments:
        classes: The number of classes to be considered for the YoloX head.
        bounding_box_format:  The format of bounding boxes of input dataset. Refer
            [to the keras.io docs](https://keras.io/api/keras_cv/bounding_box/formats/)
            for more details on supported bounding box formats.
        phi: One of `"tiny"`, `"s"`, `"m"`, `"l"` or `"x"`. Used to specify the size of
            the YoloX model. This is used to map the model to the relevant depth and
            width multiplier.
        backbone: an optional `tf.keras.Model` custom backbone model. Defaults
            to a keras_cv.models.csp_darknet.CSPDarkNet with depth and width multipiers 
            corresponding to the passed phi value with include_rescaling=True.
        label_encoder: (Optional) a keras.Layer that accepts an image Tensor, a
            bounding box Tensor and a bounding box class Tensor to its `call()` method,
            and returns YoloX training targets.  By default, a YoloX standard 
            LabelEncoder is created and used.
        prediction_decoder: (Optional)  A `keras.layer` that is responsible for
            transforming YoloX predictions into usable bounding box Tensors.  If
            not provided, a default DecodePredictions is provided. The default layer
            uses a `NonMaxSuppression` operation for box pruning.
        feature_pyramid: (Optional) A `keras.layer` representing a feature pyramid
            network (FPN).  The feature pyramid network is called on the outputs of the
            `backbone`.  The KerasCV default backbones return three outputs in a list,
            but custom backbones may be written and used with custom feature pyramid
            networks.  If not provided, a default feature pyramid network is produced
            by the library.  The default feature pyramid network is compatible with all
            standard keras_cv backbones.
        yolox_head: (Optional) A keras.layer representing the yolox_head. The design of
            this layer should mimic that of the internal default YoloX head. If None,
            a default YoloX head will be used.
        name: (Optional) the name to be passed to the model. Defaults to `"YoloX"`.
    """

    def __init__(
        self,
        classes,
        bounding_box_format,
        phi,
        backbone=None,
        label_encoder=None,
        prediction_decoder=None,
        feature_pyramid=None,
        yolox_head=None,
        name="YoloX",
        **kwargs,
    ):
        label_encoder = label_encoder or YoloXLabelEncoder(
            bounding_box_format=bounding_box_format
        )

        super().__init__(name=name, **kwargs)
        self.label_encoder = label_encoder
        self.depth_multiplier = DEPTH_MULTIPLIERS[phi]
        self.width_multiplier = WIDTH_MULTIPLIERS[phi]

        self.bounding_box_format = bounding_box_format
        self.classes = classes
        self.backbone = (
            backbone
            or keras_cv.models.csp_darknet.CSPDarkNet(
                include_top=False, include_rescaling=True,
                depth_multiplier=self.depth_multiplier,
                width_multiplier=self.width_multiplier,
            ).as_backbone(min_level=3)
        )

        suppression_layer = prediction_decoder or keras_cv.layers.MultiClassNonMaxSuppression(
            bounding_box_format=bounding_box_format,
            from_logits=False,
            confidence_threshold=0.01,
            iou_threshold=0.65,
            max_detections=100,
            max_detections_per_class=100,
        )
        self._prediction_decoder = DecodePredictions(
            bounding_box_format=bounding_box_format,
            classes=classes,
            suppression_layer=suppression_layer
        )

        self.feature_pyramid = feature_pyramid or YoloXPAFPN(
            depth_multiplier=self.depth_multiplier,
            width_multiplier=self.width_multiplier,
        )
        self.yolox_head = yolox_head or YoloXHead(classes, width_multiplier=self.width_multiplier)
        self.loss_metric = tf.keras.metrics.Mean(name="loss")

        self.classification_loss_metric = tf.keras.metrics.Mean(
            name="classification_loss"
        )
        self.objectness_loss_metric = tf.keras.metrics.Mean(name="objectness_loss")
        self.box_loss_metric = tf.keras.metrics.Mean(name="box_loss")
    
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
        y_pred = self.prediction_decoder(images, predictions)
        return bounding_box.convert_format(
            y_pred,
            source=self.prediction_decoder.bounding_box_format,
            target=self.bounding_box_format,
            images=images,
        )

    def compile(
        self,
        box_loss=None,
        objectness_loss=None,
        classification_loss=None,
        loss=None,
        **kwargs,
    ):
        """compiles the RetinaNet.

        compile() mirrors the standard Keras compile() method, but has a few key
        distinctions.  Primarily, all metrics must support bounding boxes, and
        three losses must be provided: `box_loss`, `objectness_loss` and `classification_loss`.

        Args:
            box_loss: a Keras loss to use for box offset regression.  Preconfigured
                losses are provided when the string "iou" or "giou" are passed.
            objectness_loss: a keras loss to use for objectness score (whether
                a given anchor point is a object or not). A preconfigured 
                `BinaryCrossEntropyLoss` is provided when the string 
                "binary_crossentropy" is passed.
            classification_loss: a Keras loss to use for box classification. A 
                preconfigured `BinaryCrossEntropyLoss` is provided when the string 
                "binary_crossentropy" is passed.
            kwargs: most other `keras.Model.compile()` arguments are supported and
                propagated to the `keras.Model` class.
        """
        if "metrics" in kwargs.keys():
            raise ValueError(
                "`RetinaNet` does not currently support the use of "
                "`metrics` due to performance and distribution concerns. Please us the "
                "`PyCOCOCallback` to evaluate COCO metrics."
            )
        super().compile(**kwargs)
        if loss is not None:
            raise ValueError(
                "`YoloX` does not accept a `loss` to `compile()`. "
                "Instead, please pass `box_loss`, `objectness_loss` and `classification_loss`. "
                "`loss` will be ignored during training."
            )
        box_loss = _parse_box_loss(box_loss)
        objectness_loss = _parse_objectness_loss(objectness_loss)
        classification_loss = _parse_classification_loss(classification_loss)

        if hasattr(classification_loss, "from_logits"):
            if not classification_loss.from_logits:
                raise ValueError(
                    "YoloX.compile() expects `from_logits` to be True for "
                    "`classification_loss`. Got "
                    "`classification_loss.from_logits="
                    f"{classification_loss.from_logits}`"
                )
        if hasattr(objectness_loss, "from_logits"):
            if not objectness_loss.from_logits:
                raise ValueError(
                    "YoloX.compile() expects `from_logits` to be True for "
                    "`objectness_loss`. Got "
                    "`objectness_loss.from_logits="
                    f"{objectness_loss.from_logits}`"
                )

        self.box_loss = box_loss
        self.objectness_loss = objectness_loss
        self.classification_loss = classification_loss
        # TODO: update compute_loss to support all formats
        if hasattr(self.box_loss, "bounding_box_format"):
            if self.box_loss.bounding_box_format != "center_xywh":
                raise ValueError(
                    "YoloX currently supports only `center_xywh`. "
                    f"Got `box_loss.bounding_box_format={self.box_loss.bounding_box_format}`, "
                )

    @property
    def metrics(self):
        return super().metrics + self.train_metrics

    @property
    def train_metrics(self):
        return [
            self.loss_metric,
            self.classification_loss_metric,
            self.objectness_loss_metric,
            self.box_loss_metric,
        ]

    def call(self, images, training=None):
        backbone_outputs = self.backbone(images, training=training)
        features = self.feature_pyramid(backbone_outputs)
        return self.yolox_head(features)

    def train_step(self, data):
        x, y = data
        gt_boxes, gt_classes = self.label_encoder(x, y)

        # yolox internally works on center_xywh
        gt_boxes = bounding_box.convert_format(
            gt_boxes,
            source=self.bounding_box_format,
            target="center_xywh",
            images=x,
        )
        y_true = tf.concat([gt_boxes, gt_classes], -1)

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(y_true, y_pred, input_shape=x.shape[1:3])

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return self.compute_metrics(x, {}, {}, sample_weight={})

    def compute_loss(self, y_true, y_pred, input_shape=(640, 640)):
        num_levels = len(y_pred)

        if y_true.shape[-1] != 5:
            raise ValueError(
                "gt_boxes should have shape (None, None, 5).  Got "
                f"gt_boxes.shape={tuple(y_true.shape)}"
            )

        for i in range(num_levels):
            if y_pred[i].shape[-1] != self.classes + 5:
                raise ValueError(
                    "y_pred should be a list with tensors of shape (None, None, None, classes + 5). "
                    f"Got y_pred[{i}].shape={tuple(y_pred[i].shape)}."
                )

        x_offsets = []
        y_offsets = []
        expanded_strides = []
        outputs = []

        # the following loop calculates the outputs by using tf.meshgrid to compute
        # anchor points and then using strides to scale the boxes back to image
        # shape.
        for i in range(num_levels):
            output = y_pred[i]

            grid_shape = tf.shape(output)[1:3]
            stride = input_shape[0] / tf.cast(grid_shape[0], tf.float32)

            grid_x, grid_y = tf.meshgrid(
                tf.range(grid_shape[1]), tf.range(grid_shape[0])
            )
            grid = tf.cast(
                tf.reshape(tf.stack((grid_x, grid_y), 2), (1, -1, 2)), tf.float32
            )

            output = tf.reshape(
                output, [tf.shape(y_pred[i])[0], grid_shape[0] * grid_shape[1], -1]
            )
            output_xy = (output[..., :2] + grid) * stride
            # exponential to ensure that both the width and height are positive
            output_wh = tf.exp(output[..., 2:4]) * stride
            output = tf.concat([output_xy, output_wh, output[..., 4:]], -1)

            x_offsets.append(grid[..., 0])
            y_offsets.append(grid[..., 1])
            expanded_strides.append(tf.ones_like(grid[..., 0]) * stride)
            outputs.append(output)

        x_offsets = tf.concat(x_offsets, 1)
        y_offsets = tf.concat(y_offsets, 1)
        expanded_strides = tf.concat(expanded_strides, 1)
        outputs = tf.concat(outputs, 1)

        bbox_preds = outputs[:, :, :4]
        obj_preds = outputs[:, :, 4:5]
        cls_preds = outputs[:, :, 5:]

        # this calculation would exclude the -1 boxes used to pad the input
        # RaggedTensor in label encoder.
        nlabel = tf.reduce_sum(tf.cast(tf.reduce_sum(y_true, -1) > 0, tf.float32), -1)
        num_anchor_points = tf.shape(outputs)[1]

        num_fg = 0.0
        loss_obj = 0.0
        loss_cls = 0.0
        loss_iou = 0.0

        def loop_across_batch(b, num_fg, loss_iou, loss_obj, loss_cls):
            num_gt = tf.cast(nlabel[b], tf.int32)

            gt_bboxes_per_image = y_true[b][:num_gt, :4]
            gt_classes = y_true[b][:num_gt, 4]

            bboxes_preds_per_image = bbox_preds[b]
            obj_preds_per_image = obj_preds[b]
            cls_preds_per_image = cls_preds[b]

            gt_bboxes_per_image = tf.ensure_shape(gt_bboxes_per_image, [None, 4])
            bboxes_preds_per_image = tf.ensure_shape(bboxes_preds_per_image, [None, 4])
            obj_preds_per_image = tf.ensure_shape(obj_preds_per_image, [None, 1])
            cls_preds_per_image = tf.ensure_shape(
                cls_preds_per_image, [None, self.classes]
            )

            def return_empty_boxes():
                num_fg_img = tf.constant(0.0)
                cls_target = tf.zeros((0, self.classes))
                reg_target = tf.zeros((0, 4))
                obj_target = tf.zeros((num_anchor_points, 1))
                fg_mask = tf.cast(tf.zeros(num_anchor_points), tf.bool)
                return num_fg_img, cls_target, reg_target, obj_target, fg_mask

            def perform_label_assignment():
                (
                    gt_matched_classes,
                    fg_mask,
                    pred_ious_this_matching,
                    matched_indices,
                    num_fg_img,
                ) = self.get_assignments(
                    gt_bboxes_per_image,
                    gt_classes,
                    bboxes_preds_per_image,
                    obj_preds_per_image,
                    cls_preds_per_image,
                    x_offsets,
                    y_offsets,
                    expanded_strides,
                    self.classes,
                    num_gt,
                    num_anchor_points,
                )
                reg_target = tf.gather_nd(
                    gt_bboxes_per_image, tf.reshape(matched_indices, [-1, 1])
                )
                pred_ious_this_matching = tf.expand_dims(pred_ious_this_matching, -1)
                cls_target = tf.cast(
                    tf.one_hot(tf.cast(gt_matched_classes, tf.int32), self.classes)
                    * pred_ious_this_matching,
                    tf.float32,
                )
                obj_target = tf.cast(tf.expand_dims(fg_mask, -1), tf.float32)
                return num_fg_img, cls_target, reg_target, obj_target, fg_mask

            # if no ground truths for this image, there are 0 boxes
            # else we perform the label assignment
            num_fg_img, cls_target, reg_target, obj_target, fg_mask = tf.cond(
                tf.equal(num_gt, 0), return_empty_boxes, perform_label_assignment
            )
            loss_iou_this_image = self.box_loss(
                reg_target, tf.boolean_mask(bboxes_preds_per_image, fg_mask)
            )
            loss_obj_this_image = self.objectness_loss(obj_target, obj_preds_per_image)
            loss_cls_this_image = self.classification_loss(
                cls_target, tf.boolean_mask(cls_preds_per_image, fg_mask)
            )

            # TODO: add assertions to ensure loss output shapes aren't wrong
            num_fg += num_fg_img
            loss_iou += tf.math.reduce_sum(loss_iou_this_image)
            loss_obj += tf.math.reduce_sum(loss_obj_this_image)
            loss_cls += tf.math.reduce_sum(loss_cls_this_image)
            return b + 1, num_fg, loss_iou, loss_obj, loss_cls

        _, num_fg, loss_iou, loss_obj, loss_cls = tf.while_loop(
            lambda b, *args: b < tf.shape(outputs)[0],
            loop_across_batch,
            [0, num_fg, loss_iou, loss_obj, loss_cls],
        )

        self.classification_loss_metric.update_state(loss_cls / num_fg)
        self.box_loss_metric.update_state(loss_iou / num_fg)
        self.objectness_loss_metric.update_state(loss_obj / num_fg)

        num_fg = tf.cast(tf.maximum(num_fg, 1), tf.float32)
        reg_weight = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_cls

        loss /= num_fg

        # TODO: loss_metric removed
        self.loss_metric.update_state(loss)
        return loss

    def get_assignments(
        self,
        gt_bboxes_per_image,
        gt_classes,
        bboxes_preds_per_image,
        obj_preds_per_image,
        cls_preds_per_image,
        x_offsets,
        y_offsets,
        expanded_strides,
        num_classes,
        num_gt,
        num_anchor_points,
    ):
        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
            gt_bboxes_per_image,
            x_offsets,
            y_offsets,
            expanded_strides,
            num_gt,
            num_anchor_points,
        )
        bboxes_preds_per_image = tf.boolean_mask(
            bboxes_preds_per_image, fg_mask, axis=0
        )
        obj_preds_ = tf.boolean_mask(obj_preds_per_image, fg_mask, axis=0)
        cls_preds_ = tf.boolean_mask(cls_preds_per_image, fg_mask, axis=0)
        num_in_boxes_anchor = tf.shape(bboxes_preds_per_image)[0]

        pair_wise_ious = bounding_box.compute_iou(
            gt_bboxes_per_image, bboxes_preds_per_image, "center_xywh"
        )
        pair_wise_ious_loss = -tf.math.log(pair_wise_ious + 1e-8)
        gt_cls_per_image = tf.tile(
            tf.expand_dims(tf.one_hot(tf.cast(gt_classes, tf.int32), num_classes), 1),
            (1, num_in_boxes_anchor, 1),
        )
        obj_preds_ = tf.math.sigmoid(
            tf.tile(tf.expand_dims(obj_preds_, 0), (num_gt, 1, 1))
        )
        cls_preds_ = (
            tf.math.sigmoid(tf.tile(tf.expand_dims(cls_preds_, 0), (num_gt, 1, 1)))
            * obj_preds_
        )
        pair_wise_cls_loss = tf.reduce_sum(
            K.binary_crossentropy(gt_cls_per_image, tf.sqrt(cls_preds_)), -1
        )

        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_ious_loss
            + 100000.0 * tf.cast((~is_in_boxes_and_center), tf.float32)
        )

        return self.dynamic_k_matching(
            cost, pair_wise_ious, fg_mask, gt_classes, num_gt
        )

    def get_in_boxes_info(
        self,
        gt_bboxes_per_image,
        x_offsets,
        y_offsets,
        expanded_strides,
        num_gt,
        num_anchor_points,
        center_radius=2.5,
    ):
        expanded_strides_per_image = expanded_strides[0]
        x_centers_per_image = tf.tile(
            tf.expand_dims(((x_offsets[0] + 0.5) * expanded_strides_per_image), 0),
            [num_gt, 1],
        )
        y_centers_per_image = tf.tile(
            tf.expand_dims(((y_offsets[0] + 0.5) * expanded_strides_per_image), 0),
            [num_gt, 1],
        )

        gt_bboxes_per_image_l = tf.tile(
            tf.expand_dims(
                (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2]), 1
            ),
            [1, num_anchor_points],
        )
        gt_bboxes_per_image_r = tf.tile(
            tf.expand_dims(
                (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2]), 1
            ),
            [1, num_anchor_points],
        )
        gt_bboxes_per_image_t = tf.tile(
            tf.expand_dims(
                (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3]), 1
            ),
            [1, num_anchor_points],
        )
        gt_bboxes_per_image_b = tf.tile(
            tf.expand_dims(
                (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3]), 1
            ),
            [1, num_anchor_points],
        )

        b_l = x_centers_per_image - gt_bboxes_per_image_l
        b_r = gt_bboxes_per_image_r - x_centers_per_image
        b_t = y_centers_per_image - gt_bboxes_per_image_t
        b_b = gt_bboxes_per_image_b - y_centers_per_image
        bbox_deltas = tf.stack([b_l, b_t, b_r, b_b], 2)

        is_in_boxes = tf.reduce_min(bbox_deltas, axis=-1) > 0.0
        is_in_boxes_all = tf.reduce_sum(tf.cast(is_in_boxes, tf.float32), axis=0) > 0.0

        gt_bboxes_per_image_l = tf.tile(
            tf.expand_dims(gt_bboxes_per_image[:, 0], 1), [1, num_anchor_points]
        ) - center_radius * tf.expand_dims(expanded_strides_per_image, 0)
        gt_bboxes_per_image_r = tf.tile(
            tf.expand_dims(gt_bboxes_per_image[:, 0], 1), [1, num_anchor_points]
        ) + center_radius * tf.expand_dims(expanded_strides_per_image, 0)
        gt_bboxes_per_image_t = tf.tile(
            tf.expand_dims(gt_bboxes_per_image[:, 1], 1), [1, num_anchor_points]
        ) - center_radius * tf.expand_dims(expanded_strides_per_image, 0)
        gt_bboxes_per_image_b = tf.tile(
            tf.expand_dims(gt_bboxes_per_image[:, 1], 1), [1, num_anchor_points]
        ) + center_radius * tf.expand_dims(expanded_strides_per_image, 0)

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = tf.stack([c_l, c_t, c_r, c_b], 2)

        is_in_centers = tf.reduce_min(center_deltas, axis=-1) > 0.0
        is_in_centers_all = (
            tf.reduce_sum(tf.cast(is_in_centers, tf.float32), axis=0) > 0.0
        )

        fg_mask = tf.cast(is_in_boxes_all | is_in_centers_all, tf.bool)

        is_in_boxes_and_center = tf.boolean_mask(
            is_in_boxes, fg_mask, axis=1
        ) & tf.boolean_mask(is_in_centers, fg_mask, axis=1)

        return fg_mask, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, fg_mask, gt_classes, num_gt):
        matching_matrix = tf.zeros_like(cost)

        n_candidate_k = tf.minimum(10, tf.shape(pair_wise_ious)[1])
        topk_ious, _ = tf.nn.top_k(pair_wise_ious, n_candidate_k)
        dynamic_ks = tf.maximum(tf.reduce_sum(topk_ious, 1), 1)

        def loop_across_batch_1(b, matching_matrix):
            _, pos_idx = tf.nn.top_k(-cost[b], k=tf.cast(dynamic_ks[b], tf.int32))
            matching_matrix = tf.concat(
                [
                    matching_matrix[:b],
                    tf.expand_dims(
                        tf.reduce_max(tf.one_hot(pos_idx, tf.shape(cost)[1]), 0), 0
                    ),
                    matching_matrix[b + 1 :],
                ],
                axis=0,
            )
            return b + 1, matching_matrix

        _, matching_matrix = tf.while_loop(
            lambda b, *args: b < num_gt,
            loop_across_batch_1,
            [0, matching_matrix],
        )

        anchor_matching_gt = tf.reduce_sum(matching_matrix, 0)
        anchor_indices = tf.reshape(tf.where(anchor_matching_gt > 1), [-1])

        def loop_across_batch_2(b, matching_matrix):
            anchor_index = tf.cast(anchor_indices[b], tf.int32)
            gt_index = tf.math.argmin(cost[:, anchor_index])
            matching_matrix = tf.concat(
                [
                    matching_matrix[:, :anchor_index],
                    tf.expand_dims(tf.one_hot(gt_index, tf.cast(num_gt, tf.int32)), 1),
                    matching_matrix[:, anchor_index + 1 :],
                ],
                axis=-1,
            )
            return b + 1, matching_matrix

        _, matching_matrix = tf.while_loop(
            lambda b, *args: b < tf.shape(anchor_indices)[0],
            loop_across_batch_2,
            [0, matching_matrix],
        )

        fg_mask_inboxes = tf.reduce_sum(matching_matrix, 0) > 0.0
        num_fg = tf.reduce_sum(tf.cast(fg_mask_inboxes, tf.float32))

        fg_mask_indices = tf.reshape(tf.where(fg_mask), [-1])
        fg_mask_inboxes_indices = tf.reshape(tf.where(fg_mask_inboxes), [-1, 1])
        fg_mask_select_indices = tf.gather_nd(fg_mask_indices, fg_mask_inboxes_indices)
        fg_mask = tf.cast(
            tf.reduce_max(tf.one_hot(fg_mask_select_indices, tf.shape(fg_mask)[0]), 0),
            K.dtype(fg_mask),
        )

        matched_indices = tf.math.argmax(
            tf.boolean_mask(matching_matrix, fg_mask_inboxes, axis=1), 0
        )
        gt_matched_classes = tf.gather_nd(
            gt_classes, tf.reshape(matched_indices, [-1, 1])
        )

        pred_ious_this_matching = tf.boolean_mask(
            tf.reduce_sum(matching_matrix * pair_wise_ious, 0), fg_mask_inboxes
        )
        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_indices,
            num_fg,
        )

    def test_step(self, data):
        x, y = data
        gt_boxes, gt_classes = self.label_encoder(x, y)

        # yolox internally works on center_xywh
        gt_boxes = bounding_box.convert_format(
            gt_boxes,
            source=self.bounding_box_format,
            target="center_xywh",
            images=x,
        )
        y_true = tf.concat([gt_boxes, gt_classes], -1)
        y_pred = self(x, training=False)
        _ = self.compute_loss(y_true, y_pred, input_shape=x.shape[1:3])

        return self.compute_metrics(x, {}, {}, sample_weight={})

    def _update_metrics(self, y_true, y_pred):
        y_true = bounding_box.convert_format(
            y_true,
            source=self.bounding_box_format,
            target=self._metrics_bounding_box_format,
        )
        y_pred = bounding_box.convert_format(
            y_pred,
            source=self.bounding_box_format,
            target=self._metrics_bounding_box_format,
        )
        self.compiled_metrics.update_state(y_true, y_pred)

    def predict(self, x, **kwargs):
        predictions = super().predict(x, **kwargs)
        predictions = self.decode_predictions(x, predictions)
        return predictions


def _parse_box_loss(loss):
    if not isinstance(loss, str):
        # support arbitrary callables
        return loss

    # case insensitive comparison
    if loss.lower() == "iou":
        return keras_cv.losses.IoULoss(
            bounding_box_format="center_xywh", mode="quadratic", reduction="none"
        )
    if loss.lower() == "giou":
        return keras_cv.losses.GIoULoss(
            bounding_box_format="center_xywh", reduction="none"
        )

    raise ValueError(
        "Expected `box_loss` to be either a Keras Loss, "
        f"callable, or one of ['IoU', GIoU].  Got loss={loss}."
    )


def _parse_classification_loss(loss):
    if not isinstance(loss, str):
        # support arbitrary callables
        return loss

    # case insensitive comparison
    if loss.lower() == "binary_crossentropy":
        return tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction="none")

    raise ValueError(
        "Expected `classification_loss` to be either a Keras Loss, "
        f"callable, or the string 'binary_crossentropy'.  Got loss={loss}."
    )


def _parse_objectness_loss(loss):
    if not isinstance(loss, str):
        # support arbitrary callables
        return loss

    # case insensitive comparison
    if loss.lower() == "binary_crossentropy":
        return tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction="none")

    raise ValueError(
        "Expected `objectness_loss` to be either a Keras Loss, "
        f"callable, or the string 'binary_crossentropy'.  Got loss={loss}."
    )



BASE_DOCSTRING = """Instantiates the {name} architecture using the given phi value.

    {name} is an anchor-free network that utilizes SOTA techniques such as simOTA for
    label assignment.

    Reference:
    - [YOLOX: Exceeding YOLO Series in 2021](https://arxiv.org/abs/2107.08430)
    - [Keras Implemenation of YoloX by bubbliiing](https://github.com/bubbliiiing/yolox-keras)

    Arguments:
        classes: The number of classes to be considered for the {name} head.
        bounding_box_format:  The format of bounding boxes of input dataset. Refer
            [to the keras.io docs](https://keras.io/api/keras_cv/bounding_box/formats/)
            for more details on supported bounding box formats.
        backbone: an optional `tf.keras.Model` custom backbone model. Defaults
            to a CSPDarkNet model corresponding to {name} model with include_rescaling=True.
        label_encoder: (Optional) a keras.Layer that accepts an image Tensor, a
            bounding box Tensor and a bounding box class Tensor to its `call()` method,
            and returns YoloX training targets.  By default, a YoloX standard 
            LabelEncoder is created and used.
        prediction_decoder: (Optional)  A `keras.layer` that is responsible for
            transforming {name} predictions into usable bounding box Tensors.  If
            not provided, a default DecodePredictions is provided. The default layer
            uses a `NonMaxSuppression` operation for box pruning.
        feature_pyramid: (Optional) A `keras.Model` representing a feature pyramid
            network (FPN).  The feature pyramid network is called on the outputs of the
            `backbone`.  The KerasCV default backbones return three outputs in a list,
            but custom backbones may be written and used with custom feature pyramid
            networks.  If not provided, a default feature pyramid network is produced
            by the library.  The default feature pyramid network is compatible with all
            standard keras_cv backbones.
        name: (Optional) the name to be passed to the model. Defaults to `"{name}"`.
"""


class YoloX_tiny(YoloX):
    def __init__(
        self,
        classes,
        bounding_box_format,
        backbone=None,
        label_encoder=None,
        prediction_decoder=None,
        feature_pyramid=None,
        name="YoloX-Tiny",
        **kwargs,
    ):
        super().__init__(
            classes=classes,
            bounding_box_format=bounding_box_format,
            phi="tiny",
            backbone=backbone,
            label_encoder=label_encoder,
            prediction_decoder=prediction_decoder,
            feature_pyramid=feature_pyramid,
            name=name,
            **kwargs,
        )


class YoloX_s(YoloX):
    def __init__(
        self,
        classes,
        bounding_box_format,
        backbone=None,
        label_encoder=None,
        prediction_decoder=None,
        feature_pyramid=None,
        name="YoloX-s",
        **kwargs,
    ):
        super().__init__(
            classes=classes,
            bounding_box_format=bounding_box_format,
            phi="s",
            backbone=backbone,
            label_encoder=label_encoder,
            prediction_decoder=prediction_decoder,
            feature_pyramid=feature_pyramid,
            name=name,
            **kwargs,
        )


class YoloX_m(YoloX):
    def __init__(
        self,
        classes,
        bounding_box_format,
        backbone=None,
        label_encoder=None,
        prediction_decoder=None,
        feature_pyramid=None,
        name="YoloX-m",
        **kwargs,
    ):
        super().__init__(
            classes=classes,
            bounding_box_format=bounding_box_format,
            phi="m",
            backbone=backbone,
            label_encoder=label_encoder,
            prediction_decoder=prediction_decoder,
            feature_pyramid=feature_pyramid,
            name=name,
            **kwargs,
        )


class YoloX_l(YoloX):
    def __init__(
        self,
        classes,
        bounding_box_format,
        backbone=None,
        label_encoder=None,
        prediction_decoder=None,
        feature_pyramid=None,
        name="YoloX-l",
        **kwargs,
    ):
        super().__init__(
            classes=classes,
            bounding_box_format=bounding_box_format,
            phi="l",
            backbone=backbone,
            label_encoder=label_encoder,
            prediction_decoder=prediction_decoder,
            feature_pyramid=feature_pyramid,
            name=name,
            **kwargs,
        )


class YoloX_x(YoloX):
    def __init__(
        self,
        classes,
        bounding_box_format,
        backbone=None,
        label_encoder=None,
        prediction_decoder=None,
        feature_pyramid=None,
        name="YoloX-x",
        **kwargs,
    ):
        super().__init__(
            classes=classes,
            bounding_box_format=bounding_box_format,
            phi="x",
            backbone=backbone,
            label_encoder=label_encoder,
            prediction_decoder=prediction_decoder,
            feature_pyramid=feature_pyramid,
            name=name,
            **kwargs,
        )


YoloX_tiny.__doc__ = BASE_DOCSTRING.format(name="YoloX_tiny")
YoloX_s.__doc__ = BASE_DOCSTRING.format(name="YoloX_s")
YoloX_m.__doc__ = BASE_DOCSTRING.format(name="YoloX_m")
YoloX_l.__doc__ = BASE_DOCSTRING.format(name="YoloX_l")
YoloX_x.__doc__ = BASE_DOCSTRING.format(name="YoloX_x")
