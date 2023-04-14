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
import copy

import tensorflow as tf
from keras import layers
from tensorflow import keras

import keras_cv
from keras_cv import bounding_box
from keras_cv.models.backbones.backbone_presets import backbone_presets
from keras_cv.models.backbones.backbone_presets import (
    backbone_presets_with_weights,
)
from keras_cv.models.object_detection import predict_utils
from keras_cv.models.object_detection.__internal__ import unpack_input
from keras_cv.models.object_detection.yolo_v8.yolo_v8_detector_presets import (
    yolo_v8_detector_presets,
)
from keras_cv.models.object_detection.yolo_v8.yolo_v8_iou_loss import (
    YOLOV8IoULoss,
)
from keras_cv.models.object_detection.yolo_v8.yolo_v8_label_encoder import (
    YOLOV8LabelEncoder,
)
from keras_cv.models.object_detection.yolo_v8.yolo_v8_layers import (
    apply_conv_bn,
)
from keras_cv.models.object_detection.yolo_v8.yolo_v8_layers import (
    apply_csp_block,
)
from keras_cv.models.task import Task
from keras_cv.utils.python_utils import classproperty
from keras_cv.utils.train import get_feature_extractor


def get_anchors(
    image_shape=(512, 512, 3),
    strides=[8, 16, 32],
    base_anchors=[0.5, 0.5, 0.5, 0.5],
):
    base_anchors = tf.constant(base_anchors, dtype=tf.float32)

    all_anchors = []
    all_strides = []
    for stride in strides:
        hh_centers = tf.range(start=0, limit=image_shape[0], delta=stride)
        ww_centers = tf.range(start=0, limit=image_shape[1], delta=stride)
        ww_grid, hh_grid = tf.meshgrid(ww_centers, hh_centers)
        grid = tf.cast(
            tf.reshape(
                tf.stack([hh_grid, ww_grid, hh_grid, ww_grid], 2), [-1, 1, 4]
            ),
            tf.float32,
        )
        anchors = (
            tf.expand_dims(base_anchors * [stride, stride, stride, stride], 0)
            + grid
        )
        anchors = tf.reshape(anchors, [-1, 4])
        all_anchors.append(anchors)
        all_strides.append(tf.repeat(stride, anchors.shape[0]))

    all_anchors = tf.concat(all_anchors, axis=0)
    all_anchors = tf.cast(all_anchors, tf.float32)

    all_strides = tf.concat(all_strides, axis=0)
    all_strides = tf.cast(all_strides, tf.float32)

    all_anchors = all_anchors[:, :2] / all_strides[:, None]

    all_anchors = tf.concat(
        [all_anchors[:, 1, tf.newaxis], all_anchors[:, 0, tf.newaxis]], axis=-1
    )
    return all_anchors, all_strides


def path_aggregation_fpn(features, depth=3, name=None):
    # yolov8
    # 9: p5 1024 ---+----------------------+-> 21: out2 1024
    #               v [up 1024 -> concat]  ^ [down 512 -> concat]
    # 6: p4 512 --> 12: p4p5 512 --------> 18: out1 512
    #               v [up 512 -> concat]   ^ [down 256 -> concat]
    # 4: p3 256 --> 15: p3p4p5 256 --------+--> 15: out0 128
    # features: [p3, p4, p5]
    channel_axis = -1
    upsamples = [features[-1]]
    # upsamples: [p5], features[:-1][::-1]: [p4, p3] -> [p5, p4p5, p3p4p5]
    for id, feature in enumerate(features[:-1][::-1]):
        size = tf.shape(feature)[1:-1]
        x = tf.image.resize(upsamples[-1], size, method="nearest")
        x = tf.concat([x, feature], axis=channel_axis)

        out_channel = feature.shape[channel_axis]
        x = apply_csp_block(
            x,
            channels=out_channel,
            depth=depth,
            shortcut=False,
            activation="swish",
            name=f"{name}_p{len(features) + 1 - id}",
        )
        upsamples.append(x)

    downsamples = [upsamples[-1]]
    # downsamples: [p3p4p5], upsamples[:-1][::-1]:
    # [p4p5, p5] -> [p3p4p5, p3p4p5 + p4p5, p3p4p5 + p4p5 + p5]
    for id, ii in enumerate(upsamples[:-1][::-1]):
        cur_name = f"{name}_c3n{id + 3}"
        x = apply_conv_bn(
            downsamples[-1],
            downsamples[-1].shape[channel_axis],
            kernel_size=3,
            strides=2,
            activation="swish",
            name=f"{cur_name}_down",
        )
        x = tf.concat([x, ii], axis=channel_axis)

        out_channel = ii.shape[channel_axis]
        x = apply_csp_block(
            x,
            channels=out_channel,
            depth=depth,
            shortcut=False,
            activation="swish",
            name=cur_name,
        )
        downsamples.append(x)
    return downsamples


def yolov8_head(
    inputs,
    num_classes=80,
    bbox_len=64,
    name="yolov8_head",
):
    outputs = []
    reg_channels = max(16, bbox_len, inputs[0].shape[-1] // 4)
    cls_channels = max(num_classes, inputs[0].shape[-1])
    for id, feature in enumerate(inputs):
        cur_name = f"{name}_{id+1}"

        reg_x = apply_conv_bn(
            feature,
            reg_channels,
            3,
            activation="swish",
            name=f"{cur_name}_reg_1",
        )
        reg_x = apply_conv_bn(
            reg_x,
            reg_channels,
            3,
            activation="swish",
            name=f"{cur_name}_reg_2",
        )
        reg_out = layers.Conv2D(
            filters=bbox_len,
            kernel_size=1,
            name=f"{cur_name}_reg_3_conv",
        )(reg_x)

        cls_x = apply_conv_bn(
            feature,
            cls_channels,
            3,
            activation="swish",
            name=f"{cur_name}_cls_1",
        )
        cls_x = apply_conv_bn(
            cls_x,
            cls_channels,
            3,
            activation="swish",
            name=f"{cur_name}_cls_2",
        )
        cls_out = layers.Conv2D(
            filters=num_classes,
            kernel_size=1,
            name=f"{cur_name}_cls_3_conv",
        )(cls_x)
        cls_out = layers.Activation("sigmoid", name=f"{cur_name}_classifier")(
            cls_out
        )

        out = tf.concat([reg_out, cls_out], axis=-1)
        out = layers.Reshape(
            [-1, out.shape[-1]], name=f"{cur_name}_output_reshape"
        )(out)
        outputs.append(out)

    outputs = tf.concat(outputs, axis=1)
    return outputs


def decode_regression_to_boxes(preds, regression_max=16):
    preds_bbox = tf.reshape(preds, (-1, preds.shape[1], 4, regression_max))
    preds_bbox = tf.nn.softmax(preds_bbox, axis=-1) * tf.range(
        regression_max, dtype="float32"
    )
    return tf.reduce_sum(preds_bbox, axis=-1)


def dist2bbox(distance, anchor_points):
    lt, rb = tf.split(distance, 2, axis=-1)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    return tf.concat((x1y1, x2y2), axis=-1)  # xyxy bbox


@keras.utils.register_keras_serializable(package="keras_cv")
class YOLOV8Detector(Task):
    """
    Implements the YOLOV8 architecture for object detection.

    Examples:
    ```python
    ```python
    images = tf.ones(shape=(1, 512, 512, 3))
    labels = {
        "boxes": [
            [
                [0, 0, 100, 100],
                [100, 100, 200, 200],
                [300, 300, 100, 100],
            ]
        ],
        "classes": [[1, 1, 1]],
    }
    model = keras_cv.models.YOLOV8Detector(
        num_classes=20,
        bounding_box_format="xywh",
        backbone=keras_cv.models.YOLOV8Backbone.from_preset(
            "yolov8_m_coco"
        ),
        fpn_depth=2.
    )

    # Evaluate model
    model(images)

    # Train model
    model.compile(
        classification_loss='binary_crossentropy',
        box_loss='iou',
        optimizer=tf.optimizers.SGD(global_clipnorm=10.0),
        jit_compile=False,
    )
    model.fit(images, labels)
    ```

    Args:
        num_classes: the number of classes in your dataset excluding the
            background class. Classes should be represented by integers in the
            range [0, num_classes).
        bounding_box_format: The format of bounding boxes of input dataset.
            Refer
            [to the keras.io docs](https://keras.io/api/keras_cv/bounding_box/formats/)
            for more details on supported bounding box formats.
        backbone: `keras.Model`. Must implement the `pyramid_level_inputs`
            property with keys 3, 4, and 5 and layer names as values. A
            sensible backbone to use in many cases is the
            `keras_cv.models.YOLOV8Backbone`
        fpn_depth: integer, a specification of the depth for the Feature
            Pyramid Network. This is usually 1, 2, or 3, depending on the
            size of your YOLOV8Detector model.
        label_encoder: (Optional)  A `YOLOV8LabelEncoder` that is
            responsible for transforming input boxes into trainable labels for
            YOLOV8. If not provided, a default is provided.
        prediction_decoder: (Optional)  A `keras.layers.Layer` that is
            responsible for transforming YOLOV8 predictions into usable
            bounding boxes. If not provided, a default is provided. The
            default `prediction_decoder` layer is a
            `keras_cv.layers.MultiClassNonMaxSuppression` layer, which uses
            a Non-Max Suppression for box pruning.
    """  # noqa: E501

    def __init__(
        self,
        num_classes,
        bounding_box_format,
        backbone,
        fpn_depth,
        label_encoder,
        prediction_decoder=None,
        **kwargs,
    ):
        extractor_levels = [2, 3, 4]
        extractor_layer_names = [
            backbone.pyramid_level_inputs[i] for i in extractor_levels
        ]
        feature_extractor = get_feature_extractor(
            backbone, extractor_layer_names, extractor_levels
        )

        images = layers.Input(feature_extractor.input_shape[1:])
        features = list(feature_extractor(images).values())

        # Apply the FPN
        fpn_features = path_aggregation_fpn(
            features, depth=fpn_depth, name="pa_fpn"
        )

        outputs = yolov8_head(
            fpn_features,
            num_classes,
        )
        outputs = layers.Activation(
            "linear", dtype="float32", name="outputs_fp32"
        )(outputs)
        boxes, scores = outputs[:, :, :64], outputs[:, :, 64:]

        # To make loss metrics pretty, we use a no-op layer with a good name.
        boxes = keras.layers.Concatenate(axis=1, name="box")([boxes])
        scores = keras.layers.Concatenate(axis=1, name="class")([scores])

        outputs = {"boxes": boxes, "classes": scores}
        super().__init__(inputs=images, outputs=outputs, **kwargs)

        self.bounding_box_format = bounding_box_format
        self.prediction_decoder = (
            prediction_decoder
            or keras_cv.layers.MultiClassNonMaxSuppression(
                bounding_box_format=bounding_box_format,
                from_logits=False,
                confidence_threshold=0.2,
                iou_threshold=0.5,
            )
        )
        self.backbone = backbone
        self.fpn_depth = fpn_depth
        self.num_classes = num_classes
        self.label_encoder = label_encoder or YOLOV8LabelEncoder(
            num_classes=num_classes
        )

        self.box_loss_weight = 7.5
        self.class_loss_weight = 0.5

    def compile(
        self,
        box_loss=None,
        classification_loss=None,
        metrics=None,
        **kwargs,
    ):
        """Compiles the YOLOV8Detector.

        `compile()` mirrors the standard Keras `compile()` method, but has one
        key distinction -- two losses must be provided: `box_loss` and
        `classification_loss`.

        Args:
            box_loss: a Keras loss to use for box offset regression. A
                preconfigured loss is provided when the string "iou" is passed.
            classification_loss: a Keras loss to use for box classification. A
                preconfigured loss is provided when the string
                "binary_crossentropy" is passed.
            kwargs: most other `keras.Model.compile()` arguments are supported
                and propagated to the `keras.Model` class.
        """
        if metrics is not None:
            raise ValueError("User metrics not yet supported for YOLOV8")

        self.box_loss = _parse_box_loss(box_loss)
        self.classification_loss = _parse_classification_loss(
            classification_loss
        )

        losses = {
            "box": self.box_loss,
            "class": self.classification_loss,
        }

        super().compile(loss=losses, **kwargs)

    def train_step(self, data):
        x, y = unpack_input(data)

        with tf.GradientTape() as tape:
            outputs = self(x, training=True)
            box_pred, cls_pred = outputs["boxes"], outputs["classes"]
            total_loss = self.compute_loss(x, y, box_pred, cls_pred)

        trainable_vars = self.trainable_variables

        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return super().compute_metrics(x, {}, {}, sample_weight={})

    def test_step(self, data):
        x, y = unpack_input(data)

        outputs = self(x, training=False)
        box_pred, cls_pred = outputs["boxes"], outputs["classes"]
        _ = self.compute_loss(x, y, box_pred, cls_pred)

        return super().compute_metrics(x, {}, {}, sample_weight={})

    def compute_loss(self, x, y, box_pred, cls_pred, return_early=False):
        pred_boxes = decode_regression_to_boxes(box_pred)
        pred_scores = cls_pred

        image_shape = (640, 640, 3)
        anchor_points, stride_tensor = get_anchors(image_shape)
        stride_tensor = tf.expand_dims(stride_tensor, axis=-1)

        gt_labels = y["classes"]

        mask_gt = tf.reduce_all(y["boxes"] > -1.0, axis=-1, keepdims=True)
        gt_bboxes = bounding_box.convert_format(
            y["boxes"],
            source=self.bounding_box_format,
            target="xyxy",
            image_shape=image_shape,
        )

        pred_bboxes = dist2bbox(pred_boxes, anchor_points)

        target_bboxes, target_scores, fg_mask = self.label_encoder(
            pred_scores,
            tf.cast(pred_bboxes * stride_tensor, gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_bboxes /= stride_tensor
        target_scores_sum = tf.math.maximum(tf.reduce_sum(target_scores), 1)
        box_weight = tf.expand_dims(
            tf.boolean_mask(tf.reduce_sum(target_scores, axis=-1), fg_mask),
            axis=-1,
        )

        y_true = {
            "box": target_bboxes[fg_mask],
            "class": target_scores,
        }
        y_pred = {
            "box": pred_bboxes[fg_mask],
            "class": pred_scores,
        }
        sample_weights = {
            "box": self.box_loss_weight * box_weight / target_scores_sum,
            "class": self.class_loss_weight / target_scores_sum,
        }

        return super().compute_loss(
            x=x, y=y_true, y_pred=y_pred, sample_weight=sample_weights
        )

    def decode_predictions(
        self,
        pred,
        images,
    ):
        boxes = pred["boxes"]
        scores = pred["classes"]

        boxes = decode_regression_to_boxes(boxes, 64 // 4)

        anchor_points, stride_tensor = get_anchors(image_shape=images.shape[1:])
        stride_tensor = tf.expand_dims(stride_tensor, axis=-1)

        box_preds = dist2bbox(boxes, anchor_points) * stride_tensor
        box_preds = bounding_box.convert_format(
            box_preds,
            source="xyxy",
            target=self.bounding_box_format,
            images=images,
        )

        return self.prediction_decoder(box_preds, scores)

    def make_predict_function(self, force=False):
        return predict_utils.make_predict_function(self, force=force)

    def get_config(self):
        return {
            "num_classes": self.num_classes,
            "bounding_box_format": self.bounding_box_format,
            "fpn_depth": self.fpn_depth,
            "backbone": keras.utils.serialize_keras_object(self.backbone),
            "label_encoder": self.label_encoder,
            "prediction_decoder": self.prediction_decoder,
        }

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return copy.deepcopy({**backbone_presets, **yolo_v8_detector_presets})

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return copy.deepcopy(
            {**backbone_presets_with_weights, **yolo_v8_detector_presets}
        )

    @classproperty
    def backbone_presets(cls):
        """Dictionary of preset names and configurations of compatible
        backbones."""
        return copy.deepcopy(backbone_presets)


def _parse_box_loss(loss):
    if not isinstance(loss, str):
        return loss

    if loss.lower() == "iou":
        return YOLOV8IoULoss(reduction="sum")

    raise ValueError(
        "Expected `box_loss` to be either a Keras Loss, "
        f"callable, or the string 'iou'. Got loss={loss}."
    )


def _parse_classification_loss(loss):
    if not isinstance(loss, str):
        return loss

    if loss.lower() == "binary_crossentropy":
        return keras.losses.BinaryCrossentropy(reduction="sum")

    raise ValueError(
        "Expected `classification_loss` to be either a Keras Loss, "
        f"callable, or the string 'binary_crossentropy'. Got loss={loss}."
    )
