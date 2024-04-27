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
import warnings

from keras_cv.src import bounding_box
from keras_cv.src import layers
from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.backend import keras
from keras_cv.src.backend import ops
from keras_cv.src.losses.ciou_loss import CIoULoss
from keras_cv.src.models.backbones.backbone_presets import backbone_presets
from keras_cv.src.models.backbones.backbone_presets import (
    backbone_presets_with_weights,
)
from keras_cv.src.models.object_detection.__internal__ import unpack_input
from keras_cv.src.models.object_detection.yolo_v8.yolo_v8_detector_presets import (  # noqa: E501
    yolo_v8_detector_presets,
)
from keras_cv.src.models.object_detection.yolo_v8.yolo_v8_label_encoder import (
    YOLOV8LabelEncoder,
)
from keras_cv.src.models.object_detection.yolo_v8.yolo_v8_layers import (
    apply_conv_bn,
)
from keras_cv.src.models.object_detection.yolo_v8.yolo_v8_layers import (
    apply_csp_block,
)
from keras_cv.src.models.task import Task
from keras_cv.src.utils.python_utils import classproperty
from keras_cv.src.utils.train import get_feature_extractor

BOX_REGRESSION_CHANNELS = 64


def get_anchors(
    image_shape,
    strides=[8, 16, 32],
    base_anchors=[0.5, 0.5],
):
    """Gets anchor points for YOLOV8.

    YOLOV8 uses anchor points representing the center of proposed boxes, and
    matches ground truth boxes to anchors based on center points.

    Args:
        image_shape: tuple or list of two integers representing the height and
            width of input images, respectively.
        strides: tuple of list of integers, the size of the strides across the
            image size that should be used to create anchors.
        base_anchors: tuple or list of two integers representing the offset from
            (0,0) to start creating the center of anchor boxes, relative to the
            stride. For example, using the default (0.5, 0.5) creates the first
            anchor box for each stride such that its center is half of a stride
            from the edge of the image.

    Returns:
        A tuple of anchor centerpoints and anchor strides. Multiplying the
        two together will yield the centerpoints in absolute x,y format.

    """
    base_anchors = ops.array(base_anchors, dtype="float32")

    all_anchors = []
    all_strides = []
    for stride in strides:
        hh_centers = ops.arange(0, image_shape[0], stride)
        ww_centers = ops.arange(0, image_shape[1], stride)
        ww_grid, hh_grid = ops.meshgrid(ww_centers, hh_centers)
        grid = ops.cast(
            ops.reshape(ops.stack([hh_grid, ww_grid], 2), [-1, 1, 2]),
            "float32",
        )
        anchors = (
            ops.expand_dims(
                base_anchors * ops.array([stride, stride], "float32"), 0
            )
            + grid
        )
        anchors = ops.reshape(anchors, [-1, 2])
        all_anchors.append(anchors)
        all_strides.append(ops.repeat(stride, anchors.shape[0]))

    all_anchors = ops.cast(ops.concatenate(all_anchors, axis=0), "float32")
    all_strides = ops.cast(ops.concatenate(all_strides, axis=0), "float32")

    all_anchors = all_anchors / all_strides[:, None]

    # Swap the x and y coordinates of the anchors.
    all_anchors = ops.concatenate(
        [all_anchors[:, 1, None], all_anchors[:, 0, None]], axis=-1
    )
    return all_anchors, all_strides


def apply_path_aggregation_fpn(features, depth=3, name="fpn"):
    """Applies the Feature Pyramid Network (FPN) to the outputs of a backbone.

    Args:
        features: list of tensors representing the P3, P4, and P5 outputs of the
            backbone.
        depth: integer, the depth of the CSP blocks used in the FPN.
        name: string, a prefix for names of layers used by the FPN.

    Returns:
        A list of three tensors whose shapes are the same as the three inputs,
        but which are dependent on each of the three inputs to combine the high
        resolution of the P3 inputs with the strong feature representations of
        the P5 inputs.

    """
    p3, p4, p5 = features

    # Upsample P5 and concatenate with P4, then apply a CSPBlock.
    p5_upsampled = ops.repeat(ops.repeat(p5, 2, axis=1), 2, axis=2)
    p4p5 = ops.concatenate([p5_upsampled, p4], axis=-1)
    p4p5 = apply_csp_block(
        p4p5,
        channels=p4.shape[-1],
        depth=depth,
        shortcut=False,
        activation="swish",
        name=f"{name}_p4p5",
    )

    # Upsample P4P5 and concatenate with P3, then apply a CSPBlock.
    p4p5_upsampled = ops.repeat(ops.repeat(p4p5, 2, axis=1), 2, axis=2)
    p3p4p5 = ops.concatenate([p4p5_upsampled, p3], axis=-1)
    p3p4p5 = apply_csp_block(
        p3p4p5,
        channels=p3.shape[-1],
        depth=depth,
        shortcut=False,
        activation="swish",
        name=f"{name}_p3p4p5",
    )

    # Downsample P3P4P5, concatenate with P4P5, and apply a CSP Block.
    p3p4p5_d1 = apply_conv_bn(
        p3p4p5,
        p3p4p5.shape[-1],
        kernel_size=3,
        strides=2,
        activation="swish",
        name=f"{name}_p3p4p5_downsample1",
    )
    p3p4p5_d1 = ops.concatenate([p3p4p5_d1, p4p5], axis=-1)
    p3p4p5_d1 = apply_csp_block(
        p3p4p5_d1,
        channels=p4p5.shape[-1],
        shortcut=False,
        activation="swish",
        name=f"{name}_p3p4p5_downsample1_block",
    )

    # Downsample the resulting P3P4P5 again, concatenate with P5, and apply
    # another CSP Block.
    p3p4p5_d2 = apply_conv_bn(
        p3p4p5_d1,
        p3p4p5_d1.shape[-1],
        kernel_size=3,
        strides=2,
        activation="swish",
        name=f"{name}_p3p4p5_downsample2",
    )
    p3p4p5_d2 = ops.concatenate([p3p4p5_d2, p5], axis=-1)
    p3p4p5_d2 = apply_csp_block(
        p3p4p5_d2,
        channels=p5.shape[-1],
        shortcut=False,
        activation="swish",
        name=f"{name}_p3p4p5_downsample2_block",
    )

    return [p3p4p5, p3p4p5_d1, p3p4p5_d2]


def apply_yolo_v8_head(
    inputs,
    num_classes,
    name="yolo_v8_head",
):
    """Applies a YOLOV8 head.

    Makes box and class predictions based on the output of a feature pyramid
    network.

    Args:
        inputs: list of tensors output by the Feature Pyramid Network, should
            have the same shape as the P3, P4, and P5 outputs of the backbone.
        num_classes: integer, the number of classes that a bounding box could
            possibly be assigned to.
        name: string, a prefix for names of layers used by the head.

    Returns: A dictionary with two entries. The "boxes" entry contains box
        regression predictions, while the "classes" entry contains class
        predictions.
    """
    # 64 is the default number of channels, as 16 components are used to predict
    # each of the 4 offsets for corner points of a bounding box with respect
    # to the center point. In cases where the input has much higher resolution
    # (e.g. the P3 input has >256 channels), we use additional channels for
    # the intermediate conv layers. This is only true for very large backbones.
    box_channels = max(BOX_REGRESSION_CHANNELS, inputs[0].shape[-1] // 4)

    # We use at least num_classes channels for intermediate conv layer for class
    # predictions. In most cases, the P3 input has many more channels than the
    # number of classes, so we preserve those channels until the final layer.
    class_channels = max(num_classes, inputs[0].shape[-1])

    # We compute box and class predictions for each of the feature maps from
    # the FPN and then combine them.
    outputs = []
    for id, feature in enumerate(inputs):
        cur_name = f"{name}_{id+1}"

        box_predictions = apply_conv_bn(
            feature,
            box_channels,
            kernel_size=3,
            activation="swish",
            name=f"{cur_name}_box_1",
        )
        box_predictions = apply_conv_bn(
            box_predictions,
            box_channels,
            kernel_size=3,
            activation="swish",
            name=f"{cur_name}_box_2",
        )
        box_predictions = keras.layers.Conv2D(
            filters=BOX_REGRESSION_CHANNELS,
            kernel_size=1,
            name=f"{cur_name}_box_3_conv",
        )(box_predictions)

        class_predictions = apply_conv_bn(
            feature,
            class_channels,
            kernel_size=3,
            activation="swish",
            name=f"{cur_name}_class_1",
        )
        class_predictions = apply_conv_bn(
            class_predictions,
            class_channels,
            kernel_size=3,
            activation="swish",
            name=f"{cur_name}_class_2",
        )
        class_predictions = keras.layers.Conv2D(
            filters=num_classes,
            kernel_size=1,
            name=f"{cur_name}_class_3_conv",
        )(class_predictions)
        class_predictions = keras.layers.Activation(
            "sigmoid", name=f"{cur_name}_classifier"
        )(class_predictions)

        out = ops.concatenate([box_predictions, class_predictions], axis=-1)
        out = keras.layers.Reshape(
            [-1, out.shape[-1]], name=f"{cur_name}_output_reshape"
        )(out)
        outputs.append(out)

    outputs = ops.concatenate(outputs, axis=1)
    outputs = keras.layers.Activation(
        "linear", dtype="float32", name="box_outputs"
    )(outputs)

    return {
        "boxes": outputs[:, :, :BOX_REGRESSION_CHANNELS],
        "classes": outputs[:, :, BOX_REGRESSION_CHANNELS:],
    }


def decode_regression_to_boxes(preds):
    """Decodes the results of the YOLOV8Detector forward-pass into boxes.

    Returns left / top / right / bottom predictions with respect to anchor
    points.

    Each coordinate is encoded with 16 predicted values. Those predictions are
    softmaxed and multiplied by [0..15] to make predictions. The resulting
    predictions are relative to the stride of an anchor box (and correspondingly
    relative to the scale of the feature map from which the predictions came).
    """
    preds_bbox = keras.layers.Reshape((-1, 4, BOX_REGRESSION_CHANNELS // 4))(
        preds
    )
    preds_bbox = ops.nn.softmax(preds_bbox, axis=-1) * ops.arange(
        BOX_REGRESSION_CHANNELS // 4, dtype="float32"
    )
    return ops.sum(preds_bbox, axis=-1)


def dist2bbox(distance, anchor_points):
    """Decodes distance predictions into xyxy boxes.

    Input left / top / right / bottom predictions are transformed into xyxy box
    predictions based on anchor points.

    The resulting xyxy predictions must be scaled by the stride of their
    corresponding anchor points to yield an absolute xyxy box.
    """
    left_top, right_bottom = ops.split(distance, 2, axis=-1)
    x1y1 = anchor_points - left_top
    x2y2 = anchor_points + right_bottom
    return ops.concatenate((x1y1, x2y2), axis=-1)  # xyxy bbox


@keras_cv_export(
    [
        "keras_cv.models.YOLOV8Detector",
        "keras_cv.models.object_detection.YOLOV8Detector",
    ]
)
class YOLOV8Detector(Task):
    """Implements the YOLOV8 architecture for object detection.

    Args:
        backbone: `keras.Model`, must implement the `pyramid_level_inputs`
            property with keys "P3", "P4", and "P5" and layer names as values.
            A sensible backbone to use is the `keras_cv.models.YOLOV8Backbone`.
        num_classes: integer, the number of classes in your dataset excluding the
            background class. Classes should be represented by integers in the
            range [0, num_classes).
        bounding_box_format: string, the format of bounding boxes of input dataset.
            Refer
            [to the keras.io docs](https://keras.io/api/keras_cv/bounding_box/formats/)
            for more details on supported bounding box formats.
        fpn_depth: integer, a specification of the depth of the CSP blocks in
            the Feature Pyramid Network. This is usually 1, 2, or 3, depending
            on the size of your YOLOV8Detector model. We recommend using 3 for
            "yolo_v8_l_backbone" and "yolo_v8_xl_backbone". Defaults to 2.
        label_encoder: (Optional)  A `YOLOV8LabelEncoder` that is
            responsible for transforming input boxes into trainable labels for
            YOLOV8Detector. If not provided, a default is provided.
        prediction_decoder: (Optional)  A `keras.layers.Layer` that is
            responsible for transforming YOLOV8 predictions into usable
            bounding boxes. If not provided, a default is provided. The
            default `prediction_decoder` layer is a
            `keras_cv.layers.MultiClassNonMaxSuppression` layer, which uses
            a Non-Max Suppression for box pruning.

    Example:
    ```python
    images = tf.ones(shape=(1, 512, 512, 3))
    labels = {
        "boxes": tf.constant([
            [
                [0, 0, 100, 100],
                [100, 100, 200, 200],
                [300, 300, 100, 100],
            ]
        ], dtype=tf.float32),
        "classes": tf.constant([[1, 1, 1]], dtype=tf.int64),
    }

    model = keras_cv.models.YOLOV8Detector(
        num_classes=20,
        bounding_box_format="xywh",
        backbone=keras_cv.models.YOLOV8Backbone.from_preset(
            "yolo_v8_m_backbone_coco"
        ),
        fpn_depth=2
    )

    # Evaluate model without box decoding and NMS
    model(images)

    # Prediction with box decoding and NMS
    model.predict(images)

    # Train model
    model.compile(
        classification_loss='binary_crossentropy',
        box_loss='ciou',
        optimizer=tf.optimizers.SGD(global_clipnorm=10.0),
        jit_compile=False,
    )
    model.fit(images, labels)
    ```
    """  # noqa: E501

    def __init__(
        self,
        backbone,
        num_classes,
        bounding_box_format,
        fpn_depth=2,
        label_encoder=None,
        prediction_decoder=None,
        **kwargs,
    ):
        extractor_levels = ["P3", "P4", "P5"]
        extractor_layer_names = [
            backbone.pyramid_level_inputs[i] for i in extractor_levels
        ]
        feature_extractor = get_feature_extractor(
            backbone, extractor_layer_names, extractor_levels
        )

        images = keras.layers.Input(feature_extractor.input_shape[1:])
        features = list(feature_extractor(images).values())

        fpn_features = apply_path_aggregation_fpn(
            features, depth=fpn_depth, name="pa_fpn"
        )

        outputs = apply_yolo_v8_head(
            fpn_features,
            num_classes,
        )

        # To make loss metrics pretty, we use a no-op layer with a good name.
        boxes = keras.layers.Concatenate(axis=1, name="box")([outputs["boxes"]])
        scores = keras.layers.Concatenate(axis=1, name="class")(
            [outputs["classes"]]
        )

        outputs = {"boxes": boxes, "classes": scores}
        super().__init__(inputs=images, outputs=outputs, **kwargs)

        self.bounding_box_format = bounding_box_format
        self._prediction_decoder = (
            prediction_decoder
            or layers.NonMaxSuppression(
                bounding_box_format=bounding_box_format,
                from_logits=False,
                confidence_threshold=0.2,
                iou_threshold=0.7,
            )
        )
        self.backbone = backbone
        self.fpn_depth = fpn_depth
        self.num_classes = num_classes
        self.label_encoder = label_encoder or YOLOV8LabelEncoder(
            num_classes=num_classes
        )

    def compile(
        self,
        box_loss,
        classification_loss,
        box_loss_weight=7.5,
        classification_loss_weight=0.5,
        metrics=None,
        **kwargs,
    ):
        """Compiles the YOLOV8Detector.

        `compile()` mirrors the standard Keras `compile()` method, but has one
        key distinction -- two losses must be provided: `box_loss` and
        `classification_loss`.

        Args:
            box_loss: a Keras loss to use for box offset regression. A
                preconfigured loss is provided when the string "ciou" is passed.
            classification_loss: a Keras loss to use for box classification. A
                preconfigured loss is provided when the string
                "binary_crossentropy" is passed.
            box_loss_weight: (optional) float, a scaling factor for the box
                loss. Defaults to 7.5.
            classification_loss_weight: (optional) float, a scaling factor for
                the classification loss. Defaults to 0.5.
            kwargs: most other `keras.Model.compile()` arguments are supported
                and propagated to the `keras.Model` class.
        """
        if metrics is not None:
            raise ValueError("User metrics not yet supported for YOLOV8")

        if isinstance(box_loss, str):
            if box_loss == "ciou":
                box_loss = CIoULoss(bounding_box_format="xyxy", reduction="sum")
            elif box_loss == "iou":
                warnings.warn(
                    "YOLOV8 recommends using CIoU loss, but was configured to "
                    "use standard IoU. Consider using `box_loss='ciou'` "
                    "instead."
                )
            else:
                raise ValueError(
                    f"Invalid box loss for YOLOV8Detector: {box_loss}. Box "
                    "loss should be a keras.Loss or the string 'ciou'."
                )
        if isinstance(classification_loss, str):
            if classification_loss == "binary_crossentropy":
                classification_loss = keras.losses.BinaryCrossentropy(
                    reduction="sum"
                )
            else:
                raise ValueError(
                    "Invalid classification loss for YOLOV8Detector: "
                    f"{classification_loss}. Classification loss should be a "
                    "keras.Loss or the string 'binary_crossentropy'."
                )

        self.box_loss = box_loss
        self.classification_loss = classification_loss
        self.box_loss_weight = box_loss_weight
        self.classification_loss_weight = classification_loss_weight

        losses = {
            "box": self.box_loss,
            "class": self.classification_loss,
        }

        super().compile(loss=losses, **kwargs)

    def train_step(self, *args):
        data = args[-1]
        args = args[:-1]
        x, y = unpack_input(data)
        return super().train_step(*args, (x, y))

    def test_step(self, *args):
        data = args[-1]
        args = args[:-1]
        x, y = unpack_input(data)
        return super().test_step(*args, (x, y))

    def compute_loss(self, x, y, y_pred, sample_weight=None, **kwargs):
        box_pred, cls_pred = y_pred["boxes"], y_pred["classes"]

        pred_boxes = decode_regression_to_boxes(box_pred)
        pred_scores = cls_pred

        anchor_points, stride_tensor = get_anchors(image_shape=x.shape[1:])
        stride_tensor = ops.expand_dims(stride_tensor, axis=-1)

        gt_labels = y["classes"]

        mask_gt = ops.all(y["boxes"] > -1.0, axis=-1, keepdims=True)
        gt_bboxes = bounding_box.convert_format(
            y["boxes"],
            source=self.bounding_box_format,
            target="xyxy",
            images=x,
        )

        pred_bboxes = dist2bbox(pred_boxes, anchor_points)

        target_bboxes, target_scores, fg_mask = self.label_encoder(
            pred_scores,
            ops.cast(pred_bboxes * stride_tensor, gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_bboxes /= stride_tensor
        target_scores_sum = ops.maximum(ops.sum(target_scores), 1)
        box_weight = ops.expand_dims(
            ops.sum(target_scores, axis=-1) * fg_mask,
            axis=-1,
        )

        y_true = {
            "box": target_bboxes * fg_mask[..., None],
            "class": target_scores,
        }
        y_pred = {
            "box": pred_bboxes * fg_mask[..., None],
            "class": pred_scores,
        }
        sample_weights = {
            "box": self.box_loss_weight * box_weight / target_scores_sum,
            "class": self.classification_loss_weight / target_scores_sum,
        }

        return super().compute_loss(
            x=x, y=y_true, y_pred=y_pred, sample_weight=sample_weights, **kwargs
        )

    def decode_predictions(
        self,
        pred,
        images,
    ):
        boxes = pred["boxes"]
        scores = pred["classes"]

        boxes = decode_regression_to_boxes(boxes)

        anchor_points, stride_tensor = get_anchors(image_shape=images.shape[1:])
        stride_tensor = ops.expand_dims(stride_tensor, axis=-1)

        box_preds = dist2bbox(boxes, anchor_points) * stride_tensor
        box_preds = bounding_box.convert_format(
            box_preds,
            source="xyxy",
            target=self.bounding_box_format,
            images=images,
        )

        return self.prediction_decoder(box_preds, scores)

    def predict_step(self, *args):
        outputs = super().predict_step(*args)
        if isinstance(outputs, tuple):
            return self.decode_predictions(outputs[0], args[-1]), outputs[1]
        else:
            return self.decode_predictions(outputs, args[-1])

    @property
    def prediction_decoder(self):
        return self._prediction_decoder

    @prediction_decoder.setter
    def prediction_decoder(self, prediction_decoder):
        if prediction_decoder.bounding_box_format != self.bounding_box_format:
            raise ValueError(
                "Expected `prediction_decoder` and YOLOV8Detector to "
                "use the same `bounding_box_format`, but got "
                "`prediction_decoder.bounding_box_format="
                f"{prediction_decoder.bounding_box_format}`, and "
                "`self.bounding_box_format="
                f"{self.bounding_box_format}`."
            )
        self._prediction_decoder = prediction_decoder
        self.make_predict_function(force=True)
        self.make_train_function(force=True)
        self.make_test_function(force=True)

    def get_config(self):
        return {
            "num_classes": self.num_classes,
            "bounding_box_format": self.bounding_box_format,
            "fpn_depth": self.fpn_depth,
            "backbone": keras.saving.serialize_keras_object(self.backbone),
            "label_encoder": keras.saving.serialize_keras_object(
                self.label_encoder
            ),
            "prediction_decoder": keras.saving.serialize_keras_object(
                self._prediction_decoder
            ),
        }

    @classmethod
    def from_config(cls, config):
        config["backbone"] = keras.saving.deserialize_keras_object(
            config["backbone"]
        )
        label_encoder = config.get("label_encoder")
        if label_encoder is not None and isinstance(label_encoder, dict):
            config["label_encoder"] = keras.saving.deserialize_keras_object(
                label_encoder
            )
        prediction_decoder = config.get("prediction_decoder")
        if prediction_decoder is not None and isinstance(
            prediction_decoder, dict
        ):
            config["prediction_decoder"] = (
                keras.saving.deserialize_keras_object(prediction_decoder)
            )
        return cls(**config)

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
