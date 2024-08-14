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
from keras_cv.src.models.object_detection.yolo_v8.yolo_v8_detector import (
    apply_path_aggregation_fpn,
)
from keras_cv.src.models.object_detection.yolo_v8.yolo_v8_detector import (
    apply_yolo_v8_head as build_YOLOV8_detection_head,
)
from keras_cv.src.models.object_detection.yolo_v8.yolo_v8_detector import (
    decode_regression_to_boxes,
)
from keras_cv.src.models.object_detection.yolo_v8.yolo_v8_detector import (
    dist2bbox,
)
from keras_cv.src.models.object_detection.yolo_v8.yolo_v8_detector import (
    get_anchors,
)
from keras_cv.src.models.object_detection.yolo_v8.yolo_v8_detector_presets import (  # noqa: E501
    yolo_v8_detector_presets,
)
from keras_cv.src.models.object_detection.yolo_v8.yolo_v8_label_encoder import (  # noqa: E501
    YOLOV8LabelEncoder,
)
from keras_cv.src.models.object_detection.yolo_v8.yolo_v8_layers import (
    apply_conv_bn,
)
from keras_cv.src.models.task import Task
from keras_cv.src.utils.python_utils import classproperty
from keras_cv.src.utils.train import get_feature_extractor


def build_mask_prototypes(x, dimension, num_prototypes, name="prototypes"):
    """Builds protonet module. The outputs of this tensor are linearly combined
    with the regressed mask coefficients to produce the predicted masks.
    This is an implementation of the module proposed in YOLACT
    https://arxiv.org/abs/1904.02689.

    Args:
        x: tensor, representing the output of a low backone featuremap i.e. P3.
        dimension: integer, inner number of channels used for mask prototypes.
        num_prototypes: integer, number of mask prototypes to build predictions.
        name: string, a prefix for names of layers used by the prototypes.

    Returns:
        Tensor whose resolution is double than the inputted tensor.
        This tensor is used as a base to build linear combinations of masks.
    """
    x = apply_conv_bn(x, dimension, 3, name=f"{name}_0")
    x = apply_conv_bn(x, dimension, 3, name=f"{name}_1")
    x = apply_conv_bn(x, dimension, 3, name=f"{name}_2")
    upsampling_kwargs = {"interpolation": "bilinear", "name": f"{name}_3"}
    x = keras.layers.UpSampling2D((2, 2), **upsampling_kwargs)(x)
    x = apply_conv_bn(x, num_prototypes, 1, name=name)
    return x


def build_branch_mask_coefficients(x, dimension, num_prototypes, branch_arg):
    """Builds mask coefficients of a single branch as in Figure 4 of
    YOLACT https://arxiv.org/abs/1904.02689.

    Args:
        x: tensor, representing the outputs of a single branch of FPN i.e. P3.
        dimension: integer, inner number of channels used for mask coefficients.
        num_prototypes: integer, number of mask prototypes to build predictions.
        branch_arg: integer, representing the branch number. This is used to
        build the name of the tensors.

    Returns:
        Tensor representing the coefficients used to regress the outputted masks
        of a single branch.
    """
    name = f"branch_{branch_arg}_mask_coefficients"
    x = apply_conv_bn(x, dimension, 3, name=f"{name}_0")
    x = apply_conv_bn(x, dimension, 3, name=f"{name}_1")
    x = keras.layers.Conv2D(num_prototypes, 1, name=f"{name}_2")(x)
    x = keras.layers.Reshape((-1, num_prototypes), name=f"{name}_3")(x)
    return x


def build_mask_coefficients(branches, num_prototypes, dimension=32):
    """Builds all mask coefficients used to combine the prototypes masks.

    Args:
        branches: list of tensors, representing the outputs of a backbone model.
        num_prototypes: integer, number of mask prototypes to build predictions.
        dimension: integer, inner number of channels used for mask coefficients.

    Returns:
        Tensor representing the linear coefficients for regressing masks.
    """
    coefficients = []
    for branch_arg, branch in enumerate(branches):
        branch_coefficients = build_branch_mask_coefficients(
            branch, dimension, num_prototypes, branch_arg
        )
        coefficients.append(branch_coefficients)
    return keras.layers.Concatenate(axis=1, name="coefficients")(coefficients)


def combine_linearly_prototypes(coefficients, prototypes):
    """Linearly combines prototypes masks using the predicted coefficients.
    This applies equation 1 of YOLACT https://arxiv.org/abs/1904.02689.

    Args:
        coefficients: tensor representing the linear coefficients of the
            prototypes masks.
        prototypes: tensor representing a base of masks that can be
            linearly combined to produce predicted masks.

    Returns:
        Tensor representing all the predicted masks.
    """
    masks = ops.sigmoid(ops.einsum("bnm,bhwm->bnhw", coefficients, prototypes))
    return masks


def build_segmentation_head(branches, dimension, num_prototypes):
    """Builds a YOLACT https://arxiv.org/abs/1904.02689 segmentation head
    by predicting prototype masks, their linear coefficients, and combining
    them to build the predicted masks.

    Args:
        branches: list of tensors, representing the outputs of a backbone model.
        dimension: integer, inner number of channels used for mask prototypes.
        num_prototypes: integer, number of mask prototypes to build predictions.

    Returns:
        Tensor representing all the predicted masks.
    """
    prototypes = build_mask_prototypes(branches[0], dimension, num_prototypes)
    coefficients = build_mask_coefficients(branches, num_prototypes)
    masks = combine_linearly_prototypes(coefficients, prototypes)
    return masks


def split_masks(masks, num_classes):
    """Splits single channel segmentation mask into different class channels.

    Args:
        masks: tensor representing ground truth masks using a single
        channel consisting of integers representing the pixel class.
        num_classes: integer, total number of classes in the dataset.

    Returns:
        tensor representing each class mask in a different channel.
    """
    splitted_masks = []
    for class_arg in range(num_classes):
        splitted_masks.append(masks == class_arg)
    splitted_masks = ops.concatenate(splitted_masks, axis=-1)
    splitted_masks = ops.cast(splitted_masks, float)
    return splitted_masks


def repeat_masks(masks, class_labels, num_classes):
    """Repeats ground truth masks by gathering each ground truth mask
    channel using the assigned class label. This is used to build a
    tensor with the same shape as the predicted masks in order to
    compute the loss.

    Args:
        masks: tensor representing ground truth masks using a single
        channel consisting of integers representing the pixel class.
        class_labels: tensor, with the assigned class labels in each anchor box.
            The class labels are in a one-hot encoding vector form.
        num_classes: integer, total number of classes in the dataset.

    Returns:
        tensor representing each class mask in a different channel.
    """
    class_args = ops.argmax(class_labels, axis=-1)
    batch_shape = class_args.shape[0]
    class_args = ops.reshape(class_args, (batch_shape, 1, 1, -1))
    masks = split_masks(masks, num_classes)
    repeated_masks = ops.take_along_axis(masks, class_args, axis=-1)
    return repeated_masks


def unpack_input(data):
    classes = data["bounding_boxes"]["classes"]
    boxes = data["bounding_boxes"]["boxes"]
    segmentation_masks = data["segmentation_masks"]
    y = {
        "classes": classes,
        "boxes": boxes,
        "segmentation_masks": segmentation_masks,
    }
    return data["images"], y


@keras_cv_export(
    [
        "keras_cv.models.YOLOV8Segmentation",
        "keras_cv.models.segmentation.YOLOV8Segmentation",
    ]
)
class YOLOV8Segmentation(Task):
    """Implements the YOLOV8 architecture for instance segmentation."""

    def __init__(
        self,
        backbone,
        num_classes,
        bounding_box_format,
        fpn_depth=2,
        label_encoder=None,
        prediction_decoder=None,
        prototype_dimension=256,
        num_prototypes=32,
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

        branches = apply_path_aggregation_fpn(
            features, fpn_depth, name="pa_fpn"
        )

        masks = build_segmentation_head(
            branches, prototype_dimension, num_prototypes
        )

        detection_head = build_YOLOV8_detection_head(branches, num_classes)
        boxes, classes = detection_head["boxes"], detection_head["classes"]

        # TODO remove no-op layer to overwrite metric name for pretty printing.
        boxes = keras.layers.Concatenate(axis=1, name="box")([boxes])
        scores = keras.layers.Concatenate(axis=1, name="class")([classes])
        masks = keras.layers.Concatenate(axis=1, name="masks")([masks])

        outputs = {"boxes": boxes, "classes": scores, "masks": masks}
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
        segmentation_loss,
        box_loss_weight=7.5,
        classification_loss_weight=0.5,
        segmentation_loss_weight=6.125,
        metrics=None,
        **kwargs,
    ):
        """Compiles the YOLOV8Detector.

        `compile()` mirrors the standard Keras `compile()` method, but has one
        key distinction -- two losses must be provided: `box_loss` and
        `classification_loss`.

        Args:
            box_loss: a Keras loss to use for box offset regression. A
                preconfigured loss is given when the string "ciou" is passed.
            classification_loss: a Keras loss to use for box classification. A
                preconfigured loss is provided when the string
                "binary_crossentropy" is passed.
            segmentation_loss:a Keras loss for segmentation.
            box_loss_weight: (optional) float, a scaling factor for the box
                loss. Defaults to 7.5.
            classification_loss_weight: (optional) float, a scaling factor for
                the classification loss. Defaults to 0.5.
            segmentation_loss_weight: (optional) float, a scaling factor for
                the classification loss. Defaults to 6.125.
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

        if isinstance(segmentation_loss, str):
            if segmentation_loss == "binary_crossentropy":
                segmentation_loss = keras.losses.BinaryCrossentropy(
                    reduction="sum"
                )
            else:
                raise ValueError(
                    "Invalid segmentation loss for YOLOV8Detector: "
                    f"{classification_loss}. Classification loss should be a "
                    "keras.Loss or the string 'binary_crossentropy'."
                )

        self.box_loss = box_loss
        self.classification_loss = classification_loss
        self.segmentation_loss = segmentation_loss
        self.box_loss_weight = box_loss_weight
        self.classification_loss_weight = classification_loss_weight
        self.segmentation_loss_weight = segmentation_loss_weight

        losses = {
            "box": self.box_loss,
            "class": self.classification_loss,
            "masks": self.segmentation_loss,
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

        target_masks = y["segmentation_masks"]
        _, num_priors, H_mask, W_mask = y_pred["masks"].shape
        target_masks = ops.image.resize(target_masks, (H_mask, W_mask))
        target_masks = repeat_masks(
            target_masks, target_scores, self.num_classes
        )
        batch_size, H_mask, W_mask, num_anchors = target_masks.shape
        target_masks = ops.reshape(
            target_masks, (batch_size, num_anchors, H_mask, W_mask)
        )

        y_true = {
            "box": target_bboxes * fg_mask[..., None],
            "class": target_scores,
            "masks": target_masks * fg_mask[..., None, None],
        }
        y_pred = {
            "box": pred_bboxes * fg_mask[..., None],
            "class": pred_scores,
            "masks": y_pred["masks"] * fg_mask[..., None, None],
        }
        sample_weights = {
            "box": self.box_loss_weight * box_weight / target_scores_sum,
            "class": self.classification_loss_weight / target_scores_sum,
            "masks": self.segmentation_loss_weight / target_scores_sum,
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
