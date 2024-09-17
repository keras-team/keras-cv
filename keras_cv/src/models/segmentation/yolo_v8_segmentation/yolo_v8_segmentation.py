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

from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Conv2D
from keras.layers import Input
from keras.layers import Reshape
from keras.layers import UpSampling2D
from keras.losses import BinaryCrossentropy

from keras_cv.src import bounding_box
from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.backend import keras
from keras_cv.src.backend import ops
from keras_cv.src.layers import NonMaxSuppression
from keras_cv.src.losses.ciou_loss import CIoULoss
from keras_cv.src.models.backbones.backbone_presets import backbone_presets
from keras_cv.src.models.backbones.backbone_presets import (
    backbone_presets_with_weights,
)
from keras_cv.src.models.object_detection.yolo_v8.yolo_v8_detector_presets import (  # noqa: E501
    yolo_v8_detector_presets,
)
from keras_cv.src.models.object_detection.yolo_v8.yolo_v8_layers import (
    apply_conv_bn,
)
from keras_cv.src.models.segmentation.yolo_v8_segmentation.yolo_v8_backbone import (  # noqa: E501
    apply_path_aggregation_fpn,
)
from keras_cv.src.models.segmentation.yolo_v8_segmentation.yolo_v8_backbone import (  # noqa: E501
    apply_yolo_v8_head,
)
from keras_cv.src.models.segmentation.yolo_v8_segmentation.yolo_v8_backbone import (  # noqa: E501
    decode_regression_to_boxes,
)
from keras_cv.src.models.segmentation.yolo_v8_segmentation.yolo_v8_backbone import (  # noqa: E501
    dist2bbox,
)
from keras_cv.src.models.segmentation.yolo_v8_segmentation.yolo_v8_backbone import (  # noqa: E501
    get_anchors,
)
from keras_cv.src.models.segmentation.yolo_v8_segmentation.yolo_v8_label_encoder import (  # noqa: E501
    YOLOV8LabelEncoder,
)
from keras_cv.src.models.task import Task
from keras_cv.src.utils.python_utils import classproperty
from keras_cv.src.utils.train import get_feature_extractor


def build_mask_prototypes(x, dimension, num_prototypes, name="prototypes"):
    """Builds mask prototype network.

    The outputs of this module are linearly combined with the regressed mask
    coefficients to produce the predicted masks. This is an implementation of
    the module proposed in YOLACT https://arxiv.org/abs/1904.02689.

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
    x = UpSampling2D((2, 2), "channels_last", "bilinear", name=f"{name}_3")(x)
    x = apply_conv_bn(x, dimension, 3, name=f"{name}_4")
    x = Conv2D(num_prototypes, 1, padding="same", name=f"{name}_5")(x)
    x = Activation("relu", name=name)(x)
    return x


def build_branch_mask_coefficients(x, dimension, num_prototypes, branch_arg):
    """Builds mask coefficients of a single branch.

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
    x = Conv2D(num_prototypes, 1, name=f"{name}_2")(x)
    x = Activation("tanh", name=f"{name}_3")(x)
    x = Reshape((-1, num_prototypes), name=f"{name}_4")(x)
    return x


def build_mask_coefficients(branches, num_prototypes, dimension):
    """Builds all mask coefficients.

    This coefficients represent the linear terms used to combine the masks.

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
    return Concatenate(axis=1, name="coefficients")(coefficients)


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


def build_segmentation_head(
    branches, prototype_dimension, num_prototypes, coefficient_dimension
):
    """Builds a YOLACT https://arxiv.org/abs/1904.02689 segmentation head.

    The proposed segmentation head of YOLACT https://arxiv.org/abs/1904.02689
    predicts prototype masks, their linear coefficients, and combines them to
    build the predicted masks.

    Args:
        branches: list of tensors, representing the outputs of a backbone model.
        prototype_dimension: integer, inner number of channels used for mask
        prototypes.
        num_prototypes: integer, number of mask prototypes to build predictions.
        coefficient_dimension: integer, inner number of channels used for
        predicting the mask coefficients.

    Returns:
        Tensor representing all the predicted masks.
    """
    prototypes = build_mask_prototypes(
        branches[0], prototype_dimension, num_prototypes
    )
    coefficients = build_mask_coefficients(
        branches, num_prototypes, coefficient_dimension
    )
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
    for class_arg in range(1, num_classes + 1):
        splitted_masks.append(masks == class_arg)
    splitted_masks = ops.concatenate(splitted_masks, axis=-1)
    splitted_masks = ops.cast(splitted_masks, float)
    return splitted_masks


def repeat_masks(masks, class_labels, num_classes):
    """Repeats ground truth masks.

    Each ground truth mask channel is gathered using the assigned class label.
    This is used to build a tensor with the same shape as the predicted masks
    in order to compute the mask loss.

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
    splitted_masks = split_masks(masks, num_classes)
    repeated_masks = ops.take_along_axis(splitted_masks, class_args, axis=-1)
    return repeated_masks


def build_target_masks(true_masks, true_scores, H_mask, W_mask, num_classes):
    """Build target masks by resizing and repeating ground truth masks.

    Resizes ground truth masks to the predicted tensor mask shape, and repeats
    masks using the largest true score value.

    Args:
        true_masks: tensor representing the ground truth masks.
        true_scores: tensor with the class scores assigned by the label encoder.
        num_classes: integer indicating the total number of classes.

    Returns:
        Tensor with resized and repeated target masks.
    """
    true_masks = ops.image.resize(true_masks, (H_mask, W_mask), "nearest")
    true_masks = repeat_masks(true_masks, true_scores, num_classes)
    true_masks = ops.moveaxis(true_masks, 3, 1)
    return true_masks


def compute_box_areas(boxes):
    """Computes area for bounding boxes

    Args:
        boxes: (N, 4) or (batch_size, N, 4) float tensor, either batched
        or unbatched boxes.

    Returns:
        a float Tensor of [N] or [batch_size, N]
    """
    y_min, x_min, y_max, x_max = ops.split(boxes[..., :4], 4, axis=-1)
    box_areas = ops.squeeze((y_max - y_min) * (x_max - x_min), axis=-1)
    return box_areas


def normalize_box_areas(box_areas, H, W):
    """Normalizes box areas by dividing by the total image area.

    Args:
        boxes: tensor of shape (B, N, 4) with bounding boxes in xyxy format.
        H: integer indicating the mask height.
        W: integer indicating the mask width.

    Returns:
        Tensor of shape (B, N, 4).
    """
    return box_areas / (H * W)


def get_backbone_pyramid_layer_names(backbone, level_names):
    """Gets actual layer names from the provided pyramid levels inside backbone.

    Args:
        backbone: Keras backbone model with the field "pyramid_level_inputs".
        level_names: list of strings indicating the level names.

    Returns:
        List of layer strings indicating the layer names of each level.
    """
    layer_names = []
    for level_name in level_names:
        layer_names.append(backbone.pyramid_level_inputs[level_name])
    return layer_names


def build_feature_extractor(backbone, level_names):
    """Builds feature extractor directly from the level names

    Args:
        backbone: Keras backbone model with the field "pyramid_level_inputs".
        level_names: list of strings indicating the level names.

    Returns:
        Keras Model with level names as outputs.
    """
    layer_names = get_backbone_pyramid_layer_names(backbone, level_names)
    extractor = get_feature_extractor(backbone, layer_names, level_names)
    return extractor


def extend_branches(inputs, extractor, FPN_depth):
    """Extends extractor model with a feature pyramid network.

    Args:
        inputs: tensor, with image input.
        extractor: Keras Model with level names as outputs.
        FPN_depth: integer representing the feature pyramid depth.

    Returns:
        List of extended branch tensors.
    """
    features = list(extractor(inputs).values())
    branches = apply_path_aggregation_fpn(features, FPN_depth, name="pa_fpn")
    return branches


def extend_backbone(backbone, level_names, trainable, FPN_depth):
    """Extends backbone levels with a feature pyramid network.

    Args:
        backbone: Keras backbone model with the field "pyramid_level_inputs".
        level_names: list of strings indicating the level names.
        trainable: boolean indicating if backbone should be optimized.
        FPN_depth: integer representing the feature pyramid depth.

    Return:
        Tuple with input image tensor, and list of extended branch tensors.
    """
    feature_extractor = build_feature_extractor(backbone, level_names)
    feature_extractor.trainable = trainable
    inputs = Input(feature_extractor.input_shape[1:])
    branches = extend_branches(inputs, feature_extractor, FPN_depth)
    return inputs, branches


def add_no_op_for_pretty_print(x, name):
    """Wrap tensor with dummy operation to change tensor name.

    # Args:
        x: tensor.
        name: string name given to the tensor.

    Return:
        Tensor with new wrapped name.
    """
    return Concatenate(axis=1, name=name)([x])


def unpack_input(data):
    """Unpacks standard keras-cv data dictionary into inputs and outputs.

    Args:
        data: Dictionary with the standard key-value pairs of keras-cv

    Returns:
       Tuple containing inputs and outputs.
    """
    classes = data["bounding_boxes"]["classes"]
    boxes = data["bounding_boxes"]["boxes"]
    segmentation_masks = data["segmentation_masks"]
    y = {
        "classes": classes,
        "boxes": boxes,
        "segmentation_masks": segmentation_masks,
    }
    return data["images"], y


def boxes_to_masks(boxes, H_mask, W_mask):
    """Build mask with True values inside the bounding box and False elsewhere.

    Args:
        boxes: tensor of shape (N, 4) with bounding boxes in xyxy format.
        H_mask: integer indicating the height of the mask.
        W_mask: integer indicating the width of the mask.

    Returns:
        A mask of the specified shape with True values inside bounding box.
    """
    x_min, y_min, x_max, y_max = ops.split(boxes, 4, 1)

    y_range = ops.arange(H_mask)
    x_range = ops.arange(W_mask)
    y_indices, x_indices = ops.meshgrid(y_range, x_range, indexing="ij")

    y_indices = ops.expand_dims(y_indices, 0)
    x_indices = ops.expand_dims(x_indices, 0)

    x_min = ops.expand_dims(x_min, axis=1)
    y_min = ops.expand_dims(y_min, axis=1)
    x_max = ops.expand_dims(x_max, axis=1)
    y_max = ops.expand_dims(y_max, axis=1)

    in_x_min_to_x_max = ops.logical_and(x_indices >= x_min, x_indices < x_max)
    in_y_min_to_y_max = ops.logical_and(y_indices >= y_min, y_indices < y_max)
    masks = ops.logical_and(in_x_min_to_x_max, in_y_min_to_y_max)
    return masks


def batch_boxes_to_masks(boxes, H_mask, W_mask):
    """Converts boxes to masks over the batch dimension.

    Args:
        boxes: tensor of shape (B, N, 4) with bounding boxes in xyxy format.
        H_mask: integer indicating the height of the mask.
        W_mask: integer indicating the width of the mask.

    Returns:
        Batch of masks with True values inside the bounding box.
    """
    batch_size = boxes.shape[0]
    crop_masks = []
    for batch_arg in range(batch_size):
        boxes_sample = ops.cast(boxes[batch_arg], "int32")
        crop_mask = boxes_to_masks(boxes_sample, H_mask, W_mask)
        crop_masks.append(crop_mask[None])
    crop_masks = ops.concatenate(crop_masks)
    crop_masks = ops.cast(crop_masks, "float32")
    return crop_masks


def build_mask_weights(weight, boxes, H_mask, W_mask):
    """Build mask sample weights used to scale the loss at every batch.

    To balance the loss of masks with different shapes, YOLACT assigns a weight
    to each mask that is inversely proportional to its area.

    Args:
        weight: float, weight multiplied to the mask loss.
        boxes: tensor of shape (B, N, 4) with bounding boxes in xyxy format.
        H_image: integer indicating the inputted image height.
        W_image: integer indicating the inputted image width.
        H_mask: integer indicating the predicted mask height.
        W_mask: integer indicating the predicted mask width.

    Returns:
        Tensor of shape [B, num_anchors, 1, 1] containing the mask weights.
    """
    box_areas = compute_box_areas(boxes)
    box_areas = normalize_box_areas(box_areas, H_mask, W_mask)
    weights = ops.divide_no_nan(weight, box_areas)
    weights = weights / (H_mask * W_mask)
    return weights[..., None, None]


@keras_cv_export(
    [
        "keras_cv.models.YOLOV8Segmentation",
        "keras_cv.models.segmentation.YOLOV8Segmentation",
    ]
)
class YOLOV8Segmentation(Task):
    """Implements the YOLOV8 instance segmentation model.

    Args:
        backbone: `keras.Model`, must implement the `pyramid_level_inputs`
            property with keys "P3", "P4", and "P5" and layer names as values.
            A sensible backbone to use is the `keras_cv.models.YOLOV8Backbone`.
        num_classes: integer, the number of classes in your dataset excluding
            the background class. Classes should be represented by integers in
            the range [0, num_classes).
        bounding_box_format: string, the format of bounding boxes of input
            dataset.
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
        prototype_dimension: integer, inner number of channels used for mask
        prototypes. Defaults to 256.
        num_prototypes: integer, number of mask prototypes to build predictions.
            Defaults to 32.
        coefficient_dimension: integer, inner number of channels used for
            predicting the mask coefficients. Defaults to 32
        trainable_backbone: boolean indicating if the provided backbone should
            be trained as well. Defaults to False.

    Example:
    ```python
    images = tf.ones(shape=(1, 512, 512, 3))

    model = keras_cv.models.YOLOV8Segmentation(
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
    ```
    """

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
        coefficient_dimension=32,
        trainable_backbone=False,
        **kwargs,
    ):
        level_names = ["P3", "P4", "P5"]
        images, branches = extend_backbone(
            backbone, level_names, trainable_backbone, fpn_depth
        )
        masks = build_segmentation_head(
            branches, prototype_dimension, num_prototypes, coefficient_dimension
        )
        detection_head = apply_yolo_v8_head(branches, num_classes)
        boxes, classes = detection_head["boxes"], detection_head["classes"]
        boxes = add_no_op_for_pretty_print(boxes, "box")
        masks = add_no_op_for_pretty_print(masks, "masks")
        classes = add_no_op_for_pretty_print(classes, "class")
        outputs = {"boxes": boxes, "classes": classes, "masks": masks}
        super().__init__(inputs=images, outputs=outputs, **kwargs)

        self.bounding_box_format = bounding_box_format
        self._prediction_decoder = prediction_decoder or NonMaxSuppression(
            bounding_box_format=bounding_box_format,
            from_logits=False,
            confidence_threshold=0.2,
            iou_threshold=0.7,
        )
        self.backbone = backbone
        self.fpn_depth = fpn_depth
        self.num_classes = num_classes
        self.label_encoder = label_encoder or YOLOV8LabelEncoder(
            num_classes=num_classes
        )
        self.prototype_dimension = prototype_dimension
        self.num_prototypes = num_prototypes
        self.coefficient_dimension = coefficient_dimension
        self.trainable_backbone = trainable_backbone

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
        """Compiles the YOLOV8Segmentation.

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
                classification_loss = BinaryCrossentropy(reduction="sum")
            else:
                raise ValueError(
                    "Invalid classification loss for YOLOV8Detector: "
                    f"{classification_loss}. Classification loss should be a "
                    "keras.Loss or the string 'binary_crossentropy'."
                )

        if isinstance(segmentation_loss, str):
            if segmentation_loss == "binary_crossentropy":
                segmentation_loss = BinaryCrossentropy(reduction="sum")
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

        true_masks = y["segmentation_masks"]
        pred_masks = y_pred["masks"]
        batch_size, _, H_mask, W_mask = pred_masks.shape
        true_masks = build_target_masks(
            true_masks, target_scores, H_mask, W_mask, self.num_classes
        )

        crop_masks = batch_boxes_to_masks(target_bboxes, H_mask, W_mask)
        H_image, W_image = x.shape[1:3]
        mask_weights = build_mask_weights(
            self.segmentation_loss_weight, target_bboxes, H_mask, W_mask
        )

        y_true = {
            "box": target_bboxes * fg_mask[..., None],
            "class": target_scores,
            "masks": true_masks * crop_masks * fg_mask[..., None, None],
        }
        y_pred = {
            "box": pred_bboxes * fg_mask[..., None],
            "class": pred_scores,
            "masks": pred_masks * crop_masks * fg_mask[..., None, None],
        }
        sample_weights = {
            "box": self.box_loss_weight * box_weight / target_scores_sum,
            "class": self.classification_loss_weight / target_scores_sum,
            "masks": mask_weights,
        }

        return super().compute_loss(
            x=x, y=y_true, y_pred=y_pred, sample_weight=sample_weights, **kwargs
        )

    def decode_predictions(self, pred, images):
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
        decoded_outputs = self.decode_predictions(outputs, args[-1])
        selected_args = decoded_outputs["idx"][..., None, None]
        masks = outputs["masks"]
        masks = ops.take_along_axis(masks, selected_args, axis=1)
        is_valid_output = decoded_outputs["confidence"] > -1
        masks = ops.where(is_valid_output[..., None, None], masks, -1)
        decoded_outputs["masks"] = masks
        return decoded_outputs

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
            "backbone": keras.saving.serialize_keras_object(self.backbone),
            "num_classes": self.num_classes,
            "bounding_box_format": self.bounding_box_format,
            "fpn_depth": self.fpn_depth,
            "label_encoder": keras.saving.serialize_keras_object(
                self.label_encoder
            ),
            "prediction_decoder": keras.saving.serialize_keras_object(
                self._prediction_decoder
            ),
            "prototype_dimension": self.prototype_dimension,
            "num_prototypes": self.num_prototypes,
            "coefficient_dimension": self.coefficient_dimension,
            "trainable_backbone": self.trainable_backbone,
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
