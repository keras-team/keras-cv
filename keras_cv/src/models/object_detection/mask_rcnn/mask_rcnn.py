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


import tensorflow as tf
import tree

from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.backend import keras
from keras_cv.src.backend import ops
from keras_cv.src.bounding_box import convert_format
from keras_cv.src.bounding_box.converters import _decode_deltas_to_boxes
from keras_cv.src.bounding_box.utils import _clip_boxes
from keras_cv.src.layers.object_detection.roi_sampler import ROISampler
from keras_cv.src.models.object_detection.faster_rcnn import MaskHead
from keras_cv.src.models.object_detection.faster_rcnn.faster_rcnn import (
    BOX_VARIANCE,
)
from keras_cv.src.models.object_detection.faster_rcnn.faster_rcnn import (
    _parse_box_loss,
)
from keras_cv.src.models.object_detection.faster_rcnn.faster_rcnn import (
    _parse_classification_loss,
)
from keras_cv.src.models.object_detection.faster_rcnn.faster_rcnn import (
    _parse_rpn_classification_loss,
)
from keras_cv.src.models.object_detection.faster_rcnn.faster_rcnn import (
    unpack_input,
)
from keras_cv.src.models.task import Task


@keras_cv_export(
    [
        "keras_cv.models.MaskRCNN",
        "keras_cv.models.object_detection.MaskRCNN",
    ]
)
class MaskRCNN(Task):
    """A Keras model implementing the Mask R-CNN architecture.

    Mask R-CNN is a straightforward extension of Faster R-CNN, providing an
    additional mask head that predicts segmentation masks. The constructor
    therefore requires a FasterRCNN model as backend.

    This model is compatible with Keras 3 only.

    Args:
         backbone: `keras.Model`. A FasterRCNN model that is used for
            object detection.
        mask_head: (Optional) A `keras.Layer` that performs regression of
            the segmentation masks.
            If not provided, a network with 2 convolutional layers, an
            upsampling layer and a class-specific layer will be used.
    """  # noqa: E501

    def __init__(self, backbone, mask_head=None, **kwargs):
        mask_head = mask_head or MaskHead(
            backbone.num_classes, name="mask_head"
        )

        images = backbone.input
        feature_map = backbone.get_layer("roi_align").output

        segmask_pred = mask_head(feature_map=feature_map)
        segmask_pred = keras.layers.Concatenate(axis=1, name="segmask")(
            [segmask_pred]
        )

        inputs = {"images": images}
        outputs = {"segmask": segmask_pred, **backbone.output}

        super().__init__(
            inputs=inputs,
            outputs=outputs,
            **kwargs,
        )

        self.backbone = backbone
        self.mask_head = mask_head
        self.roi_sampler = ROISampler(
            roi_bounding_box_format="yxyx",
            gt_bounding_box_format=backbone.bounding_box_format,
            roi_matcher=backbone.box_matcher,
            num_sampled_rois=64,
            mask_shape=(14, 14),
        )

    def compile(
        self,
        rpn_box_loss=None,
        rpn_classification_loss=None,
        box_loss=None,
        classification_loss=None,
        mask_loss=None,
        weight_decay=0.0001,
        loss=None,
        metrics=None,
        **kwargs,
    ):
        if loss is not None:
            raise ValueError(
                "`MaskRCNN` does not accept a `loss` to `compile()`. "
                "Instead, please pass `box_loss` and `classification_loss`. "
                "`loss` will be ignored during training."
            )
        rpn_box_loss = _parse_box_loss(rpn_box_loss)
        rpn_classification_loss = _parse_rpn_classification_loss(
            rpn_classification_loss
        )

        if hasattr(rpn_classification_loss, "from_logits"):
            if not rpn_classification_loss.from_logits:
                raise ValueError(
                    "FasterRCNN.compile() expects `from_logits` to be True for "
                    "`rpn_classification_loss`. Got "
                    "`rpn_classification_loss.from_logits="
                    f"{rpn_classification_loss.from_logits}`"
                )
        box_loss = _parse_box_loss(box_loss)
        classification_loss = _parse_classification_loss(classification_loss)
        mask_loss = _parse_mask_loss(mask_loss)

        if hasattr(classification_loss, "from_logits"):
            if not classification_loss.from_logits:
                raise ValueError(
                    "FasterRCNN.compile() expects `from_logits` to be True for "
                    "`classification_loss`. Got "
                    "`classification_loss.from_logits="
                    f"{classification_loss.from_logits}`"
                )
        if hasattr(box_loss, "bounding_box_format"):
            if (
                box_loss.bounding_box_format
                != self.backbone.bounding_box_format
            ):
                raise ValueError(
                    "Wrong `bounding_box_format` passed to `box_loss` in "
                    "`FasterRCNN.compile()`. Got "
                    "`box_loss.bounding_box_format="
                    f"{box_loss.bounding_box_format}`, want "
                    "`box_loss.bounding_box_format="
                    f"{self.backbone.bounding_box_format}`"
                )

        self.rpn_box_loss = rpn_box_loss
        self.rpn_cls_loss = rpn_classification_loss
        self.box_loss = box_loss
        self.cls_loss = classification_loss
        self.mask_loss = mask_loss
        self.weight_decay = weight_decay
        losses = {
            "rpn_box": self.rpn_box_loss,
            "rpn_classification": self.rpn_cls_loss,
            "box": self.box_loss,
            "classification": self.cls_loss,
            "segmask": self.mask_loss,
        }
        self._has_user_metrics = metrics is not None and len(metrics) != 0
        self._user_metrics = metrics
        super().compile(loss=losses, **kwargs)

    def compute_loss(
        self, x, y, y_pred, sample_weight, training=True, **kwargs
    ):

        # 1. Unpack the inputs
        images = x
        gt_boxes = y["boxes"]
        if ops.ndim(y["classes"]) != 2:
            raise ValueError(
                "Expected 'classes' to be a Tensor of rank 2. "
                f"Got y['classes'].shape={ops.shape(y['classes'])}."
            )

        gt_classes = y["classes"]
        gt_classes = ops.expand_dims(gt_classes, axis=-1)

        gt_masks = y["segmask"]

        #######################################################################
        # Generate  Anchors and Generate RPN Targets
        #######################################################################
        local_batch = ops.shape(images)[0]
        image_shape = ops.shape(images)[1:]
        anchors = self.backbone.anchor_generator(image_shape=image_shape)

        # 2. Label with the anchors -- exclusive to compute_loss
        (
            rpn_box_targets,
            rpn_box_weights,
            rpn_cls_targets,
            rpn_cls_weights,
        ) = self.backbone.label_encoder(
            anchors_dict=ops.concatenate(
                tree.flatten(anchors),
                axis=0,
            ),
            gt_boxes=gt_boxes,
            gt_classes=gt_classes,
        )

        # 3. Computing the weights
        rpn_box_weights /= (
            self.backbone.label_encoder.samples_per_image * local_batch * 0.25
        )
        rpn_cls_weights /= (
            self.backbone.label_encoder.samples_per_image * local_batch
        )

        #######################################################################
        # Call Backbone, FPN and RPN Head
        #######################################################################

        backbone_outputs = self.backbone.feature_extractor(images)
        feature_map = self.backbone.feature_pyramid(backbone_outputs)
        rpn_boxes, rpn_scores = self.backbone.rpn_head(feature_map)

        for lvl in rpn_boxes:
            rpn_boxes[lvl] = keras.layers.Reshape(target_shape=(-1, 4))(
                rpn_boxes[lvl]
            )

        for lvl in rpn_scores:
            rpn_scores[lvl] = keras.layers.Reshape(target_shape=(-1, 1))(
                rpn_scores[lvl]
            )

        # [BS, num_anchors, 4], [BS, num_anchors, 1]
        rpn_cls_pred = keras.layers.Concatenate(
            axis=1, name="rpn_classification"
        )(tree.flatten(rpn_scores))
        rpn_box_pred = keras.layers.Concatenate(axis=1, name="rpn_box")(
            tree.flatten(rpn_boxes)
        )

        #######################################################################
        # Generate RoI's and RoI Sampling
        #######################################################################

        decoded_rpn_boxes = _decode_deltas_to_boxes(
            anchors=anchors,
            boxes_delta=rpn_boxes,
            anchor_format="yxyx",
            box_format="yxyx",
            variance=BOX_VARIANCE,
        )

        rois, _ = self.backbone.roi_generator(
            decoded_rpn_boxes, rpn_scores, training=training
        )
        rois = _clip_boxes(rois, "yxyx", image_shape)

        # 4. Stop gradient from flowing into the ROI
        # -- exclusive to compute_loss
        rois = ops.stop_gradient(rois)

        # 5. Sample the ROIS -- exclusive to compute_loss
        (
            rois,
            box_targets,
            box_weights,
            cls_targets,
            cls_weights,
            segmask_targets,
            segmask_weights,
        ) = self.roi_sampler(rois, gt_boxes, gt_classes, gt_masks)

        cls_targets = ops.squeeze(cls_targets, axis=-1)
        cls_weights = ops.squeeze(cls_weights, axis=-1)

        # 6. Box and class weights -- exclusive to compute loss
        box_weights /= self.roi_sampler.num_sampled_rois * local_batch * 0.25
        cls_weights /= self.roi_sampler.num_sampled_rois * local_batch
        cls_targets_numeric = cls_targets
        cls_targets = ops.one_hot(
            cls_targets, num_classes=self.backbone.num_classes + 1
        )

        #######################################################################
        # Call RoI Aligner and RCNN Head
        #######################################################################

        feature_map = self.backbone.roi_aligner(
            features=feature_map, boxes=rois
        )

        segmask_pred = self.mask_head(feature_map=feature_map)
        # we only consider the mask prediction for the groundtruth class
        segmask_pred = ops.reshape(
            segmask_pred, (-1, *ops.shape(segmask_pred)[2:])
        )
        segmask_pred_ind = ops.reshape(
            cls_targets_numeric, (ops.shape(segmask_pred)[0], 1, 1, -1)
        )
        segmask_pred_ind = ops.cast(segmask_pred_ind, "int32")
        segmask_pred = ops.take_along_axis(
            segmask_pred, segmask_pred_ind, axis=-1
        )
        # flatten each ROI's segmask to perform averaging (instead of
        # summation) over all pixels during loss computation
        segmask_pred = ops.reshape(
            segmask_pred, (local_batch, self.roi_sampler.num_sampled_rois, -1)
        )
        segmask_targets = ops.reshape(
            segmask_targets, (*segmask_targets.shape[:2], -1)
        )

        # [BS, H*W*K]
        feature_map = ops.reshape(
            feature_map,
            newshape=ops.shape(rois)[:2] + (-1,),
        )

        # [BS, H*W*K, 4], [BS, H*W*K, num_classes + 1]
        box_pred, cls_pred = self.backbone.rcnn_head(feature_map=feature_map)

        y_true = {
            "rpn_box": rpn_box_targets,
            "rpn_classification": rpn_cls_targets,
            "box": box_targets,
            "classification": cls_targets,
            "segmask": segmask_targets,
        }
        y_pred = {
            "rpn_box": rpn_box_pred,
            "rpn_classification": rpn_cls_pred,
            "box": box_pred,
            "classification": cls_pred,
            "segmask": segmask_pred,
        }
        weights = {
            "rpn_box": rpn_box_weights,
            "rpn_classification": rpn_cls_weights,
            "box": box_weights,
            "classification": cls_weights,
            "segmask": segmask_weights,
        }

        return super().compute_loss(
            x=x, y=y_true, y_pred=y_pred, sample_weight=weights, **kwargs
        )

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

    def decode_predictions(self, predictions, images):
        y_pred = self.backbone.decode_predictions(predictions, images)
        image_shape = ops.shape(images)[1:]
        y_pred["segmask"] = self.decode_segmentation_masks(
            segmask_pred=y_pred["segmask"],
            class_pred=y_pred["classes"],
            decoded_boxes=y_pred["boxes"],
            bbox_foramt=self.bounding_box_format,
            image_shape=image_shape,
        )
        return y_pred

    def predict_step(self, *args):
        outputs = super().predict_step(*args)
        if type(outputs) is tuple:
            return self.decode_predictions(outputs[0], args[-1]), outputs[1]
        else:
            return self.decode_predictions(outputs, args[-1])

    def _resize_and_pad_mask(
        self, segmask_pred, class_pred, decoded_boxes, image_shape
    ):
        num_rois = ops.shape(segmask_pred)[0]
        image_height, image_width = image_shape[:2]

        # Reshape segmask_pred to (num_rois, mask_height, mask_width, 1) to
        # use with image resizing functions
        segmask_pred = ops.expand_dims(segmask_pred, 1)

        # Initialize a list to store the padded masks
        padded_masks_list = []

        # Iterate over the batch and place the resized masks into the correct
        # position
        for i in range(num_rois):
            if class_pred[i] == -1:
                continue
            y1, x1, y2, x2 = ops.maximum(ops.cast(decoded_boxes[i], "int32"), 0)
            y1, y2 = ops.minimum([y1, y2], image_height)
            x1, x2 = ops.minimum([x1, x2], image_width)
            box_height = y2 - y1
            box_width = x2 - x1

            # Resize the mask to the size of the bounding box
            resized_mask = tf.image.resize(
                segmask_pred[i], size=(box_height, box_width)
            )

            # Place the resized mask into the correct position in the final mask
            padded_mask = tf.image.pad_to_bounding_box(
                resized_mask, y1, x1, image_height, image_width
            )
            padded_mask = ops.squeeze(padded_mask, axis=-1)

            # Append the padded mask to the list
            padded_masks_list.append(padded_mask)

        # Stack the list of masks into a single tensor
        final_masks = ops.max(padded_masks_list, axis=0)

        return final_masks

    def decode_segmentation_masks(
        self, segmask_pred, class_pred, decoded_boxes, bbox_format, image_shape
    ):
        """Decode the predicted segmentation mask output, combining all
        masks in one mask for each image."""

        decoded_boxes = convert_format(
            decoded_boxes, source=bbox_format, target="yxyx"
        )
        # pick the mask prediction for the predicted class
        segmask_pred = ops.take_along_axis(
            segmask_pred, class_pred[:, :, None, None, None] + 1, axis=-1
        )

        final_masks = []
        for i in range(segmask_pred.shape[0]):
            # resize the mask according to the bounding box
            image_masks = self._resize_and_pad_mask(
                segmask_pred[i], class_pred[i], decoded_boxes[i], image_shape
            )
            final_masks.append(image_masks)

        return ops.stack(final_masks, axis=0)

    def compute_metrics(self, x, y, y_pred, sample_weight):
        metrics = {}
        metrics.update(super().compute_metrics(x, {}, {}, sample_weight={}))

        if not self._has_user_metrics:
            return metrics

        y_pred = self.decode_predictions(y_pred, x)

        for metric in self._user_metrics:
            metric.update_state(y, y_pred, sample_weight=sample_weight)

        for metric in self._user_metrics:
            result = metric.result()
            if isinstance(result, dict):
                metrics.update(result)
            else:
                metrics[metric.name] = result
        return metrics

    def get_config(self):
        return {
            "backbone": keras.saving.serialize_keras_object(self.backbone),
            "mask_head": self.mask_head,
        }

    @classmethod
    def from_config(cls, config):
        if "backbone" in config and isinstance(config["backbone"], dict):
            config["backbone"] = keras.layers.deserialize(config["backbone"])
        if "mask_head" in config and isinstance(config["mask_head"], dict):
            config["mask_head"] = keras.layers.deserialize(config["mask_head"])
        return super().from_config(config)


def _parse_mask_loss(loss):
    if not isinstance(loss, str):
        # support arbitrary callables
        return loss

    if loss.lower() == "binarycrossentropy":
        return keras.losses.BinaryCrossentropy(
            reduction="sum", from_logits=True
        )

    raise ValueError(
        f"Expected `mask_loss` to be either BinaryCrossentropy"
        f" loss callable, or the string 'BinaryCrossentropy'. Got loss={loss}."
    )