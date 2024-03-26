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
from keras_cv.bounding_box.converters import _decode_deltas_to_boxes
from keras_cv.bounding_box.utils import _clip_boxes
from keras_cv.layers.object_detection.anchor_generator import AnchorGenerator
from keras_cv.layers.object_detection.box_matcher import BoxMatcher
from keras_cv.layers.object_detection.roi_align import _ROIAligner
from keras_cv.layers.object_detection.roi_generator import ROIGenerator
from keras_cv.layers.object_detection.roi_sampler import _ROISampler
from keras_cv.layers.object_detection.rpn_label_encoder import _RpnLabelEncoder
from keras_cv.models.object_detection.__internal__ import unpack_input
from keras_cv.models.object_detection.faster_rcnn import FeaturePyramid
from keras_cv.models.object_detection.faster_rcnn import RCNNHead
from keras_cv.models.object_detection.faster_rcnn import RPNHead
from keras_cv.models.task import Task
from keras_cv.utils.train import get_feature_extractor

BOX_VARIANCE = [0.1, 0.1, 0.2, 0.2]


# TODO(tanzheny): add more configurations
@keras_cv_export("keras_cv.models.FasterRCNN")
class FasterRCNN(Task):
    def __init__(
        self,
        batch_size,
        backbone,
        num_classes,
        bounding_box_format,
        anchor_generator=None,
        feature_pyramid=None,
        rcnn_head=None,
        label_encoder=None,
        *args,
        **kwargs,
    ):

        # 1. Create the Input Layer
        extractor_levels = ["P2", "P3", "P4", "P5"]
        extractor_layer_names = [
            backbone.pyramid_level_inputs[i] for i in extractor_levels
        ]
        feature_extractor = get_feature_extractor(
            backbone, extractor_layer_names, extractor_levels
        )
        feature_pyramid = feature_pyramid or FeaturePyramid(
            name="feature_pyramid"
        )
        image_shape = feature_extractor.input_shape[
            1:
        ]  # exclude the batch size
        images = keras.layers.Input(
            image_shape,
            batch_size=batch_size,
            name="images",
        )

        # 2. Create the anchors
        scales = [2**x for x in [0]]
        aspect_ratios = [0.5, 1.0, 2.0]
        anchor_generator = anchor_generator or AnchorGenerator(
            bounding_box_format=bounding_box_format,
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
            name="anchor_generator",
        )
        # Note: `image_shape` should not be of NoneType
        # Need to assert before this line
        anchors = anchor_generator(image_shape=image_shape)

        #######################################################################
        # Call RPN
        #######################################################################

        # 3. Get the backbone outputs
        backbone_outputs = feature_extractor(images)
        feature_map = feature_pyramid(backbone_outputs)

        # 4. Get the Region Proposal Boxes and Scores
        num_anchors_per_location = len(scales) * len(aspect_ratios)
        rpn_head = RPNHead(
            num_anchors_per_location=num_anchors_per_location, name="rpn_head"
        )
        # [BS, num_anchors, 4], [BS, num_anchors, 1]
        rpn_boxes, rpn_scores = rpn_head(feature_map)

        # 5. Decode the deltas to boxes
        decoded_rpn_boxes = _decode_deltas_to_boxes(
            anchors=anchors,
            boxes_delta=rpn_boxes,
            anchor_format=bounding_box_format,
            box_format=bounding_box_format,
            variance=BOX_VARIANCE,
        )

        # 6. Generate the Region of Interests
        roi_generator = ROIGenerator(
            bounding_box_format=bounding_box_format,
            nms_score_threshold_train=float("-inf"),
            nms_score_threshold_test=float("-inf"),
            name="roi_generator",
        )
        rois, _ = roi_generator(decoded_rpn_boxes, rpn_scores)
        rois = _clip_boxes(rois, bounding_box_format, image_shape)
        rpn_box_pred = keras.ops.concatenate(tree.flatten(rpn_boxes), axis=1)
        rpn_cls_pred = keras.ops.concatenate(tree.flatten(rpn_scores), axis=1)

        #######################################################################
        # Call RCNN
        #######################################################################

        # 7. Pool the region of interests
        roi_pooler = _ROIAligner(bounding_box_format="yxyx", name="roi_pooler")
        feature_map = roi_pooler(features=feature_map, boxes=rois)

        # 8. Reshape the feature map [BS, H*W*K]
        feature_map = keras.ops.reshape(
            feature_map,
            newshape=keras.ops.shape(rois)[:2] + (-1,),
        )

        # 9. Pass the feature map to RCNN head
        # [BS, H*W*K, 4], [BS, H*W*K, num_classes + 1]
        rcnn_head = rcnn_head or RCNNHead(
            num_classes=num_classes, name="rcnn_head"
        )
        box_pred, cls_pred = rcnn_head(feature_map=feature_map)

        # 10. Create the model using Functional API
        inputs = {"images": images}
        box_pred = keras.layers.Concatenate(axis=1, name="box")([box_pred])
        cls_pred = keras.layers.Concatenate(axis=1, name="classification")(
            [cls_pred]
        )
        rpn_box_pred = keras.layers.Concatenate(axis=1, name="rpn_box")(
            [rpn_box_pred]
        )
        rpn_cls_pred = keras.layers.Concatenate(
            axis=1, name="rpn_classification"
        )([rpn_cls_pred])
        outputs = {
            "box": box_pred,
            "classification": cls_pred,
            "rpn_box": rpn_box_pred,
            "rpn_classification": rpn_cls_pred,
        }

        super().__init__(inputs=inputs, outputs=outputs, *args, **kwargs)

        # Define the model parameters
        self.bounding_box_format = bounding_box_format
        self.anchor_generator = anchor_generator
        self.rpn_labeler = label_encoder or _RpnLabelEncoder(
            anchor_format="yxyx",
            ground_truth_box_format="yxyx",
            positive_threshold=0.7,
            negative_threshold=0.3,
            samples_per_image=256,
            positive_fraction=0.5,
            box_variance=BOX_VARIANCE,
        )
        self.feature_extractor = feature_extractor
        self.feature_pyramid = feature_pyramid
        self.roi_generator = roi_generator
        self.rpn_head = rpn_head
        self.box_matcher = BoxMatcher(
            thresholds=[0.0, 0.5], match_values=[-2, -1, 1]
        )
        self.roi_sampler = _ROISampler(
            bounding_box_format="yxyx",
            roi_matcher=self.box_matcher,
            background_class=num_classes,
            num_sampled_rois=512,
        )
        self.roi_pooler = roi_pooler
        self.rcnn_head = rcnn_head

    def compile(
        self,
        box_loss=None,
        classification_loss=None,
        rpn_box_loss=None,
        rpn_classification_loss=None,
        weight_decay=0.0001,
        loss=None,
        metrics=None,
        **kwargs,
    ):
        if loss is not None:
            raise ValueError(
                "`FasterRCNN` does not accept a `loss` to `compile()`. "
                "Instead, please pass `box_loss` and `classification_loss`. "
                "`loss` will be ignored during training."
            )
        box_loss = _parse_box_loss(box_loss)
        classification_loss = _parse_classification_loss(classification_loss)

        rpn_box_loss = _parse_box_loss(rpn_box_loss)
        rpn_classification_loss = _parse_classification_loss(
            rpn_classification_loss
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
        self._has_user_metrics = metrics is not None and len(metrics) != 0
        self._user_metrics = metrics
        super().compile(loss=losses, **kwargs)

    def compute_loss(self, x, y, y_pred, sample_weight, **kwargs):
        # 1. Unpack the inputs
        images = x
        gt_boxes = y["boxes"]
        if keras.ops.ndim(y["classes"]) != 2:
            raise ValueError(
                "Expected 'classes' to be a Tensor of rank 2. "
                f"Got y['classes'].shape={keras.ops.shape(y['classes'])}."
            )
        # TODO(tanzhenyu): remove this hack and perform broadcasting elsewhere
        # gt_classes = keras.ops.expand_dims(y["classes"], axis=-1)
        gt_classes = y["classes"]

        # Generate anchors
        # image shape must not contain the batch size
        local_batch = keras.ops.shape(images)[0]
        image_shape = keras.ops.shape(images)[1:]
        anchors = self.anchor_generator(image_shape=image_shape)

        # 2. Label with the anchors -- exclusive to compute_loss
        (
            rpn_box_targets,
            rpn_box_weights,
            rpn_cls_targets,
            rpn_cls_weights,
        ) = self.rpn_labeler(
            anchors_dict=keras.ops.concatenate(
                tree.flatten(anchors),
                axis=0,
            ),
            gt_boxes=gt_boxes,
            gt_classes=gt_classes,
        )

        # 3. Computing the weights
        rpn_box_weights /= (
            self.rpn_labeler.samples_per_image * local_batch * 0.25
        )
        rpn_cls_weights /= self.rpn_labeler.samples_per_image * local_batch

        #######################################################################
        # Call RPN
        #######################################################################

        backbone_outputs = self.feature_extractor(images)
        feature_map = self.feature_pyramid(backbone_outputs)

        # [BS, num_anchors, 4], [BS, num_anchors, 1]
        rpn_boxes, rpn_scores = self.rpn_head(feature_map)

        decoded_rpn_boxes = _decode_deltas_to_boxes(
            anchors=anchors,
            boxes_delta=rpn_boxes,
            anchor_format=self.bounding_box_format,
            box_format=self.bounding_box_format,
            variance=BOX_VARIANCE,
        )

        rois, _ = self.roi_generator(decoded_rpn_boxes, rpn_scores)
        rois = _clip_boxes(rois, self.bounding_box_format, image_shape)
        rpn_box_pred = keras.ops.concatenate(tree.flatten(rpn_boxes), axis=1)
        rpn_cls_pred = keras.ops.concatenate(tree.flatten(rpn_scores), axis=1)

        # 4. Stop gradient from flowing into the ROI -- exclusive to compute_loss
        rois = keras.ops.stop_gradient(rois)

        # 5. Sample the ROIS -- exclusive to compute_loss -- exclusive to compute loss
        (
            rois,
            box_targets,
            box_weights,
            cls_targets,
            cls_weights,
        ) = self.roi_sampler(rois, gt_boxes, gt_classes)

        # 6. Box and class weights -- exclusive to compute loss
        box_weights /= self.roi_sampler.num_sampled_rois * local_batch * 0.25
        cls_weights /= self.roi_sampler.num_sampled_rois * local_batch

        #######################################################################
        # Call RCNN
        #######################################################################

        feature_map = self.roi_pooler(features=feature_map, boxes=rois)

        # [BS, H*W*K]
        feature_map = keras.ops.reshape(
            feature_map,
            newshape=keras.ops.shape(rois)[:2] + (-1,),
        )

        # [BS, H*W*K, 4], [BS, H*W*K, num_classes + 1]
        box_pred, cls_pred = self.rcnn_head(feature_map=feature_map)

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


def _parse_box_loss(loss):
    if not isinstance(loss, str):
        # support arbitrary callables
        return loss

    # case insensitive comparison
    if loss.lower() == "smoothl1":
        return keras.losses.SmoothL1Loss(l1_cutoff=1.0, reduction="sum")
    if loss.lower() == "huber":
        return keras.losses.Huber(reduction="sum")

    raise ValueError(
        "Expected `box_loss` to be either a Keras Loss, "
        f"callable, or the string 'SmoothL1', 'Huber'. Got loss={loss}."
    )


def _parse_classification_loss(loss):
    if not isinstance(loss, str):
        # support arbitrary callables
        return loss

    # case insensitive comparison
    if loss.lower() == "focal":
        return keras.losses.FocalLoss(from_logits=True, reduction="sum")

    raise ValueError(
        "Expected `classification_loss` to be either a Keras Loss, "
        f"callable, or the string 'Focal'. Got loss={loss}."
    )
