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

from typing import Optional

from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.backend import keras
from keras_cv.src.backend import ops
from keras_cv.src.layers import NonMaxSuppression


@keras_cv_export("keras_cv.layers.ROIGenerator")
class ROIGenerator(keras.layers.Layer):
    """
    Generates region of interests (ROI, or proposal) from scores.

    Mainly used in Region CNN (RCNN) networks.

    This works for a multi-level input, both boxes and scores are dictionary
    inputs with the same set of keys.

    Users can configure top k and threshold differently in train and inference.

    Users can choose to combine all levels if NMS across all levels are desired.

    The following steps are applied to pair of (boxes, scores):
    1) pre_nms_topk scores and boxes sorted and selected per level
    2) nms applied and selected post_nms_topk scores and ROIs per level
    3) combined scores and ROIs across all levels
    4) post_nms_topk scores and ROIs sorted and selected

    Args:
        bounding_box_format: a case-insensitive string.
            For detailed information on the supported format, see the
            [KerasCV bounding box documentation](https://keras.io/api/keras_cv/bounding_box/formats/).
        pre_nms_topk_train: int. number of top k scoring proposals to keep
            before applying NMS in training mode. When RPN is run on multiple
            feature maps / levels (as in FPN) this number is per
            feature map / level.
        nms_score_threshold_train: float. score threshold to use for NMS in
            training mode.
        nms_iou_threshold_train: float. IOU threshold to use for NMS in training
            mode.
        post_nms_topk_train: int. number of top k scoring proposals to keep
            after applying NMS in training mode. When RPN is run on multiple
            feature maps / levels (as in FPN) this number is per
            feature map / level.
        pre_nms_topk_test: int. number of top k scoring proposals to keep before
            applying NMS in inference mode. When RPN is run on multiple
            feature maps / levels (as in FPN) this number is per
            feature map / level.
        nms_score_threshold_test: float. score threshold to use for NMS in
            inference mode.
        nms_iou_threshold_test: float. IOU threshold to use for NMS in inference
            mode.
        post_nms_topk_test: int. number of top k scoring proposals to keep after
            applying NMS in inference mode. When RPN is run on multiple
            feature maps / levels (as in FPN) this number is per
            feature map / level.

    Example:
    ```python
    roi_generator = ROIGenerator("xyxy")
    boxes = {2: tf.random.normal([32, 5, 4])}
    scores = {2: tf.random.normal([32, 5])}
    rois, roi_scores = roi_generator(boxes, scores, training=True)
    ```

    """  # noqa: E501

    def __init__(
        self,
        bounding_box_format,
        pre_nms_topk_train: int = 2000,
        nms_score_threshold_train: float = 0.0,
        nms_iou_threshold_train: float = 0.7,
        post_nms_topk_train: int = 1000,
        pre_nms_topk_test: int = 1000,
        nms_score_threshold_test: float = 0.0,
        nms_iou_threshold_test: float = 0.7,
        post_nms_topk_test: int = 1000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.bounding_box_format = bounding_box_format
        self.pre_nms_topk_train = pre_nms_topk_train
        self.nms_score_threshold_train = nms_score_threshold_train
        self.nms_iou_threshold_train = nms_iou_threshold_train
        self.post_nms_topk_train = post_nms_topk_train
        self.pre_nms_topk_test = pre_nms_topk_test
        self.nms_score_threshold_test = nms_score_threshold_test
        self.nms_iou_threshold_test = nms_iou_threshold_test
        self.post_nms_topk_test = post_nms_topk_test
        self.built = True

    def call(
        self,
        multi_level_boxes,
        multi_level_scores,
        training: Optional[bool] = None,
    ):
        """
        Args:
          multi_level_boxes: float Tensor. A dictionary or single Tensor of
            boxes, one per level. Shape is [batch_size, num_boxes, 4] each
            level, in `bounding_box_format`. The boxes from RPNs are usually
            encoded as deltas w.r.t to anchors, they need to be decoded before
            passing in here.
          multi_level_scores: float Tensor. A dictionary or single Tensor of
            scores, typically confidence scores, one per level. Shape is
            [batch_size, num_boxes] each level.

        Returns:
          rois: float Tensor of [batch_size, post_nms_topk, 4]
          roi_scores: float Tensor of [batch_size, post_nms_topk]
        """
        if training:
            pre_nms_topk = self.pre_nms_topk_train
            post_nms_topk = self.post_nms_topk_train
            nms_score_threshold = self.nms_score_threshold_train
            nms_iou_threshold = self.nms_iou_threshold_train
        else:
            pre_nms_topk = self.pre_nms_topk_test
            post_nms_topk = self.post_nms_topk_test
            nms_score_threshold = self.nms_score_threshold_test
            nms_iou_threshold = self.nms_iou_threshold_test

        def per_level_gen(boxes, scores):
            boxes = ops.convert_to_tensor(boxes, dtype="float32")
            scores = ops.convert_to_tensor(scores, dtype="float32")
            scores_shape = ops.shape(scores)
            # Check if scores is a 3-dimensional tensor
            # ([batch_size, num_boxes, 1])
            # If so, remove the last dimension to make it 2D
            if len(scores_shape) == 3:
                scores = ops.squeeze(scores, axis=-1)
                scores_shape = ops.shape(scores)
            _, num_boxes = scores_shape
            level_pre_nms_topk = min(num_boxes, pre_nms_topk)
            level_post_nms_topk = min(num_boxes, post_nms_topk)
            scores, sorted_indices = ops.top_k(
                scores, k=level_pre_nms_topk, sorted=True
            )
            boxes = ops.take_along_axis(
                boxes, sorted_indices[..., None], axis=1
            )
            # TODO(tanzhenyu): consider supporting soft / batched nms for accl
            boxes = NonMaxSuppression(
                bounding_box_format=self.bounding_box_format,
                from_logits=False,
                iou_threshold=nms_iou_threshold,
                confidence_threshold=nms_score_threshold,
                max_detections=level_post_nms_topk,
            )(
                box_prediction=boxes,
                class_prediction=scores[..., None],
            )
            return boxes["boxes"], boxes["confidence"]

        if not isinstance(multi_level_boxes, dict):
            return per_level_gen(multi_level_boxes, multi_level_scores)

        rois = []
        roi_scores = []
        for level in sorted(multi_level_scores.keys()):
            boxes = multi_level_boxes[level]
            scores = multi_level_scores[level]
            level_rois, level_roi_scores = per_level_gen(boxes, scores)
            rois.append(level_rois)
            roi_scores.append(level_roi_scores)

        rois = ops.concatenate(rois, axis=1)
        roi_scores = ops.concatenate(roi_scores, axis=1)
        _, num_valid_rois = ops.shape(roi_scores)
        overall_top_k = min(num_valid_rois, post_nms_topk)
        roi_scores, sorted_indices = ops.top_k(
            roi_scores, k=overall_top_k, sorted=True
        )
        rois = ops.take_along_axis(rois, sorted_indices[..., None], axis=1)

        return rois, roi_scores

    def get_config(self):
        config = {
            "bounding_box_format": self.bounding_box_format,
            "pre_nms_topk_train": self.pre_nms_topk_train,
            "nms_score_threshold_train": self.nms_score_threshold_train,
            "nms_iou_threshold_train": self.nms_iou_threshold_train,
            "post_nms_topk_train": self.post_nms_topk_train,
            "pre_nms_topk_test": self.pre_nms_topk_test,
            "nms_score_threshold_test": self.nms_score_threshold_test,
            "nms_iou_threshold_test": self.nms_iou_threshold_test,
            "post_nms_topk_test": self.post_nms_topk_test,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
