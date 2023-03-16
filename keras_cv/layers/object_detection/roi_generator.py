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

from typing import Mapping
from typing import Optional
from typing import Tuple
from typing import Union

import tensorflow as tf

from keras_cv import bounding_box


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class ROIGenerator(tf.keras.layers.Layer):
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
        pre_nms_topk_train: int. number of top k scoring proposals to keep before applying NMS in training mode.
            When RPN is run on multiple feature maps / levels (as in FPN) this number is per
            feature map / level.
        nms_score_threshold_train: float. score threshold to use for NMS in training mode.
        nms_iou_threshold_train: float. IOU threshold to use for NMS in training mode.
        post_nms_topk_train: int. number of top k scoring proposals to keep after applying NMS in training mode.
            When RPN is run on multiple feature maps / levels (as in FPN) this number is per
            feature map / level.
        pre_nms_topk_test: int. number of top k scoring proposals to keep before applying NMS in inference mode.
            When RPN is run on multiple feature maps / levels (as in FPN) this number is per
            feature map / level.
        nms_score_threshold_test: float. score threshold to use for NMS in inference mode.
        nms_iou_threshold_test: float. IOU threshold to use for NMS in inference mode.
        post_nms_topk_test: int. number of top k scoring proposals to keep after applying NMS in inference mode.
            When RPN is run on multiple feature maps / levels (as in FPN) this number is per
            feature map / level.

    Usage:
    ```python
    roi_generator = ROIGenerator("xyxy")
    boxes = {2: tf.random.normal([32, 5, 4])}
    scores = {2: tf.random.normal([32, 5])}
    rois, roi_scores = roi_generator(boxes, scores, training=True)
    ```

    """

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
        multi_level_boxes: Union[tf.Tensor, Mapping[int, tf.Tensor]],
        multi_level_scores: Union[tf.Tensor, Mapping[int, tf.Tensor]],
        training: Optional[bool] = None,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Args:
          multi_level_boxes: float Tensor. A dictionary or single Tensor of boxes, one per level. shape is
            [batch_size, num_boxes, 4] each level, in `bounding_box_format`.
            The boxes from RPNs are usually encoded as deltas w.r.t to anchors,
            they need to be decoded before passing in here.
          multi_level_scores: float Tensor. A dictionary or single Tensor of scores, usually confidence scores,
            one per level. shape is [batch_size, num_boxes] each level.

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
            scores_shape = scores.get_shape().as_list()
            # scores can also be [batch_size, num_boxes, 1]
            if len(scores_shape) == 3:
                scores = tf.squeeze(scores, axis=-1)
            _, num_boxes = scores.get_shape().as_list()
            level_pre_nms_topk = min(num_boxes, pre_nms_topk)
            level_post_nms_topk = min(num_boxes, post_nms_topk)
            scores, sorted_indices = tf.nn.top_k(
                scores, k=level_pre_nms_topk, sorted=True
            )
            boxes = tf.gather(boxes, sorted_indices, batch_dims=1)
            # convert from input format to yxyx for the TF NMS operation
            boxes = bounding_box.convert_format(
                boxes,
                source=self.bounding_box_format,
                target="yxyx",
            )
            # TODO(tanzhenyu): consider supporting soft / batched nms for accl
            selected_indices, num_valid = tf.image.non_max_suppression_padded(
                boxes,
                scores,
                max_output_size=level_post_nms_topk,
                iou_threshold=nms_iou_threshold,
                score_threshold=nms_score_threshold,
                pad_to_max_output_size=True,
                sorted_input=True,
                canonicalized_coordinates=True,
            )
            # convert back to input format
            boxes = bounding_box.convert_format(
                boxes,
                source="yxyx",
                target=self.bounding_box_format,
            )
            level_rois = tf.gather(boxes, selected_indices, batch_dims=1)
            level_roi_scores = tf.gather(scores, selected_indices, batch_dims=1)
            level_rois = level_rois * tf.cast(
                tf.reshape(tf.range(level_post_nms_topk), [1, -1, 1])
                < tf.reshape(num_valid, [-1, 1, 1]),
                level_rois.dtype,
            )
            level_roi_scores = level_roi_scores * tf.cast(
                tf.reshape(tf.range(level_post_nms_topk), [1, -1])
                < tf.reshape(num_valid, [-1, 1]),
                level_roi_scores.dtype,
            )
            return level_rois, level_roi_scores

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

        rois = tf.concat(rois, axis=1)
        roi_scores = tf.concat(roi_scores, axis=1)
        _, num_valid_rois = roi_scores.get_shape().as_list()
        overall_top_k = min(num_valid_rois, post_nms_topk)
        roi_scores, sorted_indices = tf.nn.top_k(
            roi_scores, k=overall_top_k, sorted=True
        )
        rois = tf.gather(rois, sorted_indices, batch_dims=1)

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
