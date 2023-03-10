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
import warnings

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.initializers as initializers

from keras_cv import bounding_box
from keras_cv.bounding_box import iou as iou_lib
from keras_cv.metrics.coco import utils


class BoxRecall(keras.metrics.Metric):
    """BoxRecall computes recall based on varying true positive IoU thresholds.

    BoxRecall is analagous to traditional Recall.  The primary distinction is
    that when operating in the problem domain of object detection there exists
    ambiguity in what is considered a true positive.  The BoxRecall metric
    works by using the Intersection over Union (IoU) metric to determine whether
    or not a detection is a true positive or a false positive.  For each
    detection the IoU metric is computed for all ground truth boxes of the same
    category.  If the IoU is above the selected threshold `t`, then the box is
    considered a true positive.  If not, it is marked as a false positive. An
    average is taken across many `t`, or IoU thresholds.  These thresholds are
    specified in the `iou_thresholds` argument.

    Args:
        class_ids: The class IDs to evaluate the metric for.  To evaluate for
            all classes in over a set of sequentially labelled classes, pass
            `range(num_classes)`.
        bounding_box_format: Format of the incoming bounding boxes.  Supported values
            are "xywh", "center_xywh", "xyxy".
        iou_thresholds: IoU thresholds over which to evaluate the recall.  Must
            be a tuple of floats, defaults to [0.5:0.05:0.95].
        area_range: area range to constrict the considered bounding boxes in
            metric computation. Defaults to `None`, which makes the metric
            count all bounding boxes.  Must be a tuple of floats.  The first
            number in the tuple represents a lower bound for areas, while the
            second value represents an upper bound.  For example, when
            `(0, 32**2)` is passed to the metric, recall is only evaluated for
            objects with areas less than `32*32`.  If `(32**2, 1000000**2)` is
            passed the metric will only be evaluated for boxes with areas larger
            than `32**2`, and smaller than `1000000**2`.
        max_detections: number of maximum detections a model is allowed to make.
            Must be an integer, defaults to `100`.

    Usage:

    BoxRecall accepts two dictionaries that comply with KerasCV's bounding box
    specification as inputs to it's `update_state` method.
    These dictionaries represent bounding boxes in the specified
    `bounding_box_format`.

    ```python
    coco_recall = keras_cv.metrics.BoxRecall(
        bounding_box_format='xyxy',
        max_detections=100,
        class_ids=[1]
    )
    od_model.compile(metrics=[coco_recall])
    od_model.fit(my_dataset)
    ```
    """

    def __init__(
        self,
        class_ids,
        bounding_box_format,
        iou_thresholds=None,
        area_range=None,
        max_detections=100,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Initialize parameter values
        self.bounding_box_format = bounding_box_format
        iou_thresholds = iou_thresholds or [
            x / 100.0 for x in range(50, 100, 5)
        ]

        self.iou_thresholds = np.array(iou_thresholds)
        self.class_ids = list(class_ids)
        self.area_range = area_range
        self.max_detections = max_detections

        # Initialize result counters
        num_thresholds = len(iou_thresholds)
        num_categories = len(class_ids)
        self.num_thresholds = num_thresholds
        if any([c < 0 for c in class_ids]):
            raise ValueError(
                "class_ids must be positive.  Got " f"class_ids={class_ids}"
            )

        self.true_positives = np.zeros(shape=(num_thresholds, num_categories), dtype='int32')
        self.ground_truth_boxes = np.zeros(shape=(num_categories), dtype='int32')

    def reset_state(self):
        self.true_positives = np.zeros(shape=(num_thresholds, num_categories), dtype='int32')
        self.ground_truth_boxes = np.zeros(shape=(num_categories), dtype='int32')

    def update_state(self, y_true, y_pred, sample_weight=None):
        for imgId in range(y_true)['boxes'].shape[0]:
            for catId, c_i in enumerate(self.class_ids):
                true_boxes = _gather_by_image_and_category(y_true, imgId, catId)
                pred_boxes = _gather_by_image_and_category(y_pred, imgId, catId)
                result = self.true_positives_for_image(catId, imgId)
                self.true_positives[:, c_i] += result

            self.ground_truth_boxes += y_true['boxes'].shape[0]

    def true_positives_for_image(self, y_true, y_pred):
        ious = iou_lib.compute_iou(y_true, y_pred, bounding_box_format=self.bounding_box_format)
        recalls_for_image = np.zeros((self.num_thresholds))

    def result(self):
        pass

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "class_ids": self.class_ids,
                "bounding_box_format": self.bounding_box_format,
                "iou_thresholds": self.iou_thresholds,
                "area_range": self.area_range,
                "max_detections": self.max_detections,
            }
        )
        return config



def _gather_by_image_and_category(bounding_boxes, image_index, category_id):
    bounding_boxes = utils.get_boxes_for_image(bounding_boxes, image_index)
    inds = bounding_boxes['classes'] == category_id
    bounding_boxes = utils.gather_nd(bounding_boxes, inds)
    return bounding_boxes
