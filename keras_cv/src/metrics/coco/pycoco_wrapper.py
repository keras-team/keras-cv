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
import copy

import numpy as np

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError:
    COCO = object
    COCOeval = None

from keras_cv.src.utils.conditional_imports import assert_pycocotools_installed

METRIC_NAMES = [
    "AP",
    "AP50",
    "AP75",
    "APs",
    "APm",
    "APl",
    "ARmax1",
    "ARmax10",
    "ARmax100",
    "ARs",
    "ARm",
    "ARl",
]


class PyCOCOWrapper(COCO):
    """COCO wrapper class.
    This class wraps COCO API object, which provides the following additional
    functionalities:
      1. Support string type image id.
      2. Support loading the groundtruth dataset using the external annotation
         dictionary.
      3. Support loading the prediction results using the external annotation
         dictionary.
    """

    def __init__(self, gt_dataset=None):
        """Instantiates a COCO-style API object.
        Args:
          eval_type: either 'box' or 'mask'.
          annotation_file: a JSON file that stores annotations of the eval
            dataset. This is required if `gt_dataset` is not provided.
          gt_dataset: the groundtruth eval dataset in COCO API format.
        """
        assert_pycocotools_installed("PyCOCOWrapper")
        COCO.__init__(self, annotation_file=None)
        self._eval_type = "box"
        if gt_dataset:
            self.dataset = gt_dataset
            self.createIndex()

    def loadRes(self, predictions):
        """Loads result file and return a result api object.
        Args:
          predictions: a list of dictionary each representing an annotation in
            COCO format. The required fields are `image_id`, `category_id`,
            `score`, `bbox`, `segmentation`.
        Returns:
          res: result COCO api object.
        Raises:
          ValueError: if the set of image id from predictions is not the subset
            of the set of image id of the groundtruth dataset.
        """
        res = COCO()
        res.dataset["images"] = copy.deepcopy(self.dataset["images"])
        res.dataset["categories"] = copy.deepcopy(self.dataset["categories"])

        image_ids = [ann["image_id"] for ann in predictions]
        if set(image_ids) != (set(image_ids) & set(self.getImgIds())):
            raise ValueError(
                "Results do not correspond to the current dataset!"
            )
        for ann in predictions:
            x1, x2, y1, y2 = [
                ann["bbox"][0],
                ann["bbox"][0] + ann["bbox"][2],
                ann["bbox"][1],
                ann["bbox"][1] + ann["bbox"][3],
            ]

            ann["area"] = ann["bbox"][2] * ann["bbox"][3]
            ann["segmentation"] = [[x1, y1, x1, y2, x2, y2, x2, y1]]

        res.dataset["annotations"] = copy.deepcopy(predictions)
        res.createIndex()
        return res


def _yxyx_to_xywh(boxes):
    if boxes.shape[-1] != 4:
        raise ValueError(
            "boxes.shape[-1] is {:d}, but must be 4.".format(boxes.shape[-1])
        )

    boxes_ymin = boxes[..., 0]
    boxes_xmin = boxes[..., 1]
    boxes_width = boxes[..., 3] - boxes[..., 1]
    boxes_height = boxes[..., 2] - boxes[..., 0]
    new_boxes = np.stack(
        [boxes_xmin, boxes_ymin, boxes_width, boxes_height], axis=-1
    )

    return new_boxes


def _convert_predictions_to_coco_annotations(predictions):
    coco_predictions = []
    num_batches = len(predictions["source_id"])
    for i in range(num_batches):
        batch_size = predictions["source_id"][i].shape[0]
        predictions["detection_boxes"][i] = predictions["detection_boxes"][
            i
        ].copy()
        for j in range(batch_size):
            max_num_detections = predictions["num_detections"][i][j]
            predictions["detection_boxes"][i][j] = _yxyx_to_xywh(
                predictions["detection_boxes"][i][j]
            )
            for k in range(max_num_detections):
                ann = {}
                ann["image_id"] = predictions["source_id"][i][j]
                ann["category_id"] = predictions["detection_classes"][i][j][k]
                ann["bbox"] = predictions["detection_boxes"][i][j][k]
                ann["score"] = predictions["detection_scores"][i][j][k]
                coco_predictions.append(ann)

    for i, ann in enumerate(coco_predictions):
        ann["id"] = i + 1

    return coco_predictions


def _convert_groundtruths_to_coco_dataset(groundtruths, label_map=None):
    source_ids = np.concatenate(groundtruths["source_id"], axis=0)
    gt_images = [{"id": i} for i in source_ids]

    gt_annotations = []
    num_batches = len(groundtruths["source_id"])
    for i in range(num_batches):
        max_num_instances = max(x.shape[0] for x in groundtruths["classes"][i])
        batch_size = groundtruths["source_id"][i].shape[0]
        for j in range(batch_size):
            num_instances = groundtruths["num_detections"][i][j]
            if num_instances > max_num_instances:
                num_instances = max_num_instances
            for k in range(int(num_instances)):
                ann = {}
                ann["image_id"] = groundtruths["source_id"][i][j]
                ann["iscrowd"] = 0
                ann["category_id"] = int(groundtruths["classes"][i][j][k])
                boxes = groundtruths["boxes"][i]
                ann["bbox"] = [
                    float(boxes[j][k][1]),
                    float(boxes[j][k][0]),
                    float(boxes[j][k][3] - boxes[j][k][1]),
                    float(boxes[j][k][2] - boxes[j][k][0]),
                ]
                ann["area"] = float(
                    (boxes[j][k][3] - boxes[j][k][1])
                    * (boxes[j][k][2] - boxes[j][k][0])
                )
                gt_annotations.append(ann)

    for i, ann in enumerate(gt_annotations):
        ann["id"] = i + 1

    if label_map:
        gt_categories = [{"id": i, "name": label_map[i]} for i in label_map]
    else:
        category_ids = [gt["category_id"] for gt in gt_annotations]
        gt_categories = [{"id": i} for i in set(category_ids)]

    gt_dataset = {
        "images": gt_images,
        "categories": gt_categories,
        "annotations": copy.deepcopy(gt_annotations),
    }
    return gt_dataset


def _concat_numpy(groundtruths, predictions):
    """Converts tensors to numpy arrays."""
    numpy_groundtruths = {}
    for key, val in groundtruths.items():
        if isinstance(val, tuple):
            val = np.concatenate(val)
        numpy_groundtruths[key] = val

    numpy_predictions = {}
    for key, val in predictions.items():
        if isinstance(val, tuple):
            val = np.concatenate(val)
        numpy_predictions[key] = val

    return numpy_groundtruths, numpy_predictions


def compute_pycoco_metrics(groundtruths, predictions):
    assert_pycocotools_installed("compute_pycoco_metrics")

    groundtruths, predictions = _concat_numpy(groundtruths, predictions)

    gt_dataset = _convert_groundtruths_to_coco_dataset(groundtruths)
    coco_gt = PyCOCOWrapper(gt_dataset=gt_dataset)
    coco_predictions = _convert_predictions_to_coco_annotations(predictions)
    coco_dt = coco_gt.loadRes(predictions=coco_predictions)
    image_ids = [ann["image_id"] for ann in coco_predictions]

    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    coco_metrics = coco_eval.stats

    metrics = coco_metrics

    metrics_dict = {}
    for i, name in enumerate(METRIC_NAMES):
        metrics_dict[name] = metrics[i].astype(np.float32)

    return metrics_dict
