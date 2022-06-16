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

"""Tests to ensure that COCOrecall computes the correct values.."""
import os

import numpy as np
import tensorflow as tf

from keras_cv import bounding_box
from keras_cv.metrics import COCORecall

SAMPLE_FILE = os.path.dirname(os.path.abspath(__file__)) + "/sample_boxes.npz"

delta = 0.04


class RecallCorrectnessTest(tf.test.TestCase):
    """Unit tests that test Keras COCO metric results against the known good ones of
    cocoeval.py.  The bounding boxes in sample_boxes.npz were given to cocoeval.py
    which output the following values:
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.478
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.645
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.648
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.628
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.653
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.641
    """

    def test_recall_correctness_maxdets_1(self):
        y_true, y_pred, categories = load_samples(SAMPLE_FILE)

        # Area range all
        recall = COCORecall(
            bounding_box_format="xyxy",
            class_ids=categories + [1000],
            max_detections=1,
        )

        recall.update_state(y_true, y_pred)
        result = recall.result().numpy()
        self.assertAlmostEqual(result, 0.478, delta=delta)

    def test_recall_correctness_maxdets_10(self):
        y_true, y_pred, categories = load_samples(SAMPLE_FILE)

        # Area range all
        recall = COCORecall(
            bounding_box_format="xyxy",
            class_ids=categories + [1000],
            max_detections=10,
        )

        recall.update_state(y_true, y_pred)
        result = recall.result().numpy()
        self.assertAlmostEqual(result, 0.645, delta=delta)

    def test_recall_correctness_maxdets_100(self):
        y_true, y_pred, categories = load_samples(SAMPLE_FILE)

        # Area range all
        recall = COCORecall(
            bounding_box_format="xyxy",
            class_ids=categories + [1000],
            max_detections=100,
        )

        recall.update_state(y_true, y_pred)
        result = recall.result().numpy()
        self.assertAlmostEqual(result, 0.648, delta=delta)

    def test_recall_correctness_small_objects(self):
        y_true, y_pred, categories = load_samples(SAMPLE_FILE)
        recall = COCORecall(
            bounding_box_format="xyxy",
            class_ids=categories + [1000],
            max_detections=100,
            area_range=(0, 32**2),
        )

        recall.update_state(y_true, y_pred)
        result = recall.result().numpy()
        self.assertAlmostEqual(result, 0.628, delta=delta)

    def test_recall_correctness_medium_objects(self):
        y_true, y_pred, categories = load_samples(SAMPLE_FILE)
        recall = COCORecall(
            bounding_box_format="xyxy",
            class_ids=categories + [1000],
            max_detections=100,
            area_range=(32**2, 96**2),
        )

        recall.update_state(y_true, y_pred)
        result = recall.result().numpy()
        self.assertAlmostEqual(result, 0.653, delta=delta)

    def test_recall_correctness_large_objects(self):
        y_true, y_pred, categories = load_samples(SAMPLE_FILE)
        recall = COCORecall(
            bounding_box_format="xyxy",
            class_ids=categories + [1000],
            max_detections=100,
            area_range=(96**2, 1e5**2),
        )

        recall.update_state(y_true, y_pred)
        result = recall.result().numpy()
        self.assertAlmostEqual(result, 0.641, delta=delta)


def load_samples(fname):
    npzfile = np.load(fname)
    y_true = npzfile["arr_0"].astype(np.float32)
    y_pred = npzfile["arr_1"].astype(np.float32)

    y_true = bounding_box.convert_format(y_true, source="xywh", target="xyxy")
    y_pred = bounding_box.convert_format(y_pred, source="xywh", target="xyxy")

    categories = set(int(x) for x in y_true[:, :, 4].numpy().flatten())
    categories = [x for x in categories if x != -1]

    return y_true, y_pred, categories
