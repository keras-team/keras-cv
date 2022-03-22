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
import os

import numpy as np
import tensorflow as tf

from keras_cv.metrics.coco import COCOMeanAveragePrecision
from keras_cv.utils import bounding_box

SAMPLE_FILE = os.path.dirname(os.path.abspath(__file__)) + "/sample_boxes.npz"


class MeanAveragePrecisionTest(tf.test.TestCase):
    """Numerical testing for COCOMeanAveragePrecision.

    Unit tests that test Keras COCO metric results against the known values of
    cocoeval.py.  The bounding boxes in sample_boxes.npz were given to
    cocoeval.py, which computed the following values:
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.643
        Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 1.000
        Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.729
        Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.644
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.633
        Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.689
    """

    def test_mean_average_precision_correctness_default(self):
        y_true, y_pred, categories = load_samples(SAMPLE_FILE)

        # Area range all
        mean_average_precision = COCOMeanAveragePrecision(
            class_ids=categories + [1000], max_detections=100, num_buckets=1000
        )

        mean_average_precision.update_state(y_true, y_pred)
        result = mean_average_precision.result().numpy()
        self.assertAlmostEqual(result, 0.643, delta=0.06)

    def test_mean_average_precision_correctness_medium(self):
        y_true, y_pred, categories = load_samples(SAMPLE_FILE)

        mean_average_precision = COCOMeanAveragePrecision(
            class_ids=categories + [1000],
            max_detections=100,
            area_range=(0, 32**2),
        )

        mean_average_precision.update_state(y_true, y_pred)
        result = mean_average_precision.result().numpy()
        self.assertAlmostEqual(result, 0.633, delta=0.06)

    def test_mean_average_precision_correctness_small(self):
        y_true, y_pred, categories = load_samples(SAMPLE_FILE)

        mean_average_precision = COCOMeanAveragePrecision(
            class_ids=categories + [1000],
            max_detections=100,
            area_range=(0, 32**2),
        )

        mean_average_precision.update_state(y_true, y_pred)
        result = mean_average_precision.result().numpy()
        self.assertAlmostEqual(result, 0.644, delta=0.06)

    def test_mean_average_precision_correctness_iou_05(self):
        y_true, y_pred, categories = load_samples(SAMPLE_FILE)

        mean_average_precision = COCOMeanAveragePrecision(
            class_ids=categories + [1000],
            iou_thresholds=[0.5],
            max_detections=100,
            area_range=(0, 1e5**2),
        )

        mean_average_precision.update_state(y_true, y_pred)
        result = mean_average_precision.result().numpy()
        self.assertAlmostEqual(result, 1.0, delta=0.06)

    def test_mean_average_precision_correctness_iou_75(self):
        y_true, y_pred, categories = load_samples(SAMPLE_FILE)

        mean_average_precision = COCOMeanAveragePrecision(
            class_ids=categories + [1000],
            iou_thresholds=[0.75],
            max_detections=100,
            area_range=(0, 1e5**2),
        )

        mean_average_precision.update_state(y_true, y_pred)
        result = mean_average_precision.result().numpy()
        self.assertAlmostEqual(result, 0.729, delta=0.06)

    # TODO(lukewood): re-enable after performance testing
    # def test_mean_average_precision_correctness_large(self):
    #     y_true, y_pred, categories = load_samples(SAMPLE_FILE)
    #
    #     mean_average_precision = COCOMeanAveragePrecision(
    #         class_ids=categories + [1000],
    #         max_detections=100,
    #         area_range=(0, 32**2),
    #     )
    #
    #     mean_average_precision.update_state(y_true, y_pred)
    #     result = mean_average_precision.result().numpy()
    #     self.assertAlmostEqual(result, 0.689, delta=0.06)


def load_samples(fname):
    npzfile = np.load(fname)
    y_true = npzfile["arr_0"].astype(np.float32)
    y_pred = npzfile["arr_1"].astype(np.float32)

    y_true = bounding_box.xywh_to_corners(y_true)
    y_pred = bounding_box.xywh_to_corners(y_pred)

    categories = set(int(x) for x in y_true[:, :, 4].numpy().flatten())
    categories = [x for x in categories if x != -1]

    return y_true, y_pred, categories
