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
import os

import numpy as np
import tensorflow as tf

from keras_cv import bounding_box
from keras_cv.metrics import BoxCOCOMetrics

SAMPLE_FILE = (
    os.path.dirname(os.path.abspath(__file__)) + "/test_data/sample_boxes.npz"
)


def load_samples(fname):
    npzfile = np.load(fname)
    y_true = npzfile["arr_0"].astype(np.float32)
    y_pred = npzfile["arr_1"].astype(np.float32)

    y_true = {
        "boxes": y_true[:, :, :4],
        "classes": y_true[:, :, 4],
    }
    y_pred = {
        "boxes": y_pred[:, :, :4],
        "classes": y_pred[:, :, 4],
        "confidence": y_pred[:, :, 5],
    }

    y_true = bounding_box.convert_format(y_true, source="xywh", target="xyxy")
    y_pred = bounding_box.convert_format(y_pred, source="xywh", target="xyxy")

    categories = set(int(x) for x in y_true["classes"].flatten())
    categories = [x for x in categories if x != -1]

    return y_true, y_pred, categories


golden_metrics = {
    "MaP": 0.6194297,
    "MaP@[IoU=50]": 1.0,
    "MaP@[IoU=75]": 0.7079766,
    "MaP@[area=small]": 0.6045385,
    "MaP@[area=medium]": 0.6283987,
    "MaP@[area=large]": 0.6143586,
    "Recall@[max_detections=1]": 0.47537246,
    "Recall@[max_detections=10]": 0.6450954,
    "Recall@[max_detections=100]": 0.6484465,
    "Recall@[area=small]": 0.62842655,
    "Recall@[area=medium]": 0.65336424,
    "Recall@[area=large]": 0.6405466,
}


class BoxCOCOMetricsTest(tf.test.TestCase):
    def test_coco_metric_suite_returns_all_coco_metrics(self):
        suite = BoxCOCOMetrics(bounding_box_format="xyxy", evaluate_freq=1)
        y_true, y_pred, categories = load_samples(SAMPLE_FILE)

        suite.update_state(y_true, y_pred)
        metrics = suite.result()

        self.assertAllEqual(metrics, golden_metrics)

    def test_coco_metric_suite_evaluate_freq(self):
        suite = BoxCOCOMetrics(bounding_box_format="xyxy", evaluate_freq=2)
        y_true, y_pred, categories = load_samples(SAMPLE_FILE)

        suite.update_state(y_true, y_pred)
        metrics = suite.result()
        self.assertAllEqual(metrics, {key: 0 for key in golden_metrics})

        suite.update_state(y_true, y_pred)
        metrics = suite.result()
        #
        for metric in metrics:
            # The metrics do not match golden metrics because two batches were
            # passed which actually modifies the final area under curve value.
            self.assertNotEqual(metrics[metric], 0.0)

    def test_name_parameter(self):
        suite = BoxCOCOMetrics(
            bounding_box_format="xyxy", evaluate_freq=1, name="coco_metrics"
        )
        y_true, y_pred, categories = load_samples(SAMPLE_FILE)
        suite.update_state(y_true, y_pred)
        metrics = suite.result()

        for metric in golden_metrics:
            self.assertAlmostEqual(
                metrics["coco_metrics_" + metric], golden_metrics[metric]
            )
