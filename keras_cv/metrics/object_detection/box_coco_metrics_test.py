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
    "MaP": 0.61690974,
    "MaP@[IoU=50]": 1.0,
    "MaP@[IoU=75]": 0.70687747,
    "MaP@[area=small]": 0.6041764,
    "MaP@[area=medium]": 0.6262922,
    "MaP@[area=large]": 0.61016285,
    "Recall@[max_detections=1]": 0.47804594,
    "Recall@[max_detections=10]": 0.6451851,
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

    def test_coco_metric_graph_mode(self):
        suite = BoxCOCOMetrics(bounding_box_format="xyxy", evaluate_freq=1)
        y_true, y_pred, categories = load_samples(SAMPLE_FILE)

        @tf.function()
        def update_state(y_true, y_pred):
            suite.update_state(y_true, y_pred)

        @tf.function()
        def result():
            return suite.result()

        metrics = result()
        self.assertAllEqual(metrics, {key: 0 for key in golden_metrics})

        update_state(y_true, y_pred)
        metrics = result()
        for metric in metrics:
            self.assertNotEqual(metrics[metric], 0.0)

    def test_coco_metric_suite_force_eval(self):
        suite = BoxCOCOMetrics(bounding_box_format="xyxy", evaluate_freq=512)
        y_true, y_pred, categories = load_samples(SAMPLE_FILE)

        suite.update_state(y_true, y_pred)
        metrics = suite.result()
        self.assertAllEqual(metrics, {key: 0 for key in golden_metrics})

        suite.update_state(y_true, y_pred)
        metrics = suite.result(force=True)
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

    def test_coco_metric_suite_ragged_prediction(self):
        bounding_boxes = {
            "boxes": tf.ragged.constant(
                [
                    [[10, 10, 20, 20], [100, 100, 150, 150]],  # small, medium
                    [
                        [200, 200, 400, 400],  # large
                    ],
                ],
                dtype=tf.float32,
            ),
            "classes": tf.ragged.constant(
                [
                    [0, 1],
                    [2],
                ],
                dtype=tf.float32,
            ),
            "confidence": tf.ragged.constant(
                [
                    [0.7, 0.8],
                    [0.9],
                ],
                dtype=tf.float32,
            ),
        }
        dense_bounding_boxes = bounding_box.to_dense(bounding_boxes)
        ragged_bounding_boxes = bounding_box.to_ragged(dense_bounding_boxes)
        suite = BoxCOCOMetrics(bounding_box_format="xyxy", evaluate_freq=1)
        y_true = dense_bounding_boxes
        y_pred = ragged_bounding_boxes

        suite.update_state(y_true, y_pred)
        metrics = suite.result(force=True)

        for metric in metrics:
            # The metrics will be all 1.0 because the prediction and ground
            # truth is identical.
            self.assertAllEqual(metrics[metric], 1.0)
