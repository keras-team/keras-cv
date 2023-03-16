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
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tqdm import tqdm

from keras_cv import bounding_box
from keras_cv import layers as cv_layers
from keras_cv.metrics import ObjectDetectionMetricSuite

SAMPLE_FILE = os.path.dirname(os.path.abspath(__file__)) + "/sample_boxes.npz"


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


class AnchorGeneratorTest(tf.test.TestCase):
    def test_coco_metric_suite_returns_all_coco_metrics(self):
        suite = ObjectDetectionMetricSuite(bounding_box_format="xyxy")
        y_true, y_pred, categories = load_samples(SAMPLE_FILE)

        suite.update_state(y_true, y_pred)
        metrics = suite.result()
        print(metrics)
