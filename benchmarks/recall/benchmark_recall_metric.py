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

"""Tests to ensure that BoxRecall computes the correct values."""
import os

from tqdm import tqdm
import time
import numpy as np
import tensorflow as tf
import json

from keras_cv import bounding_box
from keras_cv.metrics import BoxRecall
import pandas as pd
import in_graph_recall
import pymetric_recall
import matplotlib.pyplot as plt
import seaborn as sns
import pycocotools_recall

SAMPLE_FILE = os.path.dirname(os.path.abspath(__file__)) + "/sample_boxes.npz"

name = "v2"


def trial(metric, y_true, y_pred, expected_result):
    # Warmup!!!
    metric.update_state(y_true, y_pred)
    result = metric.result()
    metric.reset_state()

    t0 = time.time()
    metric.update_state(y_true, y_pred)
    result = metric.result()
    t1 = time.time()

    assert np.abs(result - expected_result) < 0.06, "Numerical accuracy check failed"
    return t1 - t0


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


y_true, y_pred, categories = load_samples(SAMPLE_FILE)

cases = [
    (
        {
            "bounding_box_format": "xyxy",
            "class_ids": categories + [1000],
            "max_detections": 1,
        },
        0.478,
        "ARmax1",
    ),
    (
        {
            "bounding_box_format": "xyxy",
            "class_ids": categories + [1000],
            "max_detections": 10,
        },
        0.645,
        "ARmax10",
    ),
    (
        {
            "bounding_box_format": "xyxy",
            "class_ids": categories + [1000],
            "max_detections": 100,
        },
        0.648,
        "ARmax100",
    ),
    (
        {
            "bounding_box_format": "xyxy",
            "class_ids": categories + [1000],
            "max_detections": 100,
            "area_range": (0, 32**2),
        },
        0.628,
        "ARs",
    ),
    (
        {
            "bounding_box_format": "xyxy",
            "class_ids": categories + [1000],
            "max_detections": 100,
            "area_range": (32**2, 96**2),
        },
        0.653,
        "ARm",
    ),
    (
        {
            "bounding_box_format": "xyxy",
            "class_ids": categories + [1000],
            "max_detections": 100,
            "area_range": (96**2, 1e5**2),
        },
        0.641,
        "ARl",
    ),
]

ingraph_runtimes = []
pymetric_runtimes = []
pycocotools_runtimes = []

for args, target, cocotoolskey in tqdm(cases):
    # igr_metric = in_graph_recall.InGraphBoxRecall(**args)
    pymetric_metric = pymetric_recall.PyMetricRecall(**args)
    # pycocotools_metric = pycocotools_recall.COCOToolsRecall(cocotoolskey)

    # runtime_pycocotools = trial(pycocotools_metric, y_true, y_pred, target)
    # runtime_ingraph = trial(igr_metric, y_true, y_pred, target)
    runtime_pymetric = trial(pymetric_metric, y_true, y_pred, target)

    # ingraph_runtimes.append(runtime_ingraph)
    pymetric_runtimes.append(runtime_pymetric)
    # pycocotools_runtimes.append(runtime_pycocotools)

with open("history.json", "r") as f:
    history = json.load(f)

# history["ingraph"] = ingraph_runtimes
history["pymetric"] = pymetric_runtimes
# history["pycocotools"] = pycocotools_runtimes

with open("history.json", "w") as f:
    json.dump(history, f)

df = pd.DataFrame(data=history)

sns.violinplot(df)
plt.show()
