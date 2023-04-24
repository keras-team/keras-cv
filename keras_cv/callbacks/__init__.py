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
try:
    from keras_cv.callbacks.pycoco_callback import PyCOCOCallback
except ImportError:
    print(
        "You do not have pyococotools installed, so the `PyCOCOCallback` API is"
        "not available."
    )

from keras_cv.callbacks.waymo_evaluation_callback import WaymoEvaluationCallback

class EvaluateCOCOMetricsCallback(keras.callbacks.Callback):
    def __init__(self, data, bounding_box_format):
        super().__init__()
        self.data = data
        self.bounding_box_format = bounding_box_format

    def on_epoch_end(self, epoch, logs):
        from keras_cv.metrics.coco.pycoco_wrapper import compute_dataset_pycoco_metrics
        metrics = compute_dataset_pycoco_metrics(self.model, self.data, self.bounding_box_format)
        logs.update(metrics)
        return logs
