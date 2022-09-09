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
import tensorflow as tf

import keras_cv


class VisualizeObjectDetectionPredictions(tf.keras.callbacks.Callback):
    def __init__(self, x, y, value_range, bounding_box_format, artifacts_dir=None):
        self.x = x
        self.y = y
        self.artifacts_dir = artifacts_dir
        self.value_range = value_range
        self.bounding_box_format = bounding_box_format
        keras_cv.utils.ensure_exists(artifacts_dir)

    def on_epoch_end(self, epoch, logs):
        y_pred = self.model.predict(self.x)
        path = (
            None if self.artifacts_dir is None else f"{self.artifacts_dir}/{epoch}.png"
        )
        keras_cv.visualization.plot_bounding_box_gallery(
            self.x,
            value_range=self.value_range,
            bounding_box_format=self.bounding_box_format,
            y_true=self.y,
            y_pred=y_pred,
            path=path,
        )
