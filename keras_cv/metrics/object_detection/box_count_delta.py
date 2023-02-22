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
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.initializers as initializers

from keras_cv import bounding_box


def box_count_delta(y_true, y_pred, mode):
    y_true = bounding_box.ensure_tensor(y_true)
    y_pred = bounding_box.ensure_tensor(y_pred)
    y_pred = bounding_box.to_dense(y_pred)
    y_true = bounding_box.to_dense(y_true)

    ground_truth_boxes = tf.cast(y_true["classes"] != -1, tf.int32)
    predicted_boxes = tf.cast(y_pred["classes"] != -1, tf.int32)

    ground_truth_boxes = tf.math.reduce_sum(ground_truth_boxes, axis=-1)
    predicted_boxes = tf.math.reduce_sum(predicted_boxes, axis=-1)

    if mode == "relative":
        return ground_truth_boxes - predicted_boxes
    elif mode == "absolute":
        return tf.math.abs(ground_truth_boxes - predicted_boxes)
    raise ValueError(
        f"`BoxCountDelta` received unimplemented `mode`={mode}. "
        "Expected either 'relative' or 'absolute'."
    )


class BoxCountDelta(keras.metrics.MeanMetricWrapper):
    """BoxCountDelta counts the difference of counts of predicted and true boxes.

    BoxCountDelta looks at the number of boxes in the ground truth dataset
    and counts the delta between that number and the number of boxes your object
    detection model predicted for that image. This is primarily useful when
    attempting to tune the confidence threshold of your
    `MultiClassNonMaxSuppression` layer in an object detection model.  If this
    metric is high, it indicates that your model is making too many or too few
    predictions.  Ideally this metric will be zero.

    Args:
      mode: (Optional) either 'relative' or 'absolute'.  When set to 'absolute',
        the metric will measure the absolute value of the BoxCountDelta.
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.

    Usage:

    ```python
    box_count_delta = keras_cv.metrics.BoxCountDelta()
    y_true = {
        "classes": [[0, -1, -1]],
        "boxes": [[[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]]],
    }
    y_pred = {
        "classes": [[0, 1, 1, -1]],
        "boxes": [[[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]]],
    }
    box_count_delta.update_state(y_true, y_pred)
    print(box_count_delta.result())
    # > 2.0
    """

    def __init__(
        self,
        mode="relative",
        dtype=None,
        name="box_count_delta",
        **kwargs,
    ):
        super().__init__(
            lambda y_true, y_pred: box_count_delta(y_true, y_pred, mode),
            dtype=dtype,
            name=name,
            **kwargs,
        )
