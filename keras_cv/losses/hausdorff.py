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
from tensorflow import keras


class Hausdorff(keras.losses.Loss):
    """
    Hausdorff computes the hausdorff distance between two sets of n_dimensional points.

    Standalone usage:

    >>> y_true = np.array([[0, 1, 0, 2], [0, 1, 0, 3], [1, 1, 1, 1]], dtype=np.float32)
    >>> y_pred = np.array([[1, 2, 3, 4], [1, 1, 1, 1]], dtype=np.float32)
    >>> hausdorff_distance = keras_cv.losses.Hausdorff()
    >>> hausdorff_distance(y_true, y_pred).numpy()
    2.4494898

    Usage with the `compile()` API:
    ```python
    model.compile(optimizer="Adam", loss=keras_cv.losses.Hausdorff())
    ```

    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _pairwise_distance(self, set_a, set_b):
        """
        Args:
            set_a: A tf tensor where the last axis
            represents points in D dimensional space
            set_b: A tf tensor where the last axis
                represents points in D dimentional space
        """
        set_a = tf.expand_dims(set_a, -2)
        set_b = tf.expand_dims(set_b, -3)

        distances = tf.reduce_sum(tf.square(set_a - set_b), axis=-1)

        return tf.sqrt(distances)

    def call(self, y_true, y_pred):
        """
        Args:
            y_true: ground truth Tensor.
            y_pred: predicted Tensor.
        """
        y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)

        if len(y_true.shape) != len(y_pred.shape):
            raise ValueError(f"Dimension mismatch, Expected ndims of \
y_true == ndims of y_pred, Got ndims of y_true:{len(y_true.shape)} \
, ndims of y_pred:{len(y_pred.shape)}.")

        if y_true.shape[-1] != y_pred.shape[-1]:
            raise ValueError(f"Dimension mismatch in last axis of y_true \
and y_pred, Expected y_true.shape[-1] == y_pred.shape[-1], Got {y_true.shape} \
, {y_pred.shape}.")

        distance = self._pairwise_distance(y_true, y_pred)

        minimum_distance_a_to_b = tf.reduce_min(distance, axis=-1)

        return tf.reduce_max(minimum_distance_a_to_b, axis=-1)
