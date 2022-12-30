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

"Argmax-based box matching"

from typing import List
from typing import Tuple

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class ArgmaxBoxMatcher(tf.keras.layers.Layer):
    """Box matching logic based on argmax of highest value (e.g., IOU).

    This class computes matches from a similarity matrix. Each row will be
    matched to at least one column, the matched result can either be positive
     / negative, or simply ignored depending on the setting.

    The settings include `thresholds` and `match_values`, for example if:
    1) thresholds=[negative_threshold, positive_threshold], and
       match_values=[negative_value=0, ignore_value=-1, positive_value=1]: the rows will
       be assigned to positive_value if its argmax result >=
       positive_threshold; the rows will be assigned to negative_value if its
       argmax result < negative_threshold, and the rows will be assigned
       to ignore_value if its argmax result is between [negative_threshold, positive_threshold).
    2) thresholds=[negative_threshold, positive_threshold], and
       match_values=[ignore_value=-1, negative_value=0, positive_value=1]: the rows will
       be assigned to positive_value if its argmax result >=
       positive_threshold; the rows will be assigned to ignore_value if its
       argmax result < negative_threshold, and the rows will be assigned
       to negative_value if its argmax result is between [negative_threshold ,positive_threshold).
       This is different from case 1) by swapping first two
       values.
    3) thresholds=[positive_threshold], and
       match_values=[negative_values, positive_value]: the rows will be assigned to
       positive value if its argmax result >= positive_threshold; the rows
       will be assigned to negative_value if its argmax result < negative_threshold.

    Args:
        thresholds: A sorted list of floats to classify the matches into
          different results (e.g. positive or negative or ignored match). The
          list will be prepended with -Inf and and appended with +Inf.
        match_values: A list of integers representing matched results (e.g.
          positive or negative or ignored match). len(`match_values`) must
          equal to len(`thresholds`) + 1.
        force_match_for_each_col: each row will be argmax matched to at
          least one column. This means some columns will be matched to
          multiple rows while some columns will not be matched to any rows.
          Filtering by `thresholds` will make less columns match to positive
          result. Setting this to True guarantees that each column will be
          matched to positive result to at least one row.

    Raises:
        ValueError: if `thresholds` not sorted or
        len(`match_values`) != len(`thresholds`) + 1

    Usage:

    ```python
    box_matcher = keras_cv.ops.ArgmaxBoxMatcher([0.3, 0.7], [-1, 0, 1])
    iou_metric = keras_cv.bounding_box.compute_iou(anchors, gt_boxes)
    matched_columns, matched_match_values = box_matcher(iou_metric)
    cls_mask = tf.less_equal(matched_match_values, 0)
    ```

    TODO(tanzhenyu): document when to use which mode.

    """

    def __init__(
        self,
        thresholds: List[float],
        match_values: List[int],
        force_match_for_each_col: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if sorted(thresholds) != thresholds:
            raise ValueError(f"`threshold` must be sorted, got {thresholds}")
        self.match_values = match_values
        if len(match_values) != len(thresholds) + 1:
            raise ValueError(
                f"len(`match_values`) must be len(`thresholds`) + 1, got "
                f"match_values {match_values}, thresholds {thresholds}"
            )
        thresholds.insert(0, -float("inf"))
        thresholds.append(float("inf"))
        self.thresholds = thresholds
        self.force_match_for_each_col = force_match_for_each_col
        self.built = True

    def call(self, similarity_matrix: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Matches each row to a column based on argmax

        TODO(tanzhenyu): consider swapping rows and cols.
        Args:
          similarity_matrix: A float Tensor of shape [num_rows, num_cols] or
            [batch_size, num_rows, num_cols] representing any similarity metric.

        Returns:
          matched_columns: An integer tensor of shape [num_rows] or [batch_size,
            num_rows] storing the index of the matched colum for each row.
          matched_values: An integer tensor of shape [num_rows] or [batch_size,
            num_rows] storing the match result (positive match, negative match,
            ignored match).
        """
        squeeze_result = False
        if len(similarity_matrix.shape) == 2:
            squeeze_result = True
            similarity_matrix = tf.expand_dims(similarity_matrix, axis=0)
        static_shape = similarity_matrix.shape.as_list()
        num_rows = static_shape[1] or tf.shape(similarity_matrix)[1]
        batch_size = static_shape[0] or tf.shape(similarity_matrix)[0]

        def _match_when_cols_are_empty():
            """Performs matching when the rows of similarity matrix are empty.
            When the rows are empty, all detections are false positives. So we return
            a tensor of -1's to indicate that the rows do not match to any columns.
            Returns:
                matched_columns: An integer tensor of shape [batch_size, num_rows]
                  storing the index of the matched column for each row.
                matched_values: An integer tensor of shape [batch_size, num_rows]
                  storing the match type indicator (e.g. positive or negative
                  or ignored match).
            """
            with tf.name_scope("empty_gt_boxes"):
                matched_columns = tf.zeros([batch_size, num_rows], dtype=tf.int32)
                matched_values = -tf.ones([batch_size, num_rows], dtype=tf.int32)
                return matched_columns, matched_values

        def _match_when_cols_are_non_empty():
            """Performs matching when the rows of similarity matrix are non empty.
            Returns:
                matched_columns: An integer tensor of shape [batch_size, num_rows]
                  storing the index of the matched column for each row.
                matched_values: An integer tensor of shape [batch_size, num_rows]
                  storing the match type indicator (e.g. positive or negative
                  or ignored match).
            """
            with tf.name_scope("non_empty_gt_boxes"):
                matched_columns = tf.argmax(
                    similarity_matrix, axis=-1, output_type=tf.int32
                )

                # Get logical indices of ignored and unmatched columns as tf.int64
                matched_vals = tf.reduce_max(similarity_matrix, axis=-1)
                matched_values = tf.zeros([batch_size, num_rows], tf.int32)

                match_dtype = matched_vals.dtype
                for (ind, low, high) in zip(
                    self.match_values, self.thresholds[:-1], self.thresholds[1:]
                ):
                    low_threshold = tf.cast(low, match_dtype)
                    high_threshold = tf.cast(high, match_dtype)
                    mask = tf.logical_and(
                        tf.greater_equal(matched_vals, low_threshold),
                        tf.less(matched_vals, high_threshold),
                    )
                    matched_values = self._set_values_using_indicator(
                        matched_values, mask, ind
                    )

                if self.force_match_for_each_col:
                    # [batch_size, num_cols], for each column (groundtruth_box), find the
                    # best matching row (anchor).
                    matching_rows = tf.argmax(
                        input=similarity_matrix, axis=1, output_type=tf.int32
                    )
                    # [batch_size, num_cols, num_rows], a transposed 0-1 mapping matrix M,
                    # where M[j, i] = 1 means column j is matched to row i.
                    column_to_row_match_mapping = tf.one_hot(
                        matching_rows, depth=num_rows
                    )
                    # [batch_size, num_rows], for each row (anchor), find the matched
                    # column (groundtruth_box).
                    force_matched_columns = tf.argmax(
                        input=column_to_row_match_mapping, axis=1, output_type=tf.int32
                    )
                    # [batch_size, num_rows]
                    force_matched_column_mask = tf.cast(
                        tf.reduce_max(column_to_row_match_mapping, axis=1), tf.bool
                    )
                    # [batch_size, num_rows]
                    matched_columns = tf.where(
                        force_matched_column_mask,
                        force_matched_columns,
                        matched_columns,
                    )
                    matched_values = tf.where(
                        force_matched_column_mask,
                        self.match_values[-1]
                        * tf.ones([batch_size, num_rows], dtype=tf.int32),
                        matched_values,
                    )

                return matched_columns, matched_values

        num_gt_boxes = (
            similarity_matrix.shape.as_list()[-1] or tf.shape(similarity_matrix)[-1]
        )
        matched_columns, matched_values = tf.cond(
            pred=tf.greater(num_gt_boxes, 0),
            true_fn=_match_when_cols_are_non_empty,
            false_fn=_match_when_cols_are_empty,
        )

        if squeeze_result:
            matched_columns = tf.squeeze(matched_columns, axis=0)
            matched_values = tf.squeeze(matched_values, axis=0)

        return matched_columns, matched_values

    def _set_values_using_indicator(self, x, indicator, val):
        """Set the indicated fields of x to val.

        Args:
          x: tensor.
          indicator: boolean with same shape as x.
          val: scalar with value to set.
        Returns:
          modified tensor.
        """
        indicator = tf.cast(indicator, x.dtype)
        return tf.add(tf.multiply(x, 1 - indicator), val * indicator)

    def get_config(self):
        config = {
            "thresholds": self.thresholds[1:-1],
            "match_values": self.match_values,
            "force_match_for_each_col": self.force_match_for_each_col,
        }
        return config
