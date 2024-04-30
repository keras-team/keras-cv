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

from typing import List

from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.backend import keras
from keras_cv.src.backend import ops


@keras_cv_export("keras_cv.layers.BoxMatcher")
class BoxMatcher(keras.layers.Layer):
    """Box matching logic based on argmax of highest value (e.g., IOU).

    This class computes matches from a similarity matrix. Each row will be
    matched to at least one column, the matched result can either be positive
     / negative, or simply ignored depending on the setting.

    The settings include `thresholds` and `match_values`, for example if:
    1) thresholds=[negative_threshold, positive_threshold], and
       match_values=[negative_value=0, ignore_value=-1, positive_value=1]: the
       rows will be assigned to positive_value if its argmax result >=
       positive_threshold; the rows will be assigned to negative_value if its
       argmax result < negative_threshold, and the rows will be assigned to
       ignore_value if its argmax result is between [negative_threshold,
       positive_threshold).
    2) thresholds=[negative_threshold, positive_threshold], and
       match_values=[ignore_value=-1, negative_value=0, positive_value=1]: the
       rows will be assigned to positive_value if its argmax result >=
       positive_threshold; the rows will be assigned to ignore_value if its
       argmax result < negative_threshold, and the rows will be assigned to
       negative_value if its argmax result is between [negative_threshold,
       positive_threshold). This is different from case 1) by swapping first two
       values.
    3) thresholds=[positive_threshold], and
       match_values=[negative_values, positive_value]: the rows will be assigned
       to positive value if its argmax result >= positive_threshold; the rows
       will be assigned to negative_value if its argmax result <
       negative_threshold.

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

    Example:

    ```python
    box_matcher = keras_cv.layers.BoxMatcher([0.3, 0.7], [-1, 0, 1])
    iou_metric = keras_cv.bounding_box.compute_iou(anchors, boxes)
    matched_columns, matched_match_values = box_matcher(iou_metric)
    cls_mask = ops.less_equal(matched_match_values, 0)
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

    def call(self, similarity_matrix):
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
            similarity_matrix = ops.expand_dims(similarity_matrix, axis=0)
        static_shape = list(similarity_matrix.shape)
        num_rows = static_shape[1] or ops.shape(similarity_matrix)[1]
        batch_size = static_shape[0] or ops.shape(similarity_matrix)[0]

        def _match_when_cols_are_empty():
            """Performs matching when the rows of similarity matrix are empty.
            When the rows are empty, all detections are false positives. So we
            return a tensor of -1's to indicate that the rows do not match to
            any columns.
            Returns:
                matched_columns: An integer tensor of shape [batch_size,
                    num_rows] storing the index of the matched column for each
                    row.
                matched_values: An integer tensor of shape [batch_size,
                    num_rows] storing the match type indicator (e.g. positive or
                    negative or ignored match).
            """
            with ops.name_scope("empty_boxes"):
                matched_columns = ops.zeros(
                    [batch_size, num_rows], dtype="int32"
                )
                matched_values = -ops.ones(
                    [batch_size, num_rows], dtype="int32"
                )
                return matched_columns, matched_values

        def _match_when_cols_are_non_empty():
            """Performs matching when the rows of similarity matrix are
            non-empty.
            Returns:
                matched_columns: An integer tensor of shape [batch_size,
                    num_rows] storing the index of the matched column for each
                    row.
                matched_values: An integer tensor of shape [batch_size,
                    num_rows] storing the match type indicator (e.g. positive or
                    negative or ignored match).
            """
            with ops.name_scope("non_empty_boxes"):
                # Jax traces this function even when running eagerly and the
                # columns are non-empty. Therefore, we need to handle the case
                # where the similarity matrix is empty. We do this by padding
                # some -1s to the end. -1s are guaranteed to not affect argmax
                # matching because all values in a similarity matrix are [0,1]
                # and the indexing won't change because these are added at the
                # end.
                padded_similarity_matrix = ops.concatenate(
                    [similarity_matrix, -ops.ones((batch_size, num_rows, 1))],
                    axis=-1,
                )

                matched_columns = ops.argmax(
                    padded_similarity_matrix,
                    axis=-1,
                )

                # Get logical indices of ignored and unmatched columns as int32
                matched_vals = ops.max(padded_similarity_matrix, axis=-1)
                matched_values = ops.zeros([batch_size, num_rows], "int32")

                match_dtype = matched_vals.dtype
                for ind, low, high in zip(
                    self.match_values, self.thresholds[:-1], self.thresholds[1:]
                ):
                    low_threshold = ops.cast(low, match_dtype)
                    high_threshold = ops.cast(high, match_dtype)
                    mask = ops.logical_and(
                        ops.greater_equal(matched_vals, low_threshold),
                        ops.less(matched_vals, high_threshold),
                    )
                    matched_values = self._set_values_using_indicator(
                        matched_values, mask, ind
                    )

                if self.force_match_for_each_col:
                    # [batch_size, num_cols], for each column (groundtruth_box),
                    # find the best matching row (anchor).
                    matching_rows = ops.argmax(
                        padded_similarity_matrix,
                        axis=1,
                    )
                    # [batch_size, num_cols, num_rows], a transposed 0-1 mapping
                    # matrix M, where M[j, i] = 1 means column j is matched to
                    # row i.
                    column_to_row_match_mapping = ops.one_hot(
                        matching_rows, num_rows
                    )
                    # [batch_size, num_rows], for each row (anchor), find the
                    # matched column (groundtruth_box).
                    force_matched_columns = ops.argmax(
                        column_to_row_match_mapping,
                        axis=1,
                    )
                    # [batch_size, num_rows]
                    force_matched_column_mask = ops.cast(
                        ops.max(column_to_row_match_mapping, axis=1),
                        "bool",
                    )
                    # [batch_size, num_rows]
                    matched_columns = ops.where(
                        force_matched_column_mask,
                        force_matched_columns,
                        matched_columns,
                    )
                    matched_values = ops.where(
                        force_matched_column_mask,
                        self.match_values[-1]
                        * ops.ones([batch_size, num_rows], dtype="int32"),
                        matched_values,
                    )

                return ops.cast(matched_columns, "int32"), matched_values

        num_boxes = (
            similarity_matrix.shape[-1] or ops.shape(similarity_matrix)[-1]
        )
        matched_columns, matched_values = ops.cond(
            pred=ops.greater(num_boxes, 0),
            true_fn=_match_when_cols_are_non_empty,
            false_fn=_match_when_cols_are_empty,
        )

        if squeeze_result:
            matched_columns = ops.squeeze(matched_columns, axis=0)
            matched_values = ops.squeeze(matched_values, axis=0)

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
        indicator = ops.cast(indicator, x.dtype)
        return ops.add(ops.multiply(x, 1 - indicator), val * indicator)

    def get_config(self):
        config = {
            "thresholds": self.thresholds[1:-1],
            "match_values": self.match_values,
            "force_match_for_each_col": self.force_match_for_each_col,
        }
        return config
