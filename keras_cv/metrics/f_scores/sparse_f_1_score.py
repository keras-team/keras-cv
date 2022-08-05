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

from keras_cv.metrics.f_scores import SparseFBetaScore


class SparseF1Score(SparseFBetaScore):
    r"""Computes F-1 Score.

    It is the harmonic mean of precision and recall.
    Output range is `[0, 1]`. Works for
    sparse multi-class classification.

    $$
    F_1 = 2 \cdot \frac{\textrm{precision} \cdot \textrm{recall}}{\textrm{precision} + \textrm{recall}}
    $$

    Args:
        num_classes: Number of unique classes in the dataset.
        average: Type of averaging to be performed on data.
            Acceptable values are `None`, `micro`, `macro`
            and `weighted`. Default value is None.
        threshold: Elements of `y_pred` above threshold are
            considered to be 1, and the rest 0. If threshold is
            None, the argmax is converted to 1, and the rest 0.
        name: (Optional) String name of the metric instance.
        dtype: (Optional) Data type of the metric result.

    Returns:
        F-1 Score: float.

    Raises:
        ValueError: If the `average` has values other than
        [None, 'micro', 'macro', 'weighted'].

    `average` parameter behavior:

        None: Scores for each class are returned

        micro: True positivies, false positives and
            false negatives are computed globally.

        macro: True positivies, false positives and
            false negatives are computed for each class
            and their unweighted mean is returned.

        weighted: Metrics are computed for each class
            and returns the mean weighted by the
            number of true instances in each class.

    Usage:

    >>> metric = keras_cv.metrics.SparseF1Score(num_classes=3, threshold=0.5)
    >>> y_true = np.array([2, 0, 1], np.int32)
    >>> y_pred = np.array([[0.2, 0.6, 0.7],
    ...                    [0.2, 0.6, 0.6],
    ...                    [0.6, 0.8, 0.0]], np.float32)
    >>> metric.update_state(y_true, y_pred)
    >>> result = metric.result()
    >>> result.numpy()
    array([0.       , 0.5      , 0.6666667], dtype=float32)
    """

    def __init__(
        self,
        num_classes,
        average = None,
        threshold = None,
        name = "sparse_f1_score",
        dtype = None,
        **kwargs
    ):
        super().__init__(num_classes, average, 1.0, threshold, name=name, dtype=dtype, **kwargs)

    def get_config(self):
        base_config = super().get_config()
        del base_config["beta"]
        return base_config