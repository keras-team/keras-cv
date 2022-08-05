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

from keras_cv.metrics.f_scores import FBetaScore


class F2Score(FBetaScore):
    r"""Computes F-2 Score.

    It is the harmonic mean of precision and recall.
    Output range is `[0, 1]`. Works for both multi-class
    and multi-label classification.

    $$
    F_2 = 2 \cdot \frac{\textrm{precision} \cdot \textrm{recall}}{\textrm{precision} + \textrm{recall}}
    $$

    Args:
        num_classes: Number of unique classes in the dataset.
        average: Type of averaging to be performed on data.
            Acceptable values are `None`, `micro`, `macro`
            and `weighted`. Default value is None.
        threshold: Elements of `y_pred` above threshold are
            considered to be 1, and the rest 0. If threshold is
            None, the argmax is converted to 1, and the rest 0.

    Returns:
        F-2 Score: float.

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

    >>> metric = keras_cv.metrics.F2Score(num_classes=3, threshold=0.5)
    >>> y_true = np.array([[1, 1, 1],
    ...                    [1, 0, 0],
    ...                    [1, 1, 0]], np.int32)
    >>> y_pred = np.array([[0.2, 0.6, 0.7],
    ...                    [0.2, 0.6, 0.6],
    ...                    [0.6, 0.8, 0.0]], np.float32)
    >>> metric.update_state(y_true, y_pred)
    >>> result = metric.result()
    >>> result.numpy()
    array([0.3846154 , 0.90909094, 0.8333334], dtype=float32)
    """

    def __init__(
        self,
        num_classes,
        average=None,
        threshold=None,
        **kwargs
    ):
        super().__init__(num_classes, average, 2.0, threshold, **kwargs)

    def get_config(self):
        base_config = super().get_config()
        del base_config["beta"]
        return base_config