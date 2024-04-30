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

from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.backend import keras
from keras_cv.src.backend import ops


@keras_cv_export("keras_cv.losses.BinaryPenaltyReducedFocalCrossEntropy")
class BinaryPenaltyReducedFocalCrossEntropy(keras.losses.Loss):
    """Implements CenterNet modified Focal loss.

    Compared with `keras.losses.BinaryFocalCrossentropy`, this loss discounts
    for negative labels that have value less than `positive_threshold`, the
    larger value the negative label is, the more discount to the final loss.

    User can choose to divide the number of keypoints outside the loss
    computation, or by passing in `sample_weight` as 1.0/num_key_points.

    Args:
      alpha: a focusing parameter used to compute the focal factor.
        Defaults to 2.0. Note, this is equivalent to the `gamma` parameter in
        `keras.losses.BinaryFocalCrossentropy`.
      beta: a float parameter, penalty exponent for negative labels, defaults to
        4.0.
      from_logits: Whether `y_pred` is expected to be a logits tensor, defaults
        to `False`.
      positive_threshold: Anything bigger than this is treated as positive
        label, defaults to 0.99.
      positive_weight: single scalar weight on positive examples, defaults to
        1.0.
      negative_weight: single scalar weight on negative examples, defaults to
        1.0.

    Inputs:
      y_true: [batch_size, ...] float tensor
      y_pred: [batch_size, ...] float tensor with same shape as y_true.

    References:
        - [Objects as Points](https://arxiv.org/pdf/1904.07850.pdf) Eq 1.
        - [Cornernet: Detecting objects as paired keypoints](https://arxiv.org/abs/1808.01244) for `alpha` and
            `beta`.
    """  # noqa: E501

    def __init__(
        self,
        alpha=2.0,
        beta=4.0,
        from_logits=False,
        positive_threshold=0.99,
        positive_weight=1.0,
        negative_weight=1.0,
        reduction="sum_over_batch_size",
        name="binary_penalty_reduced_focal_cross_entropy",
    ):
        super().__init__(reduction=reduction, name=name)
        self.alpha = alpha
        self.beta = beta
        self.from_logits = from_logits
        self.positive_threshold = positive_threshold
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight

    def call(self, y_true, y_pred):
        y_pred = ops.convert_to_tensor(y_pred)
        y_true = ops.cast(y_true, y_pred.dtype)

        if self.from_logits:
            y_pred = ops.sigmoid(y_pred)

        # TODO(tanzhenyu): Evaluate whether we need clipping after model is
        #  trained.
        y_pred = ops.clip(y_pred, 1e-4, 0.9999)
        y_true = ops.clip(y_true, 0.0, 1.0)

        pos_loss = ops.power(1.0 - y_pred, self.alpha) * ops.log(y_pred)
        neg_loss = (
            ops.power(1.0 - y_true, self.beta)
            * ops.power(y_pred, self.alpha)
            * ops.log(1.0 - y_pred)
        )

        positive_mask = y_true > self.positive_threshold

        loss = ops.where(
            positive_mask,
            self.positive_weight * pos_loss,
            self.negative_weight * neg_loss,
        )

        return -1.0 * loss

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "alpha": self.alpha,
                "beta": self.beta,
                "from_logits": self.from_logits,
                "positive_threshold": self.positive_threshold,
                "positive_weight": self.positive_weight,
                "negative_weight": self.negative_weight,
            }
        )
        return config
