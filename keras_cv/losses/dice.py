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

from keras_cv.utils import losses_utils


def _calculate_dice_numerator_denominator(y_true, y_pred, beta, axis, epsilon):
    # Calculate Dice Loss' numerator and denominator from equation.
    true_positive = tf.reduce_sum(y_true * y_pred, axis=axis)
    false_positive = tf.reduce_sum(y_pred, axis=axis) - true_positive
    false_negative = tf.reduce_sum(y_true, axis=axis) - true_positive

    power_beta = 1 + beta**2
    numerator = power_beta * true_positive + epsilon
    denominator = (
        (power_beta * true_positive)
        + (beta**2 * false_negative)
        + false_positive
        + epsilon
    )
    return (numerator, denominator)


def _generalized_dice_score(y_true, numerator, denominator, axis):
    # Calculate the volume of ground truth labels.
    weight = tf.math.reciprocal(tf.square(tf.reduce_sum(y_true, axis=axis)))

    # Calculate the weighted dice score and normalizer.
    weighted_numerator = tf.reduce_sum(weight * numerator)
    weighted_denominator = tf.reduce_sum(weight * denominator)
    general_dice_score = weighted_numerator / weighted_denominator

    return general_dice_score


def _adaptive_dice_score(numerator, denominator):
    # Calculate the dice scores
    dice_score = numerator / denominator
    # Calculate weights based on Dice scores.
    weights = tf.exp(-1.0 * dice_score)
    # Multiply weights by corresponding scores and get sum.
    weighted_dice = tf.reduce_sum(weights * dice_score)
    # Calculate normalization factor.
    normalizer = tf.cast(tf.size(input=dice_score), dtype=tf.float32) * tf.exp(-1.0)
    # normalize the dice score
    norm_dice_score = weighted_dice / normalizer

    return norm_dice_score


def _check_input_params(beta, loss_type=None, class_ids=None):
    if beta <= 0.0:
        raise ValueError(f"`beta` value should be greater than zero. Got {beta}")

    if loss_type is not None:
        if loss_type not in ["generalized", "adaptive"]:
            raise ValueError(
                "The `loss_type` is not valid. "
                "If `loss_type` is not `None`, It should be either "
                f"`generalized` or `adaptive`. Got {loss_type}"
            )

    if class_ids is not None:
        if isinstance(class_ids, float) or any(isinstance(x, float) for x in class_ids):
            raise ValueError(
                f"The indices should be int or a list of integer. Got {class_ids}"
            )


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class SparseDice(tf.keras.losses.Loss):
    """Compute the dice loss between the sparse labels and predictions.

    Use this sparse dice loss function when there are 2D and 3D semantic
    segmentaiton task. We expect labels to be provided in a `sparse`
    representation (integer-encoded).

    In the snippet below, there are `num_classes` times channels per
    sample. The shape of `y_true` and `y_pred` are
    `[batch_size, height, width, class]` and
    `[batch_size, height, width, num_classes]`.

    Standalone usage:

    >>> y_true = tf.random.uniform([5, 10, 10, 1], 0, maxval=4, dtype=tf.int32)
    >>> y_pred = tf.random.uniform([5, 10, 10, 4], 0, maxval=4)
    >>> dice = SparseDice()
    >>> dice(y_true, y_pred).numpy()
    0.5692612

    >>> # Calling with 'sample_weight'.
    >>> dice(y_true, y_pred, sample_weight=tf.constant([[0.5, 0.5]])).numpy()
    0.2846306

    Usage with the `compile()` API:
    ```python
    model.compile(optimizer='adam', loss=keras_cv.losses.SparseDice())
    ```
    """

    def __init__(
        self,
        beta=1,
        from_logits=False,
        class_ids=None,
        axis=[1, 2],
        loss_type=None,
        label_smoothing=0.0,
        epsilon=1e-07,
        per_image=False,
        name="sparse_dice",
        **kwargs,
    ):
        """Initializes `Dice` instance.

        Args:
            beta: A float or integer coefficient for balancing the precision
                and recall. It determines the weight of recall and precision
                in the combined score. The value of `beta` should be greater
                than `0`. If `beta < 1`, precision will dominate; if `beta > 1`,
                recall will dominate. Default to `1`.
            from_logits: Whether `y_pred` is expected to be a logits tensor. By
                default, we assume that `y_pred` encodes a probability distribution.
                Default to `False`.
            class_ids: An integer or a list of integers within `range(num_classes)`
                to evaluate the loss. If it's `None`, all classes will be used to
                calculate the loss. Default to `None`.
            axis: An optional sequence of `int` specifying the axis to perform reduce
                ops for raw dice score. For 2D models, it should be [1,2] or [2,3]
                for the `channels_last` or `channels_first` format respectively. And for
                3D model, it should be [1,2,3] or [2,3,4] for the `channels_last` or
                `channel_first` format respectively.
            loss_type: An optional `str` specifying the type of the dice score to
                compute. Compute generalized or adaptive dice score if loss type is
                `generalized` or `adaptive`; otherwise compute original dice score.
                Defaults to `None`.
            label_smoothing: Float in [0, 1]. When > 0, label values are smoothed,
                meaning the confidence on label values are relaxed. For example, if
                `0.1`, use `0.1 / num_classes` for non-target labels and
                `0.9 + 0.1 / num_classes` for target labels. Default to `0.0`.
            per_image: if True, calculates score as mean of all scores for each image in the batch,
                respecting the channels_first or channels_last format. Default to False.
            epsilon: Small float added to dice score to avoid dividing by zero.
                Defaults to `1e-07` (`backend.epsilon`).
            name: Optional name for the instance.
                Defaults to 'dice'.
        """
        super().__init__(name=name, **kwargs)

        _check_input_params(beta, loss_type, class_ids)

        if isinstance(class_ids, int):
            class_ids = [class_ids]

        self.beta = beta
        self.from_logits = from_logits
        self.loss_type = loss_type
        self.label_smoothing = label_smoothing
        self.epsilon = epsilon
        self.class_ids = class_ids
        self.per_image = per_image
        self.axis = axis

    def _smooth_labels(self, y_true, y_pred, label_smoothing):
        num_classes = tf.cast(tf.shape(y_true)[-1], y_pred.dtype)
        return y_true * (1.0 - label_smoothing) + (label_smoothing / num_classes)

    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)

        """
        If axis is [1, 2] or [1, 2, 3] - the input vector is in the channels_last format.
        The number of channels is thus y_pred.shape[-1].
        If axis is [2, 3] or [2, 3, 4] - the input vector is in the channels_first format.
        The number of channels is thus y_pred.shape[0].
        If none of these hold true - raise exception.
        """

        if self.axis in [[1, 2], [1, 2, 3]]:
            num_channels = tf.cast(y_pred.shape[-1], dtype=tf.int32)
        elif self.axis in [[2, 3], [2, 3, 4]]:
            num_channels = tf.cast(y_pred.shape[0], dtype=tf.int32)
        else:
            raise ValueError(
                f"`axis` value should be [1, 2] or [1, 2, 3] for 2D and 3D channels_last input respectively, and [2, 3] or [2, 3, 4] for 2D and 3D channels_first input respectively. Got {self.axis}"
            )

        y_true = tf.one_hot(
            tf.reshape(
                y_true, shape=(y_true.shape[0], y_true.shape[1], y_true.shape[2])
            ),
            depth=num_channels,
        )

        y_true = tf.cast(y_true, y_pred.dtype)

        label_smoothing = tf.convert_to_tensor(self.label_smoothing, dtype=y_pred.dtype)

        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred)

        if tf.cast(label_smoothing, dtype=tf.bool):
            y_true = self._smooth_labels(y_true, y_pred, label_smoothing)

        if self.class_ids is not None:
            y_true, y_pred = losses_utils.gather_channels(
                y_true, y_pred, indices=self.class_ids
            )

        # loss calculation: Fβ-score (in terms of Type I and type II error).
        numerator, denominator = _calculate_dice_numerator_denominator(
            y_true, y_pred, self.beta, self.axis, self.epsilon
        )

        if self.loss_type == "generalized":
            dice_score = _generalized_dice_score(
                y_true, numerator, denominator, self.axis
            )
        elif self.loss_type == "adaptive":
            dice_score = _adaptive_dice_score(numerator, denominator)
        else:
            dice_score = numerator / denominator

        if self.per_image and self.axis in [[1, 2], [1, 2, 3]]:
            dice_score = tf.reduce_mean(dice_score, axis=0)
        elif self.per_image and self.axis in [[2, 3], [2, 3, 4]]:
            dice_score = tf.reduce_mean(dice_score, axis=1)
        else:
            dice_score = tf.reduce_mean(dice_score)

        return 1 - dice_score

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta": self.beta,
                "from_logits": self.from_logits,
                "class_ids": self.class_ids,
                "loss_type": self.loss_type,
                "label_smoothing": self.label_smoothing,
                "epsilon": self.epsilon,
                "per_image": self.per_image,
                "axis": self.axis,
            }
        )
        return config
