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

from keras_cv.utils.fill_utils import gather_channels


class Dice(keras.losses.Loss):
    """Compute the dice loss between the labels and predictions.

      Use this dice loss function when there are 2D and 3D semantic
      segmentaiton task. We expect labels to be provided in a `one_hot`
      representation.

      In the snippet below, there are `num_classes` times channels per
      sample. The shape of `y_true` and `y_pred` are
      `[batch_size, height, width, num_classes]` or
      `[batch_size, height, width, depth, num_classes]`.

      Standalone usage:

      >>> y_true = tf.random.uniform([5, 10, 10], 0, maxval=4, dtype=tf.int32)
      >>> y_true = tf.one_hot(y_true, depth=4)
      >>> y_pred = tf.random.uniform([5, 10, 10, 4], 0, maxval=4)
      >>> dice = Dice()
      >>> dice(y_true, y_pred).numpy()
      0.5549314

      >>> # Calling with 'sample_weight'.
      >>> dice(y_true, y_pred, sample_weight=tf.constant([[0.5, 0.5]])).numpy()
      0.24619368

      Usage with the `compile()` API:
      ```python
      model.compile(optimizer='adam', loss=keras_cv.losses.Dice())
    ```
    """

    def __init__(
        self,
        beta=1,
        from_logits=False,
        class_ids=None,
        label_smoothing=0.0,
        per_sample=False,
        epsilon=keras.backend.epsilon(),
        name="dice",
        **kwargs,
    ):
        """Initializes `Dice` instance.

        Args:
            beta: A float or integer coefficient for balancing the precision
                and recall. It determines the weight of recall and precision
                in the combined score. If `beta < 1`, precisoin will doninate;
                if `beta > 1`, recall will dominate. Default to `1`.
            from_logits: Whether `y_pred` is expected to be a logits tensor. By
                default, we assume that `y_pred` encodes a probability distribution.
                Default to `False`.
            class_ids: An interger or a lost of intergers within `range(num_classes)`
                to evaluate the loss. If it's `None`, all classes will beu used to
                calculate the loss. Default to `None`.
            label_smoothing: Float in [0, 1]. When > 0, label values are smoothed,
                meaning the confidence on label values are relaxed. For example, if
                `0.1`, use `0.1 / num_classes` for non-target labels and
                `0.9 + 0.1 / num_classes` for target labels. Default to `0.0`.
            per_sample: If `True`, the loss will be calculated for each sample in
                batch and then averaged. Otherwise the loss will be calculated for
                the whole batch. Default to `False`.
            epsilon: Small float added to dice score to avoid dividing by zero.
                Default to `keras.backend.epsilon()` or `1e-07`.
            name: Optional name for the instance.
                Defaults to 'dice'.
        """
        super().__init__(name=name, **kwargs)
        self.beta = beta
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing
        self.per_sample = per_sample
        self.epsilon = epsilon

        if class_ids is not None:
            if isinstance(class_ids, float) or any(
                isinstance(x, float) for x in class_ids
            ):
                raise ValueError(
                    f"The indices should be int or a list of integer. Got {class_ids}"
                )
            elif isinstance(class_ids, int):
                class_ids = [class_ids]

        self.class_ids = class_ids

    def _smooth_labels(self, y_true, y_pred, label_smoothing):
        num_classes = tf.cast(tf.shape(y_true)[-1], y_pred.dtype)
        return y_true * (1.0 - label_smoothing) + (label_smoothing / num_classes)

    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        label_smoothing = tf.convert_to_tensor(self.label_smoothing, dtype=y_pred.dtype)

        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred)

        if tf.cast(label_smoothing, dtype=tf.bool):
            y_true = self._smooth_labels(y_true, y_pred, label_smoothing)

        if self.class_ids is not None:
            y_true, y_pred = gather_channels(y_true, y_pred, indices=self.class_ids)

        axes = (
            tf.constant([1, 2])
            if keras.backend.image_data_format() == "channels_last"
            else tf.constant([2, 3])
        )
        if not self.per_sample:
            axes = tf.concat([tf.constant([0]), axes], axis=0)

        # loss calculation: Fβ-score (in terms of Type I and type II erro
        true_positive = keras.backend.sum(y_true * y_pred, axis=axes)
        false_positive = keras.backend.sum(y_pred, axis=axes) - true_positive
        false_negative = keras.backend.sum(y_true, axis=axes) - true_positive

        power_beta = 1 + self.beta**2
        numerator = power_beta * true_positive + self.epsilon
        denominator = (
            (power_beta * true_positive)
            + (self.beta**2 * false_negative)
            + false_positive
            + self.epsilon
        )
        dice_score = numerator / denominator

        if self.per_sample:
            dice_score = keras.backend.mean(dice_score, axis=0)
        else:
            dice_score = keras.backend.mean(dice_score)

        return 1 - dice_score

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta": self.beta,
                "from_logits": self.from_logits,
                "class_ids": self.class_ids,
                "label_smoothing": self.label_smoothing,
                "per_sample": self.per_sample,
                "epsilon": self.epsilon,
            }
        )
        return config
