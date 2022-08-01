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
import tensorflow.keras.backend as K


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class FocalLoss(tf.keras.losses.Loss):
    """Implements Focal loss

    Focal loss is a modified cross-entropy designed to perform better with
    class imbalance. For this reason, it's commonly used with object detectors.

    Args:
        alpha: a float value between 0 and 1 representing a weighting factor
            used to deal with class imbalance. Positive classes and negative
            classes have alpha and (1 - alpha) as their weighting factors
            respectively. Defaults to 0.25.
        gamma: a positive float value representing the tunable focusing
            parameter. Defaults to 2.
        from_logits: Whether `y_pred` is expected to be a logits tensor. By
            default, `y_pred` is assumed to encode a probability distribution.
            Default to `False`.
        label_smoothing: Float in `[0, 1]`. If higher than 0 then smooth the
            labels by squeezing them towards `0.5`, i.e., using `1. - 0.5 * label_smoothing`
            for the target class and `0.5 * label_smoothing` for the non-target
            class.

    References:
        - [Focal Loss paper](https://arxiv.org/abs/1708.02002)

    Standalone usage:
    ```python
    y_true = tf.random.uniform([10], 0, maxval=4)
    y_pred = tf.random.uniform([10], 0, maxval=4)
    loss = FocalLoss()
    loss(y_true, y_pred).numpy()
    ```
    Usage with the `compile()` API:
    ```python
    model.compile(optimizer='adam', loss=keras_cv.losses.FocalLoss())
    ```
    """

    def __init__(
        self,
        alpha=0.25,
        gamma=2,
        from_logits=False,
        label_smoothing=0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._alpha = float(alpha)
        self._gamma = float(gamma)
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing

    def _smooth_labels(self, y_true):
        return y_true * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)

        if self.label_smoothing:
            y_true = self._smooth_labels(y_true)

        if self.from_logits:
            y_pred = tf.nn.sigmoid(y_pred)

        cross_entropy = K.binary_crossentropy(y_true, y_pred)

        alpha = tf.where(tf.equal(y_true, 1.0), self._alpha, (1.0 - self._alpha))
        pt = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        loss = alpha * tf.pow(1.0 - pt, self._gamma) * cross_entropy

        return tf.reduce_mean(loss, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "alpha": self.alpha,
                "gamma": self.gamma,
                "from_logits": self.from_logits,
                "label_smoothing": self.label_smoothing,
            }
        )
        return config
