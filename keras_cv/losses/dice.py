import tensorflow as tf
from tensorflow import keras

from keras_cv.utils.fill_utils import gather_channels
from keras_cv.utils.fill_utils import get_reduce_axes


class Dice(keras.losses.Loss):
    """Dice loss class for semantic segmentation task.

    Input shape:
        ...
    Output shape:
        ...

    Args:
        ...

    Call arguments:
        y_true:
        y_pred:

    Usages:
    ```python

    ```

    """

    def __init__(
        self,
        beta=1,
        class_ids=None,
        per_image=False,
        smooth=keras.backend.epsilon(),
        label_smoothing=0.0,
        from_logits=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.beta = beta
        self.class_ids = class_ids
        self.per_image = per_image
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing
        self.smooth = smooth

    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        label_smoothing = tf.convert_to_tensor(self.label_smoothing, dtype=y_pred.dtype)

        y_pred = tf.__internal__.smart_cond.smart_cond(
            self.from_logits, lambda: tf.nn.softmax(y_pred), lambda: y_pred
        )

        def _smooth_labels():
            num_classes = tf.cast(tf.shape(y_true)[-1], y_pred.dtype)
            return y_true * (1.0 - label_smoothing) + (label_smoothing / num_classes)

        y_true = tf.__internal__.smart_cond.smart_cond(
            label_smoothing, _smooth_labels, lambda: y_true
        )

        y_true, y_pred = tf.__internal__.smart_cond.smart_cond(
            self.class_indexes == None,
            lambda: (y_true, y_pred),
            lambda: gather_channels(y_true, y_pred, indexes=self.class_ids),
        )

        axes = get_reduce_axes(self.per_image)

        true_positive = keras.backend.sum(y_true * y_pred, axis=axes)
        false_positive = keras.backend.sum(y_pred, axis=axes) - true_positive
        false_negative = keras.backend.sum(y_true, axis=axes) - true_positive

        # Type I and type II errors - f-score forumula
        power_beta = 1 + self.beta**2
        numerator = power_beta * true_positive + self.smooth
        denominator = (
            (power_beta * true_positive)
            + (self.beta**2 * false_negative)
            + false_positive
            + self.smooth
        )

        dice_score = numerator / denominator
        dice_score = tf.cond(
            self.per_image,
            lambda: keras.backend.mean(dice_score, axis=0),
            lambda: keras.backend.mean(dice_score),
        )

        return 1 - dice_score
