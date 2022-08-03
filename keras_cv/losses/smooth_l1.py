import tensorflow as tf


# --- Implementing Smooth L1 loss and Focal Loss as keras custom losses ---
class SmoothL1Loss(tf.keras.losses.Loss):
    """Implements Smooth L1 loss.

    SmoothL1Loss implements the SmoothL1 function, where values less than `cutoff`
    contribute to the overall loss based on their squared difference, and values greater
    than cutoff contribute based on their raw difference.

    Args:
        cutoff: differences between y_true and y_pred that are larger than `cutoff` are
            treated as `L1` values
    """

    def __init__(self, cutoff=1.0, reduction="none", **kwargs):
        super().__init__(reduction=reduction, **kwargs)
        self.cutoff = cutoff

    def call(self, y_true, y_pred):
        difference = y_true - y_pred
        absolute_difference = tf.abs(difference)
        squared_difference = difference**2
        loss = tf.where(
            tf.less(absolute_difference, self.cutoff),
            0.5 * squared_difference,
            absolute_difference - 0.5,
        )
        return tf.reduce_sum(loss, axis=-1)

    def get_config(self):
        config = {
            "cutoff": self.cutoff,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
