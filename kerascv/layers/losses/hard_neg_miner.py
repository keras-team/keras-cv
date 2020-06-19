import tensorflow as tf


class HardNegativeMining(tf.keras.layers.Layer):
    """Hard Negative Mining to keep balanced positive and negative samples.

    This layer will keep a balanced positive and negative samples and return
    the summed results.

    # Attributes:
        negative_positive_ratio: An int to represent the maximum ratio between
            negative samples and positive samples. For example, if the inputs
            has negative samples : positive samples = 100 : 20, it will sort all
            negative samples by `values` and only keep the top 60. Defaults to `3`.
        minimum_negative_examples: An int to represent the minimum negative examples
            to have. For example if negative samples = 100, positive samples = 20,
            `negative_positive_ratio=1` and `minimum_negative_examples=30`, it will
            keep the top 30 results.

    Inputs:
        values: The values to be sorted, usually classification or regression losses,
            shape [batch_size, n_boxes]
        positive_mask: The mask for positive samples, 1 if the value should be
            treated as positive sample, or 0 if the value should be masked out.
            shape [batch_size, n_boxes]
        positive_mask: The mask for negative samples, 1 if the value should be
            treated as negative sample, or 0 if the value should be masked out.
            shape [batch_size, n_boxes]

    Returns:
        summed positive and negative values across the boxes, shape [batch_size]
    """

    def __init__(
        self,
        negative_positive_ratio=3,
        minimum_negative_examples=0,
        name=None,
        **kwargs
    ):
        if not isinstance(negative_positive_ratio, int):
            raise ValueError(
                "`negative_positive_ratio` should be an int, got {}".format(
                    negative_positive_ratio
                )
            )
        if not isinstance(minimum_negative_examples, int):
            raise ValueError(
                "`minimum_negative_examples` should be an int, got {}".format(
                    minimum_negative_examples
                )
            )
        self.negative_positive_ratio = negative_positive_ratio
        self.minimum_negative_examples = minimum_negative_examples
        super(HardNegativeMining, self).__init__(name=name, **kwargs)

    # values [batch_size, n_boxes]
    # positive_mask [batch_size, n_boxes]
    # negative_mask [batch_size, n_boxes]
    # positive_mask and negative_mask should be mutually exclusive, though we don't
    # check that here.
    def call(self, values, positive_mask, negative_mask):
        positive_mask = tf.cast(positive_mask, values.dtype)
        negative_mask = tf.cast(negative_mask, values.dtype)
        values_shape = tf.shape(values)
        batch_size = values_shape[0]
        n_boxes = values_shape[1]
        # Summing across the batch, a single scalar
        num_positives = tf.reduce_sum(positive_mask)
        # [batch_size]
        positive_values = tf.reduce_sum(values * positive_mask, axis=-1)
        # [batch_size, n_boxes]
        negative_values_unreduced = values * negative_mask
        # Only sort non zero negative samples, otherwise positive indices might get
        # leaked into the sorted result.
        n_non_zero_negative_values = tf.math.count_nonzero(
            negative_values_unreduced, dtype=tf.int32
        )
        top_k_negatives = tf.maximum(
            self.negative_positive_ratio * num_positives, self.minimum_negative_examples
        )
        top_k_negatives = tf.minimum(
            tf.cast(top_k_negatives, tf.int32), n_non_zero_negative_values
        )

        # [batch_size * n_boxes]
        negative_values_unreduced = tf.reshape(negative_values_unreduced, [-1])
        _, negative_indices = tf.nn.top_k(
            negative_values_unreduced, k=top_k_negatives, sorted=False
        )
        top_k_negative_mask = tf.scatter_nd(
            indices=tf.expand_dims(negative_indices, axis=1),
            updates=tf.ones_like(negative_indices, dtype=values.dtype),
            shape=tf.shape(negative_values_unreduced),
        )
        top_k_negative_mask = tf.reshape(top_k_negative_mask, [batch_size, n_boxes])
        negative_values = tf.reduce_sum(values * top_k_negative_mask, axis=-1)

        return positive_values + negative_values

    def get_config(self):
        config = {
            "negative_positive_ratio": self.negative_positive_ratio,
            "minimum_negative_examples": self.minimum_negative_examples,
        }
        base_config = super(HardNegativeMining, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
