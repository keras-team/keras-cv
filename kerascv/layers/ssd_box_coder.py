import tensorflow as tf


class SSDBoxCoder(tf.keras.layers.Layer):
    """Defines a SSDBoxEncoder that converts encodes the ground_truth_boxes using anchors.

    Mathematically, the encoding is:
        $ \hat{cx_gt} = (cx_gt - cx_a) / width_a
        $ \hat{cy_gt} = (cy_gt - cy_a) / height_a
        $ \hat{width_gt} = log(width_gt / width_a)
        $ \hat{height_gt} = log(height_gt / height_a)

    where cx, cy, width, height represents center of width, center of height, width, height respectively,
    and subscript `gt` represents ground truth box, `a` represents anchor.

    The `boxes` must have the same shape as `anchors`, this is typically the result of assigning
    `ground_truth_boxes` to anchors based on a certain matching strategy (argmax, bipartite)

    # Attributes:
        variances: The 1-D scaling factor with 4 floats. This is used to represent the variance of
            [y_center, x_center, height, width] in Gaussian distribution when labeling the ground truth boxes.
            During encoding, the result will be divided by the variances. During decoding, the result will be
            multiplied by the variances. Defaults to `None` where no variance is applied.
            The SSD paper uses [.1, .1, .2, .2].
        invert: Boolean to indicate whether the layer should encode the `boxes`, i.e., convert from
            [y_min, x_min, y_max, x_max] format to [ty, tx, h, w] format, if True, or the other way around,
            if False. Defaults to 'False'.

    # References
        [Wei Liu et al., 2015](https://arxiv.org/abs/1512.02325)
    """

    def __init__(self, variances=None, invert=False, name=None, **kwargs):
        self.variances = variances
        self.invert = invert
        super(SSDBoxCoder, self).__init__(name=name, **kwargs)

    def call(self, boxes, anchors):
        def corner_to_centroids(box_tensor):
            box_tensor = tf.cast(box_tensor, tf.float32)
            y_min, x_min, y_max, x_max = tf.split(
                box_tensor, num_or_size_splits=4, axis=-1
            )
            height = y_max - y_min
            width = x_max - x_min
            cy = y_min + 0.5 * height
            cx = x_min + 0.5 * width
            return (
                cy,
                cx,
                height + tf.keras.backend.epsilon(),
                width + tf.keras.backend.epsilon(),
            )

        cy_a, cx_a, height_a, width_a = corner_to_centroids(anchors)

        if not self.invert:
            cy_gt, cx_gt, height_gt, width_gt = corner_to_centroids(boxes)
            ty = (cy_gt - cy_a) / height_a
            tx = (cx_gt - cx_a) / width_a
            th = tf.math.log(height_gt / height_a)
            tw = tf.math.log(width_gt / width_a)

            if self.variances is not None:
                ty = ty / tf.cast(self.variances[0], dtype=ty.dtype)
                tx = tx / tf.cast(self.variances[1], dtype=tx.dtype)
                th = th / tf.cast(self.variances[2], dtype=th.dtype)
                tw = tw / tf.cast(self.variances[3], dtype=tw.dtype)

            return tf.concat([ty, tx, th, tw], axis=-1)

        else:
            ty, tx, th, tw = tf.split(boxes, num_or_size_splits=4, axis=-1)
            if self.variances is not None:
                ty = ty * tf.cast(self.variances[0], dtype=ty.dtype)
                tx = tx * tf.cast(self.variances[1], dtype=tx.dtype)
                th = th * tf.cast(self.variances[2], dtype=th.dtype)
                tw = tw * tf.cast(self.variances[3], dtype=tw.dtype)

            height_gt = tf.math.exp(th) * height_a
            width_gt = tf.math.exp(tw) * width_a
            cy_gt = ty * height_a + cy_a
            cx_gt = tx * width_a + cx_a
            y_min_gt = cy_gt - 0.5 * height_gt
            y_max_gt = cy_gt + 0.5 * height_gt
            x_min_gt = cx_gt - 0.5 * width_gt
            x_max_gt = cx_gt + 0.5 * width_gt

            return tf.concat([y_min_gt, x_min_gt, y_max_gt, x_max_gt], axis=-1)

    def get_config(self):
        config = {"variances": self.variances, "invert": self.invert}
        base_config = super(SSDBoxCoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
