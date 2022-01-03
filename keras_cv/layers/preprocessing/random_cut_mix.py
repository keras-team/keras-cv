import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers


class RandomCutMix(layers.Layer):
    def __init__(
        self, num_classes, alpha=0.8, probability=1.0, label_smoothing=0.0, **kwargs
    ):
        super(RandomCutMix, self).__init__(*kwargs)
        self.alpha = alpha
        self.probability = probability
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing

    @staticmethod
    def _sample_from_beta(alpha, beta, shape):
        sample_alpha = tf.random.gamma(shape, 1.0, beta=alpha)
        sample_beta = tf.random.gamma(shape, 1.0, beta=beta)
        return sample_alpha / (sample_alpha + sample_beta)

    def call(self, images, labels):
        augment_cond = tf.less(
            tf.random.uniform(shape=[], minval=0.0, maxval=1.0), self.probability
        )
        # pylint: disable=g-long-lambda
        augment_a = lambda: self._update_labels(*self._cutmix(images, labels))
        no_augment = lambda: (images, self._smooth_labels(labels))
        return tf.cond(augment_cond, augment_a, no_augment)

    def _cutmix(self, images, labels):
        """Apply cutmix."""
        lam = RandomCutMix._sample_from_beta(
            self.alpha, self.alpha, labels.shape
        )

        ratio = tf.math.sqrt(1 - lam)

        batch_size = tf.shape(images)[0]
        image_height, image_width = tf.shape(images)[1], tf.shape(images)[2]

        cut_height = tf.cast(
            ratio * tf.cast(image_height, dtype=tf.float32), dtype=tf.int32
        )
        cut_width = tf.cast(
            ratio * tf.cast(image_height, dtype=tf.float32), dtype=tf.int32
        )

        random_center_height = tf.random.uniform(
            shape=[batch_size], minval=0, maxval=image_height, dtype=tf.int32
        )
        random_center_width = tf.random.uniform(
            shape=[batch_size], minval=0, maxval=image_width, dtype=tf.int32
        )

        bbox_area = cut_height * cut_width
        lam = 1.0 - bbox_area / (image_height * image_width)
        lam = tf.cast(lam, dtype=tf.float32)

        images = tf.map_fn(
            lambda x: _fill_rectangle(*x),
            (
                images,
                random_center_width,
                random_center_height,
                cut_width // 2,
                cut_height // 2,
                tf.reverse(images, [0]),
            ),
            dtype=(tf.float32, tf.int32, tf.int32, tf.int32, tf.int32, tf.float32),
            fn_output_signature=tf.TensorSpec(images.shape[1:], dtype=tf.float32),
        )

        return images, labels, lam

    def _update_labels(self, images, labels, lam):
        labels_1 = self._smooth_labels(labels)
        labels_2 = tf.reverse(labels_1, [0])

        lam = tf.reshape(lam, [-1, 1])
        labels = lam * labels_1 + (1.0 - lam) * labels_2

        return images, labels

    def _smooth_labels(self, labels):
        label_smoothing = self.label_smoothing or 0.0
        off_value = label_smoothing / self.num_classes
        on_value = 1.0 - label_smoothing + off_value

        smooth_labels = tf.one_hot(
            labels, self.num_classes, on_value=on_value, off_value=off_value
        )
        return smooth_labels


def _fill_rectangle(
    image, center_width, center_height, half_width, half_height, replace=None
):
    """Fill blank area."""
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]

    lower_pad = tf.maximum(0, center_height - half_height)
    upper_pad = tf.maximum(0, image_height - center_height - half_height)
    left_pad = tf.maximum(0, center_width - half_width)
    right_pad = tf.maximum(0, image_width - center_width - half_width)

    cutout_shape = [
        image_height - (lower_pad + upper_pad),
        image_width - (left_pad + right_pad),
    ]
    padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]
    mask = tf.pad(
        tf.zeros(cutout_shape, dtype=image.dtype), padding_dims, constant_values=1
    )
    mask = tf.expand_dims(mask, -1)
    mask = tf.tile(mask, [1, 1, 3])

    if replace is None:
        fill = tf.random.normal(tf.shape(image), dtype=image.dtype)
    elif isinstance(replace, tf.Tensor):
        fill = replace
    else:
        fill = tf.ones_like(image, dtype=image.dtype) * replace
    image = tf.where(tf.equal(mask, 0), fill, image)
    return image
