import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers


class CutMix(layers.Layer):
    """
    CutMix implements the CutMix data augmentation technique as proposed in https://arxiv.org/abs/1905.04899.

    Args:
        alpha: alpha parameter for the sample distribution.
        probability: probability to apply the CutMix augmentation.
        label_smoothing: coefficient used in label smoothing.

    Sample usage:
    ```python
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    cutmix = CutMix(10)
    augmented_data, updated_labels = cutmix(x_train, y_train)
    ```
    """

    def __init__(self, alpha=0.8, probability=1.0, label_smoothing=0.0, **kwargs):
        super(CutMix, self).__init__(*kwargs)
        self.alpha = alpha
        self.probability = probability
        self.label_smoothing = label_smoothing

    @staticmethod
    def _sample_from_beta(alpha, beta, shape):
        sample_alpha = tf.random.gamma(shape, 1.0, beta=alpha)
        sample_beta = tf.random.gamma(shape, 1.0, beta=beta)
        return sample_alpha / (sample_alpha + sample_beta)

    def call(self, images, labels):
        """
        call method for the CutMix layer.

        Args:
            images: Tensor representing images of shape [batch_size, width, height, channels].
            labels: One hot encoded tensor of labels for the images.
        Returns:
            images: augmented images, same shape as input.
            labels: updated labels with both label smoothing and the cutmix updates applied.
        """
        augment_cond = tf.less(
            tf.random.uniform(shape=[], minval=0.0, maxval=1.0), self.probability
        )
        # pylint: disable=g-long-lambda
        cutmix_augment = lambda: self._update_labels(*self._cutmix(images, labels))
        no_augment = lambda: (images, self._smooth_labels(labels))
        return tf.cond(augment_cond, cutmix_augment, no_augment)

    def _cutmix(self, images, labels):
        """Apply cutmix."""
        lambda_sample = CutMix._sample_from_beta(
            self.alpha, self.alpha, (tf.shape(labels)[0],)
        )

        ratio = tf.math.sqrt(1 - lambda_sample)

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
        lambda_sample = 1.0 - bbox_area / (image_height * image_width)
        lambda_sample = tf.cast(lambda_sample, dtype=tf.float32)

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

        return images, labels, lambda_sample

    def _update_labels(self, images, labels, lambda_sample):
        labels_1 = self._smooth_labels(labels)
        labels_2 = tf.reverse(labels_1, [0])

        lambda_sample = tf.reshape(lambda_sample, [-1, 1])
        labels = lambda_sample * labels_1 + (1.0 - lambda_sample) * labels_2

        return images, labels

    def _smooth_labels(self, labels):
        label_smoothing = self.label_smoothing or 0.0
        off_value = label_smoothing / tf.cast(tf.shape(labels)[1], tf.float32)
        on_value = 1.0 - label_smoothing + off_value
        return labels * on_value + (1 - labels) * off_value


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
