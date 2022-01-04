import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers


class RandomMixUp(layers.Layer):
    def __init__(
        self,
        alpha=0.8,
        probability=1.0,
        num_classes=None,
        label_smoothing=0.0,
        **kwargs
    ):
        super(RandomMixUp, self).__init__(*kwargs)
        if num_classes is None:
            raise ValueError(
                "num_classes is required.  Got MixUp.__init__(num_classes=None)"
            )
        self.alpha = alpha
        self.probability = probability
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing

    @staticmethod
    def _sample_from_beta(alpha, beta, shape):
        sample_alpha = tf.random.gamma(shape, 1.0, beta=alpha)
        sample_beta = tf.random.gamma(shape, 1.0, beta=beta)
        return sample_alpha / (sample_alpha + sample_beta)

    def call(self, inputs):
        images, labels = inputs
        augment_cond = tf.less(
            tf.random.uniform(shape=[], minval=0.0, maxval=1.0), self.probability
        )
        # pylint: disable=g-long-lambda
        augment_a = lambda: self._update_labels(*self._mixup(images, labels))
        no_augment = lambda: (images, self._smooth_labels(labels))
        return tf.cond(augment_cond, augment_a, no_augment)

    def _mixup(self, images, labels):
        lam = RandomMixUp._sample_from_beta(self.alpha, self.alpha, labels.shape)
        lam = tf.reshape(lam, [-1, 1, 1, 1])
        images = lam * images + (1.0 - lam) * tf.reverse(images, [0])

        return images, labels, tf.squeeze(lam)

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
