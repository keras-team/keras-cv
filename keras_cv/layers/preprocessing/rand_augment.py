from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers

from keras_cv.layers import preprocessing as cv_preprocessing
from keras_cv.utils import preprocessing as preprocessing_utils

class RandAugmentPolicy:
    pass

class RandAugment(keras.layers.Layer):
    """RandAugment performs the Rand Augment operation on input images.

    This layer can be thought of as an all in one image augmentation layer.  The policy
    implemented by this layer has been benchmarked extensively and is effective on a
    wide variety of datasets.

    The policy operates as follows:

    For each `layer`, the policy selects a random operation from a list of operations.
    It then samples a random number and if that number is less than `prob_to_apply`
    applies it to the given image.

    References:

    Args:
        num_layers:
        magnitude:
        probability_to_apply:
        value_range:

    Usage:
    ```python
    ```
    """

    def __init__(
        self,
        num_layers=3,
        magnitude=7.0,
        probability_to_apply=None,
        value_range=(0, 255),
    ):
        super().__init__()
        self.num_layers = num_layers
        self.magnitude = magnitude
        addition = magnitude * 2
        # TODO autogenerate parameters based on magnitude.

        self.auto_contrast = cv_preprocessing.AutoContrast()
        self.equalize = cv_preprocessing.Equalization()

        # solarize = solarize add layer
        self.solarize = cv_preprocessing.Solarization()
        self.solarize_add = cv_preprocessing.Solarization(addition=addition)
        self.invert = cv_preprocessing.Solarization()
        self.color = cv_preprocessing.RandomColorDegeneration(0.5)
        self.contrast = cv_preprocessing.RandomContrast(0.5)
        self.brightness = cv_preprocessing.RandomBrightness(0.5)
        self.shear_x = cv_preprocessing.RandomShear(x=0.5)
        self.shear_y = cv_preprocessing.RandomShear(y=0.5)
        self.translate_x = cv_preprocessing.RandomTranslation(width_factor=0.2, height_factor=0)
        self.translate_y = cv_preprocessing.RandomTranslation(height_factor=0.2, width_factor=0)
        self.cutout = cv_preprocessing.RandomCutout(height_factor=0.2, width_factor=0.2)

        self.augmentation_layers = [
            self.auto_contrast,
            self.equalize,
            self.solarize,
            self.solarize_add,
            self.invert,
            self.color,
            self.contrast,
            self.brightness,
            self.shear_x,
            self.shear_y,
            self.translate_x,
            self.translate_y,
            self.cutout,
        ]

    def augment_sample(self, sample):
        for _ in range(self.num_layers):
            selected_op = tf.random.uniform(
                (), maxval=len(self.augmentation_layers) + 1, dtype=tf.int32
            )
            branch_fns = []
            for (i, layer) in enumerate(self.augmentation_layers):
                branch_fns.append((i, lambda: layer(sample)))

            augmented_sample = tf.switch_case(
                branch_index=selected_op, branch_fns=branch_fns, default=lambda: sample
            )
            if self.probability_to_apply is not None:
                augmented_sample = tf.cond(
                    tf.random.uniform(shape=(), dtype=tf.float32)
                    < self.probability_to_apply,
                    lambda: augmented_sample,
                    lambda: sample,
                )
            sample = augmented_sample
        return sample

    def call(self, inputs):
        return tf.map_fn(lambda sample: self.augment_sample(sample), inputs)
