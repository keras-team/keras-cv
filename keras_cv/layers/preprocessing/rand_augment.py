from tensorflow import keras
from tensorflow.keras import layers

from keras_cv.layers import preprocessing as cv_preprocessing
from keras_cv.utils import preprocessing as preprocessing_utils


class _SolarizationAdd(keras.__internal__.layers.BaseImageAugmentationLayer):
    def __init__(threshold, addition, value_range=(0, 255) ** kwargs):
        super().__init__(**kwargs)
        self.addition = addition
        self.value_range = value_range
        self.solarization = cv_preprocessing.Solarization(threshold=threshold)

    def augment_image(self, image, transformation=None):
        image = utils.transform_value_range(
            image, original_range=self.value_range, target_range=(0, 255)
        )
        image = tf.clip_by_value(image + self.addition, 0, 255)
        result = self.solarization(image)
        result = utils.transform_value_range(
            result, original_range=(0, 255), target_range=self.value_range
        )
        return result


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

    Usage:
    ```python
    ```


    """

    def __init__(self,
        num_layers=3,
        magnitude=7.0,
        value_range=(0, 255),
        prob_to_apply=None):

        self.num_layers = num_layers
        self.magnitude = magnitude

        self.auto_contrast = cv_preprocessing.AutoContrast()
        self.equalize = cv_preprocessing.Equalize()

        # solarize = solarize add layer
        self.solarize = cv_preprocessing.Solarization()
        self.invert = cv_preprocessing.Solarization()
        self.color = cv_preprocessing.RandomColorDegeneration()
        self.contrast = cv_preprocessing.RandomContrast()
        self.brightness = cv_preprocessing.RandomBrightness()
        self.shear_x = cv_preprocessing.RandomShear()
        self.shear_y = cv_preprocessing.RandomShear()
        self.translate_x = cv_preprocessing.RandomTranslate()
        self.translate_y = cv_preprocessing.RandomTranslate()
        self.cutout = cv_preprocessing.RandomCutout()

    def call(self, inputs):
        pass
