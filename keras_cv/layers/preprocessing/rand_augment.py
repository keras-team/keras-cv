from tensorflow import keras
from tensorflow.keras import layers

from keras_cv.layers import preprocessing as cv_preprocessing
from keras_cv.utils import preprocessing as preprocessing_utils

class _RandomAugmentTransformation:
    selected_augments = []

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

        # TODO autogenerate parameters based on magnitude.

        self.auto_contrast = cv_preprocessing.AutoContrast()
        self.equalize = cv_preprocessing.Equalize()

        # solarize = solarize add layer
        self.solarize = cv_preprocessing.Solarization(addition=addition)
        self.solarize_add = cv_preprocessing.Solarization()
        self.invert = cv_preprocessing.Solarization()
        self.color = cv_preprocessing.RandomColorDegeneration()
        self.contrast = cv_preprocessing.RandomContrast()
        self.brightness = cv_preprocessing.RandomBrightness()
        self.shear_x = cv_preprocessing.RandomShear()
        self.shear_y = cv_preprocessing.RandomShear()
        self.translate_x = cv_preprocessing.RandomTranslate()
        self.translate_y = cv_preprocessing.RandomTranslate()
        self.cutout = cv_preprocessing.RandomCutout()

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
            self.cutout
        ]

    def get_random_transformation(self, image=None, label=None, bounding_box=None):
        pass

    def augment_image(self, image, transformation=None):
        branch_fns = []
        for augmentation_layer in self.augmentation_layers:
            lambda img: augmentation_layer.augment_image(img, transformation=augmentation_layer.get_random_transformation())

    def call(self, inputs):
        batch=tf.shape(images)
        minibatches = tf.random.uniform((), min_value=0, max_value=len(self.augmentation_layers))
