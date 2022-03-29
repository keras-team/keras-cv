from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers

from keras_cv.layers import preprocessing as cv_preprocessing
from keras_cv.utils import preprocessing as preprocessing_utils


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
        self.probability_to_apply = probability_to_apply

        policy = create_rand_augment_policy(magnitude)

        self.auto_contrast = cv_preprocessing.AutoContrast(**policy["auto_contrast"])
        self.equalize = cv_preprocessing.Equalization(**policy["equalize"])

        self.solarize = cv_preprocessing.Solarization(**policy["solarize"])
        self.solarize_add = cv_preprocessing.Solarization(**policy["solarize_add"])
        self.invert = cv_preprocessing.Solarization(**policy["invert"])

        self.color = cv_preprocessing.RandomColorDegeneration(**policy["color"])
        self.contrast = cv_preprocessing.RandomContrast(**policy["contrast"])
        self.brightness = cv_preprocessing.RandomBrightness(**policy["brightness"])
        self.shear_x = cv_preprocessing.RandomShear(**policy["shear_x"])
        self.shear_y = cv_preprocessing.RandomShear(**policy["shear_y"])
        self.translate_x = cv_preprocessing.RandomTranslation(**policy["translate_x"])
        self.translate_y = cv_preprocessing.RandomTranslation(**policy["translate_y"])
        self.cutout = cv_preprocessing.RandomCutout(**policy["cutout"])

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

    @tf.function
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

def auto_contrast_policy(magnitude):
    return {}


def equalize_policy(magnitude):
    return {}


def solarize_policy(magnitude):
    return {"threshold": magnitude/10 * 256}


def solarize_add_policy(magnitude):
    return {"addition": magnitude/10 * 110,"threshold": 128}


def invert_policy(magnitude):
    return {"addition": 0, "threshold": 0}


def color_policy(magnitude):
    return {"factor": (magnitude/10.0)}


def contrast_policy(magnitude):
    return {"factor": (magnitude/10.0)}


def brightness_policy(magnitude):
    return {"factor": (magnitude/10.0)}


def shear_x_policy(magnitude):
    return {"x_factor": magnitude/10, "y_factor": 0}


def shear_y_policy(magnitude):
    return {"x_factor": 0, "y_factor":  magnitude/10}


def translate_x_policy(magnitude):
    return {"width_factor": magnitude/10, "height_factor": 0}

def translate_y_policy(magnitude):
    return {"width_factor": 0, "height_factor":  magnitude/10}


def cutout_policy(magnitude):
    return {"width_factor": 0.5 * magnitude/10, "height_factor":  0.5 * magnitude/10}


policy_pairs = [
    ("auto_contrast", auto_contrast_policy),
    ("equalize", equalize_policy),
    ("solarize", solarize_policy),
    ("solarize_add", solarize_add_policy),
    ("invert", invert_policy),
    ("color", color_policy),
    ("contrast", contrast_policy),
    ("brightness", brightness_policy),
    ("shear_x", shear_x_policy),
    ("shear_y", shear_y_policy),
    ("translate_x", translate_x_policy),
    ("translate_y", translate_y_policy),
    ("cutout", cutout_policy),
]


def create_rand_augment_policy(magnitude):
    result = {}
    for name, policy_fn in policy_pairs:
        result[name] = policy_fn(magnitude)
    return result
