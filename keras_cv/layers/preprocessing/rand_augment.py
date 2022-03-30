# Copyright 2022 The KerasCV Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import tensorflow as tf
from tensorflow import keras

from keras_cv import core
from keras_cv.layers import preprocessing as cv_preprocessing
from keras_cv.utils import preprocessing as preprocessing_utils


class RandAugment(keras.layers.Layer):
    """RandAugment performs the Rand Augment operation on input images.

    This layer can be thought of as an all in one image augmentation layer.  The policy
    implemented by this layer has been benchmarked extensively and is effective on a
    wide variety of datasets.

    The policy operates as follows:

    For each `layer`, the policy selects a random operation from a list of operations.
    It then samples a random number and if that number is less than
    `probability_to_apply` applies it to the given image.

    References:
        - [RandAugment](https://arxiv.org/abs/1909.13719)

    Args:
        num_layers: the number of layers to use in the rand augment policy.
        magnitude: the magnitude to use for each of the augmentation.
        magnitude_standard_deviation: the standard deviation to use when drawing values
            for the perturbations.
        probability_to_apply:  the probability to apply an augmentation at each layer.
        value_range: the value range of the incoming images

    Usage:
    # TODO(lukewood): document fully.
    ```python
    ```
    """

    def __init__(
        self,
        num_layers=3,
        magnitude=5.0,
        magnitude_standard_deviation=1.5,
        probability_to_apply=None,
        value_range=(0, 255),
        seed=None,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.magnitude = magnitude
        self.value_range = value_range
        if magnitude > 10.0:
            raise ValueError(
                f"`magnitude` must be in the range [0, 10], got `magnitude={magnitude}`"
            )
        self.magnitude_standard_deviation = magnitude_standard_deviation
        self.probability_to_apply = probability_to_apply

        policy = create_rand_augment_policy(magnitude, magnitude_standard_deviation)

        self.auto_contrast = cv_preprocessing.AutoContrast(
            **policy["auto_contrast"], seed=seed
        )
        self.equalize = cv_preprocessing.Equalization(**policy["equalize"], seed=seed)

        self.solarize = cv_preprocessing.Solarization(**policy["solarize"], seed=seed)
        self.solarize_add = cv_preprocessing.Solarization(
            **policy["solarize_add"], seed=seed
        )
        self.invert = cv_preprocessing.Solarization(**policy["invert"], seed=seed)

        self.color = cv_preprocessing.RandomColorDegeneration(
            **policy["color"], seed=seed
        )
        self.contrast = cv_preprocessing.RandomContrast(**policy["contrast"], seed=seed)
        self.brightness = cv_preprocessing.RandomBrightness(
            **policy["brightness"], seed=seed
        )
        self.shear_x = cv_preprocessing.RandomShear(**policy["shear_x"], seed=seed)
        self.shear_y = cv_preprocessing.RandomShear(**policy["shear_y"], seed=seed)
        self.translate_x = cv_preprocessing.RandomTranslation(
            **policy["translate_x"], seed=seed
        )
        self.translate_y = cv_preprocessing.RandomTranslation(
            **policy["translate_y"], seed=seed
        )
        self.cutout = cv_preprocessing.RandomCutout(**policy["cutout"], seed=seed)

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
        inputs = tf.cast(inputs, self.compute_dtype)
        inputs = preprocessing_utils.transform_value_range(
            inputs, self.value_range, (0, 255)
        )
        result = tf.map_fn(lambda sample: self.augment_sample(sample), inputs)
        result = preprocessing_utils.transform_value_range(
            result, (0, 255), self.value_range
        )
        return result

    def get_config(self):
        return {
            "num_layers": self.num_layers,
            "magnitude": self.magnitude,
            "magnitude_standard_deviation": self.magnitude_standard_deviation,
            "probability_to_apply": self.probability_to_apply,
            "value_range": self.value_range,
        }


def auto_contrast_policy(magnitude, magnitude_std):
    return {}


def equalize_policy(magnitude, magnitude_std):
    return {}


def solarize_policy(magnitude, magnitude_std):
    # TODO(lukewood): this should support a sample-able factor.
    return {"threshold": magnitude / 10 * 256}


def solarize_add_policy(magnitude, magnitude_std):
    # TODO(lukewood): this should support a sample-able factor.
    return {"addition": magnitude / 10 * 110, "threshold": 128}


def invert_policy(magnitude, magnitude_std):
    return {"addition": 0, "threshold": 0}


def color_policy(magnitude, magnitude_std):
    factor = core.NormalFactor(
        mean=magnitude / 10.0,
        standard_deviation=magnitude_std / 10.0,
        min_value=0,
        max_value=1,
    )
    return {"factor": factor}


def contrast_policy(magnitude, magnitude_std):
    # TODO(lukewood): should we integrate RandomContrast with `factor`?
    # RandomContrast layer errors when factor=0
    factor = max(magnitude / 10, 0.001)
    return {"factor": factor}


def brightness_policy(magnitude, magnitude_std):
    # TODO(lukewood): should we integrate RandomBrightness with `factor`?
    return {"factor": magnitude / 10.0}


def shear_x_policy(magnitude, magnitude_std):
    factor = core.NormalFactor(
        mean=magnitude / 10.0,
        standard_deviation=magnitude_std / 10.0,
        min_value=0,
        max_value=1,
    )
    return {"x_factor": factor, "y_factor": 0}


def shear_y_policy(magnitude, magnitude_std):
    factor = core.NormalFactor(
        mean=magnitude / 10.0,
        standard_deviation=magnitude_std / 10.0,
        min_value=0,
        max_value=1,
    )
    return {"x_factor": 0, "y_factor": factor}


def translate_x_policy(magnitude, magnitude_std):
    # TODO(lukewood): should we integrate RandomTranslation with `factor`?
    return {"width_factor": magnitude / 10, "height_factor": 0}


def translate_y_policy(magnitude, magnitude_std):
    # TODO(lukewood): should we integrate RandomTranslation with `factor`?
    return {"width_factor": 0, "height_factor": magnitude / 10}


def cutout_policy(magnitude, magnitude_std):
    factor = core.NormalFactor(
        mean=(magnitude / 10.0),
        standard_deviation=(magnitude_std / 10.0),
        min_value=0,
        max_value=1,
    )
    return {"width_factor": factor, "height_factor": factor}


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


def create_rand_augment_policy(magnitude, magnitude_standard_deviation):
    result = {}
    for name, policy_fn in policy_pairs:
        result[name] = policy_fn(magnitude, magnitude_standard_deviation)
    return result
