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

from keras_cv import core
from keras_cv.layers import preprocessing as cv_preprocessing
from keras_cv.utils import preprocessing as preprocessing_utils


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class RandAugment(tf.keras.__internal__.layers.BaseImageAugmentationLayer):
    """RandAugment performs the Rand Augment operation on input images.

    This layer can be thought of as an all in one image augmentation layer.  The policy
    implemented by this layer has been benchmarked extensively and is effective on a
    wide variety of datasets.

    The policy operates as follows:

    For each `layer`, the policy selects a random operation from a list of operations.
    It then samples a random number and if that number is less than
    `rate` applies it to the given image.

    References:
        - [RandAugment](https://arxiv.org/abs/1909.13719)

    Args:
        value_range: the range of values the incoming images will have.
            Represented as a two number tuple written [low, high].
            This is typically either `[0, 1]` or `[0, 255]` depending
            on how your preprocessing pipeline is setup.
        distortions: the number of layers to use in the rand augment policy.
        magnitude: the magnitude to use for each of the augmentation.  magnitude should
            be a float in the range `[0, 1]`.  A magnitude of `0` indicates that the
            augmentations are as weak as possible (not recommended), while a value of
            `1.0` implies use of the strongest possible augmentation.  Defaults to
            `0.5`.
        magnitude_stddev: the standard deviation to use when drawing values
            for the perturbations.  Keep in mind magnitude will still be clipped to the
            range `[0, 1]` after samples are drawn from the uniform distribution.
        rate:  the rate at which to apply each augmentation.  This parameter is applied
            on a per-distortion layer, per image.

    Usage:
    ```python
    (x_test, y_test), _ = tf.keras.datasets.cifar10.load_data()
    rand_augment = keras_cv.layers.RandAugment(
        value_range=(0, 255), distortions=3, magnitude=5.0
    )
    x_test = rand_augment(x_test)
    ```
    """

    def __init__(
        self,
        value_range,
        distortions=3,
        magnitude=0.5,
        magnitude_stddev=1.5,
        rate=None,
        seed=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.distortions = distortions
        self.magnitude = float(magnitude)
        self.value_range = value_range
        self.seed = seed
        if magnitude < 0.0 or magnitude > 1:
            raise ValueError(
                f"`magnitude` must be in the range [0, 1], got `magnitude={magnitude}`"
            )
        self.magnitude_stddev = float(magnitude_stddev)
        self.rate = rate

        policy = create_rand_augment_policy(magnitude, magnitude_stddev)

        self.auto_contrast = cv_preprocessing.AutoContrast(
            **policy["auto_contrast"], value_range=(0, 255), seed=seed
        )
        self.equalize = cv_preprocessing.Equalization(
            **policy["equalize"], value_range=(0, 255), seed=seed
        )

        self.solarize = cv_preprocessing.Solarization(
            **policy["solarize"], value_range=(0, 255), seed=seed
        )
        self.solarize_add = cv_preprocessing.Solarization(
            **policy["solarize_add"], value_range=(0, 255), seed=seed
        )
        self.invert = cv_preprocessing.Solarization(
            **policy["invert"], value_range=(0, 255), seed=seed
        )

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
        self.auto_vectorize = False

    def _augment(self, sample):
        sample["images"] = preprocessing_utils.transform_value_range(
            sample["images"], self.value_range, (0, 255)
        )
        augmented_sample = sample
        for _ in range(self.distortions):
            selected_op = self._random_generator.random_uniform(
                (), minval=0, maxval=len(self.augmentation_layers) + 1, dtype=tf.int32
            )
            branch_fns = []
            for (i, layer) in enumerate(self.augmentation_layers):
                branch_fns.append((i, lambda: layer(augmented_sample)))

            sample_augmented_by_this_layer = tf.switch_case(
                branch_index=selected_op,
                branch_fns=branch_fns,
                default=lambda: augmented_sample,
            )
            if self.rate is not None:
                augmented_sample = tf.cond(
                    self._random_generator.random_uniform(shape=(), dtype=tf.float32)
                    < self.rate,
                    lambda: sample_augmented_by_this_layer,
                    lambda: augmented_sample,
                )
            augmented_sample = sample_augmented_by_this_layer

        augmented_sample["images"] = preprocessing_utils.transform_value_range(
            augmented_sample["images"], (0, 255), self.value_range
        )
        return augmented_sample

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "value_range": self.value_range,
                "distortions": self.distortions,
                "magnitude": self.magnitude,
                "magnitude_stddev": self.magnitude_stddev,
                "rate": self.rate,
                "seed": self.seed,
            }
        )
        return config


def auto_contrast_policy(magnitude, magnitude_stddev):
    return {}


def equalize_policy(magnitude, magnitude_stddev):
    return {}


def solarize_policy(magnitude, magnitude_stddev):
    threshold_factor = core.NormalFactorSampler(
        mean=magnitude * 255,
        standard_deviation=magnitude_stddev * 256,
        min_value=0,
        max_value=255,
    )
    return {"threshold_factor": threshold_factor}


def solarize_add_policy(magnitude, magnitude_stddev):
    # We cap additions at 110, because if we add more than 110 we will be nearly
    # nullifying the information contained in the image, making the model train on noise
    maximum_addition_value = 110
    addition_factor = core.NormalFactorSampler(
        mean=magnitude * maximum_addition_value,
        standard_deviation=magnitude_stddev * maximum_addition_value,
        min_value=0,
        max_value=maximum_addition_value,
    )
    return {"addition_factor": addition_factor, "threshold_factor": 128}


def invert_policy(magnitude, magnitude_stddev):
    return {"addition_factor": 0, "threshold_factor": 0}


def color_policy(magnitude, magnitude_stddev):
    factor = core.NormalFactorSampler(
        mean=magnitude,
        standard_deviation=magnitude_stddev,
        min_value=0,
        max_value=1,
    )
    return {"factor": factor}


def contrast_policy(magnitude, magnitude_stddev):
    # TODO(lukewood): should we integrate RandomContrast with `factor`?
    # RandomContrast layer errors when factor=0
    factor = max(magnitude, 0.001)
    return {"factor": factor}


def brightness_policy(magnitude, magnitude_stddev):
    # TODO(lukewood): should we integrate RandomBrightness with `factor`?
    return {"factor": magnitude}


def shear_x_policy(magnitude, magnitude_stddev):
    factor = core.NormalFactorSampler(
        mean=magnitude,
        standard_deviation=magnitude_stddev,
        min_value=0,
        max_value=1,
    )
    return {"x_factor": factor, "y_factor": 0}


def shear_y_policy(magnitude, magnitude_stddev):
    factor = core.NormalFactorSampler(
        mean=magnitude,
        standard_deviation=magnitude_stddev,
        min_value=0,
        max_value=1,
    )
    return {"x_factor": 0, "y_factor": factor}


def translate_x_policy(magnitude, magnitude_stddev):
    # TODO(lukewood): should we integrate RandomTranslation with `factor`?
    return {"width_factor": magnitude, "height_factor": 0}


def translate_y_policy(magnitude, magnitude_stddev):
    # TODO(lukewood): should we integrate RandomTranslation with `factor`?
    return {"width_factor": 0, "height_factor": magnitude}


def cutout_policy(magnitude, magnitude_stddev):
    factor = core.NormalFactorSampler(
        mean=0.75 * magnitude,
        standard_deviation=0.75 * magnitude_stddev,
        min_value=0,
        max_value=1,
    )
    return {"width_factor": factor, "height_factor": factor}


POLICY_PAIRS = {
    "auto_contrast": auto_contrast_policy,
    "equalize": equalize_policy,
    "solarize": solarize_policy,
    "solarize_add": solarize_add_policy,
    "invert": invert_policy,
    "color": color_policy,
    "contrast": contrast_policy,
    "brightness": brightness_policy,
    "shear_x": shear_x_policy,
    "shear_y": shear_y_policy,
    "translate_x": translate_x_policy,
    "translate_y": translate_y_policy,
    "cutout": cutout_policy,
}


def create_rand_augment_policy(magnitude, magnitude_stddev):
    result = {}
    for name, policy_fn in POLICY_PAIRS.items():
        result[name] = policy_fn(magnitude, magnitude_stddev)
    return result
