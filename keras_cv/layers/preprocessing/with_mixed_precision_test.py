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
from absl.testing import parameterized

from keras_cv.layers import preprocessing

TEST_CONFIGURATIONS = [
    ("AutoContrast", preprocessing.AutoContrast, {"value_range": (0, 255)}),
    ("ChannelShuffle", preprocessing.ChannelShuffle, {}),
    ("Equalization", preprocessing.Equalization, {"value_range": (0, 255)}),
    (
        "RandomCropAndResize",
        preprocessing.RandomCropAndResize,
        {
            "target_size": (224, 224),
            "crop_area_factor": (0.8, 1.0),
            "aspect_ratio_factor": (3 / 4, 4 / 3),
        },
    ),
    (
        "RandomlyZoomedCrop",
        preprocessing.RandomlyZoomedCrop,
        {
            "height": 224,
            "width": 224,
            "zoom_factor": (0.8, 1.0),
            "aspect_ratio_factor": (3 / 4, 4 / 3),
        },
    ),
    ("Grayscale", preprocessing.Grayscale, {}),
    ("GridMask", preprocessing.GridMask, {}),
    (
        "Posterization",
        preprocessing.Posterization,
        {"bits": 3, "value_range": (0, 255)},
    ),
    (
        "RandomColorDegeneration",
        preprocessing.RandomColorDegeneration,
        {"factor": 0.5},
    ),
    (
        "RandomCutout",
        preprocessing.RandomCutout,
        {"height_factor": 0.2, "width_factor": 0.2},
    ),
    (
        "RandomHue",
        preprocessing.RandomHue,
        {"factor": 0.5, "value_range": (0, 255)},
    ),
    (
        "RandomChannelShift",
        preprocessing.RandomChannelShift,
        {"value_range": (0, 255), "factor": 0.5},
    ),
    (
        "RandomColorJitter",
        preprocessing.RandomColorJitter,
        {
            "value_range": (0, 255),
            "brightness_factor": (-0.2, 0.5),
            "contrast_factor": (0.5, 0.9),
            "saturation_factor": (0.5, 0.9),
            "hue_factor": (0.5, 0.9),
            "seed": 1,
        },
    ),
    (
        "RandomGaussianBlur",
        preprocessing.RandomGaussianBlur,
        {"kernel_size": 3, "factor": (0.0, 3.0)},
    ),
    ("RandomJpegQuality", preprocessing.RandomJpegQuality, {"factor": (75, 100)}),
    ("RandomSaturation", preprocessing.RandomSaturation, {"factor": 0.5}),
    (
        "RandomSharpness",
        preprocessing.RandomSharpness,
        {"factor": 0.5, "value_range": (0, 255)},
    ),
    ("RandomShear", preprocessing.RandomShear, {"x_factor": 0.3, "x_factor": 0.3}),
    ("Solarization", preprocessing.Solarization, {"value_range": (0, 255)}),
]

NO_CPU_FP16_KERNEL_LAYERS = [
    preprocessing.RandomSaturation,
    preprocessing.RandomColorJitter,
]


class WithMixedPrecisionTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(
        *TEST_CONFIGURATIONS,
        ("CutMix", preprocessing.CutMix, {}),
        ("Mosaic", preprocessing.Mosaic, {}),
    )
    def test_can_run_in_mixed_precision(self, layer_cls, init_args):
        if not tf.config.list_physical_devices("GPU"):
            if layer_cls in NO_CPU_FP16_KERNEL_LAYERS:
                self.skipTest(
                    "`RandomSaturation` and `RandomColorJitter` both use "
                    "`tf.image.adjust_saturation`, which doesn't have FLOAT16 CPU "
                    "kernel registered. Skipping."
                )

        tf.keras.mixed_precision.set_global_policy("mixed_float16")

        img = tf.random.uniform(
            shape=(3, 512, 512, 3), minval=0, maxval=255, dtype=tf.float32
        )
        labels = tf.ones((3,), dtype=tf.float32)
        inputs = {"images": img, "labels": labels}

        layer = layer_cls(**init_args)
        layer(inputs)

    @classmethod
    def tearDownClass(cls) -> None:
        # Do NOT affect other tests
        tf.keras.mixed_precision.set_global_policy("float32")


if __name__ == "__main__":
    tf.test.main()
