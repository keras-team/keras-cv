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

from keras_cv import layers

TEST_CONFIGURATIONS = [
    ("AutoContrast", layers.AutoContrast, {"value_range": (0, 255)}),
    ("ChannelShuffle", layers.ChannelShuffle, {}),
    ("Equalization", layers.Equalization, {"value_range": (0, 255)}),
    (
        "RandomCropAndResize",
        layers.RandomCropAndResize,
        {
            "target_size": (224, 224),
            "crop_area_factor": (0.8, 1.0),
            "aspect_ratio_factor": (3 / 4, 4 / 3),
        },
    ),
    (
        "RandomlyZoomedCrop",
        layers.RandomlyZoomedCrop,
        {
            "height": 224,
            "width": 224,
            "zoom_factor": (0.8, 1.0),
            "aspect_ratio_factor": (3 / 4, 4 / 3),
        },
    ),
    ("Grayscale", layers.Grayscale, {}),
    ("GridMask", layers.GridMask, {}),
    (
        "Posterization",
        layers.Posterization,
        {"bits": 3, "value_range": (0, 255)},
    ),
    ("RandomBrightness", layers.RandomBrightness, {"factor": 0.5}),
    (
        "RandomColorDegeneration",
        layers.RandomColorDegeneration,
        {"factor": 0.5},
    ),
    (
        "RandomCutout",
        layers.RandomCutout,
        {"height_factor": 0.2, "width_factor": 0.2},
    ),
    (
        "RandomHue",
        layers.RandomHue,
        {"factor": 0.5, "value_range": (0, 255)},
    ),
    (
        "RandomChannelShift",
        layers.RandomChannelShift,
        {"value_range": (0, 255), "factor": 0.5},
    ),
    (
        "RandomColorJitter",
        layers.RandomColorJitter,
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
        layers.RandomGaussianBlur,
        {"kernel_size": 3, "factor": (0.0, 3.0)},
    ),
    ("RandomJpegQuality", layers.RandomJpegQuality, {"factor": (75, 100)}),
    ("RandomSaturation", layers.RandomSaturation, {"factor": 0.5}),
    (
        "RandomSharpness",
        layers.RandomSharpness,
        {"factor": 0.5, "value_range": (0, 255)},
    ),
    ("RandomAspectRatio", layers.RandomAspectRatio, {"factor": (0.9, 1.1)}),
    ("RandomShear", layers.RandomShear, {"x_factor": 0.3, "x_factor": 0.3}),
    ("Solarization", layers.Solarization, {"value_range": (0, 255)}),
    ("Mosaic", layers.Mosaic, {}),
    ("CutMix", layers.CutMix, {}),
    ("MixUp", layers.MixUp, {}),
    (
        "Resizing",
        layers.Resizing,
        {
            "height": 224,
            "width": 224,
        },
    ),
    (
        "JitteredResize",
        layers.JitteredResize,
        {
            "target_size": (224, 224),
            "scale_factor": (0.8, 1.25),
            "bounding_box_format": "xywh",
        },
    ),
    (
        "RandomZoom",
        layers.RandomZoom,
        {"height_factor": 0.2, "width_factor": 0.5},
    ),
    (
        "RandomCrop",
        layers.RandomCrop,
        {"height": 224, "width": 224},
    ),
    (
        "Rescaling",
        layers.Rescaling,
        {
            "scale": 1,
            "offset": 0.5,
        },
    ),
]

NO_CPU_FP16_KERNEL_LAYERS = [
    layers.RandomSaturation,
    layers.RandomColorJitter,
    layers.RandomHue,
    layers.RandomContrast,
]


class WithMixedPrecisionTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(*TEST_CONFIGURATIONS)
    def test_can_run_in_mixed_precision(self, layer_cls, init_args):
        if not tf.config.list_physical_devices("GPU"):
            if layer_cls in NO_CPU_FP16_KERNEL_LAYERS:
                self.skipTest(
                    "There is currently no float16 CPU kernel registered for operations"
                    " `tf.image.adjust_saturation`, and `tf.image.adjust_hue`. "
                    "Skipping."
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
        # Do not affect other tests
        tf.keras.mixed_precision.set_global_policy("float32")


if __name__ == "__main__":
    tf.test.main()
