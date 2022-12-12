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

CONSISTENT_OUTPUT_TEST_CONFIGURATIONS = [
    ("AutoContrast", layers.AutoContrast, {"value_range": (0, 255)}),
    ("ChannelShuffle", layers.ChannelShuffle, {}),
    ("Equalization", layers.Equalization, {"value_range": (0, 255)}),
    ("Grayscale", layers.Grayscale, {}),
    ("GridMask", layers.GridMask, {}),
    (
        "Posterization",
        layers.Posterization,
        {"bits": 3, "value_range": (0, 255)},
    ),
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
    ("RandomFlip", layers.RandomFlip, {"mode": "horizontal"}),
    ("RandomJpegQuality", layers.RandomJpegQuality, {"factor": (75, 100)}),
    ("RandomSaturation", layers.RandomSaturation, {"factor": 0.5}),
    (
        "RandomSharpness",
        layers.RandomSharpness,
        {"factor": 0.5, "value_range": (0, 255)},
    ),
    ("RandomShear", layers.RandomShear, {"x_factor": 0.3, "x_factor": 0.3}),
    ("Solarization", layers.Solarization, {"value_range": (0, 255)}),
]

DENSE_OUTPUT_TEST_CONFIGURATIONS = [
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
        "RandomlyZoomedCrop",
        layers.RandomlyZoomedCrop,
        {
            "height": 224,
            "width": 224,
            "zoom_factor": (0.8, 1.0),
            "aspect_ratio_factor": (3 / 4, 4 / 3),
        },
    ),
]

RAGGED_OUTPUT_TEST_CONFIGURATIONS = [
    ("RandomAspectRatio", layers.RandomAspectRatio, {"factor": (0.9, 1.1)}),
]


class RaggedImageTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(*CONSISTENT_OUTPUT_TEST_CONFIGURATIONS)
    def test_preserves_ragged_status(self, layer_cls, init_args):
        layer = layer_cls(**init_args)
        inputs = tf.ragged.stack(
            [
                tf.ones((512, 512, 3)),
                tf.ones((600, 300, 3)),
            ]
        )
        outputs = layer(inputs)
        self.assertTrue(isinstance(outputs, tf.RaggedTensor))

    @parameterized.named_parameters(*DENSE_OUTPUT_TEST_CONFIGURATIONS)
    def test_converts_ragged_to_dense(self, layer_cls, init_args):
        layer = layer_cls(**init_args)
        inputs = tf.ragged.stack(
            [
                tf.ones((512, 512, 3)),
                tf.ones((600, 300, 3)),
            ]
        )
        outputs = layer(inputs)
        self.assertTrue(isinstance(outputs, tf.Tensor))

    @parameterized.named_parameters(*RAGGED_OUTPUT_TEST_CONFIGURATIONS)
    def test_dense_to_ragged(self, layer_cls, init_args):
        layer = layer_cls(**init_args)
        inputs = tf.ones((8, 512, 512, 3))
        outputs = layer(inputs)
        self.assertTrue(isinstance(outputs, tf.RaggedTensor))
