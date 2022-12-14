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
        "RandomHue",
        preprocessing.RandomHue,
        {"factor": 0.5, "value_range": (0, 255)},
    ),
    ("RandomBrightness", preprocessing.RandomBrightness, {"factor": 0.5}),
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
    ("RandomContrast", preprocessing.RandomContrast, {"factor": 0.5}),
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
    ("Solarization", preprocessing.Solarization, {"value_range": (0, 255)}),
]


class WithSegmentationMasksTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(*TEST_CONFIGURATIONS)
    def test_can_run_with_segmentation_masks(self, layer_cls, init_args):
        classes = 10
        layer = layer_cls(**init_args)

        img = tf.random.uniform(
            shape=(3, 512, 512, 3), minval=0, maxval=1, dtype=tf.float32
        )
        segmentation_masks = tf.random.uniform(
            shape=(3, 512, 512, 1), minval=0, maxval=classes, dtype=tf.int32
        )

        inputs = {"images": img, "segmentation_masks": segmentation_masks}
        outputs = layer(inputs)

        self.assertIn("segmentation_masks", outputs)
        # This currently asserts that all layers are no-ops.
        # When preprocessing layers are updated to mutate segmentation masks,
        # this condition should only be asserted for no-op layers.
        self.assertAllClose(inputs["segmentation_masks"], outputs["segmentation_masks"])

    # This has to be a separate test case to exclude CutMix and MixUp
    # (which are not yet supported for segmentation mask augmentation)
    @parameterized.named_parameters(*TEST_CONFIGURATIONS)
    def test_can_run_with_segmentation_mask_single_image(self, layer_cls, init_args):
        classes = 10
        layer = layer_cls(**init_args)
        img = tf.random.uniform(
            shape=(512, 512, 3), minval=0, maxval=1, dtype=tf.float32
        )
        segmentation_mask = tf.random.uniform(
            shape=(512, 512, 1), minval=0, maxval=classes, dtype=tf.int32
        )

        inputs = {"images": img, "segmentation_masks": segmentation_mask}
        outputs = layer(inputs)

        self.assertIn("segmentation_masks", outputs)
        # This currently asserts that all layers are no-ops.
        # When preprocessing layers are updated to mutate segmentation masks,
        # this condition should only be asserted for no-op layers.
        self.assertAllClose(inputs["segmentation_masks"], outputs["segmentation_masks"])
