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

from keras_cv import core
from keras_cv.layers import preprocessing
from keras_cv.layers import regularization


def custom_compare(obj1, obj2):
    if isinstance(obj1, core.FactorSampler):
        return obj1.get_config() == obj2.get_config()
    else:
        return obj1 == obj2


def config_equals(config1, config2):
    for key in list(config1.keys()) + list(config2.keys()):
        v1, v2 = config1[key], config2[key]
        if not custom_compare(v1, v2):
            return False
    return True


class SerializationTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(
        ("AutoContrast", preprocessing.AutoContrast, {"value_range": (0, 255)}),
        ("ChannelShuffle", preprocessing.ChannelShuffle, {"seed": 1}),
        ("CutMix", preprocessing.CutMix, {"seed": 1}),
        ("Equalization", preprocessing.Equalization, {"value_range": (0, 255)}),
        ("Grayscale", preprocessing.Grayscale, {}),
        ("GridMask", preprocessing.GridMask, {"seed": 1}),
        ("MixUp", preprocessing.MixUp, {"seed": 1}),
        (
            "RandomChannelShift",
            preprocessing.RandomChannelShift,
            {"value_range": (0, 255), "factor": 0.5},
        ),
        (
            "Posterization",
            preprocessing.Posterization,
            {"bits": 3, "value_range": (0, 255)},
        ),
        (
            "RandomColorDegeneration",
            preprocessing.RandomColorDegeneration,
            {"factor": 0.5, "seed": 1},
        ),
        (
            "RandomCutout",
            preprocessing.RandomCutout,
            {"height_factor": 0.2, "width_factor": 0.2, "seed": 1},
        ),
        (
            "RandomHue",
            preprocessing.RandomHue,
            {"factor": 0.5, "value_range": (0, 255), "seed": 1},
        ),
        (
            "RandomSaturation",
            preprocessing.RandomSaturation,
            {"factor": 0.5, "seed": 1},
        ),
        (
            "RandomSharpness",
            preprocessing.RandomSharpness,
            {"factor": 0.5, "value_range": (0, 255), "seed": 1},
        ),
        (
            "RandomShear",
            preprocessing.RandomShear,
            {"x_factor": 0.3, "x_factor": 0.3, "seed": 1},
        ),
        ("Solarization", preprocessing.Solarization, {"value_range": (0, 255)}),
        (
            "RandAugment",
            preprocessing.RandAugment,
            {
                "value_range": (0, 255),
                "magnitude": 0.5,
                "augmentations_per_image": 3,
                "rate": 0.3,
                "magnitude_stddev": 0.1,
            },
        ),
        (
            "RandomAugmentationPipeline",
            preprocessing.RandomAugmentationPipeline,
            {
                "layers": [],
                "augmentations_per_image": 1,
                "rate": 1.0,
            },
        ),
        (
            "RandomChoice",
            preprocessing.RandomChoice,
            {
                "layers": [],
                "seed": 3,
                "auto_vectorize": False,
            },
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
            "DropBlock2D",
            regularization.DropBlock2D,
            {
                "rate": 0.1,
                "block_size": (7, 7),
                "seed": 1234,
            },
        ),
        (
            "MaybeApply",
            preprocessing.MaybeApply,
            {
                "rate": 0.5,
                "layer": None,
                "seed": 1234,
            },
        ),
    )
    def test_layer_serialization(self, layer_cls, init_args):
        layer = layer_cls(**init_args)
        if "seed" in init_args:
            self.assertIn("seed", layer.get_config())

        model = tf.keras.models.Sequential(layer)
        model_config = model.get_config()

        reconstructed_model = tf.keras.Sequential().from_config(model_config)
        reconstructed_layer = reconstructed_model.layers[0]

        self.assertTrue(
            config_equals(layer.get_config(), reconstructed_layer.get_config())
        )
