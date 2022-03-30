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


class SerializationTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(
        ("AutoContrast", preprocessing.AutoContrast, {}),
        ("ChannelShuffle", preprocessing.ChannelShuffle, {}),
        ("CutMix", preprocessing.CutMix, {}),
        ("Equalization", preprocessing.Equalization, {}),
        ("Grayscale", preprocessing.Grayscale, {}),
        ("GridMask", preprocessing.GridMask, {}),
        ("MixUp", preprocessing.MixUp, {}),
        ("Posterization", preprocessing.Posterization, {"bits": 3}),
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
        ("RandomHue", preprocessing.RandomHue, {"factor": 0.5}),
        ("RandomSaturation", preprocessing.RandomSaturation, {"factor": 0.5}),
        ("RandomSharpness", preprocessing.RandomSharpness, {"factor": 0.5}),
        ("RandomShear", preprocessing.RandomShear, {"x": 0.3, "y": 0.3}),
        ("Solarization", preprocessing.Solarization, {}),
    )
    def test_layer_serialization(self, layer_cls, init_args):
        layer = layer_cls(**init_args)
        model = tf.keras.models.Sequential(layer)
        model_config = model.get_config()
        reconstructed_model = tf.keras.Sequential().from_config(model_config)
        reconstructed_layer = reconstructed_model.layers[0]
        self.assertEqual(layer.get_config(), reconstructed_layer.get_config())
