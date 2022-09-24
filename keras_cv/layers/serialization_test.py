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
import inspect

import tensorflow as tf
from absl.testing import parameterized

from keras_cv import core
from keras_cv import layers as cv_layers
from keras_cv.models.segmentation.__internal__ import SegmentationHead


def exhaustive_compare(obj1, obj2):
    classes_supporting_get_config = (
        core.FactorSampler,
        tf.keras.layers.Layer,
        cv_layers.BaseImageAugmentationLayer,
    )

    # If both objects are either one of list or tuple then their individual
    # elements also must be checked exhaustively.
    if isinstance(obj1, (list, tuple)) and isinstance(obj2, (list, tuple)):
        # Length based checks.
        if len(obj1) == 0 and len(obj2) == 0:
            return True
        if len(obj1) != len(obj2):
            return False

        # Exhaustive check for all elements.
        for v1, v2 in list(zip(obj1, obj2)):
            return exhaustive_compare(v1, v2)

    # If the objects are dicts then we simply call the `config_equals` function
    # which supports dicts.
    elif isinstance(obj1, (dict)) and isinstance(obj2, (dict)):
        return config_equals(v1, v2)

    # If both objects are subclasses of Keras classes that support `get_config`
    # method, then we compare their individual attributes using `config_equals`.
    elif isinstance(obj1, classes_supporting_get_config) and isinstance(
        obj2, classes_supporting_get_config
    ):
        return config_equals(obj1.get_config(), obj2.get_config())

    # Following checks are if either of the objects are _functions_, not methods
    # or callables, since Layers and other unforeseen objects may also fit into
    # this category. Specifically for Keras activation functions.
    elif inspect.isfunction(obj1) and inspect.isfunction(obj2):
        return tf.keras.utils.serialize_keras_object(
            obj1
        ) == tf.keras.utils.serialize_keras_object(obj2)
    elif inspect.isfunction(obj1) and not inspect.isfunction(obj2):
        return tf.keras.utils.serialize_keras_object(obj1) == obj2
    elif inspect.isfunction(obj2) and not inspect.isfunction(obj1):
        return obj1 == tf.keras.utils.serialize_keras_object(obj2)

    # Lastly check for primitive datatypes and objects that don't need
    # additional preprocessing.
    else:
        return obj1 == obj2


def config_equals(config1, config2):
    # Both `config1` and `config2` are python dicts. So the first check is to
    # see if both of them have same keys.
    if config1.keys() != config2.keys():
        return False

    # Iterate over all keys of the configs and compare each entry exhaustively.
    for key in list(config1.keys()):
        v1, v2 = config1[key], config2[key]
        if not exhaustive_compare(v1, v2):
            return False
    return True


class SerializationTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(
        ("Augmenter", cv_layers.Augmenter, {"layers": [cv_layers.Grayscale()]}),
        ("AutoContrast", cv_layers.AutoContrast, {"value_range": (0, 255)}),
        ("ChannelShuffle", cv_layers.ChannelShuffle, {"seed": 1}),
        ("CutMix", cv_layers.CutMix, {"seed": 1}),
        ("Equalization", cv_layers.Equalization, {"value_range": (0, 255)}),
        ("Grayscale", cv_layers.Grayscale, {}),
        ("GridMask", cv_layers.GridMask, {"seed": 1}),
        ("MixUp", cv_layers.MixUp, {"seed": 1}),
        (
            "RandomChannelShift",
            cv_layers.RandomChannelShift,
            {"value_range": (0, 255), "factor": 0.5},
        ),
        (
            "Posterization",
            cv_layers.Posterization,
            {"bits": 3, "value_range": (0, 255)},
        ),
        (
            "RandomColorDegeneration",
            cv_layers.RandomColorDegeneration,
            {"factor": 0.5, "seed": 1},
        ),
        (
            "RandomCutout",
            cv_layers.RandomCutout,
            {"height_factor": 0.2, "width_factor": 0.2, "seed": 1},
        ),
        (
            "RandomHue",
            cv_layers.RandomHue,
            {"factor": 0.5, "value_range": (0, 255), "seed": 1},
        ),
        (
            "RandomSaturation",
            cv_layers.RandomSaturation,
            {"factor": 0.5, "seed": 1},
        ),
        (
            "RandomSharpness",
            cv_layers.RandomSharpness,
            {"factor": 0.5, "value_range": (0, 255), "seed": 1},
        ),
        (
            "RandomShear",
            cv_layers.RandomShear,
            {"x_factor": 0.3, "x_factor": 0.3, "seed": 1},
        ),
        ("Solarization", cv_layers.Solarization, {"value_range": (0, 255)}),
        (
            "RandAugment",
            cv_layers.RandAugment,
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
            cv_layers.RandomAugmentationPipeline,
            {
                "layers": [
                    cv_layers.RandomSaturation(factor=0.5),
                    cv_layers.RandomColorDegeneration(factor=0.5),
                ],
                "augmentations_per_image": 1,
                "rate": 1.0,
            },
        ),
        (
            "RandomChoice",
            cv_layers.RandomChoice,
            {"layers": [], "seed": 3, "auto_vectorize": False},
        ),
        (
            "RandomColorJitter",
            cv_layers.RandomColorJitter,
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
            "RandomCropAndResize",
            cv_layers.RandomCropAndResize,
            {
                "target_size": (224, 224),
                "crop_area_factor": (0.8, 1.0),
                "aspect_ratio_factor": (3 / 4, 4 / 3),
            },
        ),
        (
            "RandomlyZoomedCrop",
            cv_layers.RandomlyZoomedCrop,
            {
                "height": 224,
                "width": 224,
                "zoom_factor": (0.8, 1.0),
                "aspect_ratio_factor": (3 / 4, 4 / 3),
            },
        ),
        (
            "DropBlock2D",
            cv_layers.DropBlock2D,
            {"rate": 0.1, "block_size": (7, 7), "seed": 1234},
        ),
        (
            "StochasticDepth",
            cv_layers.StochasticDepth,
            {"rate": 0.1},
        ),
        (
            "SqueezeAndExcite2D",
            cv_layers.SqueezeAndExcite2D,
            {
                "filters": 16,
                "ratio": 0.25,
                "squeeze_activation": tf.keras.layers.ReLU(),
                "excite_activation": tf.keras.activations.relu,
            },
        ),
        (
            "DropPath",
            cv_layers.DropPath,
            {
                "rate": 0.2,
            },
        ),
        (
            "MaybeApply",
            cv_layers.MaybeApply,
            {
                "rate": 0.5,
                "layer": None,
                "seed": 1234,
            },
        ),
        (
            "RandomJpegQuality",
            cv_layers.RandomJpegQuality,
            {"factor": (75, 100)},
        ),
        (
            "AugMix",
            cv_layers.AugMix,
            {
                "value_range": (0, 255),
                "severity": 0.3,
                "num_chains": 3,
                "chain_depth": -1,
                "alpha": 1.0,
                "seed": 1,
            },
        ),
        (
            "NonMaxSuppression",
            cv_layers.NonMaxSuppression,
            {
                "classes": 5,
                "bounding_box_format": "xyxy",
                "confidence_threshold": 0.5,
                "iou_threshold": 0.5,
                "max_detections": 100,
                "max_detections_per_class": 100,
            },
        ),
        (
            "RandomRotation",
            cv_layers.RandomRotation,
            {
                "factor": 0.5,
            },
        ),
        (
            "SegmentationHead",
            SegmentationHead,
            {
                "classes": 11,
                "convs": 3,
                "filters": 256,
                "activations": tf.keras.activations.relu,
            },
        ),
    )
    def test_layer_serialization(self, layer_cls, init_args):
        layer = layer_cls(**init_args)
        config = layer.get_config()
        self.assertAllInitParametersAreInConfig(layer_cls, config)

        model = tf.keras.models.Sequential(layer)
        model_config = model.get_config()

        reconstructed_model = tf.keras.Sequential().from_config(model_config)
        reconstructed_layer = reconstructed_model.layers[0]

        self.assertTrue(
            config_equals(layer.get_config(), reconstructed_layer.get_config())
        )

    def assertAllInitParametersAreInConfig(self, layer_cls, config):
        excluded_name = ["args", "kwargs", "*"]
        parameter_names = {
            v
            for v in inspect.signature(layer_cls).parameters.keys()
            if v not in excluded_name
        }

        intersection_with_config = {v for v in config.keys() if v in parameter_names}

        self.assertSetEqual(parameter_names, intersection_with_config)
