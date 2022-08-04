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
"""Integration tests for KerasCV models."""

import tensorflow as tf
from absl.testing import parameterized
from tensorflow import keras
from tensorflow.keras import backend

from keras_cv.models import csp_darknet
from keras_cv.models import darknet
from keras_cv.models import densenet
from keras_cv.models import mlp_mixer
from keras_cv.models import resnet_v1
from keras_cv.models import resnet_v2
from keras_cv.models import vgg19

MODEL_LIST = [
    (csp_darknet.CSPDarkNet, 1024, {}),
    (darknet.DarkNet21, 512, {}),
    (darknet.DarkNet53, 512, {}),
    (densenet.DenseNet121, 1024, {}),
    (densenet.DenseNet169, 1664, {}),
    (densenet.DenseNet201, 1920, {}),
    (resnet_v1.ResNet50, 2048, {}),
    (resnet_v1.ResNet101, 2048, {}),
    (resnet_v1.ResNet152, 2048, {}),
    (resnet_v2.ResNet50V2, 2048, {}),
    (resnet_v2.ResNet101V2, 2048, {}),
    (resnet_v2.ResNet152V2, 2048, {}),
    (
        mlp_mixer.MLPMixerB16,
        768,
        {"patch_size": (16, 16), "input_shape": (224, 224, 3)},
    ),
    (
        mlp_mixer.MLPMixerB32,
        768,
        {"patch_size": (32, 32), "input_shape": (224, 224, 3)},
    ),
    (
        mlp_mixer.MLPMixerL16,
        1024,
        {"patch_size": (16, 16), "input_shape": (224, 224, 3)},
    ),
    (vgg19.VGG19, 512, {}),
]


class ApplicationsTest(tf.test.TestCase, parameterized.TestCase):
    def assertShapeEqual(self, shape1, shape2):
        self.assertEqual(tf.TensorShape(shape1), tf.TensorShape(shape2))

    @parameterized.parameters(*MODEL_LIST)
    def test_application_base(self, app, _, args):
        # Can be instantiated with default arguments
        model = app(include_top=True, classes=1000, include_rescaling=False, **args)

        # Can be serialized and deserialized
        config = model.get_config()
        reconstructed_model = model.__class__.from_config(config)
        self.assertEqual(len(model.weights), len(reconstructed_model.weights))

        # There is no rescaling layer bcause include_rescaling=False
        with self.assertRaises(ValueError):
            model.get_layer(name="rescaling")

        backend.clear_session()

    @parameterized.parameters(*MODEL_LIST)
    def test_application_with_rescaling(self, app, last_dim, args):
        model = app(include_rescaling=True, include_top=False, **args)

        output_shape = model.output_shape

        if "VGG" in app.__name__:
            self.assertShapeEqual(output_shape, (None, 7, 7, last_dim))
        elif "Mixer" not in app.__name__:
            self.assertShapeEqual(output_shape, (None, None, None, last_dim))
        elif "MixerB16" in app.__name__ or "MixerL16" in app.__name__:
            num_patches = 196
            self.assertShapeEqual(output_shape, (None, num_patches, last_dim))
        elif "MixerB32" in app.__name__:
            num_patches = 49
            self.assertShapeEqual(output_shape, (None, num_patches, last_dim))

        self.assertIsNotNone(model.get_layer(name="rescaling"))

        backend.clear_session()

    @parameterized.parameters(*MODEL_LIST)
    def test_application_notop(self, app, last_dim, args):
        model = app(include_rescaling=False, include_top=False, **args)

        output_shape = model.output_shape

        if "VGG" in app.__name__:
            self.assertShapeEqual(output_shape, (None, 7, 7, last_dim))
        elif "Mixer" not in app.__name__:
            self.assertShapeEqual(output_shape, (None, None, None, last_dim))
        elif "MixerB16" in app.__name__ or "MixerL16" in app.__name__:
            num_patches = 196
            self.assertShapeEqual(output_shape, (None, num_patches, last_dim))
        elif "MixerB32" in app.__name__:
            num_patches = 49
            self.assertShapeEqual(output_shape, (None, num_patches, last_dim))

        backend.clear_session()

    @parameterized.parameters(*MODEL_LIST)
    def test_application_pooling(self, app, last_dim, args):
        model = app(include_rescaling=False, include_top=False, pooling="avg", **args)

        self.assertShapeEqual(model.output_shape, (None, last_dim))

    @parameterized.parameters(*MODEL_LIST)
    def test_application_variable_input_channels(self, app, last_dim, args):
        # Make a local copy of args because we modify them in the test
        args = dict(args)

        input_shape = (None, None, 3)

        # Avoid passing this parameter twice to the app function
        if "input_shape" in args:
            input_shape = args["input_shape"]
            del args["input_shape"]

        single_channel_input_shape = (input_shape[0], input_shape[1], 1)
        model = app(
            include_rescaling=False,
            include_top=False,
            input_shape=single_channel_input_shape,
            **args
        )

        output_shape = model.output_shape

        if "Mixer" not in app.__name__:
            self.assertShapeEqual(output_shape, (None, None, None, last_dim))
        elif "MixerB16" in app.__name__ or "MixerL16" in app.__name__:
            num_patches = 196
            self.assertShapeEqual(output_shape, (None, num_patches, last_dim))
        elif "MixerB32" in app.__name__:
            num_patches = 49
            self.assertShapeEqual(output_shape, (None, num_patches, last_dim))

        backend.clear_session()

        four_channel_input_shape = (input_shape[0], input_shape[1], 4)
        model = app(
            include_rescaling=False,
            include_top=False,
            input_shape=four_channel_input_shape,
            **args
        )

        output_shape = model.output_shape

        if "Mixer" not in app.__name__:
            self.assertShapeEqual(output_shape, (None, None, None, last_dim))
        elif "MixerB16" in app.__name__ or "MixerL16" in app.__name__:
            num_patches = 196
            self.assertShapeEqual(output_shape, (None, num_patches, last_dim))
        elif "MixerB32" in app.__name__:
            num_patches = 49
            self.assertShapeEqual(output_shape, (None, num_patches, last_dim))

        backend.clear_session()

    @parameterized.parameters(*MODEL_LIST)
    def test_model_can_be_used_as_backbone(self, app, last_dim, args):
        inputs = keras.layers.Input(shape=(224, 224, 3))
        backbone = app(
            include_rescaling=False,
            include_top=False,
            input_tensor=inputs,
            pooling="avg",
            **args
        )

        x = inputs
        x = backbone(x)

        backbone_output = backbone.get_layer(index=-1).output

        model = keras.Model(inputs=inputs, outputs=[backbone_output])
        model.compile()


if __name__ == "__main__":
    tf.test.main()
