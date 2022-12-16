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

import pytest
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend


class ModelsTest:
    def assertShapeEqual(self, shape1, shape2):
        self.assertEqual(tf.TensorShape(shape1), tf.TensorShape(shape2))

    @pytest.fixture(autouse=True)
    def cleanup_global_session(self):
        # Code before yield runs before the test
        yield
        tf.keras.backend.clear_session()

    def _test_application_base(self, app, _, args):
        # Can be instantiated with default arguments
        model = app(include_top=True, classes=1000, include_rescaling=False, **args)

        # Can be serialized and deserialized
        config = model.get_config()
        reconstructed_model = model.__class__.from_config(config)
        self.assertEqual(len(model.weights), len(reconstructed_model.weights))

        # There is no rescaling layer bcause include_rescaling=False
        with self.assertRaises(ValueError):
            model.get_layer(name="rescaling")

    def _test_application_with_rescaling(self, app, last_dim, args):
        model = app(include_rescaling=True, include_top=False, **args)
        self.assertIsNotNone(model.get_layer(name="rescaling"))

    def _test_application_pooling(self, app, last_dim, args):
        model = app(include_rescaling=False, include_top=False, pooling="avg", **args)

        self.assertShapeEqual(model.output_shape, (None, last_dim))

    def _test_application_variable_input_channels(self, app, last_dim, args):
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

        if "Mixer" not in app.__name__ and "ViT" not in app.__name__:
            self.assertShapeEqual(output_shape, (None, None, None, last_dim))
        elif "MixerB16" in app.__name__ or "MixerL16" in app.__name__:
            num_patches = 196
            self.assertShapeEqual(output_shape, (None, num_patches, last_dim))
        elif "MixerB32" in app.__name__:
            num_patches = 49
            self.assertShapeEqual(output_shape, (None, num_patches, last_dim))
        elif (
            "ViTTiny16" in app.__name__
            or "ViTS16" in app.__name__
            or "ViTB16" in app.__name__
            or "ViTL16" in app.__name__
            or "ViTH16" in app.__name__
        ):
            num_patches = 197
            self.assertShapeEqual(output_shape, (None, num_patches, last_dim))
        elif (
            "ViTTiny32" in app.__name__
            or "ViTS32" in app.__name__
            or "ViTB32" in app.__name__
            or "ViTL32" in app.__name__
            or "ViTH32" in app.__name__
        ):
            num_patches = 50
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

        if "Mixer" not in app.__name__ and "ViT" not in app.__name__:
            self.assertShapeEqual(output_shape, (None, None, None, last_dim))
        elif "MixerB16" in app.__name__ or "MixerL16" in app.__name__:
            num_patches = 196
            self.assertShapeEqual(output_shape, (None, num_patches, last_dim))
        elif "MixerB32" in app.__name__:
            num_patches = 49
            self.assertShapeEqual(output_shape, (None, num_patches, last_dim))
        elif (
            "ViTTiny16" in app.__name__
            or "ViTS16" in app.__name__
            or "ViTB16" in app.__name__
            or "ViTL16" in app.__name__
            or "ViTH16" in app.__name__
        ):
            num_patches = 197
            self.assertShapeEqual(output_shape, (None, num_patches, last_dim))
        elif (
            "ViTTiny32" in app.__name__
            or "ViTS32" in app.__name__
            or "ViTB32" in app.__name__
            or "ViTL32" in app.__name__
            or "ViTH32" in app.__name__
        ):
            num_patches = 50
            self.assertShapeEqual(output_shape, (None, num_patches, last_dim))

    def _test_model_can_be_used_as_backbone(self, app, last_dim, args):
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
