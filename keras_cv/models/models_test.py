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
from tensorflow.keras import backend

from keras_cv.models import csp_darknet
from keras_cv.models import darknet
from keras_cv.models import densenet
from keras_cv.models import mlp_mixer
from keras_cv.models import vgg19

MODEL_LIST = [
    (csp_darknet.CSPDarkNet, 1024),
    (darknet.DarkNet21, 512),
    (darknet.DarkNet53, 512),
    (densenet.DenseNet121, 1024),
    (densenet.DenseNet169, 1664),
    (densenet.DenseNet201, 1920),
    (mlp_mixer.MLPMixerB16, 768),
    (mlp_mixer.MLPMixerB32, 768),
    (mlp_mixer.MLPMixerL16, 1024),
    (vgg19.VGG19, 512),
]


class ApplicationsTest(tf.test.TestCase, parameterized.TestCase):
    def assertShapeEqual(self, shape1, shape2):
        self.assertEqual(tf.TensorShape(shape1), tf.TensorShape(shape2))

    @parameterized.parameters(*MODEL_LIST)
    def test_application_base(self, app, _):
        # Can be instantiated with default arguments
        if "Mixer" not in app.__name__:
            model = app(
                input_shape=(224, 224, 3),
                include_top=True,
                num_classes=1000,
                include_rescaling=False,
                weights=None,
            )
        elif "MixerB16" in app.__name__ or "MixerL16" in app.__name__:
            model = app(
                input_shape=(224, 224, 3),
                patch_size=(16, 16),
                include_top=True,
                num_classes=1000,
                include_rescaling=False,
                weights=None,
            )
        elif "MixerB32" in app.__name__:
            model = app(
                input_shape=(224, 224, 3),
                patch_size=(32, 32),
                include_top=True,
                num_classes=1000,
                include_rescaling=False,
                weights=None,
            )
        # Can be serialized and deserialized
        config = model.get_config()
        reconstructed_model = model.__class__.from_config(config)
        self.assertEqual(len(model.weights), len(reconstructed_model.weights))
        backend.clear_session()

    @parameterized.parameters(*MODEL_LIST)
    def test_application_with_rescaling(self, app, last_dim):
        if "Mixer" not in app.__name__:
            output_shape = _get_output_shape(
                lambda: app(
                    include_rescaling=True,
                    include_top=False,
                    weights=None,
                    input_shape=(None, None, 3),
                )
            )
        elif "MixerB16" in app.__name__ or "MixerL16" in app.__name__:
            output_shape = _get_output_shape(
                lambda: app(
                    include_rescaling=True,
                    include_top=False,
                    weights=None,
                    input_shape=(224, 224, 3),
                    patch_size=(16, 16),
                )
            )
        elif "MixerB32" in app.__name__:
            output_shape = _get_output_shape(
                lambda: app(
                    include_rescaling=True,
                    include_top=False,
                    weights=None,
                    input_shape=(224, 224, 3),
                    patch_size=(32, 32),
                )
            )

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
    def test_application_notop(self, app, last_dim):
        if "Mixer" not in app.__name__:
            output_shape = _get_output_shape(
                lambda: app(
                    include_rescaling=False,
                    include_top=False,
                    weights=None,
                    input_shape=(None, None, 3),
                )
            )
        elif "MixerB16" in app.__name__ or "MixerL16" in app.__name__:
            output_shape = _get_output_shape(
                lambda: app(
                    include_rescaling=False,
                    include_top=False,
                    weights=None,
                    input_shape=(224, 224, 3),
                    patch_size=(16, 16),
                )
            )
        elif "MixerB32" in app.__name__:
            output_shape = _get_output_shape(
                lambda: app(
                    include_rescaling=False,
                    include_top=False,
                    weights=None,
                    input_shape=(224, 224, 3),
                    patch_size=(32, 32),
                )
            )

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
    def test_application_pooling(self, app, last_dim):
        if "Mixer" not in app.__name__:
            output_shape = _get_output_shape(
                lambda: app(
                    input_shape=(224, 224, 3),
                    include_rescaling=False,
                    include_top=False,
                    weights=None,
                    pooling="avg",
                )
            )
        elif "MixerB16" in app.__name__ or "MixerL16" in app.__name__:
            output_shape = _get_output_shape(
                lambda: app(
                    input_shape=(224, 224, 3),
                    patch_size=(16, 16),
                    include_rescaling=False,
                    include_top=False,
                    weights=None,
                    pooling="avg",
                )
            )
        elif "MixerB32" in app.__name__:
            output_shape = _get_output_shape(
                lambda: app(
                    input_shape=(224, 224, 3),
                    patch_size=(32, 32),
                    include_rescaling=False,
                    include_top=False,
                    weights=None,
                    pooling="avg",
                )
            )
        self.assertShapeEqual(output_shape, (None, last_dim))

    @parameterized.parameters(*MODEL_LIST)
    def test_application_variable_input_channels(self, app, last_dim):
        input_shape = (None, None, 1) if "Mixer" not in app.__name__ else (224, 224, 1)
        if "Mixer" not in app.__name__:
            output_shape = _get_output_shape(
                lambda: app(
                    include_rescaling=False,
                    weights=None,
                    include_top=False,
                    input_shape=input_shape,
                )
            )
        elif "MixerB16" in app.__name__ or "MixerL16" in app.__name__:
            output_shape = _get_output_shape(
                lambda: app(
                    include_rescaling=False,
                    weights=None,
                    include_top=False,
                    input_shape=input_shape,
                    patch_size=(16, 16),
                )
            )
        elif "MixerB32" in app.__name__:
            output_shape = _get_output_shape(
                lambda: app(
                    include_rescaling=False,
                    weights=None,
                    include_top=False,
                    input_shape=input_shape,
                    patch_size=(32, 32),
                )
            )
        if "Mixer" not in app.__name__:
            self.assertShapeEqual(output_shape, (None, None, None, last_dim))
        elif "MixerB16" in app.__name__ or "MixerL16" in app.__name__:
            num_patches = 196
            self.assertShapeEqual(output_shape, (None, num_patches, last_dim))
        elif "MixerB32" in app.__name__:
            num_patches = 49
            self.assertShapeEqual(output_shape, (None, num_patches, last_dim))

        backend.clear_session()

        input_shape = (None, None, 4) if "Mixer" not in app.__name__ else (224, 224, 4)
        if "Mixer" not in app.__name__:
            output_shape = _get_output_shape(
                lambda: app(
                    include_rescaling=False,
                    weights=None,
                    include_top=False,
                    input_shape=input_shape,
                )
            )
        elif "MixerB16" in app.__name__ or "MixerL16" in app.__name__:
            output_shape = _get_output_shape(
                lambda: app(
                    include_rescaling=False,
                    weights=None,
                    include_top=False,
                    input_shape=input_shape,
                    patch_size=(16, 16),
                )
            )
        elif "MixerB32" in app.__name__:
            output_shape = _get_output_shape(
                lambda: app(
                    include_rescaling=False,
                    weights=None,
                    include_top=False,
                    input_shape=input_shape,
                    patch_size=(32, 32),
                )
            )

        if "Mixer" not in app.__name__:
            self.assertShapeEqual(output_shape, (None, None, None, last_dim))
        elif "MixerB16" in app.__name__ or "MixerL16" in app.__name__:
            num_patches = 196
            self.assertShapeEqual(output_shape, (None, num_patches, last_dim))
        elif "MixerB32" in app.__name__:
            num_patches = 49
            self.assertShapeEqual(output_shape, (None, num_patches, last_dim))

        backend.clear_session()


def _get_output_shape(model_fn):
    model = model_fn()
    return model.output_shape


if __name__ == "__main__":
    tf.test.main()
