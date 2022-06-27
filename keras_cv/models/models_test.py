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

from keras_cv.models import densenet

MODEL_LIST = [
    (densenet.DenseNet121, 1024),
    (densenet.DenseNet169, 1664),
    (densenet.DenseNet201, 1920),
]


class ApplicationsTest(tf.test.TestCase, parameterized.TestCase):
    def assertShapeEqual(self, shape1, shape2):
        self.assertEqual(tf.TensorShape(shape1), tf.TensorShape(shape2))

    @parameterized.parameters(*MODEL_LIST)
    def test_application_base(self, app, _):
        # Can be instantiated with default arguments
        model = app(
            input_shape=(224, 224, 3),
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
        output_shape = _get_output_shape(
            lambda: app(
                include_rescaling=True,
                include_top=False,
                weights=None,
            )
        )
        self.assertShapeEqual(output_shape, (None, None, None, last_dim))
        backend.clear_session()

    @parameterized.parameters(*MODEL_LIST)
    def test_application_notop(self, app, last_dim):
        output_shape = _get_output_shape(
            lambda: app(
                include_rescaling=False,
                include_top=False,
                weights=None,
            )
        )
        self.assertShapeEqual(output_shape, (None, None, None, last_dim))
        backend.clear_session()

    @parameterized.parameters(*MODEL_LIST)
    def test_application_pooling(self, app, last_dim):
        output_shape = _get_output_shape(
            lambda: app(
                input_shape=(224, 224, 3),
                include_rescaling=False,
                include_top=False,
                weights=None,
                pooling="avg",
            )
        )
        self.assertShapeEqual(output_shape, (None, last_dim))

    @parameterized.parameters(*MODEL_LIST)
    def test_application_variable_input_channels(self, app, last_dim):
        input_shape = (None, None, 1)
        output_shape = _get_output_shape(
            lambda: app(
                include_rescaling=False,
                weights=None,
                include_top=False,
                input_shape=input_shape,
            )
        )
        self.assertShapeEqual(output_shape, (None, None, None, last_dim))
        backend.clear_session()

        input_shape = (None, None, 4)
        output_shape = _get_output_shape(
            lambda: app(
                include_rescaling=False,
                weights=None,
                include_top=False,
                input_shape=input_shape,
            )
        )
        self.assertShapeEqual(output_shape, (None, None, None, last_dim))
        backend.clear_session()


def _get_output_shape(model_fn):
    model = model_fn()
    return model.output_shape


if __name__ == "__main__":
    tf.test.main()
