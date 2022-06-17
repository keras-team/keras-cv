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

from keras_cv.layers.preprocessing.base_image_augmentation_layer import (
    BaseImageAugmentationLayer,
)
from keras_cv.layers.preprocessing.maybe_apply import MaybeApply


class ZeroOut(BaseImageAugmentationLayer):
    """Zero out all entries, for testing purposes."""

    def __init__(self):
        super(ZeroOut, self).__init__()

    def augment_image(self, image, transformation=None, **kwargs):
        return 0 * image

    def augment_label(self, label, transformation=None, **kwargs):
        return 0 * label


class MaybeApplyTest(tf.test.TestCase, parameterized.TestCase):
    rng = tf.random.Generator.from_seed(seed=1234)

    @parameterized.parameters([-0.5, 1.7])
    def test_raises_error_on_invalid_rate_parameter(self, invalid_rate):
        with self.assertRaises(ValueError):
            MaybeApply(rate=invalid_rate, layer=ZeroOut())

    def test_works_with_batched_input(self):
        batch_size = 32
        dummy_inputs = self.rng.uniform(shape=(batch_size, 224, 224, 3))
        layer = MaybeApply(rate=0.5, layer=ZeroOut(), seed=1234)

        outputs = layer(dummy_inputs)
        num_zero_inputs = self._num_zero_batches(dummy_inputs)
        num_zero_outputs = self._num_zero_batches(outputs)

        self.assertEqual(num_zero_inputs, 0)
        self.assertLess(num_zero_outputs, batch_size)
        self.assertGreater(num_zero_outputs, 0)

    @staticmethod
    def _num_zero_batches(images):
        num_batches = tf.shape(images)[0]
        num_non_zero_batches = tf.math.count_nonzero(
            tf.math.count_nonzero(images, axis=[1, 2, 3]), dtype=tf.int32
        )
        return num_batches - num_non_zero_batches

    def test_inputs_unchanged_with_zero_rate(self):
        dummy_inputs = self.rng.uniform(shape=(32, 224, 224, 3))
        layer = MaybeApply(rate=0.0, layer=ZeroOut())

        outputs = layer(dummy_inputs)

        self.assertAllClose(outputs, dummy_inputs)

    def test_all_inputs_changed_with_rate_equal_to_one(self):
        dummy_inputs = self.rng.uniform(shape=(32, 224, 224, 3))
        layer = MaybeApply(rate=1.0, layer=ZeroOut())

        outputs = layer(dummy_inputs)

        self.assertAllEqual(outputs, tf.zeros_like(dummy_inputs))

    def test_works_with_single_image(self):
        dummy_inputs = self.rng.uniform(shape=(224, 224, 3))
        layer = MaybeApply(rate=1.0, layer=ZeroOut())

        outputs = layer(dummy_inputs)

        self.assertAllEqual(outputs, tf.zeros_like(dummy_inputs))

    def test_can_modify_label(self):
        dummy_inputs = self.rng.uniform(shape=(32, 224, 224, 3))
        dummy_labels = tf.ones(shape=(32, 2))
        layer = MaybeApply(rate=1.0, layer=ZeroOut())

        outputs = layer({"images": dummy_inputs, "labels": dummy_labels})

        self.assertAllEqual(outputs["labels"], tf.zeros_like(dummy_labels))

    def test_works_with_native_keras_layers(self):
        dummy_inputs = self.rng.uniform(shape=(32, 224, 224, 3))
        zero_out = tf.keras.layers.Lambda(lambda x: {"images": 0 * x["images"]})
        layer = MaybeApply(rate=1.0, layer=zero_out)

        outputs = layer(dummy_inputs)

        self.assertAllEqual(outputs, tf.zeros_like(dummy_inputs))

    def test_works_with_xla(self):
        dummy_inputs = self.rng.uniform(shape=(32, 224, 224, 3))
        # auto_vectorize=True will crash XLA
        layer = MaybeApply(rate=0.5, layer=ZeroOut(), auto_vectorize=False)

        @tf.function(jit_compile=True)
        def apply(x):
            return layer(x)

        apply(dummy_inputs)
