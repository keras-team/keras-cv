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
import pytest
import tensorflow as tf

from keras_cv.layers.preprocessing.fourier_mix import FourierMix
from keras_cv.tests.test_case import TestCase

num_classes = 10


class FourierMixTest(TestCase):
    # Cannot do basic layer test since the layer cannot be tested on single
    # image
    def test_return_shapes(self):
        xs = tf.ones((2, 512, 512, 3))
        # randomly sample labels
        ys = tf.random.categorical(tf.math.log([[0.5, 0.5]]), 2)
        ys = tf.squeeze(ys)
        ys = tf.one_hot(ys, num_classes)
        # randomly sample segmentation mask
        ys_segmentation_masks = tf.cast(
            tf.stack(
                [2 * tf.ones((512, 512)), tf.ones((512, 512))],
                axis=0,
            ),
            tf.uint8,
        )
        ys_segmentation_masks = tf.one_hot(ys_segmentation_masks, 3)

        layer = FourierMix()
        outputs = layer(
            {
                "images": xs,
                "labels": ys,
                "segmentation_masks": ys_segmentation_masks,
            }
        )
        xs, ys, ys_segmentation_masks = (
            outputs["images"],
            outputs["labels"],
            outputs["segmentation_masks"],
        )

        self.assertEqual(xs.shape, (2, 512, 512, 3))
        self.assertEqual(ys.shape, (2, 10))
        self.assertEqual(ys_segmentation_masks.shape, (2, 512, 512, 3))

    def test_fourier_mix_call_results_with_labels(self):
        xs = tf.cast(
            tf.stack(
                [2 * tf.ones((4, 4, 3)), tf.ones((4, 4, 3))],
                axis=0,
            ),
            tf.float32,
        )
        ys = tf.one_hot(tf.constant([0, 1]), 2)

        layer = FourierMix()
        outputs = layer({"images": xs, "labels": ys})
        xs, ys = outputs["images"], outputs["labels"]

        # None of the individual values should still be close to 1 or 0
        self.assertNotAllClose(xs, 1.0)
        self.assertNotAllClose(xs, 2.0)

        # No labels should still be close to their originals
        self.assertNotAllClose(ys, 1.0)
        self.assertNotAllClose(ys, 0.0)

    def test_mix_up_call_results_with_masks(self):
        xs = tf.cast(
            tf.stack(
                [2 * tf.ones((4, 4, 3)), tf.ones((4, 4, 3))],
                axis=0,
            ),
            tf.float32,
        )
        ys_segmentation_masks = tf.cast(
            tf.stack(
                [2 * tf.ones((4, 4)), tf.ones((4, 4))],
                axis=0,
            ),
            tf.uint8,
        )
        ys_segmentation_masks = tf.one_hot(ys_segmentation_masks, 3)

        layer = FourierMix()
        outputs = layer(
            {"images": xs, "segmentation_masks": ys_segmentation_masks}
        )
        xs, ys_segmentation_masks = (
            outputs["images"],
            outputs["segmentation_masks"],
        )

        # None of the individual values should still be close to 1 or 0
        self.assertNotAllClose(xs, 1.0)
        self.assertNotAllClose(xs, 2.0)

        # No masks should still be close to their originals
        self.assertNotAllClose(ys_segmentation_masks, 1.0)
        self.assertNotAllClose(ys_segmentation_masks, 0.0)

    @pytest.mark.tf_only
    def test_in_tf_function(self):
        xs = tf.cast(
            tf.stack(
                [2 * tf.ones((4, 4, 3)), tf.ones((4, 4, 3))],
                axis=0,
            ),
            tf.float32,
        )
        ys = tf.one_hot(tf.constant([0, 1]), 2)

        layer = FourierMix()

        @tf.function
        def augment(x, y):
            return layer({"images": x, "labels": y})

        outputs = augment(xs, ys)
        xs, ys = outputs["images"], outputs["labels"]

        # None of the individual values should still be close to 1 or 0
        self.assertNotAllClose(xs, 1.0)
        self.assertNotAllClose(xs, 2.0)

        # No labels should still be close to their originals
        self.assertNotAllClose(ys, 1.0)
        self.assertNotAllClose(ys, 0.0)

    def test_image_input_only(self):
        xs = tf.cast(
            tf.stack(
                [2 * tf.ones((100, 100, 1)), tf.ones((100, 100, 1))], axis=0
            ),
            tf.float32,
        )
        layer = FourierMix()
        with self.assertRaisesRegexp(
            ValueError, "expects inputs in a dictionary"
        ):
            _ = layer(xs)

    def test_single_image_input(self):
        xs = tf.ones((512, 512, 3))
        ys = tf.one_hot(tf.constant([1]), 2)
        inputs = {"images": xs, "labels": ys}
        layer = FourierMix()
        with self.assertRaisesRegexp(
            ValueError, "FourierMix received a single image to `call`"
        ):
            _ = layer(inputs)
