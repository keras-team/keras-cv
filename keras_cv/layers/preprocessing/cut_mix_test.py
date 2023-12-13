# Copyright 2023 The KerasCV Authors
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
import numpy as np
import pytest
import tensorflow as tf

from keras_cv.backend import ops
from keras_cv.layers.preprocessing.cut_mix import CutMix
from keras_cv.tests.test_case import TestCase

num_classes = 10


class CutMixTest(TestCase):
    def test_return_shapes(self):
        xs = tf.ones((2, 512, 512, 3))
        # randomly sample labels
        ys_labels = tf.random.categorical(tf.math.log([[0.5, 0.5]]), 2)
        ys_labels = tf.squeeze(ys_labels)
        ys_labels = tf.one_hot(ys_labels, num_classes)

        # randomly sample segmentation mask
        ys_segmentation_masks = tf.cast(
            tf.stack(
                [2 * tf.ones((512, 512)), tf.ones((512, 512))],
                axis=0,
            ),
            tf.uint8,
        )
        ys_segmentation_masks = tf.one_hot(ys_segmentation_masks, 3)

        layer = CutMix(seed=1)
        outputs = layer(
            {
                "images": xs,
                "labels": ys_labels,
                "segmentation_masks": ys_segmentation_masks,
            }
        )
        xs, ys_labels, ys_segmentation_masks = (
            outputs["images"],
            outputs["labels"],
            outputs["segmentation_masks"],
        )

        self.assertEqual(xs.shape, (2, 512, 512, 3))
        self.assertEqual(ys_labels.shape, (2, 10))
        self.assertEqual(ys_segmentation_masks.shape, (2, 512, 512, 3))

    def test_cut_mix_call_results_with_labels(self):
        xs = tf.cast(
            tf.stack(
                [2 * tf.ones((4, 4, 3)), tf.ones((4, 4, 3))],
                axis=0,
            ),
            tf.float32,
        )
        ys = tf.one_hot(tf.constant([0, 1]), 2)

        layer = CutMix(seed=1)
        outputs = layer({"images": xs, "labels": ys})
        xs, ys = outputs["images"], outputs["labels"]

        # At least some pixels should be replaced in the CutMix operation
        self.assertTrue(np.any(ops.convert_to_numpy(xs[0]) == 1.0))
        self.assertTrue(np.any(ops.convert_to_numpy(xs[0]) == 2.0))
        self.assertTrue(np.any(ops.convert_to_numpy(xs[1]) == 1.0))
        self.assertTrue(np.any(ops.convert_to_numpy(xs[1]) == 2.0))
        # No labels should still be close to their original values
        self.assertNotAllClose(ys, 1.0)
        self.assertNotAllClose(ys, 0.0)

    def test_cut_mix_call_results_one_channel_with_labels(self):
        xs = tf.cast(
            tf.stack(
                [2 * tf.ones((4, 4, 1)), tf.ones((4, 4, 1))],
                axis=0,
            ),
            tf.float32,
        )
        ys = tf.one_hot(tf.constant([0, 1]), 2)

        layer = CutMix(seed=1)
        outputs = layer({"images": xs, "labels": ys})
        xs, ys = outputs["images"], outputs["labels"]

        # At least some pixels should be replaced in the CutMix operation
        self.assertTrue(np.any(ops.convert_to_numpy(xs[0]) == 1.0))
        self.assertTrue(np.any(ops.convert_to_numpy(xs[0]) == 2.0))
        self.assertTrue(np.any(ops.convert_to_numpy(xs[1]) == 1.0))
        self.assertTrue(np.any(ops.convert_to_numpy(xs[1]) == 2.0))
        # No labels should still be close to their original values
        self.assertNotAllClose(ys, 1.0)
        self.assertNotAllClose(ys, 0.0)

    def test_cut_mix_call_results_with_dense_encoded_segmentation_masks(self):
        xs = tf.cast(
            tf.stack(
                [2 * tf.ones((4, 4, 3)), tf.ones((4, 4, 3))],
                axis=0,
            ),
            tf.float32,
        )

        ys_segmentation_masks = tf.cast(
            tf.stack(
                [2 * tf.ones((4, 4, 1)), tf.ones((4, 4, 1))],
                axis=0,
            ),
            tf.float32,
        )

        layer = CutMix(seed=1)
        outputs = layer(
            {"images": xs, "segmentation_masks": ys_segmentation_masks}
        )
        xs, ys_segmentation_masks = (
            outputs["images"],
            outputs["segmentation_masks"],
        )

        # At least some pixels should be replaced in the images
        self.assertTrue(np.any(ops.convert_to_numpy(xs[0]) == 1.0))
        self.assertTrue(np.any(ops.convert_to_numpy(xs[0]) == 2.0))
        self.assertTrue(np.any(ops.convert_to_numpy(xs[1]) == 1.0))
        self.assertTrue(np.any(ops.convert_to_numpy(xs[1]) == 2.0))
        # At least some pixels should be replaced in the segmentation_masks
        self.assertTrue(
            np.any(ops.convert_to_numpy(ys_segmentation_masks[0]) == 1.0)
        )
        self.assertTrue(
            np.any(ops.convert_to_numpy(ys_segmentation_masks[0]) == 2.0)
        )
        self.assertTrue(
            np.any(ops.convert_to_numpy(ys_segmentation_masks[1]) == 1.0)
        )
        self.assertTrue(
            np.any(ops.convert_to_numpy(ys_segmentation_masks[1]) == 2.0)
        )

    def test_cut_mix_call_results_with_one_hot_encoded_segmentation_masks(self):
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

        layer = CutMix(seed=1)
        outputs = layer(
            {"images": xs, "segmentation_masks": ys_segmentation_masks}
        )
        xs, ys_segmentation_masks = (
            outputs["images"],
            outputs["segmentation_masks"],
        )

        # At least some pixels should be replaced in the images
        self.assertTrue(np.any(ops.convert_to_numpy(xs[0]) == 1.0))
        self.assertTrue(np.any(ops.convert_to_numpy(xs[0]) == 2.0))
        self.assertTrue(np.any(ops.convert_to_numpy(xs[1]) == 1.0))
        self.assertTrue(np.any(ops.convert_to_numpy(xs[1]) == 2.0))
        # At least some pixels should be replaced in the segmentation_masks
        self.assertTrue(
            np.any(
                ops.convert_to_tensor(ys_segmentation_masks[0][:, :, 2]) == 1.0
            )
        )
        self.assertTrue(
            np.any(
                ops.convert_to_tensor(ys_segmentation_masks[0][:, :, 2]) == 0.0
            )
        )
        self.assertTrue(
            np.any(
                ops.convert_to_tensor(ys_segmentation_masks[1][:, :, 1]) == 1.0
            )
        )
        self.assertTrue(
            np.any(
                ops.convert_to_tensor(ys_segmentation_masks[1][:, :, 1]) == 0.0
            )
        )

    @pytest.mark.tf_only
    def test_in_tf_function(self):
        xs = tf.cast(
            tf.stack(
                [2 * tf.ones((100, 100, 1)), tf.ones((100, 100, 1))], axis=0
            ),
            tf.float32,
        )
        ys = tf.one_hot(tf.constant([0, 1]), 2)

        layer = CutMix(seed=1)

        @tf.function
        def augment(x, y):
            return layer({"images": x, "labels": y})

        outputs = augment(xs, ys)
        xs, ys = outputs["images"], outputs["labels"]

        # At least some pixels should be replaced in the CutMix operation
        self.assertTrue(np.any(ops.convert_to_numpy(xs[0]) == 1.0))
        self.assertTrue(np.any(ops.convert_to_numpy(xs[0]) == 2.0))
        self.assertTrue(np.any(ops.convert_to_numpy(xs[1]) == 1.0))
        self.assertTrue(np.any(ops.convert_to_numpy(xs[1]) == 2.0))
        # No labels should still be close to their original values
        self.assertNotAllClose(ys, 1.0)
        self.assertNotAllClose(ys, 0.0)

    def test_single_image_input(self):
        xs = tf.ones((512, 512, 3))
        ys = tf.one_hot(tf.constant([1]), 2)
        inputs = {"images": xs, "labels": ys}
        layer = CutMix()
        with self.assertRaisesRegexp(
            ValueError, "CutMix received a single image to `call`"
        ):
            _ = layer(inputs)

    def test_int_labels(self):
        xs = tf.ones((2, 512, 512, 3))
        ys = tf.one_hot(tf.constant([1, 0]), 2, dtype=tf.int32)
        inputs = {"images": xs, "labels": ys}
        layer = CutMix()
        with self.assertRaisesRegexp(
            ValueError, "CutMix received labels with type"
        ):
            _ = layer(inputs)

    def test_image_input(self):
        xs = tf.ones((2, 512, 512, 3))
        layer = CutMix()
        with self.assertRaisesRegexp(
            ValueError, "CutMix expects inputs in a dictionary with format"
        ):
            _ = layer(xs)
