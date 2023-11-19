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

from keras_cv import bounding_box
from keras_cv.backend.config import keras_3
from keras_cv.layers import preprocessing
from keras_cv.tests.test_case import TestCase

num_classes = 10


class RandomShearTest(TestCase):
    def test_aggressive_shear_fills_at_least_some_pixels(self):
        img_shape = (50, 50, 3)
        xs = tf.stack(
            [2 * tf.ones(img_shape), tf.ones(img_shape)],
            axis=0,
        )
        ys_segmentation_masks = tf.stack(
            [2 * tf.ones(img_shape), tf.ones(img_shape)],
            axis=0,
        )
        xs = tf.cast(xs, tf.float32)
        ys_segmentation_masks = tf.cast(ys_segmentation_masks, tf.float32)

        fill_value = 0.0
        layer = preprocessing.RandomShear(
            x_factor=(3, 3), seed=0, fill_mode="constant", fill_value=fill_value
        )
        xs = layer(xs)
        ys_segmentation_masks = layer(ys_segmentation_masks)

        # Some pixels should be replaced with fill value
        self.assertTrue(tf.math.reduce_any(xs[0] == fill_value))
        self.assertTrue(tf.math.reduce_any(xs[0] == 2.0))
        self.assertTrue(tf.math.reduce_any(xs[1] == fill_value))
        self.assertTrue(tf.math.reduce_any(xs[1] == 1.0))
        self.assertTrue(
            tf.math.reduce_any(ys_segmentation_masks[0] == fill_value)
        )
        self.assertTrue(tf.math.reduce_any(ys_segmentation_masks[0] == 2.0))
        self.assertTrue(
            tf.math.reduce_any(ys_segmentation_masks[1] == fill_value)
        )
        self.assertTrue(tf.math.reduce_any(ys_segmentation_masks[1] == 1.0))

    def test_return_shapes(self):
        """test return dict keys and value pairs"""
        xs = tf.ones((2, 512, 512, 3))
        # randomly sample labels
        ys_labels = tf.random.categorical(tf.math.log([[0.5, 0.5]]), 2)
        ys_labels = tf.squeeze(ys_labels)
        ys_labels = tf.one_hot(ys_labels, num_classes)

        # randomly sample bounding boxes
        ys_bounding_boxes = {
            "boxes": tf.ones((2, 3, 4)),
            "classes": tf.random.uniform((2, 3), 0, 1),
        }

        # randomly sample segmentation masks
        ys_segmentation_masks = tf.ones((2, 512, 512, 3))

        layer = preprocessing.RandomShear(
            x_factor=(0.1, 0.3),
            y_factor=(0.1, 0.3),
            seed=0,
            fill_mode="constant",
            bounding_box_format="xywh",
        )

        outputs = layer(
            {
                "images": xs,
                "targets": ys_labels,
                "bounding_boxes": ys_bounding_boxes,
                "segmentation_masks": ys_segmentation_masks,
            }
        )
        xs, ys_labels, ys_bounding_boxes, ys_segmentation_masks = (
            outputs["images"],
            outputs["targets"],
            outputs["bounding_boxes"],
            outputs["segmentation_masks"],
        )
        ys_bounding_boxes = bounding_box.to_dense(ys_bounding_boxes)
        self.assertEqual(xs.shape, [2, 512, 512, 3])
        self.assertEqual(ys_labels.shape, [2, 10])
        self.assertEqual(ys_bounding_boxes["boxes"].shape, [2, 3, 4])
        self.assertEqual(ys_bounding_boxes["classes"].shape, [2, 3])
        self.assertEqual(ys_segmentation_masks.shape, [2, 512, 512, 3])

    def test_single_image_input(self):
        """test for single image input"""
        xs = tf.ones((512, 512, 3))
        inputs = {"images": xs}
        layer = preprocessing.RandomShear(
            x_factor=(3, 3),
            seed=0,
            fill_mode="constant",
        )
        outputs = layer(inputs)
        self.assertEqual(outputs["images"].shape, [512, 512, 3])

    @pytest.mark.skip(reason="Flaky")
    def test_area(self):
        xs = tf.ones((1, 512, 512, 3))
        ys = {
            "boxes": tf.constant(
                [[[0.3, 0.4, 0.5, 0.6], [0.9, 0.8, 1.0, 1.0]]]
            ),
            "classes": tf.constant([2, 3]),
        }

        inputs = {"images": xs, "bounding_boxes": ys}
        layer = preprocessing.RandomShear(
            x_factor=(0.3, 0.7),
            y_factor=(0.4, 0.7),
            seed=0,
            fill_mode="constant",
            bounding_box_format="rel_xyxy",
        )
        outputs = layer(inputs)
        xs, ys_bounding_boxes = (
            outputs["images"],
            outputs["bounding_boxes"]["boxes"],
        )
        new_area = tf.math.multiply(
            tf.abs(
                tf.subtract(
                    ys_bounding_boxes[..., 2], ys_bounding_boxes[..., 0]
                )
            ),
            tf.abs(
                tf.subtract(
                    ys_bounding_boxes[..., 3], ys_bounding_boxes[..., 1]
                )
            ),
        )
        old_area = tf.math.multiply(
            tf.abs(tf.subtract(ys["boxes"][..., 2], ys["boxes"][..., 0])),
            tf.abs(tf.subtract(ys["boxes"][..., 3], ys["boxes"][..., 1])),
        )
        self.assertTrue(tf.math.reduce_all(new_area > old_area))

    def test_in_tf_function(self):
        """test for class works with tf function"""
        xs = tf.cast(
            tf.stack(
                [2 * tf.ones((4, 4, 3)), tf.ones((4, 4, 3))],
                axis=0,
            ),
            tf.float32,
        )
        layer = preprocessing.RandomShear(
            x_factor=0.2, y_factor=0.2, bounding_box_format="xywh"
        )
        ys = {
            "boxes": tf.random.uniform((2, 3, 4), 0, 1),
            "classes": tf.random.uniform((2, 3), 0, 1),
        }

        @tf.function
        def augment(x, y):
            return layer({"images": x, "bounding_boxes": y})

        outputs = augment(xs, ys)
        xs = outputs["images"]

        # None of the individual values should still be close to 1 or 0
        self.assertNotAllClose(xs, 1.0)
        self.assertNotAllClose(xs, 2.0)

    def test_no_augmentation(self):
        """test for no image and bbox augmentation when x_factor,y_factor is
        0,0"""
        xs = tf.cast(
            tf.stack(
                [2 * tf.ones((4, 4, 3)), tf.ones((4, 4, 3))],
                axis=0,
            ),
            tf.float32,
        )
        ys = {
            "boxes": tf.constant(
                [
                    [[0.3, 0.4, 0.5, 0.6], [0.9, 0.8, 1.0, 1.0]],
                    [[0.3, 0.4, 0.5, 0.6], [0.9, 0.8, 1.0, 1.0]],
                ],
                dtype=tf.float32,
            ),
            "classes": tf.constant([[0, 0], [0, 0]], dtype=tf.float32),
        }
        layer = preprocessing.RandomShear(
            x_factor=0, y_factor=0, bounding_box_format="rel_xyxy"
        )
        outputs = layer({"images": xs, "bounding_boxes": ys})
        output_xs, output_ys = outputs["images"], outputs["bounding_boxes"]

        ys = bounding_box.to_dense(ys)
        output_ys = bounding_box.to_dense(output_ys)
        self.assertAllEqual(xs, output_xs)
        self.assertAllEqual(ys["boxes"], output_ys["boxes"])

    # TODO re-enable when bounding box augmentation is fixed.
    def DISABLED_test_output_values(self):
        """test to verify augmented bounding box output coordinate"""
        xs = tf.cast(
            tf.stack(
                [2 * tf.ones((100, 100, 3)), tf.zeros((100, 100, 3))],
                axis=0,
            ),
            tf.float32,
        )
        ys = tf.cast(
            tf.stack(
                [
                    tf.constant(
                        [[10.0, 20.0, 40.0, 50.0], [12.0, 22.0, 42.0, 54.0]]
                    ),
                    tf.constant(
                        [[10.0, 20.0, 40.0, 50.0], [12.0, 22.0, 42.0, 54.0]]
                    ),
                ],
                axis=0,
            ),
            tf.float32,
        )
        ys = bounding_box.add_class_id(ys)
        true_ys = tf.cast(
            tf.stack(
                [
                    tf.constant(
                        [
                            [7.60, 20.43, 39.04, 51.79, 0.0],
                            [9.41, 22.52, 40.94, 55.88, 0.0],
                        ]
                    ),
                    tf.constant(
                        [
                            [13.68, 22.51, 49.20, 59.05, 0],
                            [16.04, 24.95, 51.940, 63.56, 0],
                        ]
                    ),
                ],
                axis=0,
            ),
            tf.float32,
        )
        layer = preprocessing.RandomShear(
            x_factor=0.2, y_factor=0.2, bounding_box_format="xyxy", seed=1
        )
        outputs = layer({"images": xs, "bounding_boxes": ys})
        _, output_ys = outputs["images"], outputs["bounding_boxes"].to_tensor()
        self.assertAllClose(true_ys, output_ys, rtol=1e-02, atol=1e-03)

    def test_random_shear_on_batched_images_independently(self):
        image = tf.random.uniform(shape=(100, 100, 3))
        input_images = tf.stack([image, image], axis=0)

        layer = preprocessing.RandomShear(x_factor=0.5, y_factor=0.5)

        results = layer(input_images)
        self.assertNotAllClose(results[0], results[1])

    def test_ragged_bounding_box(self):
        images = tf.random.uniform((2, 16, 16, 3))

        random_box = tf.constant(
            [[[0.1, 0.2, 1, 1], [0.4, 0.6, 1, 1]]], dtype=tf.float32
        )
        random_box = tf.squeeze(random_box, axis=0)
        random_box = tf.RaggedTensor.from_row_lengths(random_box, [1, 1])
        classes = tf.ragged.constant([[0], [0]])
        bounding_boxes = {"boxes": random_box, "classes": classes}
        inputs = {"images": images, "bounding_boxes": bounding_boxes}

        layer = preprocessing.RandomShear(
            x_factor=(0.5, 0.5),
            y_factor=(0.5, 0.5),
            bounding_box_format="rel_xywh",
        )

        layer(inputs)
