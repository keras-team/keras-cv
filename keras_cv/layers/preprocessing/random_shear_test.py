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

from keras_cv.layers import preprocessing

NUM_CLASSES = 10


class RandomShearTest(tf.test.TestCase):
    def test_aggressive_shear_fills_at_least_some_pixels(self):
        img_shape = (50, 50, 3)
        xs = tf.stack(
            [2 * tf.ones(img_shape), tf.ones(img_shape)],
            axis=0,
        )
        xs = tf.cast(xs, tf.float32)

        fill_value = 0.0
        layer = preprocessing.RandomShear(
            x_factor=(3, 3), seed=0, fill_mode="constant", fill_value=fill_value
        )
        xs = layer(xs)

        # Some pixels should be replaced with fill value
        self.assertTrue(tf.math.reduce_any(xs[0] == fill_value))
        self.assertTrue(tf.math.reduce_any(xs[0] == 2.0))
        self.assertTrue(tf.math.reduce_any(xs[1] == fill_value))
        self.assertTrue(tf.math.reduce_any(xs[1] == 1.0))

    def test_return_shapes(self):
        """test return dict keys and value pairs"""
        xs = tf.ones((2, 512, 512, 3))
        # randomly sample labels
        ys_labels = tf.random.categorical(tf.math.log([[0.5, 0.5]]), 2)
        ys_labels = tf.squeeze(ys_labels)
        ys_labels = tf.one_hot(ys_labels, NUM_CLASSES)

        # randomly sample bounding boxes
        ys_bounding_boxes = tf.random.uniform((2, 3, 4), 0, 1)

        layer = preprocessing.RandomShear(x_factor=(3, 3), seed=0, fill_mode="constant")
        # mixup on labels
        outputs = layer(
            {"images": xs, "labels": ys_labels, "bounding_boxes": ys_bounding_boxes}
        )
        xs, ys_labels, ys_bounding_boxes = (
            outputs["images"],
            outputs["labels"],
            outputs["bounding_boxes"],
        )

        self.assertEqual(xs.shape, [2, 512, 512, 3])
        self.assertEqual(ys_labels.shape, [2, 10])
        self.assertEqual(ys_bounding_boxes.shape, [2, 3, 4])

    def test_single_image_input(self):
        """test for single image input"""
        xs = tf.ones((512, 512, 3))
        ys = tf.ones(shape=(5, 4))
        inputs = {"images": xs, "bounding_boxes": ys}
        layer = preprocessing.RandomShear(
            x_factor=(3, 3), y_factor=1, seed=0, fill_mode="constant"
        )
        outputs = layer(inputs)
        xs, ys_bounding_boxes = (
            outputs["images"],
            outputs["bounding_boxes"],
        )
        self.assertEqual(xs.shape, [512, 512, 3])
        self.assertEqual(ys_bounding_boxes.shape, [5, 4])

    def test_area(self):
        """test for shear bbox transformation since new bbox will be
        greater than old bbox"""
        xs = tf.ones((512, 512, 3))
        ys = tf.constant([[0.3, 0.4, 0.5, 0.6], [0.9, 0.8, 1.0, 1.0]])
        inputs = {"images": xs, "bounding_boxes": ys}
        layer = preprocessing.RandomShear(
            x_factor=(0.3, 0.7), y_factor=(0.4,0.7), seed=0, fill_mode="constant"
        )
        outputs = layer(inputs)
        xs, ys_bounding_boxes = (
            outputs["images"],
            outputs["bounding_boxes"],
        )
        new_area = tf.math.multiply(
            tf.abs(tf.subtract(ys_bounding_boxes[..., 2], ys_bounding_boxes[..., 0])),
            tf.abs(tf.subtract(ys_bounding_boxes[..., 3], ys_bounding_boxes[..., 1])),
        )
        old_area = tf.math.multiply(
            tf.abs(tf.subtract(ys[..., 2], ys[..., 0])),
            tf.abs(tf.subtract(ys[..., 3], ys[..., 1])),
        )
        tf.debugging.assert_greater_equal(new_area, old_area)

    def test_in_tf_function(self):
        """test forclass works with tf function"""
        xs = tf.cast(
            tf.stack(
                [2 * tf.ones((4, 4, 3)), tf.ones((4, 4, 3))],
                axis=0,
            ),
            tf.float32,
        )
        ys = tf.cast(
            tf.stack(
                [
                    tf.constant([[0.3, 0.4, 0.5, 0.6], [0.9, 0.8, 1.0, 1.0]]),
                    tf.constant([[0.3, 0.4, 0.5, 0.6], [0.9, 0.8, 1.0, 1.0]]),
                ],
                axis=0,
            ),
            tf.float32,
        )

        layer = preprocessing.RandomShear(x_factor=0.2, y_factor=0.2)

        @tf.function
        def augment(x, y):
            return layer({"images": x, "bounding_boxes": y})

        outputs = augment(xs, ys)
        xs, ys = outputs["images"], outputs["bounding_boxes"]

        # None of the individual values should still be close to 1 or 0
        self.assertNotAllClose(xs, 1.0)
        self.assertNotAllClose(xs, 2.0)

        # No labels should still be close to their originals
        self.assertNotAllClose(ys, 1.0)
        self.assertNotAllClose(ys, 0.0)

    def test_no_augmentation(self):
        """test for no image and bbox augmenation when x_factor,y_factor is 0,0"""
        xs = tf.cast(
            tf.stack(
                [2 * tf.ones((4, 4, 3)), tf.ones((4, 4, 3))],
                axis=0,
            ),
            tf.float32,
        )
        ys = tf.cast(
            tf.stack(
                [
                    tf.constant([[0.3, 0.4, 0.5, 0.6], [0.9, 0.8, 1.0, 1.0]]),
                    tf.constant([[0.3, 0.4, 0.5, 0.6], [0.9, 0.8, 1.0, 1.0]]),
                ],
                axis=0,
            ),
            tf.float32,
        )

        layer = preprocessing.RandomShear(x_factor=0, y_factor=0)
        outputs = layer({"images": xs, "bounding_boxes": ys})
        output_xs, output_ys = outputs["images"], outputs["bounding_boxes"]
        self.assertAllEqual(xs, output_xs)
        self.assertAllEqual(ys, output_ys)

    def test_x_augmentation(self):
        """test for shear bbox augmentation is horizontal direction
        only i.e y_factor=0"""
        xs = tf.cast(
            tf.stack(
                [2 * tf.ones((4, 4, 3)), tf.ones((4, 4, 3))],
                axis=0,
            ),
            tf.float32,
        )
        ys = tf.cast(
            tf.stack(
                [
                    tf.constant([[0.3, 0.4, 0.5, 0.6], [0.9, 0.8, 1.0, 1.0]]),
                    tf.constant([[0.3, 0.4, 0.5, 0.6], [0.9, 0.8, 1.0, 1.0]]),
                ],
                axis=0,
            ),
            tf.float32,
        )

        layer = preprocessing.RandomShear(x_factor=0.5, y_factor=0)
        outputs = layer({"images": xs, "bounding_boxes": ys})
        _, output_ys = outputs["images"], outputs["bounding_boxes"]
        self.assertAllEqual(ys[..., 0], output_ys[..., 0])
        self.assertNotAllClose(ys[..., 1], output_ys[..., 1])
        self.assertAllEqual(ys[..., 2], output_ys[..., 2])
        self.assertNotAllClose(ys[..., 3], output_ys[..., 3])

    def test_y_augmentation(self):
        """test for shear bbox augmentation is vertical direction only i.e x_factor=0"""
        xs = tf.cast(
            tf.stack(
                [2 * tf.ones((4, 4, 3)), tf.ones((4, 4, 3))],
                axis=0,
            ),
            tf.float32,
        )
        ys = tf.cast(
            tf.stack(
                [
                    tf.constant([[0.3, 0.4, 0.5, 0.6], [0.9, 0.8, 1.0, 1.0]]),
                    tf.constant([[0.3, 0.4, 0.5, 0.6], [0.9, 0.8, 1.0, 1.0]]),
                ],
                axis=0,
            ),
            tf.float32,
        )
        layer = preprocessing.RandomShear(x_factor=0, y_factor=0.5)
        outputs = layer({"images": xs, "bounding_boxes": ys})
        _, output_ys = outputs["images"], outputs["bounding_boxes"]
        self.assertAllEqual(ys[..., 1], output_ys[..., 1])
        self.assertNotAllClose(ys[..., 0], output_ys[..., 0])
        self.assertAllEqual(ys[..., 3], output_ys[..., 3])
        self.assertNotAllClose(ys[..., 2], output_ys[..., 2])
