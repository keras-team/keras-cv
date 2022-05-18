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

from keras_cv.layers.preprocessing.mix_up import MixUp

NUM_CLASSES = 10


class MixUpTest(tf.test.TestCase):
    def test_return_shapes(self):
        xs = tf.ones((2, 512, 512, 3))
        # randomly sample labels
        ys_labels = tf.random.categorical(tf.math.log([[0.5, 0.5]]), 2)
        ys_labels = tf.squeeze(ys_labels)
        ys_labels = tf.one_hot(ys_labels, NUM_CLASSES)

        # randomly sample bounding boxes
        ys_bounding_boxes = tf.random.uniform((2, 3, 5), 0, 1)

        layer = MixUp()
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
        self.assertEqual(ys_bounding_boxes.shape, [2, 6, 5])

    def test_mix_up_call_results(self):
        xs = tf.cast(
            tf.stack(
                [2 * tf.ones((4, 4, 3)), tf.ones((4, 4, 3))],
                axis=0,
            ),
            tf.float32,
        )
        ys = tf.one_hot(tf.constant([0, 1]), 2)

        layer = MixUp()
        outputs = layer({"images": xs, "labels": ys})
        xs, ys = outputs["images"], outputs["labels"]

        # None of the individual values should still be close to 1 or 0
        self.assertNotAllClose(xs, 1.0)
        self.assertNotAllClose(xs, 2.0)

        # No labels should still be close to their originals
        self.assertNotAllClose(ys, 1.0)
        self.assertNotAllClose(ys, 0.0)

    def test_in_tf_function(self):
        xs = tf.cast(
            tf.stack(
                [2 * tf.ones((4, 4, 3)), tf.ones((4, 4, 3))],
                axis=0,
            ),
            tf.float32,
        )
        ys = tf.one_hot(tf.constant([0, 1]), 2)

        layer = MixUp()

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
            tf.stack([2 * tf.ones((100, 100, 1)), tf.ones((100, 100, 1))], axis=0),
            tf.float32,
        )
        layer = MixUp()
        with self.assertRaisesRegexp(ValueError, "expects inputs in a dictionary"):
            _ = layer(xs)

    def test_single_image_input(self):
        xs = tf.ones((512, 512, 3))
        ys = tf.one_hot(tf.constant([1]), 2)
        inputs = {"images": xs, "labels": ys}
        layer = MixUp()
        with self.assertRaisesRegexp(
            ValueError, "MixUp received a single image to `call`"
        ):
            _ = layer(inputs)

    def test_int_labels(self):
        xs = tf.ones((2, 512, 512, 3))
        ys = tf.one_hot(tf.constant([1, 0]), 2, dtype=tf.int32)
        inputs = {"images": xs, "labels": ys}
        layer = MixUp()
        with self.assertRaisesRegexp(ValueError, "MixUp received labels with type"):
            _ = layer(inputs)

    def test_image_input(self):
        xs = tf.ones((2, 512, 512, 3))
        layer = MixUp()
        with self.assertRaisesRegexp(
            ValueError, "MixUp expects inputs in a dictionary with format"
        ):
            _ = layer(xs)
