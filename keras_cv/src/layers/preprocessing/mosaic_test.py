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
import pytest
import tensorflow as tf

from keras_cv.src.layers.preprocessing.mosaic import Mosaic
from keras_cv.src.tests.test_case import TestCase

num_classes = 10


class MosaicTest(TestCase):
    def test_return_shapes(self):
        input_shape = (2, 512, 512, 3)
        xs = tf.ones(input_shape)
        # randomly sample labels
        ys_labels = tf.random.categorical(tf.math.log([[0.5, 0.5]]), 2)
        ys_labels = tf.squeeze(ys_labels)
        ys_labels = tf.one_hot(ys_labels, num_classes)

        # randomly sample bounding boxes
        ys_bounding_boxes = {
            "boxes": tf.random.uniform((2, 3, 4), 0, 1),
            "classes": tf.random.uniform((2, 3), 0, 1),
        }
        ys_segmentation_masks = tf.cast(
            2 * tf.random.uniform(input_shape), tf.int32
        )
        layer = Mosaic(bounding_box_format="xywh")
        # mosaic on labels
        outputs = layer(
            {
                "images": xs,
                "labels": ys_labels,
                "bounding_boxes": ys_bounding_boxes,
                "segmentation_masks": ys_segmentation_masks,
            }
        )
        xs, ys_labels, ys_bounding_boxes, ys_segmentation_masks = (
            outputs["images"],
            outputs["labels"],
            outputs["bounding_boxes"],
            outputs["segmentation_masks"],
        )

        self.assertEqual(xs.shape, input_shape)
        self.assertEqual(ys_labels.shape, [2, 10])
        self.assertEqual(ys_bounding_boxes["boxes"].shape, [2, None, 4])
        self.assertEqual(ys_bounding_boxes["classes"].shape, [2, None])
        self.assertEqual(ys_segmentation_masks.shape, input_shape)

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

        layer = Mosaic()

        @tf.function
        def augment(x, y):
            return layer({"images": x, "labels": y})

        outputs = augment(xs, ys)
        xs, ys = outputs["images"], outputs["labels"]

        self.assertEqual(xs.shape, [2, 4, 4, 3])
        self.assertEqual(ys.shape, [2, 2])

    def test_image_input_only(self):
        xs = tf.cast(
            tf.stack(
                [2 * tf.ones((100, 100, 1)), tf.ones((100, 100, 1))], axis=0
            ),
            tf.float32,
        )
        layer = Mosaic()
        with self.assertRaisesRegexp(
            ValueError, "expects inputs in a dictionary"
        ):
            _ = layer(xs)

    def test_single_image_input(self):
        xs = tf.ones((512, 512, 3))
        ys = tf.one_hot(tf.constant([1]), 2)
        inputs = {"images": xs, "labels": ys}
        layer = Mosaic()
        with self.assertRaisesRegexp(
            ValueError, "Mosaic received a single image to `call`"
        ):
            _ = layer(inputs)

    def test_image_input(self):
        xs = tf.ones((2, 512, 512, 3))
        layer = Mosaic()
        with self.assertRaisesRegexp(
            ValueError, "Mosaic expects inputs in a dictionary with format"
        ):
            _ = layer(xs)
