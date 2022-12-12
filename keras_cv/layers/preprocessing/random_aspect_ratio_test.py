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

from keras_cv import bounding_box
from keras_cv import layers


class RandomAspectRatioTest(tf.test.TestCase):
    def test_train_augments_image(self):
        # Checks if original and augmented images are different
        input_image_shape = (8, 100, 100, 3)
        image = tf.random.uniform(shape=input_image_shape)

        layer = layers.RandomAspectRatio(factor=(0.9, 1.1))
        output = layer(image, training=True)
        self.assertNotEqual(output.shape, image.shape)

    def test_inference_preserves_image(self):
        # Checks if original and augmented images are different
        input_image_shape = (8, 100, 100, 3)
        image = tf.random.uniform(shape=input_image_shape)

        layer = layers.RandomAspectRatio(factor=(0.9, 1.1))
        output = layer(image, training=False)
        self.assertAllClose(image, output)

    def test_grayscale(self):
        # Checks if original and augmented images are different
        input_image_shape = (8, 100, 100, 1)
        image = tf.random.uniform(shape=input_image_shape, seed=1223)

        layer = layers.RandomAspectRatio(factor=(0.9, 1.1))
        output = layer(image, training=True)
        self.assertEqual(output.shape[-1], 1)

    def test_augment_boxes_ragged(self):
        image = tf.zeros([2, 20, 20, 3])
        boxes = tf.ragged.constant(
            [[[0.2, 0.12, 1, 1], [0, 0, 0.5, 0.73]], [[0, 0, 1, 1]]], dtype=tf.float32
        )
        boxes = bounding_box.add_class_id(boxes)
        input = {"images": image, "bounding_boxes": boxes}
        layer = layers.RandomAspectRatio(
            factor=(0.9, 1.1), bounding_box_format="rel_xywh"
        )
        output = layer(input, training=True)

        self.assertAllClose(boxes.to_tensor(-1), output["bounding_boxes"].to_tensor(-1))
