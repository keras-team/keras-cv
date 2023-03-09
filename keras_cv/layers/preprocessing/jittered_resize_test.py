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

from keras_cv import bounding_box
from keras_cv import layers


class JitteredResizeTest(tf.test.TestCase, parameterized.TestCase):
    batch_size = 4
    height = 9
    width = 8
    seed = 13
    target_size = (4, 4)

    def test_train_augments_image(self):
        # Checks if original and augmented images are different

        input_image_shape = (self.batch_size, self.height, self.width, 3)
        image = tf.random.uniform(shape=input_image_shape, seed=self.seed)

        layer = layers.JitteredResize(
            target_size=self.target_size,
            scale_factor=(3 / 4, 4 / 3),
            seed=self.seed,
        )
        output = layer(image, training=True)

        input_image_resized = tf.image.resize(image, self.target_size)

        self.assertNotAllClose(output, input_image_resized)

    def test_augment_bounding_box_single(self):
        image = tf.zeros([20, 20, 3])
        boxes = {
            "boxes": tf.convert_to_tensor([[0, 0, 1, 1]], dtype=tf.float32),
            "classes": tf.convert_to_tensor([0], dtype=tf.float32),
        }
        input = {"images": image, "bounding_boxes": boxes}

        layer = layers.JitteredResize(
            target_size=self.target_size,
            scale_factor=(3 / 4, 4 / 3),
            bounding_box_format="rel_xywh",
            seed=self.seed,
        )
        output = layer(input, training=True)
        output["bounding_boxes"] = bounding_box.to_dense(
            output["bounding_boxes"]
        )
        expected_output = {
            "boxes": tf.convert_to_tensor([[0, 0, 1, 1]], dtype=tf.float32),
            "classes": tf.convert_to_tensor([0], dtype=tf.float32),
        }
        self.assertAllClose(
            expected_output["boxes"],
            output["bounding_boxes"]["boxes"],
        )
        self.assertAllClose(
            expected_output["num_classes"], output["bounding_boxes"]["num_classes"]
        )

    def test_augment_boxes_batched_input(self):
        image = tf.zeros([20, 20, 3])

        bounding_boxes = {
            "classes": tf.convert_to_tensor([[0, 0], [0, 0]]),
            "boxes": tf.convert_to_tensor(
                [
                    [[0, 0, 1, 1], [0, 0, 1, 1]],
                    [[0, 0, 1, 1], [0, 0, 1, 1]],
                ]
            ),
        }
        input = {"images": [image, image], "bounding_boxes": bounding_boxes}

        layer = layers.JitteredResize(
            target_size=self.target_size,
            scale_factor=(3 / 4, 4 / 3),
            bounding_box_format="rel_xyxy",
            seed=self.seed,
        )
        output = layer(input, training=True)
        output["bounding_boxes"] = bounding_box.to_dense(
            output["bounding_boxes"]
        )
        expected_output = {
            "classes": tf.convert_to_tensor([[0, 0], [0, 0]], dtype=tf.float32),
            "boxes": tf.convert_to_tensor(
                [
                    [[0, 0, 1, 1], [0, 0, 1, 1]],
                    [[0, 0, 1, 1], [0, 0, 1, 1]],
                ],
                dtype=tf.float32,
            ),
        }
        self.assertAllClose(
            expected_output["boxes"],
            output["bounding_boxes"]["boxes"],
        )
        self.assertAllClose(
            expected_output["num_classes"], output["bounding_boxes"]["num_classes"]
        )

    def test_augment_boxes_ragged(self):
        image = tf.zeros([2, 20, 20, 3])
        boxes = {
            "boxes": tf.ragged.constant(
                [[[0, 0, 1, 1], [0, 0, 1, 1]], [[0, 0, 1, 1]]], dtype=tf.float32
            ),
            "classes": tf.ragged.constant(
                [
                    [
                        0,
                        0,
                    ],
                    [0],
                ],
                dtype=tf.float32,
            ),
        }
        input = {"images": image, "bounding_boxes": boxes}

        layer = layers.JitteredResize(
            target_size=self.target_size,
            scale_factor=(3 / 4, 4 / 3),
            bounding_box_format="rel_xyxy",
            seed=self.seed,
        )
        output = layer(input, training=True)
        # the result boxes will still have the entire image in them
        expected_output = {
            "boxes": tf.ragged.constant(
                [[[0, 0, 1, 1], [0, 0, 1, 1]], [[0, 0, 1, 1]]], dtype=tf.float32
            ),
            "classes": tf.ragged.constant(
                [
                    [
                        0.0,
                        0.0,
                    ],
                    [0.0],
                ],
                dtype=tf.float32,
            ),
        }
        self.assertAllClose(
            expected_output["boxes"].to_tensor(),
            output["bounding_boxes"]["boxes"].to_tensor(),
        )
        self.assertAllClose(
            expected_output["num_classes"], output["bounding_boxes"]["num_classes"]
        )

    def test_augment_inference_mode(self):
        image = tf.zeros([20, 20, 3])
        boxes = {
            "boxes": tf.convert_to_tensor([[0, 0, 1, 1]], dtype=tf.float32),
            "classes": tf.convert_to_tensor([0], dtype=tf.float32),
        }
        input = {"images": image, "bounding_boxes": boxes}

        layer = layers.JitteredResize(
            target_size=self.target_size,
            scale_factor=(3 / 4, 4 / 3),
            bounding_box_format="rel_xywh",
            seed=self.seed,
        )
        output = layer(input, training=False)
        expected_output = layer._inference_resizing(output)
        self.assertAllClose(
            expected_output["bounding_boxes"]["boxes"],
            output["bounding_boxes"]["boxes"],
        )
        self.assertAllClose(
            expected_output["bounding_boxes"]["num_classes"],
            output["bounding_boxes"]["num_classes"],
        )
        self.assertAllClose(
            expected_output["images"],
            output["images"],
        )
