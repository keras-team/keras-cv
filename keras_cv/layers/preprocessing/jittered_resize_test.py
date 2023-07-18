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
import tensorflow as tf

from keras_cv import bounding_box
from keras_cv import core
from keras_cv import layers
from keras_cv.tests.test_case import TestCase


class JitteredResizeTest(TestCase):
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
            expected_output["classes"], output["bounding_boxes"]["classes"]
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
            expected_output["classes"], output["bounding_boxes"]["classes"]
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
            expected_output["classes"], output["bounding_boxes"]["classes"]
        )

    def test_independence_of_jittered_resize_on_batched_images(self):
        image = tf.random.uniform((100, 100, 3))
        batched_images = tf.stack((image, image), axis=0)
        layer = layers.JitteredResize(
            target_size=self.target_size,
            scale_factor=(3 / 4, 4 / 3),
            seed=self.seed,
        )

        results = layer(batched_images)

        self.assertNotAllClose(results[0], results[1])

    def test_augments_segmentation_masks(self):
        input_shape = (self.batch_size, self.height, self.width, 3)
        image = tf.random.uniform(shape=input_shape, seed=self.seed)
        mask = tf.cast(
            3 * tf.random.uniform(shape=input_shape, seed=self.seed),
            tf.int32,
        )

        inputs = {"images": image, "segmentation_masks": mask}

        layer = layers.JitteredResize(
            target_size=self.target_size,
            scale_factor=(3 / 4, 4 / 3),
            seed=self.seed,
        )
        output = layer(inputs, training=True)

        input_image_resized = tf.image.resize(image, self.target_size)
        input_mask_resized = tf.image.resize(
            mask, self.target_size, method="nearest"
        )

        self.assertNotAllClose(output["images"], input_image_resized)
        self.assertNotAllClose(output["segmentation_masks"], input_mask_resized)

    def test_config_with_custom_name(self):
        layer = layers.JitteredResize(
            target_size=self.target_size,
            scale_factor=(3 / 4, 4 / 3),
            name="image_preproc",
        )
        config = layer.get_config()
        layer_1 = layers.JitteredResize.from_config(config)
        self.assertEqual(layer_1.name, layer.name)

    def test_output_dtypes(self):
        inputs = np.array([[[1], [2]], [[3], [4]]], dtype="float64")
        layer = layers.JitteredResize(
            target_size=self.target_size,
            scale_factor=(3 / 4, 4 / 3),
        )
        self.assertAllEqual(layer(inputs).dtype, "float32")
        layer = layers.JitteredResize(
            target_size=self.target_size,
            scale_factor=(3 / 4, 4 / 3),
            dtype="uint8",
        )
        self.assertAllEqual(layer(inputs).dtype, "uint8")

    def test_config(self):
        layer = layers.JitteredResize(
            target_size=self.target_size,
            scale_factor=(3 / 4, 4 / 3),
            bounding_box_format="xyxy",
        )
        config = layer.get_config()
        self.assertEqual(config["target_size"], self.target_size)
        self.assertTrue(
            isinstance(config["scale_factor"], core.UniformFactorSampler)
        )
        self.assertEqual(config["bounding_box_format"], "xyxy")
