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
from keras_cv.backend import ops
from keras_cv.layers.preprocessing.random_rotation import RandomRotation
from keras_cv.tests.test_case import TestCase


class RandomRotationTest(TestCase):
    def test_random_rotation_output_shapes(self):
        input_images = np.random.random((2, 5, 8, 3)).astype(np.float32)
        expected_output = input_images
        layer = RandomRotation(0.5)
        actual_output = layer(input_images, training=True)
        self.assertEqual(expected_output.shape, actual_output.shape)

    def test_random_rotation_on_batched_images_independently(self):
        image = tf.random.uniform((100, 100, 3))
        batched_images = tf.stack((image, image), axis=0)
        layer = RandomRotation(factor=0.5)

        results = layer(batched_images)

        self.assertNotAllClose(results[0], results[1])

    def test_config_with_custom_name(self):
        layer = RandomRotation(0.5, name="image_preproc")
        config = layer.get_config()
        layer_reconstructed = RandomRotation.from_config(config)
        self.assertEqual(layer_reconstructed.name, layer.name)

    def test_unbatched_image(self):
        input_image = np.reshape(np.arange(0, 25), (5, 5, 1)).astype(np.float32)
        # 180 rotation.
        layer = RandomRotation(factor=(0.5, 0.5))
        output_image = layer(input_image)
        expected_output = np.asarray(
            [
                [24, 23, 22, 21, 20],
                [19, 18, 17, 16, 15],
                [14, 13, 12, 11, 10],
                [9, 8, 7, 6, 5],
                [4, 3, 2, 1, 0],
            ]
        ).astype(np.float32)
        expected_output = np.reshape(expected_output, (5, 5, 1))
        self.assertAllClose(expected_output, output_image)

    def test_augment_bounding_boxes(self):
        input_image = np.random.random((512, 512, 3)).astype(np.float32)
        bounding_boxes = {
            "boxes": np.array([[200, 200, 400, 400], [100, 100, 300, 300]]),
            "classes": np.array([1, 2]),
        }
        input = {"images": input_image, "bounding_boxes": bounding_boxes}
        # 180 rotation.
        layer = RandomRotation(factor=(0.5, 0.5), bounding_box_format="xyxy")
        output = layer(input)
        output["bounding_boxes"] = bounding_box.to_dense(
            output["bounding_boxes"]
        )
        expected_bounding_boxes = {
            "boxes": np.array(
                [[112.0, 112.0, 312.0, 312.0], [212.0, 212.0, 412.0, 412.0]],
            ),
            "classes": np.array([1, 2]),
        }
        self.assertAllClose(expected_bounding_boxes, output["bounding_boxes"])

    def test_output_dtypes(self):
        inputs = np.array([[[1], [2]], [[3], [4]]], dtype="float64")
        layer = RandomRotation(0.5)
        self.assertAllEqual(
            ops.convert_to_numpy(layer(inputs)).dtype, "float32"
        )
        layer = RandomRotation(0.5, dtype="uint8")
        self.assertAllEqual(ops.convert_to_numpy(layer(inputs)).dtype, "uint8")

    def test_ragged_bounding_boxes(self):
        input_image = tf.random.uniform((2, 512, 512, 3))
        bounding_boxes = {
            "boxes": tf.ragged.constant(
                [
                    [[200, 200, 400, 400], [100, 100, 300, 300]],
                    [[200, 200, 400, 400]],
                ],
                dtype=tf.float32,
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
        input = {"images": input_image, "bounding_boxes": bounding_boxes}
        layer = RandomRotation(factor=(0.5, 0.5), bounding_box_format="xyxy")
        output = layer(input)
        expected_output = {
            "boxes": tf.ragged.constant(
                [
                    [
                        [112.0, 112.0, 312.0, 312.0],
                        [212.0, 212.0, 412.0, 412.0],
                    ],
                    [[112.0, 112.0, 312.0, 312.0]],
                ],
                dtype=tf.float32,
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
        expected_output = bounding_box.to_dense(expected_output)
        output["bounding_boxes"] = bounding_box.to_dense(
            output["bounding_boxes"]
        )

        self.assertAllClose(
            expected_output["boxes"], output["bounding_boxes"]["boxes"]
        )
        self.assertAllClose(
            expected_output["classes"],
            output["bounding_boxes"]["classes"],
        )

    def test_augment_sparse_segmentation_mask(self):
        num_classes = 8

        input_images = np.random.random((2, 20, 20, 3)).astype(np.float32)
        # Masks are all 0s or 8s, to verify that when we rotate we don't do bad
        # mask interpolation to either a 0 or a 7
        masks = np.random.randint(2, size=(2, 20, 20, 1)) * (num_classes - 1)
        inputs = {"images": input_images, "segmentation_masks": masks}

        # Attempting to rotate a sparse mask without specifying num_classes
        # fails.
        bad_layer = RandomRotation(factor=(0.25, 0.25))
        with self.assertRaisesRegex(ValueError, "masks must be one-hot"):
            outputs = bad_layer(inputs)

        # 90 degree rotation.
        layer = RandomRotation(
            factor=(0.25, 0.25), segmentation_classes=num_classes
        )
        outputs = layer(inputs)
        expected_masks = np.rot90(masks, axes=(1, 2))
        self.assertAllClose(expected_masks, outputs["segmentation_masks"])

        # 45-degree rotation. Only verifies that no interpolation takes place.
        layer = RandomRotation(
            factor=(0.125, 0.125), segmentation_classes=num_classes
        )
        outputs = layer(inputs)
        self.assertAllInSet(
            ops.convert_to_numpy(outputs["segmentation_masks"]), [0, 7]
        )

    def test_augment_one_hot_segmentation_mask(self):
        num_classes = 8

        input_images = np.random.random((2, 20, 20, 3)).astype(np.float32)
        masks = np.array(
            tf.one_hot(
                np.random.randint(num_classes, size=(2, 20, 20)), num_classes
            )
        )
        inputs = {"images": input_images, "segmentation_masks": masks}

        # 90 rotation.
        layer = RandomRotation(factor=(0.25, 0.25))
        outputs = layer(inputs)

        expected_masks = np.rot90(masks, axes=(1, 2))

        self.assertAllClose(expected_masks, outputs["segmentation_masks"])
