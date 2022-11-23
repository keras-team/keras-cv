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
import numpy as np
import tensorflow as tf
from absl.testing import parameterized

from keras_cv import bounding_box
from keras_cv.layers import preprocessing


class RandomCropAndResizeTest(tf.test.TestCase, parameterized.TestCase):
    height, width = 300, 300
    batch_size = 4
    target_size = (224, 224)
    seed = 42

    def test_train_augments_image(self):
        # Checks if original and augmented images are different

        input_image_shape = (self.batch_size, self.height, self.width, 3)
        image = tf.random.uniform(shape=input_image_shape, seed=self.seed)

        layer = preprocessing.RandomCropAndResize(
            target_size=self.target_size,
            aspect_ratio_factor=(3 / 4, 4 / 3),
            crop_area_factor=(0.8, 1.0),
            seed=self.seed,
        )
        output = layer(image, training=True)

        input_image_resized = tf.image.resize(image, self.target_size)

        self.assertNotAllClose(output, input_image_resized)

    def test_grayscale(self):
        input_image_shape = (self.batch_size, self.height, self.width, 1)
        image = tf.random.uniform(shape=input_image_shape)

        layer = preprocessing.RandomCropAndResize(
            target_size=self.target_size,
            aspect_ratio_factor=(3 / 4, 4 / 3),
            crop_area_factor=(0.8, 1.0),
        )
        output = layer(image, training=True)

        input_image_resized = tf.image.resize(image, self.target_size)

        self.assertAllEqual(output.shape, (4, 224, 224, 1))
        self.assertNotAllClose(output, input_image_resized)

    def test_preserves_image(self):
        image_shape = (self.batch_size, self.height, self.width, 3)
        image = tf.random.uniform(shape=image_shape)

        layer = preprocessing.RandomCropAndResize(
            target_size=self.target_size,
            aspect_ratio_factor=(3 / 4, 4 / 3),
            crop_area_factor=(0.8, 1.0),
        )

        input_resized = tf.image.resize(image, self.target_size)
        output = layer(image, training=False)

        self.assertAllClose(output, input_resized)

    @parameterized.named_parameters(
        ("Not tuple or list", dict()),
        ("Length not equal to 2", [1, 2, 3]),
        ("Members not int", (2.3, 4.5)),
        ("Single integer", 5),
    )
    def test_target_size_errors(self, target_size):
        with self.assertRaisesRegex(
            ValueError,
            "`target_size` must be tuple of two integers. Received target_size=(.*)",
        ):
            _ = preprocessing.RandomCropAndResize(
                target_size=target_size,
                aspect_ratio_factor=(3 / 4, 4 / 3),
                crop_area_factor=(0.8, 1.0),
            )

    @parameterized.named_parameters(
        ("Not tuple or list", dict()),
        ("Single integer", 5),
        ("Single float", 5.0),
    )
    def test_aspect_ratio_factor_errors(self, aspect_ratio_factor):
        with self.assertRaisesRegex(
            ValueError,
            "`aspect_ratio_factor` must be tuple of two positive floats or "
            "keras_cv.core.FactorSampler instance. Received aspect_ratio_factor=(.*)",
        ):
            _ = preprocessing.RandomCropAndResize(
                target_size=(224, 224),
                aspect_ratio_factor=aspect_ratio_factor,
                crop_area_factor=(0.8, 1.0),
            )

    @parameterized.named_parameters(
        ("Not tuple or list", dict()),
        ("Single integer", 5),
        ("Single float", 5.0),
    )
    def test_crop_area_factor_errors(self, crop_area_factor):
        with self.assertRaisesRegex(
            ValueError,
            "`crop_area_factor` must be tuple of two positive floats less than or "
            "equal to 1 or keras_cv.core.FactorSampler instance. Received "
            "crop_area_factor=(.*)",
        ):
            _ = preprocessing.RandomCropAndResize(
                target_size=(224, 224),
                aspect_ratio_factor=(3 / 4, 4 / 3),
                crop_area_factor=crop_area_factor,
            )

    def test_augment_sparse_segmentation_mask(self):
        classes = 8

        input_image_shape = (1, self.height, self.width, 3)
        mask_shape = (1, self.height, self.width, 1)
        image = tf.random.uniform(shape=input_image_shape, seed=self.seed)
        mask = np.random.randint(2, size=mask_shape) * (classes - 1)

        inputs = {"images": image, "segmentation_masks": mask}

        # Crop-only to exactly 1/2 of the size
        layer = preprocessing.RandomCropAndResize(
            target_size=(150, 150),
            aspect_ratio_factor=(1, 1),
            crop_area_factor=(1, 1),
            seed=self.seed,
        )
        input_mask_resized = tf.image.crop_and_resize(
            mask, [[0, 0, 1, 1]], [0], (150, 150), "nearest"
        )
        output = layer(inputs, training=True)
        self.assertAllClose(output["segmentation_masks"], input_mask_resized)

        # Crop to an arbitrary size and make sure we don't do bad interpolation
        layer = preprocessing.RandomCropAndResize(
            target_size=(233, 233),
            aspect_ratio_factor=(3 / 4, 4 / 3),
            crop_area_factor=(0.8, 1.0),
            seed=self.seed,
        )
        output = layer(inputs, training=True)
        self.assertAllInSet(output["segmentation_masks"], [0, 7])

    def test_augment_one_hot_segmentation_mask(self):
        classes = 8

        input_image_shape = (1, self.height, self.width, 3)
        mask_shape = (1, self.height, self.width, 1)
        image = tf.random.uniform(shape=input_image_shape, seed=self.seed)
        mask = tf.one_hot(
            tf.squeeze(np.random.randint(2, size=mask_shape) * (classes - 1), axis=-1),
            classes,
        )

        inputs = {"images": image, "segmentation_masks": mask}

        # Crop-only to exactly 1/2 of the size
        layer = preprocessing.RandomCropAndResize(
            target_size=(150, 150),
            aspect_ratio_factor=(1, 1),
            crop_area_factor=(1, 1),
            seed=self.seed,
        )
        input_mask_resized = tf.image.crop_and_resize(
            mask, [[0, 0, 1, 1]], [0], (150, 150), "nearest"
        )
        output = layer(inputs, training=True)
        self.assertAllClose(output["segmentation_masks"], input_mask_resized)

    def test_augment_bounding_box_single(self):
        image = tf.zeros([20, 20, 3])
        boxes = tf.convert_to_tensor([[0, 0, 1, 1, 0]])
        input = {"images": image, "bounding_boxes": boxes}

        layer = preprocessing.RandomCropAndResize(
            target_size=(10, 10),
            crop_area_factor=(0.5**2, 0.5**2),
            aspect_ratio_factor=(1.0, 1.0),
            bounding_box_format="rel_xyxy",
        )
        output = layer(input, training=True)

        expected_output = np.asarray([[0, 0, 1, 1, 0]])
        self.assertAllClose(expected_output, output["bounding_boxes"].to_tensor())

    def test_augment_boxes_batched_input(self):
        image = tf.zeros([20, 20, 3])

        boxes = tf.convert_to_tensor(
            [
                [[0, 0, 1, 1, 0], [0, 0, 1, 1, 0]],
                [[0, 0, 1, 1, 0], [0, 0, 1, 1, 0]],
            ]
        )
        input = {"images": [image, image], "bounding_boxes": boxes}
        layer = preprocessing.RandomCropAndResize(
            target_size=(18, 18),
            crop_area_factor=(0.5**2, 0.5**2),
            aspect_ratio_factor=(1.0, 1.0),
            bounding_box_format="rel_xyxy",
        )
        output = layer(input, training=True)
        expected_output = np.asarray(
            [
                [[0, 0, 1, 1, 0], [0, 0, 1, 1, 0]],
                [[0, 0, 1, 1, 0], [0, 0, 1, 1, 0]],
            ]
        )
        self.assertAllClose(expected_output, output["bounding_boxes"].to_tensor())

    def test_augment_boxes_ragged(self):
        image = tf.zeros([2, 20, 20, 3])
        boxes = tf.ragged.constant(
            [[[0, 0, 1, 1], [0, 0, 1, 1]], [[0, 0, 1, 1]]], dtype=tf.float32
        )
        boxes = bounding_box.add_class_id(boxes)
        input = {"images": image, "bounding_boxes": boxes}
        layer = preprocessing.RandomCropAndResize(
            target_size=(18, 18),
            crop_area_factor=(0.5**2, 0.5**2),
            aspect_ratio_factor=(1.0, 1.0),
            bounding_box_format="rel_xyxy",
        )
        output = layer(input, training=True)

        # the result boxes will still have the entire image in them
        expected_output = tf.ragged.constant(
            [[[0, 0, 1, 1, 0], [0, 0, 1, 1, 0]], [[0, 0, 1, 1, 0]]], dtype=tf.float32
        )
        self.assertAllClose(
            expected_output.to_tensor(-1), output["bounding_boxes"].to_tensor(-1)
        )
