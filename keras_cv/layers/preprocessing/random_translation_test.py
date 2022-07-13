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

from keras_cv.layers.preprocessing.random_translation import RandomTranslation


class RandomTranslationTest(tf.test.TestCase):
    def test_random_translation_output_shapes(self):
        input_images = np.random.random((2, 5, 8, 3)).astype(np.float32)
        expected_output = input_images
        layer = RandomTranslation(height_factor=0.5, width_factor=0.5)
        actual_output = layer(input_images, training=True)
        self.assertEqual(expected_output.shape, actual_output.shape)

    def test_random_translation_inference(self):
        input_images = np.random.random((2, 5, 8, 3)).astype(np.float32)
        expected_output = input_images
        layer = RandomTranslation(height_factor=0.5, width_factor=0.5)
        actual_output = layer(input_images, training=False)
        self.assertAllClose(expected_output, actual_output)

    def test_config_with_custom_name(self):
        layer = RandomTranslation(
            height_factor=0.5, width_factor=0.0, name="image_preproc"
        )
        config = layer.get_config()
        layer_reconstructed = RandomTranslation.from_config(config)
        self.assertEqual(layer_reconstructed.name, layer.name)

    def test_unbatched_image(self):
        input_image = np.reshape(np.arange(0, 16), (4, 4, 1)).astype(np.float32)
        # 180 rotation.
        layer = RandomTranslation(
            height_factor=(0.5, 0.5),
            width_factor=(0.25, 0.25),
            fill_mode="reflect",
        )
        output_image = layer(input_image)
        expected_output = np.asarray(
            [
                [4, 4, 5, 6],
                [0, 0, 1, 2],
                [0, 0, 1, 2],
                [4, 4, 5, 6],
            ]
        ).astype(np.float32)
        expected_output = np.reshape(expected_output, (4, 4, 1))
        self.assertAllClose(expected_output, output_image)

    def test_augment_bbox_dict_input(self):
        input_image = np.random.random((512, 512, 3)).astype(np.float32)
        # Makes sure it supports additionnal data attached to bboxes,
        # like classes and confidences
        bboxes = tf.convert_to_tensor(
            [[200, 200, 400, 400, 3.0], [100, 100, 300, 300, 42.0]]
        )
        input = {"images": input_image, "bounding_boxes": bboxes}
        # 180 rotation.
        layer = RandomTranslation(
            height_factor=(0.5, 0.5),
            width_factor=(0.25, 0.25),
            bounding_box_format="xyxy",
        )
        output_bbox = layer(input)
        expected_output = np.asarray(
            [
                [200 + 512 / 4, 200 + 512 / 2, 400 + 512 / 4, 400 + 512 / 2, 3.0],
                [100 + 512 / 4, 100 + 512 / 2, 300 + 512 / 4, 300 + 512 / 2, 42.0],
            ],
        )
        expected_output = np.minimum(expected_output, 512)
        expected_output = np.reshape(expected_output, (2, 5))
        self.assertAllClose(expected_output, output_bbox["bounding_boxes"])

    def test_augment_keypoints_dict_input(self):
        input_image = np.random.random((512, 512, 3)).astype(np.float32)
        # Makes sure it supports additionnal data attached to bboxes,
        # like classes and confidences
        keypoints = tf.RaggedTensor.from_row_lengths(
            [[200.0, 200.0], [400.0, 400.0], [100.0, 100.0], [300.0, 300.0]], [2, 1, 1]
        )
        input = {"images": input_image, "keypoints": keypoints}
        # 180 rotation.
        layer = RandomTranslation(
            height_factor=(0.5, 0.5),
            width_factor=(0.25, 0.25),
            keypoint_format="xy",
        )
        output_bbox = layer(input)
        expected_output = tf.RaggedTensor.from_row_lengths(
            [
                [200 + 512 / 4, 200.0 + 512 / 2],
                [100 + 512 / 4, 100.0 + 512 / 2],
            ],
            [1, 1, 0],
        )

        self.assertAllClose(expected_output, output_bbox["keypoints"])

    def test_output_dtypes(self):
        inputs = np.array([[[1], [2]], [[3], [4]]], dtype="float64")
        layer = RandomTranslation(0.5, 0.5)
        self.assertAllEqual(layer(inputs).dtype, "float32")
        layer = RandomTranslation(0.5, 0.5, dtype="uint8")
        self.assertAllEqual(layer(inputs).dtype, "uint8")
