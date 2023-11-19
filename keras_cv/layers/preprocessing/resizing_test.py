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
import pytest
import tensorflow as tf
from absl.testing import parameterized

from keras_cv import layers as cv_layers
from keras_cv.backend.config import keras_3
from keras_cv.tests.test_case import TestCase


class ResizingTest(TestCase):
    def _run_output_shape_test(self, kwargs, height, width):
        kwargs.update({"height": height, "width": width})
        layer = cv_layers.Resizing(**kwargs)

        inputs = tf.random.uniform((2, 5, 8, 3))
        outputs = layer(inputs)
        self.assertEqual(outputs.shape, (2, height, width, 3))

    @parameterized.named_parameters(
        ("down_sample_bilinear_2_by_2", {"interpolation": "bilinear"}, 2, 2),
        ("down_sample_bilinear_3_by_2", {"interpolation": "bilinear"}, 3, 2),
        ("down_sample_nearest_2_by_2", {"interpolation": "nearest"}, 2, 2),
        ("down_sample_nearest_3_by_2", {"interpolation": "nearest"}, 3, 2),
        ("down_sample_area_2_by_2", {"interpolation": "area"}, 2, 2),
        ("down_sample_area_3_by_2", {"interpolation": "area"}, 3, 2),
        (
            "down_sample_crop_to_aspect_ratio_3_by_2",
            {
                "interpolation": "bilinear",
                "crop_to_aspect_ratio": True,
            },
            3,
            2,
        ),
    )
    def test_down_sampling(self, kwargs, height, width):
        self._run_output_shape_test(kwargs, height, width)

    @parameterized.named_parameters(
        ("up_sample_bilinear_10_by_12", {"interpolation": "bilinear"}, 10, 12),
        ("up_sample_bilinear_12_by_12", {"interpolation": "bilinear"}, 12, 12),
        ("up_sample_nearest_10_by_12", {"interpolation": "nearest"}, 10, 12),
        ("up_sample_nearest_12_by_12", {"interpolation": "nearest"}, 12, 12),
        ("up_sample_area_10_by_12", {"interpolation": "area"}, 10, 12),
        ("up_sample_area_12_by_12", {"interpolation": "area"}, 12, 12),
        (
            "up_sample_crop_to_aspect_ratio_12_by_14",
            {
                "interpolation": "bilinear",
                "crop_to_aspect_ratio": True,
            },
            12,
            14,
        ),
    )
    def test_up_sampling(self, kwargs, expected_height, expected_width):
        self._run_output_shape_test(kwargs, expected_height, expected_width)

    def test_down_sampling_numeric(self):
        for dtype in (np.int64, np.float32):
            input_image = np.reshape(np.arange(0, 16), (1, 4, 4, 1)).astype(
                dtype
            )
            layer = cv_layers.Resizing(
                height=2, width=2, interpolation="nearest"
            )
            output_image = layer(input_image)
            # pyformat: disable
            expected_output = np.asarray([[5, 7], [13, 15]]).astype(dtype)
            # pyformat: enable
            expected_output = np.reshape(expected_output, (1, 2, 2, 1))
            self.assertAllEqual(expected_output, output_image)

    def test_up_sampling_numeric(self):
        for dtype in (np.int64, np.float32):
            input_image = np.reshape(np.arange(0, 4), (1, 2, 2, 1)).astype(
                dtype
            )
            layer = cv_layers.Resizing(
                height=4, width=4, interpolation="nearest"
            )
            output_image = layer(input_image)
            # pyformat: disable
            expected_output = np.asarray(
                [[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3]]
            ).astype(dtype)
            # pyformat: enable
            expected_output = np.reshape(expected_output, (1, 4, 4, 1))
            self.assertAllEqual(expected_output, output_image)

    @parameterized.named_parameters(
        ("reshape_bilinear_10_by_4", {"interpolation": "bilinear"}, 10, 4)
    )
    def test_reshaping(self, kwargs, expected_height, expected_width):
        self._run_output_shape_test(kwargs, expected_height, expected_width)

    def test_invalid_interpolation(self):
        with self.assertRaises(NotImplementedError):
            cv_layers.Resizing(5, 5, interpolation="invalid_interpolation")

    def test_config_with_custom_name(self):
        layer = cv_layers.Resizing(5, 5, name="image_preproc")
        config = layer.get_config()
        layer_1 = cv_layers.Resizing.from_config(config)
        self.assertEqual(layer_1.name, layer.name)

    def test_crop_to_aspect_ratio(self):
        input_image = np.reshape(np.arange(0, 16), (1, 4, 4, 1)).astype(
            "float32"
        )
        layer = cv_layers.Resizing(4, 2, crop_to_aspect_ratio=True)
        output_image = layer(input_image)
        expected_output = np.asarray(
            [
                [1, 2],
                [5, 6],
                [9, 10],
                [13, 14],
            ]
        ).astype("float32")
        expected_output = np.reshape(expected_output, (1, 4, 2, 1))
        self.assertAllEqual(expected_output, output_image)

    def test_unbatched_image(self):
        input_image = np.reshape(np.arange(0, 16), (4, 4, 1)).astype("float32")
        layer = cv_layers.Resizing(2, 2, interpolation="nearest")
        output_image = layer(input_image)
        expected_output = np.asarray(
            [
                [5, 7],
                [13, 15],
            ]
        ).astype("float32")
        expected_output = np.reshape(expected_output, (2, 2, 1))
        self.assertAllEqual(expected_output, output_image)

    @parameterized.named_parameters(
        ("crop_to_aspect_ratio_false", False),
        ("crop_to_aspect_ratio_true", True),
    )
    @pytest.mark.tf_keras_only
    def test_ragged_image(self, crop_to_aspect_ratio):
        inputs = tf.ragged.constant(
            [
                np.ones((8, 8, 1)),
                np.ones((8, 4, 1)),
                np.ones((4, 8, 1)),
                np.ones((2, 2, 1)),
            ],
            dtype="float32",
        )
        layer = cv_layers.Resizing(
            2,
            2,
            interpolation="nearest",
            crop_to_aspect_ratio=crop_to_aspect_ratio,
        )
        outputs = layer(inputs)
        expected_output = [
            [[[1.0], [1.0]], [[1.0], [1.0]]],
            [[[1.0], [1.0]], [[1.0], [1.0]]],
            [[[1.0], [1.0]], [[1.0], [1.0]]],
            [[[1.0], [1.0]], [[1.0], [1.0]]],
        ]
        self.assertIsInstance(outputs, tf.Tensor)
        self.assertNotIsInstance(outputs, tf.RaggedTensor)
        self.assertAllEqual(expected_output, outputs)

    def test_output_dtypes(self):
        inputs = np.array([[[1], [2]], [[3], [4]]], dtype="float64")
        layer = cv_layers.Resizing(2, 2)
        self.assertAllEqual(layer(inputs).dtype, "float32")
        layer = cv_layers.Resizing(2, 2, dtype="uint8")
        self.assertAllEqual(layer(inputs).dtype, "uint8")

    @pytest.mark.skipif(
        keras_3(), reason="ragged tests not yet enabled for keras 3"
    )
    @parameterized.named_parameters(
        ("batch_crop_to_aspect_ratio", True, False, True),
        ("batch_dont_crop_to_aspect_ratio", False, False, True),
        ("single_sample_crop_to_aspect_ratio", True, False, False),
        ("single_sample_dont_crop_to_aspect_ratio", False, False, False),
        ("batch_pad_to_aspect_ratio", False, True, True),
        ("single_sample_pad_to_aspect_ratio", False, True, False),
    )
    def test_static_shape_inference(
        self, crop_to_aspect_ratio, pad_to_aspect_ratio, batch
    ):
        channels = 3
        input_height = 8
        input_width = 8
        target_height = 4
        target_width = 6
        layer = cv_layers.Resizing(
            target_height,
            target_width,
            crop_to_aspect_ratio=crop_to_aspect_ratio,
            pad_to_aspect_ratio=pad_to_aspect_ratio,
        )
        unit_test = self

        @tf.function
        def tf_function(img):
            unit_test.assertListEqual(
                [input_height, input_width, channels], img.shape.as_list()[-3:]
            )
            img = layer(img)
            unit_test.assertListEqual(
                [target_height, target_width, channels],
                img.shape.as_list()[-3:],
            )
            return img

        if batch:
            input_shape = (2, input_height, input_width, channels)
        else:
            input_shape = (input_height, input_width, channels)
        img_data = np.random.random(size=input_shape).astype("float32")
        tf_function(img_data)

    @pytest.mark.tf_keras_only
    def test_pad_to_size_with_bounding_boxes_ragged_images(self):
        images = tf.ragged.constant(
            [
                np.ones((8, 8, 3)),
                np.ones((8, 4, 3)),
                np.ones((4, 8, 3)),
                np.ones((2, 2, 3)),
            ],
            dtype="float32",
        )
        boxes = {
            "boxes": tf.ragged.stack(
                [
                    np.ones((3, 4), dtype="float32"),
                    np.ones((5, 4), dtype="float32"),
                    np.ones((3, 4), dtype="float32"),
                    np.ones((2, 4), dtype="float32"),
                ],
            ),
            "classes": tf.ragged.stack(
                [
                    np.ones((3,), dtype="float32"),
                    np.ones((5,), dtype="float32"),
                    np.ones((3,), dtype="float32"),
                    np.ones((2,), dtype="float32"),
                ],
            ),
        }
        layer = cv_layers.Resizing(
            4, 4, pad_to_aspect_ratio=True, bounding_box_format="xyxy"
        )
        inputs = {"images": images, "bounding_boxes": boxes}
        outputs = layer(inputs)
        self.assertListEqual(
            [4, 4, 4, 3],
            outputs["images"].shape.as_list(),
        )

    @pytest.mark.tf_keras_only
    def test_pad_to_size_with_bounding_boxes_ragged_images_upsample(self):
        images = tf.ragged.constant(
            [
                np.ones((8, 8, 3)),
                np.ones((8, 4, 3)),
                np.ones((4, 8, 3)),
                np.ones((2, 2, 3)),
            ],
            dtype="float32",
        )
        boxes = {
            "boxes": tf.ragged.stack(
                [
                    np.ones((3, 4), dtype="float32"),
                    np.ones((5, 4), dtype="float32"),
                    np.ones((3, 4), dtype="float32"),
                    np.ones((2, 4), dtype="float32"),
                ],
            ),
            "classes": tf.ragged.stack(
                [
                    np.ones((3,), dtype="float32"),
                    np.ones((5,), dtype="float32"),
                    np.ones((3,), dtype="float32"),
                    np.ones((2,), dtype="float32"),
                ],
            ),
        }
        layer = cv_layers.Resizing(
            16, 16, pad_to_aspect_ratio=True, bounding_box_format="xyxy"
        )
        inputs = {"images": images, "bounding_boxes": boxes}
        outputs = layer(inputs)
        self.assertListEqual(
            [4, 16, 16, 3],
            outputs["images"].shape.as_list(),
        )

        self.assertAllEqual(outputs["images"][1][:, :8, :], np.ones((16, 8, 3)))
        self.assertAllEqual(
            outputs["images"][1][:, -8:, :], np.zeros((16, 8, 3))
        )

    def test_resize_with_mask(self):
        input_images = np.random.normal(size=(2, 4, 4, 3))
        seg_masks = np.random.uniform(
            low=0.0, high=3.0, size=(2, 4, 4, 3)
        ).astype("int32")
        inputs = {
            "images": input_images,
            "segmentation_masks": seg_masks,
        }

        layer = cv_layers.Resizing(2, 2)
        outputs = layer(inputs)

        expected_output_images = tf.image.resize(input_images, size=(2, 2))
        expected_output_seg_masks = tf.image.resize(
            seg_masks, size=(2, 2), method="nearest"
        )

        self.assertAllEqual(expected_output_images, outputs["images"])
        self.assertAllEqual(
            expected_output_seg_masks, outputs["segmentation_masks"]
        )
