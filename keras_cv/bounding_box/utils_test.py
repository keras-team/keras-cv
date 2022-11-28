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


class BoundingBoxUtilTestCase(tf.test.TestCase):
    def test_clip_to_image(self):
        # Test xyxy format unbatched
        height = 256
        width = 256
        bboxes = tf.convert_to_tensor(
            [[200, 200, 400, 400, 0], [100, 100, 300, 300, 0]]
        )
        image = tf.ones(shape=(height, width, 3))
        bboxes_out = bounding_box.clip_to_image(
            bboxes, bounding_box_format="xyxy", images=image
        )
        self.assertAllGreaterEqual(bboxes_out, 0)
        x1, y1, x2, y2, rest = tf.split(bboxes_out, [1, 1, 1, 1, -1], axis=1)
        self.assertAllLessEqual([x1, x2], width)
        self.assertAllLessEqual([y1, y2], height)
        # Test relative format batched
        image = tf.ones(shape=(1, height, width, 3))
        bboxes = tf.convert_to_tensor(
            [[[0.2, -1, 1.2, 0.3, 0], [0.4, 1.5, 0.2, 0.3, 0]]]
        )
        bboxes_out = bounding_box.clip_to_image(
            bboxes, bounding_box_format="rel_xyxy", images=image
        )
        self.assertAllLessEqual(bboxes_out, 1)

    def test_clip_to_image_filters_fully_out_bounding_boxes(self):
        # Test xyxy format unbatched
        height = 256
        width = 256
        bounding_boxes = tf.convert_to_tensor(
            [[257, 257, 400, 400, 0], [100, 100, 300, 300, 0]]
        )
        image = tf.ones(shape=(height, width, 3))
        bounding_boxes = bounding_box.clip_to_image(
            bounding_boxes, bounding_box_format="xyxy", images=image
        )
        self.assertAllEqual(
            bounding_boxes,
            tf.convert_to_tensor([[-1, -1, -1, -1, -1], [100, 100, 256, 256, 0]]),
        )

    def test_clip_to_image_filters_fully_out_bounding_boxes_negative_area(self):
        # Test xyxy format unbatched
        height = 256
        width = 256
        bounding_boxes = tf.convert_to_tensor(
            [[0, float("NaN"), 100, 100, 0], [100, 100, 300, 300, 0]]
        )
        image = tf.ones(shape=(height, width, 3))
        bounding_boxes = bounding_box.clip_to_image(
            bounding_boxes, bounding_box_format="xyxy", images=image
        )
        self.assertAllEqual(
            bounding_boxes,
            tf.convert_to_tensor([[-1, -1, -1, -1, -1], [100, 100, 256, 256, 0]]),
        )

    def test_clip_to_image_filters_nans(self):
        # Test xyxy format unbatched
        height = 256
        width = 256
        bounding_boxes = tf.convert_to_tensor(
            [[257, 257, 100, 100, 0], [100, 100, 300, 300, 0]]
        )
        image = tf.ones(shape=(height, width, 3))
        bounding_boxes = bounding_box.clip_to_image(
            bounding_boxes, bounding_box_format="xyxy", images=image
        )
        self.assertAllEqual(
            bounding_boxes,
            tf.convert_to_tensor([[-1, -1, -1, -1, -1], [100, 100, 256, 256, 0]]),
        )

    def test_pad_with_sentinels(self):
        bounding_boxes = tf.ragged.constant(
            [[[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]], [[1, 2, 3, 4, 5]]]
        )
        padded_bounding_boxes = bounding_box.pad_with_sentinels(bounding_boxes)
        expected_output = tf.constant(
            [
                [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]],
                [[1, 2, 3, 4, 5], [-1, -1, -1, -1, -1]],
            ]
        )
        self.assertAllEqual(padded_bounding_boxes, expected_output)

    def test_filter_sentinels(self):
        bounding_boxes = tf.ragged.constant(
            [[[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, -1]], [[1, 2, 3, 4, 5]]]
        )
        filtered_bounding_boxes = bounding_box.filter_sentinels(bounding_boxes)
        expected_output = tf.ragged.constant(
            [[[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]], [[1, 2, 3, 4, 5]]], ragged_rank=1
        )
        self.assertAllEqual(filtered_bounding_boxes, expected_output)

    def test_filter_sentinels_unbatched(self):
        bounding_boxes = tf.convert_to_tensor(
            [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, -1]]
        )
        filtered_bounding_boxes = bounding_box.filter_sentinels(bounding_boxes)
        expected_output = tf.convert_to_tensor(
            [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]],
        )
        print(filtered_bounding_boxes, expected_output)
        self.assertAllEqual(filtered_bounding_boxes, expected_output)

    def test_filter_sentinels_tensor(self):
        bounding_boxes = tf.convert_to_tensor(
            [
                [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]],
                [[1, 2, 3, 4, 5], [1, 2, 3, 4, -1]],
            ]
        )
        filtered_bounding_boxes = bounding_box.filter_sentinels(bounding_boxes)

        expected_output = tf.ragged.constant(
            [[[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]], [[1, 2, 3, 4, 5]]], ragged_rank=1
        )
        self.assertAllEqual(filtered_bounding_boxes, expected_output)

    def test_pad_with_class_id_ragged(self):
        bounding_boxes = tf.ragged.constant(
            [[[1, 2, 3, 4], [1, 2, 3, 4]], [[1, 2, 3, 4]]]
        )
        padded_bounding_boxes = bounding_box.add_class_id(bounding_boxes)
        expected_output = tf.ragged.constant(
            [[[1, 2, 3, 4, 0], [1, 2, 3, 4, 0]], [[1, 2, 3, 4, 0]]]
        )
        self.assertAllEqual(padded_bounding_boxes, expected_output)

    def test_pad_with_class_id_unbatched(self):
        bounding_boxes = tf.convert_to_tensor([[1, 2, 3, 4], [1, 2, 3, 4]])
        padded_bounding_boxes = bounding_box.add_class_id(bounding_boxes)
        expected_output = tf.convert_to_tensor([[1, 2, 3, 4, 0], [1, 2, 3, 4, 0]])
        self.assertAllEqual(padded_bounding_boxes, expected_output)

    def test_pad_with_class_id_exists(self):
        bounding_boxes = tf.ragged.constant(
            [[[1, 2, 3, 4, 0], [1, 2, 3, 4, 0]], [[1, 2, 3, 4, 0]]]
        )
        with self.assertRaisesRegex(
            ValueError,
            "The number of values along the final axis of `bounding_boxes` is "
            "expected to be 4. But got 5.",
        ):
            bounding_box.add_class_id(bounding_boxes)

    def test_pad_with_class_id_wrong_rank(self):
        bounding_boxes = tf.ragged.constant(
            [[[[1, 2, 3, 4], [1, 2, 3, 4]], [[1, 2, 3, 4]]]]
        )
        with self.assertRaisesRegex(
            ValueError,
            f"`bounding_boxes` should be of rank 2 or 3. However "
            f"add_class_id received `bounding_boxes` of rank={4}",
        ):
            bounding_box.add_class_id(bounding_boxes)

    def test_preserve_rel_util(self):
        target_format = "xyxy"
        bounding_box_format = "rel_yxyx"

        target = bounding_box.preserve_rel(
            target_bounding_box_format=target_format,
            bounding_box_format=bounding_box_format,
        )
        self.assertEqual(target, "rel_xyxy")

    def test_preserve_rel_util_errors(self):
        # relative targets should throw an error
        target_format = "rel_xyxy"
        bounding_box_format = "rel_yxyx"

        with self.assertRaisesRegex(
            ValueError,
            'Expected "target_bounding_box_format" to be non-relative. '
            f"Got `target_bounding_box_format`={target_format}.",
        ):
            bounding_box.preserve_rel(
                target_bounding_box_format=target_format,
                bounding_box_format=bounding_box_format,
            )

        # bounding box format should be in list of supported formats
        target_format = "xyxy"
        bounding_box_format = "rel_zxzx"

        with self.assertRaises(ValueError):
            bounding_box.preserve_rel(
                target_bounding_box_format=target_format,
                bounding_box_format=bounding_box_format,
            )
