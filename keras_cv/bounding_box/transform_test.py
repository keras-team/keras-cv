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

from keras_cv.bounding_box.transform import transform_from_corners_fn


def transpose_points(keypoints):
    return tf.stack([keypoints[..., 1], keypoints[..., 0]], axis=-1)


bounding_boxes = tf.constant(
    [
        [[10, 20, 110, 220, 1], [20, 30, 220, 230, 2]],
        [[30, 40, 330, 340, 3], [40, 50, 440, 450, 4]],
    ],
    tf.float32,
)

bounding_boxes_transposed = tf.constant(
    [
        [[20, 10, 220, 110, 1], [30, 20, 230, 220, 2]],
        [[40, 30, 340, 330, 3], [50, 40, 450, 440, 4]],
    ],
    tf.float32,
)

images = tf.zeros((2, 500, 500, 3))


class TransformTestCase(tf.test.TestCase):
    def test_batched(self):
        output = transform_from_corners_fn(
            bounding_boxes=bounding_boxes,
            transform_corners_fn=transpose_points,
            bounding_box_format="xyxy",
            images=images,
        )
        self.assertAllClose(output, bounding_boxes_transposed)

    def test_unbatched(self):
        output = transform_from_corners_fn(
            bounding_boxes=bounding_boxes[0],
            transform_corners_fn=transpose_points,
            bounding_box_format="xyxy",
            images=images[0],
        )
        self.assertAllClose(output, bounding_boxes_transposed[0])

    def test_clipping(self):
        output = transform_from_corners_fn(
            bounding_boxes=bounding_boxes,
            transform_corners_fn=transpose_points,
            bounding_box_format="xyxy",
            # this image size ensure we do not have a height/width issue
            # inversion ( image format is HWC )
            images=tf.zeros((2, 500, 445, 3)),
        )
        expected_output = tf.minimum(bounding_boxes_transposed, 445)

        self.assertAllClose(output, expected_output)

    def test_ragged(self):
        inputs = _raggify(bounding_boxes)
        expected_output = _raggify(tf.minimum(bounding_boxes_transposed, 400.0))
        images = tf.zeros((3, 400, 400, 3))
        outputs = transform_from_corners_fn(
            bounding_boxes=inputs,
            transform_corners_fn=transpose_points,
            bounding_box_format="xyxy",
            images=images,
        )
        self.assertAllClose(outputs, expected_output)


def _raggify(tensor):
    return tf.RaggedTensor.from_row_lengths(
        tf.reshape(tensor, shape=(4, -1)), row_lengths=[3, 0, 1]
    )
