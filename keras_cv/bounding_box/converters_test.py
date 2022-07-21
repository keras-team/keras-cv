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

import itertools

import tensorflow as tf
from absl.testing import parameterized

from keras_cv import bounding_box

xyxy_box = tf.constant([[[10, 20, 110, 120], [20, 30, 120, 130]]], dtype=tf.float32)
yxyx_box = tf.constant([[[20, 10, 120, 110], [30, 20, 130, 120]]], dtype=tf.float32)
rel_xyxy_box = tf.constant(
    [[[0.01, 0.02, 0.11, 0.12], [0.02, 0.03, 0.12, 0.13]]], dtype=tf.float32
)
rel_yxyx_box = tf.constant(
    [[[0.02, 0.01, 0.12, 0.11], [0.03, 0.02, 0.13, 0.12]]], dtype=tf.float32
)
center_xywh_box = tf.constant(
    [[[60, 70, 100, 100], [70, 80, 100, 100]]], dtype=tf.float32
)
xywh_box = tf.constant([[[10, 20, 100, 100], [20, 30, 100, 100]]], dtype=tf.float32)

images = tf.ones([2, 1000, 1000, 3])

boxes = {
    "xyxy": xyxy_box,
    "center_xywh": center_xywh_box,
    "xywh": xywh_box,
    "rel_xyxy": rel_xyxy_box,
    "yxyx": yxyx_box,
    "rel_yxyx": rel_yxyx_box,
}

test_cases = [
    (f"{source}_{target}", source, target)
    for (source, target) in itertools.permutations(boxes.keys(), 2)
] + [("xyxy_xyxy", "xyxy", "xyxy")]


class ConvertersTestCase(tf.test.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(*test_cases)
    def test_converters(self, source, target):
        source_box = boxes[source]
        target_box = boxes[target]

        self.assertAllClose(
            bounding_box.convert_format(
                source_box, source=source, target=target, images=images
            ),
            target_box,
        )

    @parameterized.named_parameters(*test_cases)
    def test_converters_unbatched(self, source, target):
        source_box = boxes[source][0]
        target_box = boxes[target][0]

        self.assertAllClose(
            bounding_box.convert_format(
                source_box, source=source, target=target, images=images[0]
            ),
            target_box,
        )

    def test_raises_with_different_image_rank(self):
        source_box = boxes["xyxy"][0]
        with self.assertRaises(ValueError):
            bounding_box.convert_format(
                source_box, source="xyxy", target="xywh", images=images
            )

    def test_without_images(self):
        source_box = boxes["xyxy"]
        target_box = boxes["xywh"]
        self.assertAllClose(
            bounding_box.convert_format(source_box, source="xyxy", target="xywh"),
            target_box,
        )

    @parameterized.named_parameters(*test_cases)
    def test_ragged_bounding_box(self, source, target):
        source_box = _raggify(boxes[source])
        target_box = _raggify(boxes[target])
        self.assertAllClose(
            bounding_box.convert_format(
                source_box, source=source, target=target, images=images
            ),
            target_box,
        )


def _raggify(tensor, row_lengths=[[2, 0], [0, 0]]):
    return tf.RaggedTensor.from_row_lengths(tensor[0], [2, 0])
