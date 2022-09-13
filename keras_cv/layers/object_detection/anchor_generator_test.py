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

from keras_cv import layers as cv_layers


class AnchorGeneratorTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(
        ("unequal_lists", [0, 1, 2], [1]),
        ("unequal_levels_dicts", {"level_1": [0, 1, 2]}, {"1": [0, 1, 2]}),
    )
    def test_raises_when_strides_not_equal_to_sizes(self, sizes, strides):
        with self.assertRaises(ValueError):
            cv_layers.AnchorGenerator(
                bounding_box_format="xyxy",
                sizes=sizes,
                strides=strides,
                aspect_ratios=[3 / 4, 1, 4 / 3],
                scales=[0.5, 1.0, 1.5],
            )

    def test_raises_batched_images(self):
        strides = [4]
        scales = [1.0]
        sizes = [4]
        aspect_ratios = [1.0]
        anchor_generator = cv_layers.AnchorGenerator(
            bounding_box_format="xyxy",
            sizes=sizes,
            aspect_ratios=aspect_ratios,
            scales=scales,
            strides=strides,
        )

        image = tf.random.uniform((4, 8, 8, 3))
        with self.assertRaisesRegex(ValueError, "rank"):
            _ = anchor_generator(image=image)

    def test_output_shapes_image(self):
        strides = [2**i for i in range(3, 8)]
        scales = [2**x for x in [0, 1 / 3, 2 / 3]]
        sizes = [x**2 for x in [32.0, 64.0, 128.0, 256.0, 512.0]]
        aspect_ratios = [0.5, 1.0, 2.0]

        image_shape = (512, 512, 3)
        image = tf.random.uniform(image_shape)
        anchor_generator = cv_layers.AnchorGenerator(
            bounding_box_format="yxyx",
            sizes=sizes,
            aspect_ratios=aspect_ratios,
            scales=scales,
            strides=strides,
        )
        boxes = anchor_generator(image=image)
        boxes = tf.concat(list(boxes.values()), axis=0)

        # 49104 is a number found by using the previous internal anchor generator from
        # PR https://github.com/keras-team/keras-cv/pull/609
        # This unit test was written to ensure compatibility with the existing model.
        self.assertEqual(boxes.shape, [49104, 4])

    def test_output_shapes_image_shape(self):
        strides = [2**i for i in range(3, 8)]
        scales = [2**x for x in [0, 1 / 3, 2 / 3]]
        sizes = [x**2 for x in [32.0, 64.0, 128.0, 256.0, 512.0]]
        aspect_ratios = [0.5, 1.0, 2.0]

        image_shape = (512, 512, 3)
        anchor_generator = cv_layers.AnchorGenerator(
            bounding_box_format="yxyx",
            sizes=sizes,
            aspect_ratios=aspect_ratios,
            scales=scales,
            strides=strides,
        )
        boxes = anchor_generator(image_shape=image_shape)
        boxes = tf.concat(list(boxes.values()), axis=0)

        # 49104 is a number found by using the previous internal anchor generator from
        # PR https://github.com/keras-team/keras-cv/pull/609
        # This unit test was written to ensure compatibility with the existing model.
        self.assertEqual(boxes.shape, [49104, 4])

    def test_hand_crafted_aspect_ratios(self):
        strides = [4]
        scales = [1.0]
        sizes = [4]
        aspect_ratios = [3 / 4, 1.0, 4 / 3]
        anchor_generator = cv_layers.AnchorGenerator(
            bounding_box_format="xyxy",
            sizes=sizes,
            aspect_ratios=aspect_ratios,
            scales=scales,
            strides=strides,
        )

        image = tf.random.uniform((8, 8, 3))
        boxes = anchor_generator(image=image)
        level_0 = boxes[0]

        # width/4 * height/4 * aspect_ratios =
        self.assertAllEqual(level_0.shape, [12, 4])

        image = tf.random.uniform((4, 4, 3))
        boxes = anchor_generator(image=image)
        level_0 = boxes[0]

        expected_boxes = [
            [0.267949224, -0.309401035, 3.7320509, 4.30940104],
            [0, 0, 4, 4],
            [-0.309401035, 0.267949104, 4.30940104, 3.7320509],
        ]
        self.assertAllClose(level_0, expected_boxes)

    def test_hand_crafted_strides(self):
        strides = [4]
        scales = [1.0]
        sizes = [4]
        aspect_ratios = [1.0]
        anchor_generator = cv_layers.AnchorGenerator(
            bounding_box_format="xyxy",
            sizes=sizes,
            aspect_ratios=aspect_ratios,
            scales=scales,
            strides=strides,
        )

        image = tf.random.uniform((8, 8, 3))
        boxes = anchor_generator(image=image)
        level_0 = boxes[0]
        expected_boxes = [
            [0, 0, 4, 4],
            [4, 0, 8, 4],
            [0, 4, 4, 8],
            [4, 4, 8, 8],
        ]
        self.assertAllClose(level_0, expected_boxes)

    def test_relative_generation(self):
        strides = [8, 16, 32]

        # 0, 1 / 3, 2 / 3
        scales = [2**x for x in [0, 1 / 3, 2 / 3]]
        sizes = [32.0, 64.0, 128.0]
        aspect_ratios = [0.5, 1.0, 2.0]

        image = tf.random.uniform((512, 512, 3))
        anchor_generator = cv_layers.AnchorGenerator(
            bounding_box_format="rel_yxyx",
            sizes=sizes,
            aspect_ratios=aspect_ratios,
            scales=scales,
            strides=strides,
            clip_boxes=False,
        )
        boxes = anchor_generator(image=image)
        boxes = tf.concat(list(boxes.values()), axis=0)
        self.assertAllLessEqual(boxes, 1.5)
        self.assertAllGreaterEqual(boxes, -0.50)
