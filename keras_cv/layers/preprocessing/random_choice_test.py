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
import unittest

import tensorflow as tf
from absl.testing import parameterized

from keras_cv import bounding_box
from keras_cv import layers


class AddOneToInputs(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.call_counter = tf.Variable(initial_value=0)

    def call(self, inputs):
        result = inputs.copy()
        result["images"] = inputs["images"] + 1
        self.call_counter.assign_add(1)
        return result


class RandomAugmentationPipelineTest(tf.test.TestCase, parameterized.TestCase):
    def test_calls_layer_augmentation_per_image(self):
        layer = AddOneToInputs()
        pipeline = layers.RandomChoice(layers=[layer])
        xs = tf.random.uniform((2, 5, 5, 3), 0, 100, dtype=tf.float32)
        os = pipeline(xs)

        self.assertAllClose(xs + 1, os)

    def test_calls_layer_augmentation_in_graph(self):
        layer = AddOneToInputs()
        pipeline = layers.RandomChoice(layers=[layer])

        @tf.function()
        def call_pipeline(xs):
            return pipeline(xs)

        xs = tf.random.uniform((2, 5, 5, 3), 0, 100, dtype=tf.float32)
        os = call_pipeline(xs)

        self.assertAllClose(xs + 1, os)

    def test_calls_layer_augmentation_single_image(self):
        layer = AddOneToInputs()
        pipeline = layers.RandomChoice(layers=[layer])
        xs = tf.random.uniform((5, 5, 3), 0, 100, dtype=tf.float32)
        os = pipeline(xs)

        self.assertAllClose(xs + 1, os)

    def test_calls_choose_one_layer_augmentation(self):
        batch_size = 10
        pipeline = layers.RandomChoice(layers=[AddOneToInputs(), AddOneToInputs()])
        xs = tf.random.uniform((batch_size, 5, 5, 3), 0, 100, dtype=tf.float32)
        os = pipeline(xs)

        self.assertAllClose(xs + 1, os)

        total_calls = pipeline.layers[0].call_counter + pipeline.layers[1].call_counter
        self.assertEqual(total_calls, batch_size)

    def test_augmentation_bounding_boxes(self):
        images = tf.zeros([2, 10, 10, 3])
        bboxes = tf.constant(
            [
                [[0.0, 0.1, 0.5, 0.5], [0.0, 0.2, 0.5, 0.6]],
                [[0.0, 0.15, 0.55, 0.55], [0.0, 0.25, 0.55, 0.65]],
            ],
            dtype=tf.float32,
        )
        bboxes = bounding_box.add_class_id(bboxes, 0)
        inputs = {"images": images, "bounding_boxes": bboxes}

        layer = layers.RandomChoice(
            layers=[
                layers.RandomFlip(mode="horizontal", bounding_box_format="rel_xyxy"),
                layers.RandomFlip(mode="vertical", bounding_box_format="rel_xyxy"),
            ]
        )

        with unittest.mock.patch.object(
            layer._random_generator,
            "random_uniform",
            side_effect=[tf.constant(0), tf.constant(1)],  # first, second
        ):
            with unittest.mock.patch.object(
                layer.layers[0]._random_generator,
                "random_uniform",
                side_effect=[0.0],  # no flip
            ):
                with unittest.mock.patch.object(
                    layer.layers[1]._random_generator,
                    "random_uniform",
                    side_effect=[1.0],  # no flip
                ):
                    output = layer(inputs)

        expected_output1 = tf.constant(
            [
                [0.0, 1.0 - 0.55, 0.55, 1.0 - 0.15, 0.0],
                [0.0, 1.0 - 0.65, 0.55, 1.0 - 0.25, 0.0],
            ]
        )

        # the first one should not be transformed
        self.assertAllClose(output["bounding_boxes"][0], inputs["bounding_boxes"][0])
        self.assertAllClose(output["bounding_boxes"][1], expected_output1)

    def test_augmentation_bounding_boxes_ragged(self):
        images = tf.zeros([2, 10, 10, 3])
        bboxes = tf.ragged.constant(
            [
                [[0.0, 0.1, 0.5, 0.5], [0.0, 0.2, 0.5, 0.6]],
                [
                    [0.0, 0.15, 0.55, 0.55],
                ],
            ],
            dtype=tf.float32,
        )
        bboxes = bounding_box.add_class_id(bboxes, 0)
        inputs = {"images": images, "bounding_boxes": bboxes}

        layer = layers.RandomChoice(
            layers=[
                layers.RandomFlip(mode="horizontal", bounding_box_format="rel_xyxy"),
                layers.RandomFlip(mode="vertical", bounding_box_format="rel_xyxy"),
            ]
        )

        with unittest.mock.patch.object(
            layer._random_generator,
            "random_uniform",
            side_effect=[tf.constant(0), tf.constant(1)],  # first, second
        ):
            with unittest.mock.patch.object(
                layer.layers[0]._random_generator,
                "random_uniform",
                side_effect=[0.0],  # no flip
            ):
                with unittest.mock.patch.object(
                    layer.layers[1]._random_generator,
                    "random_uniform",
                    side_effect=[1.0],  # no flip
                ):
                    output = layer(inputs)

        expected_output1 = tf.constant([[0.0, 1.0 - 0.55, 0.55, 1.0 - 0.15, 0.0]])

        self.assertAllClose(
            output["bounding_boxes"][0], inputs["bounding_boxes"][0].to_tensor()
        )
        self.assertAllClose(output["bounding_boxes"][1], expected_output1)
