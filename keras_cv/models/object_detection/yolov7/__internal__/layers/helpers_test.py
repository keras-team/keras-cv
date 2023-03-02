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
from helpers import (
    ReOrganise,
    Shortcut,
    DownC,
    FusedConvolution,
    ImplicitAddition,
    ImplicitMultiplication,
)


class ReOrganiseTest(tf.test.TestCase):
    def test_return_type_and_shape(self):
        layer = ReOrganise()
        c4 = tf.ones([2, 16, 16, 3])

        inputs = c4
        output = layer(inputs, training=False)
        self.assertEquals(output.shape, [2, 8, 8, 12])

    def test_with_keras_tensor(self):
        layer = ReOrganise()
        c4 = tf.keras.layers.Input([16, 16, 3])

        inputs = c4
        output = layer(inputs, training=True)
        self.assertEquals(output.shape, [8, 8, 12])


class ShortcutTest(tf.test.TestCase):
    def test_return_type_and_shape(self):
        layer = Shortcut()
        c4 = tf.ones([2, 16, 16, 3])

        inputs = c4
        output = layer(inputs, training=True)
        self.assertEquals(output.shape, [16, 16, 3])

    def test_with_keras_input(self):
        layer = Shortcut()
        c4 = tf.keras.layers.Input([3, 16, 16, 3])

        inputs = c4
        output = layer(inputs, training=False)
        self.assertEquals(output.shape, [16, 16, 3])


class DownCTest(tf.test.TestCase):
    def test_return_type_and_shape(self):
        layer = Down(filters=322)
        c4 = tf.ones([2, 16, 16, 3])

        inputs = c4
        output = layer(inputs, training=False)
        self.assertEquals(output.shape, [2, 8, 8, 16])

    def test_with_keras_input(self):
        layer = DownC()
        c4 = tf.keras.layers.Input([2, 16, 16, 3])

        inputs = c4
        output = layer(inputs, training=False)
        self.assertEquals(output.shape, [2, 8, 8, 16])


class FusedConvolutionTest(tf.test.TestCase):
    

class ImplicitAdditionTest(tf.test.TestCase):
    def test_return_type_and_shape(self):
        layer = ImplicitAddition()
        c4 = tf.ones([16, 16, 3])

        inputs = c4
        output = layer(inputs, training=False)
        self.assertEquals(output.shape, [1, 16, 16, 3])

    def test_with_keras_input(self):
        layer = ImplicitAddition()
        c4 = tf.keras.layers.Input([16, 16, 3])

        inputs = c4
        output = layer(inputs, training=False)
        self.assertEquals(output.shape, [1, 16, 16, 3])


class ImplicitMultiplicationTest(tf.test.TestCase):
    def test_return_type_and_shape(self):
        layer = ImplicitMultiplication()
        c4 = tf.ones([16, 16, 3])

        inputs = c4
        output = layer(inputs, training=False)
        self.assertEquals(output.shape, [1, 16, 16, 3])

    def test_with_keras_input(self):
        layer = ImplicitMultiplication()
        c4 = tf.keras.layers.Input([16, 16, 3])

        inputs = c4
        output = layer(inputs, training=False)
        self.assertEquals(output.shape, [1, 16, 16, 3])

