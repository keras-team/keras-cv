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

from keras_cv.layers import preprocessing


class RandomResizedCropTest(tf.test.TestCase):
    def train_augments_image(self):
        # Checks if original and augmented images are different

        input_image_shape = (4, 300, 300, 3)
        image = tf.random.uniform(shape=input_image_shape)

        layer = preprocessing.RandomResizedCrop(
            target_size=(224, 224), area_factor=(0.8, 1.0)
        )
        output = layer(image, training=True)

        output_image_resized = tf.image.resize(output, input_image_shape)

        self.assertNotEqual(image, output_image_resized)

    def test_preserves_image(self):
        # Checks if resultant image stays the same at inference time

        image_shape = (4, 300, 300, 3)
        image = tf.random.uniform(shape=image_shape)

        layer = preprocessing.RandomResizedCrop(
            target_size=(214, 214), area_factor=(0.8, 1.0)
        )
        output = layer(image, training=False)

        self.assertEqual(output.shape, (4, 214, 214, 3))
