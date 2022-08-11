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

from keras_cv.models.object_detection.retina_net.__internal__.layers import (
    DecodePredictions,
)


class RetinaNetTest(tf.test.TestCase):
    def test_decode_predictions_output_shapes(self):
        classes = 10
        images_shape = (8, 512, 1024, 3)
        predictions_shape = (8, 98208, 4 + classes)

        images = tf.random.uniform(shape=images_shape)
        predictions = tf.random.uniform(
            shape=predictions_shape, minval=0.0, maxval=1.0, dtype=tf.float32
        )
        layer = DecodePredictions(classes=classes, bounding_box_format="rel_xyxy")

        result = layer(images=images, predictions=predictions)

        self.assertEqual(result.shape, [8, None, 6])
