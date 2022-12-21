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

from keras_cv.models import ResNet50V2
from keras_cv.models.object_detection.faster_rcnn import FasterRCNN


class FasterRCNNTest(tf.test.TestCase):
    def test_faster_rcnn_infer(self):
        model = FasterRCNN(
            classes=80, bounding_box_format="xyxy", backbone=self._build_backbone()
        )
        images = tf.random.normal([2, 512, 512, 3])
        outputs = model(images, training=False)
        # 1000 proposals in inference
        self.assertAllEqual([2, 1000, 81], outputs[1].shape)
        self.assertAllEqual([2, 1000, 4], outputs[0].shape)

    def test_faster_rcnn_train(self):
        model = FasterRCNN(
            classes=80, bounding_box_format="xyxy", backbone=self._build_backbone()
        )
        images = tf.random.normal([2, 512, 512, 3])
        outputs = model(images, training=True)
        self.assertAllEqual([2, 1000, 81], outputs[1].shape)
        self.assertAllEqual([2, 1000, 4], outputs[0].shape)

    def test_invalid_compile(self):
        model = FasterRCNN(
            classes=80, bounding_box_format="yxyx", backbone=self._build_backbone()
        )
        with self.assertRaisesRegex(ValueError, "only accepts"):
            model.compile(rpn_box_loss="binary_crossentropy")
        with self.assertRaisesRegex(ValueError, "only accepts"):
            model.compile(
                rpn_classification_loss=tf.keras.losses.BinaryCrossentropy(
                    from_logits=False
                )
            )

    def _build_backbone(self):
        return ResNet50V2(include_top=False, include_rescaling=True).as_backbone()
