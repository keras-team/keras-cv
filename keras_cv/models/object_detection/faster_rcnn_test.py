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

from keras_cv.models.object_detection.faster_rcnn import FasterRCNN


class FasterRCNNTest(tf.test.TestCase):
    def test_faster_rcnn_infer(self):
        model = FasterRCNN(classes=80, bounding_box_format="xyxy")
        images = tf.random.normal([2, 512, 512, 3])
        gt_boxes = tf.constant(
            [
                [[32.0, 32.0, 64.0, 64.0], [-1.0, -1.0, -1.0, -1.0]],
                [[32.0, 32.0, 128.0, 128.0], [128.0, 128.0, 156.0, 156.0]],
            ]
        )
        gt_classes = tf.constant(
            [
                [[1.0], [-1.0]],
                [
                    [
                        25.0,
                    ],
                    [76.0],
                ],
            ]
        )
        outputs = model(images, gt_boxes, gt_classes, training=False)
        # 1000 proposals in inference
        self.assertAllEqual([2, 1000, 81], outputs["rcnn_cls_pred"].shape)
        self.assertAllEqual([2, 1000, 4], outputs["rcnn_box_pred"].shape)

    def test_faster_rcnn_train(self):
        model = FasterRCNN(classes=80, bounding_box_format="xyxy")
        images = tf.random.normal([2, 512, 512, 3])
        gt_boxes = tf.constant(
            [
                [[32.0, 32.0, 64.0, 64.0], [-1.0, -1.0, -1.0, -1.0]],
                [[32.0, 32.0, 128.0, 128.0], [128.0, 128.0, 156.0, 156.0]],
            ]
        )
        gt_classes = tf.constant(
            [
                [[1.0], [-1.0]],
                [
                    [
                        25.0,
                    ],
                    [76.0],
                ],
            ]
        )
        outputs = model(images, gt_boxes, gt_classes, training=True)
        # 512 sampled proposals in inference
        self.assertAllEqual([2, 512, 81], outputs["rcnn_cls_pred"].shape)
        self.assertAllEqual([2, 512, 4], outputs["rcnn_box_pred"].shape)
        # (128*128 + 64*64 + 32*32+16*16+8*8) * 3 = 65472
        self.assertAllEqual([2, 65472, 4], outputs["rpn_box_pred"].shape)
        self.assertAllEqual([2, 65472, 1], outputs["rpn_cls_pred"].shape)
