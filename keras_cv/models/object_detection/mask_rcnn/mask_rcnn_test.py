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

import pytest
import tensorflow as tf
import tensorflow_models as tfm

import keras_cv


class MaskRCNNTest(tf.test.TestCase):

    @pytest.fixture(autouse=True)
    def cleanup_global_session(self):
        # Code before yield runs before the test
        yield
        tf.keras.backend.clear_session()

    def test_maskrcnn_construction(self):
        mask_rcnn_model = keras_cv.models.mask_rcnn(
            classes=91,     # default value for coco
            input_size=[512, 512, 3],
            backbone="resnet",
            backbone_weights=None,
        )
        images = tf.random.uniform(shape=(2, 512, 512, 3))
        image_shape = tf.constant([[512, 512], [512, 512]])
        anchor_generator = tfm.vision.anchor.build_anchor_generator(
            min_level=2,
            max_level=6,
            num_scales=1,
            aspect_ratios=[0.5, 1.0, 2.0],
            anchor_size=8)
        anchor_boxes = anchor_generator(image_size=(512, 512))
        gt_boxes = tf.constant([[[0, 0, 100, 100], [100, 100, 200, 200]],
                                [[200, 200, 300, 300], [300, 300, 400, 400]]])
        gt_classes = tf.constant([[1, 2], [3, 4]])
        gt_masks = tf.constant(1, shape=[2, 2, 512, 512])
        output = mask_rcnn_model(
            images=images,
            image_shape=image_shape,
            anchor_boxes=anchor_boxes,
            gt_boxes=gt_boxes,
            gt_classes=gt_classes,
            gt_masks=gt_masks,
            training=True)
        self.assertLen(output, 11)
        self.assertEquals(output['backbone_features'].keys(), {'2', '3', '4', '5'})
        self.assertEquals(output['decoder_features'].keys(), {'2', '3', '4', '5', '6'})
        self.assertEquals(output['rpn_boxes'].keys(), {'2', '3', '4', '5', '6'})
        self.assertEquals(output['rpn_scores'].keys(), {'2', '3', '4', '5', '6'})
        self.assertEquals(output['class_targets'].shape, [2, 512])
        self.assertEquals(output['box_targets'].shape, [2, 512, 4])
        self.assertEquals(output['class_outputs'].shape, [2, 512, 91])
        self.assertEquals(output['box_outputs'].shape, [2, 512, 364])
        self.assertEquals(output['mask_class_targets'].shape, [2, 128])
        self.assertEquals(output['mask_targets'].shape, [2, 128, 28, 28])
        self.assertEquals(output['mask_outputs'].shape, [2, 128, 28, 28])
