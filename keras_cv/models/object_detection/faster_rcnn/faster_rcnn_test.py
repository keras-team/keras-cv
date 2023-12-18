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

import os

import numpy as np
import pytest
import tensorflow as tf
from absl.testing import parameterized

import keras_cv
from keras_cv.backend import keras
from keras_cv.backend import ops

# from keras_cv.models.backbones.test_backbone_presets import (
#     test_backbone_presets,
# )
from keras_cv.models.object_detection.__test_utils__ import (
    _create_bounding_box_dataset,
)
from keras_cv.models.object_detection.faster_rcnn.faster_rcnn import FasterRCNN
from keras_cv.tests.test_case import TestCase


class FasterRCNNTest(TestCase):
    def test_faster_rcnn_construction(self):
        faster_rcnn = FasterRCNN(
            num_classes=80,
            bounding_box_format="xyxy",
            backbone=keras_cv.models.ResNet18V2Backbone(),
        )
        faster_rcnn.compile(
            optimizer=keras.optimizers.Adam(),
            box_loss="Huber",
            classification_loss="SparseCategoricalCrossentropy",
            rpn_box_loss="Huber",
            rpn_classification_loss="BinaryCrossentropy",
        )

    @pytest.mark.large  # Fit is slow, so mark these large.
    def test_faster_rcnn_call(self):
        faster_rcnn = keras_cv.models.FasterRCNN(
            num_classes=80,
            bounding_box_format="xywh",
            backbone=keras_cv.models.ResNet18V2Backbone(input_shape=(512, 512, 3)),
        )
        images = np.random.uniform(size=(2, 512, 512, 3))
        _ = faster_rcnn(images)
        _ = faster_rcnn.predict(images)

    def test_wrong_logits(self):
        faster_rcnn = keras_cv.models.FasterRCNN(
            num_classes=80,
            bounding_box_format="xywh",
            backbone=keras_cv.models.ResNet18V2Backbone(input_shape=(512, 512, 3)),
        )

        with self.assertRaisesRegex(
            ValueError,
            "from_logits",
        ):
            faster_rcnn.compile(
                optimizer=keras.optimizers.SGD(learning_rate=0.25),
                box_loss=keras_cv.losses.SmoothL1Loss(
                    l1_cutoff=1.0, reduction="none"
                ),
                classification_loss=keras_cv.losses.FocalLoss(
                    from_logits=False, reduction="none"
                ),
                rpn_box_loss=keras_cv.losses.SmoothL1Loss(
                    l1_cutoff=1.0, reduction="none"
                ),
                rpn_classification_loss=keras_cv.losses.FocalLoss(
                    from_logits=False, reduction="none"
                ),
            )

    def test_weights_contained_in_trainable_variables(self):
        bounding_box_format = "xyxy"
        faster_rcnn = keras_cv.models.FasterRCNN(
            num_classes=80,
            bounding_box_format=bounding_box_format,
            backbone=keras_cv.models.ResNet18V2Backbone(input_shape=(512, 512, 3)),
        )
        faster_rcnn.backbone.trainable = False
        faster_rcnn.compile(
            optimizer=keras.optimizers.Adam(),
            box_loss="Huber",
            classification_loss="SparseCategoricalCrossentropy",
            rpn_box_loss="Huber",
            rpn_classification_loss="BinaryCrossentropy",
        )
        xs, ys = _create_bounding_box_dataset(bounding_box_format)

        # call once
        _ = faster_rcnn(xs)
        self.assertEqual(len(faster_rcnn.trainable_variables), 32)

    @pytest.mark.large  # Fit is slow, so mark these large.
    def test_no_nans(self):
        faster_rcnn = keras_cv.models.FasterRCNN(
            num_classes=80,
            bounding_box_format="xyxy",
            backbone=keras_cv.models.ResNet18V2Backbone(input_shape=(512, 512, 3)),
        )
        faster_rcnn.compile(
            optimizer=keras.optimizers.Adam(),
            box_loss="Huber",
            classification_loss="SparseCategoricalCrossentropy",
            rpn_box_loss="Huber",
            rpn_classification_loss="BinaryCrossentropy",
        )

        # only a -1 box
        xs = np.ones((1, 512, 512, 3), "float32")
        ys = {
            "classes": np.array([[-1]], "float32"),
            "boxes": np.array([[[0, 0, 0, 0]]], "float32"),
        }
        ds = tf.data.Dataset.from_tensor_slices((xs, ys))
        ds = ds.repeat(2)
        ds = ds.batch(2, drop_remainder=True)
        faster_rcnn.fit(ds, epochs=1)

        weights = faster_rcnn.get_weights()
        for weight in weights:
            self.assertFalse(ops.any(ops.isnan(weight)))

    @pytest.mark.large  # Fit is slow, so mark these large.
    def test_weights_change(self):
        faster_rcnn = keras_cv.models.FasterRCNN(
            num_classes=80,
            bounding_box_format="xyxy",
            backbone=keras_cv.models.ResNet18V2Backbone(input_shape=(512, 512, 3)),
        )
        faster_rcnn.compile(
            optimizer=keras.optimizers.Adam(),
            box_loss="Huber",
            classification_loss="SparseCategoricalCrossentropy",
            rpn_box_loss="Huber",
            rpn_classification_loss="BinaryCrossentropy",
        )

        images, boxes = _create_bounding_box_dataset("xyxy")
        ds = tf.data.Dataset.from_tensor_slices(
            {"images": images, "bounding_boxes": boxes}
        ).batch(5, drop_remainder=True)

        # call once
        _ = faster_rcnn(ops.ones((1, 512, 512, 3)))
        original_fpn_weights = faster_rcnn.feature_pyramid.get_weights()
        original_rpn_head_weights = faster_rcnn.rpn_head.get_weights()
        original_rcnn_head_weights = faster_rcnn.rcnn_head.get_weights()

        faster_rcnn.fit(ds, epochs=1)
        fpn_after_fit = faster_rcnn.feature_pyramid.get_weights()
        rpn_head_after_fit_weights = faster_rcnn.rpn_head.get_weights()
        rcnn_head_after_fit_weights = faster_rcnn.rcnn_head.get_weights()

        for w1, w2 in zip(
            original_rcnn_head_weights,
            rcnn_head_after_fit_weights,
        ):
            self.assertNotAllClose(w1, w2)

        for w1, w2 in zip(
            original_rpn_head_weights, rpn_head_after_fit_weights
        ):
            self.assertNotAllClose(w1, w2)

        for w1, w2 in zip(original_fpn_weights, fpn_after_fit):
            self.assertNotAllClose(w1, w2)

    @pytest.mark.large  # Saving is slow, so mark these large.
    def test_saved_model(self):
        model = keras_cv.models.FasterRCNN(
            num_classes=80,
            bounding_box_format="xyxy",
            backbone=keras_cv.models.ResNet18V2Backbone(input_shape=(512, 512, 3)),
        )
        input_batch = ops.ones(shape=(1, 512, 512, 3))
        model_output = model(input_batch)
        save_path = os.path.join(self.get_temp_dir(), "faster_rcnn.keras")
        model.save(save_path)
        restored_model = keras.models.load_model(save_path)

        # Check we got the real object back.
        self.assertIsInstance(restored_model, keras_cv.models.FasterRCNN)

        # Check that output matches.
        restored_output = restored_model(input_batch)
        self.assertAllClose(
            tf.nest.map_structure(ops.convert_to_numpy, model_output),
            tf.nest.map_structure(ops.convert_to_numpy, restored_output),
        )

    # TODO(ianstenbit): Make FasterRCNN support shapes that are not multiples
    # of 128, perhaps by adding a flag to the anchor generator for whether to
    # include anchors centered outside of the image. (RetinaNet does use those,
    # while FasterRCNN doesn't). For more context on why this is the case, see
    # https://github.com/keras-team/keras-cv/pull/1882
    @parameterized.parameters(
        ((2, 640, 384, 3),),
        ((2, 512, 512, 3),),
        ((2, 128, 128, 3),),
    )
    def test_faster_rcnn_infer(self, batch_shape):
        model = FasterRCNN(
            num_classes=80,
            bounding_box_format="xyxy",
            backbone=keras_cv.models.ResNet18V2Backbone(input_shape=(512, 512, 3)),
        )
        images = ops.random.normal(batch_shape)
        outputs = model(images, training=False)
        # 1000 proposals in inference
        self.assertAllEqual([2, 1000, 81], outputs["classes"].shape)
        self.assertAllEqual([2, 1000, 4], outputs["boxes"].shape)

    @parameterized.parameters(
        ((2, 640, 384, 3),),
        ((2, 512, 512, 3),),
        ((2, 128, 128, 3),),
    )
    def test_faster_rcnn_train(self, batch_shape):
        model = FasterRCNN(
            num_classes=80,
            bounding_box_format="xyxy",
            backbone=keras_cv.models.ResNet18V2Backbone(input_shape=(512, 512, 3)),
        )
        images = ops.random.normal(batch_shape)
        outputs = model(images, training=True)
        self.assertAllEqual([2, 1000, 81], outputs["classes"].shape)
        self.assertAllEqual([2, 1000, 4], outputs["boxes"].shape)

    def test_invalid_compile(self):
        model = FasterRCNN(
            num_classes=80,
            bounding_box_format="yxyx",
            backbone=keras_cv.models.ResNet18V2Backbone(input_shape=(512, 512, 3)),
        )
        with self.assertRaisesRegex(ValueError, "only accepts"):
            model.compile(rpn_box_loss="binary_crossentropy")
        with self.assertRaisesRegex(ValueError, "only accepts"):
            model.compile(
                rpn_classification_loss=keras.losses.BinaryCrossentropy(
                    from_logits=False
                )
            )

    @pytest.mark.large  # Fit is slow, so mark these large.
    def test_faster_rcnn_with_dictionary_input_format(self):
        faster_rcnn = FasterRCNN(
            num_classes=20,
            bounding_box_format="xywh",
            backbone=keras_cv.models.ResNet18V2Backbone(input_shape=(512, 512, 3)),
        )

        images, boxes = _create_bounding_box_dataset("xywh")
        dataset = tf.data.Dataset.from_tensor_slices(
            {"images": images, "bounding_boxes": boxes}
        ).batch(5, drop_remainder=True)

        faster_rcnn.compile(
            optimizer=keras.optimizers.Adam(),
            box_loss="Huber",
            classification_loss="SparseCategoricalCrossentropy",
            rpn_box_loss="Huber",
            rpn_classification_loss="BinaryCrossentropy",
        )

        faster_rcnn.fit(dataset, epochs=1)
        faster_rcnn.evaluate(dataset)
