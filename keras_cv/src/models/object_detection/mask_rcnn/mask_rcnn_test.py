# Copyright 2024 The KerasCV Authors
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

import keras_cv
from keras_cv.src.backend import keras
from keras_cv.src.backend import ops
from keras_cv.src.backend.config import keras_3
from keras_cv.src.models.object_detection.mask_rcnn import MaskRCNN
from keras_cv.src.tests.test_case import TestCase


def _create_bounding_box_segmask_dataset(
    bounding_box_format,
    image_shape=(512, 512, 3),
    use_dictionary_box_format=False,
):
    # Just about the easiest dataset you can have, all classes are 0, all boxes
    # are exactly the same. [1, 1, 2, 2] are the coordinates in xyxy.
    # segmentation masks cover the entire bounding box of the respective object
    xs = np.random.normal(size=(1,) + image_shape)
    xs = np.tile(xs, [5, 1, 1, 1])

    y_classes = np.zeros((5, 3), "float32")

    ys = np.array(
        [
            [0.1, 0.1, 0.23, 0.23],
            [0.67, 0.75, 0.23, 0.23],
            [0.25, 0.25, 0.23, 0.23],
        ],
        "float32",
    )

    ys = np.expand_dims(ys, axis=0)

    ys_yxyx = ops.convert_to_numpy(
        keras_cv.bounding_box.convert_format(
            ys,
            source="rel_xywh",
            target="yxyx",
            images=xs,
            dtype="float32",
        )
    )
    ys_yxyx = ys_yxyx.astype(int)
    segmask = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)
    for object_idx, (obj_y1, obj_x1, obj_y2, obj_x2) in enumerate(ys_yxyx[0]):
        segmask[obj_y1:obj_y2, obj_x1:obj_x2] = object_idx + 1
    segmask = np.expand_dims(segmask, axis=0)

    ys = np.tile(ys, [5, 1, 1])
    segmask = np.tile(segmask, [5, 1, 1])
    ys = ops.convert_to_numpy(
        keras_cv.bounding_box.convert_format(
            ys,
            source="rel_xywh",
            target=bounding_box_format,
            images=xs,
            dtype="float32",
        )
    )
    num_dets = np.ones([5])

    if use_dictionary_box_format:
        return tf.data.Dataset.from_tensor_slices(
            {
                "images": xs,
                "bounding_boxes": {
                    "boxes": ys,
                    "classes": y_classes,
                    "num_dets": num_dets,
                    "segmask": segmask,
                },
            }
        ).batch(5, drop_remainder=True)
    else:
        return xs, {"boxes": ys, "classes": y_classes, "segmask": segmask}


class MaskRCNNTest(TestCase):
    @pytest.mark.skipif(not keras_3(), reason="disabling test for Keras 2")
    def test_mask_rcnn_construction(self):
        mask_rcnn = MaskRCNN(
            num_classes=80,
            bounding_box_format="xyxy",
            backbone=keras_cv.models.ResNet18V2Backbone(
                input_shape=(32, 32, 3)
            ),
            num_sampled_rois=256,
        )
        mask_rcnn.compile(
            optimizer=keras.optimizers.Adam(),
            box_loss="Huber",
            classification_loss="CategoricalCrossentropy",
            rpn_box_loss="Huber",
            rpn_classification_loss="BinaryCrossentropy",
            mask_loss="BinaryCrossentropy",
        )

    @pytest.mark.extra_large()
    @pytest.mark.skipif(not keras_3(), reason="disabling test for Keras 2")
    def test_mask_rcnn_call(self):
        mask_rcnn = MaskRCNN(
            num_classes=3,
            bounding_box_format="xywh",
            backbone=keras_cv.models.ResNet18V2Backbone(
                input_shape=(32, 32, 3)
            ),
            num_sampled_rois=256,
        )
        images = np.random.uniform(size=(1, 32, 32, 3))
        _ = mask_rcnn(images)
        _ = mask_rcnn.predict(images)

    @pytest.mark.skipif(not keras_3(), reason="disabling test for Keras 2")
    def test_wrong_logits(self):
        mask_rcnn = MaskRCNN(
            num_classes=80,
            bounding_box_format="xywh",
            backbone=keras_cv.models.ResNet18V2Backbone(
                input_shape=(32, 32, 3)
            ),
            num_sampled_rois=256,
        )

        with self.assertRaisesRegex(
            ValueError,
            "from_logits",
        ):
            mask_rcnn.compile(
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
                mask_loss="BinaryCrossentropy",
            )
        with self.assertRaisesRegex(
            ValueError,
            "from_logits",
        ):
            mask_rcnn.compile(
                optimizer=keras.optimizers.SGD(learning_rate=0.25),
                box_loss="Huber",
                classification_loss="CategoricalCrossentropy",
                rpn_box_loss="Huber",
                rpn_classification_loss="BinaryCrossentropy",
                mask_loss=keras_cv.losses.FocalLoss(
                    from_logits=False, reduction="none"
                ),
            )

    @pytest.mark.large()
    @pytest.mark.skipif(not keras_3(), reason="disabling test for Keras 2")
    def test_weights_contained_in_trainable_variables(self):
        bounding_box_format = "xyxy"
        mask_rcnn = MaskRCNN(
            num_classes=80,
            bounding_box_format=bounding_box_format,
            backbone=keras_cv.models.ResNet18V2Backbone(
                input_shape=(32, 32, 3)
            ),
            num_sampled_rois=256,
        )
        mask_rcnn.backbone.trainable = False
        mask_rcnn.compile(
            optimizer=keras.optimizers.Adam(),
            box_loss="Huber",
            classification_loss="CategoricalCrossentropy",
            rpn_box_loss="Huber",
            rpn_classification_loss="BinaryCrossentropy",
            mask_loss="BinaryCrossentropy",
        )
        xs, ys = _create_bounding_box_segmask_dataset(
            bounding_box_format, image_shape=(32, 32, 3)
        )

        # call once
        _ = mask_rcnn(xs)
        self.assertEqual(len(mask_rcnn.trainable_variables), 42)

    @pytest.mark.extra_large  # Fit is slow, so mark these large.
    @pytest.mark.skipif(not keras_3(), reason="disabling test for Keras 2")
    def test_no_nans(self):
        mask_rcnn = MaskRCNN(
            num_classes=5,
            bounding_box_format="xyxy",
            backbone=keras_cv.models.ResNet18V2Backbone(
                input_shape=(32, 32, 3)
            ),
            num_sampled_rois=16,
        )
        mask_rcnn.compile(
            optimizer=keras.optimizers.Adam(),
            box_loss="Huber",
            classification_loss="CategoricalCrossentropy",
            rpn_box_loss="Huber",
            rpn_classification_loss="BinaryCrossentropy",
            mask_loss="BinaryCrossentropy",
        )

        # only a -1 box
        xs = np.ones((1, 32, 32, 3), "float32")
        ys = {
            "classes": np.array([[-1]], "float32"),
            "boxes": np.array([[[0, 0, 0, 0]]], "float32"),
            "segmask": np.zeros((1, 32, 32), dtype="float32"),
        }
        ds = tf.data.Dataset.from_tensor_slices((xs, ys))
        ds = ds.repeat(1)
        ds = ds.batch(1, drop_remainder=True)
        mask_rcnn.fit(ds, epochs=1)

        weights = mask_rcnn.get_weights()
        for weight in weights:
            self.assertFalse(ops.any(ops.isnan(weight)))

    @pytest.mark.extra_large  # Fit is slow, so mark these large.
    @pytest.mark.skipif(not keras_3(), reason="disabling test for Keras 2")
    def test_weights_change(self):
        mask_rcnn = MaskRCNN(
            num_classes=3,
            bounding_box_format="xyxy",
            backbone=keras_cv.models.ResNet18V2Backbone(
                input_shape=(128, 128, 3)
            ),
            num_sampled_rois=16,
        )
        mask_rcnn.compile(
            optimizer=keras.optimizers.Adam(),
            box_loss="Huber",
            classification_loss="CategoricalCrossentropy",
            rpn_box_loss="Huber",
            rpn_classification_loss="BinaryCrossentropy",
            mask_loss="BinaryCrossentropy",
        )

        ds = _create_bounding_box_segmask_dataset(
            "xyxy", image_shape=(128, 128, 3), use_dictionary_box_format=True
        )

        # call once
        _ = mask_rcnn(ops.ones((1, 128, 128, 3)))
        original_fpn_weights = mask_rcnn.feature_pyramid.get_weights()
        original_rpn_head_weights = mask_rcnn.rpn_head.get_weights()
        original_rcnn_head_weights = mask_rcnn.rcnn_head.get_weights()
        original_mask_head_weights = mask_rcnn.mask_head.get_weights()

        mask_rcnn.fit(ds, epochs=1)
        fpn_after_fit = mask_rcnn.feature_pyramid.get_weights()
        rpn_head_after_fit_weights = mask_rcnn.rpn_head.get_weights()
        rcnn_head_after_fit_weights = mask_rcnn.rcnn_head.get_weights()
        mask_head_after_fit_weights = mask_rcnn.mask_head.get_weights()

        for w1, w2 in zip(
            original_rcnn_head_weights,
            rcnn_head_after_fit_weights,
        ):
            self.assertNotAllClose(w1, w2)
        for w1, w2 in zip(
            original_mask_head_weights,
            mask_head_after_fit_weights,
        ):
            self.assertNotAllClose(w1, w2)
        for w1, w2 in zip(
            original_rpn_head_weights, rpn_head_after_fit_weights
        ):
            self.assertNotAllClose(w1, w2)

        for w1, w2 in zip(original_fpn_weights, fpn_after_fit):
            self.assertNotAllClose(w1, w2)

    @pytest.mark.large  # Saving is slow, so mark these large.
    @pytest.mark.skipif(not keras_3(), reason="disabling test for Keras 2")
    def test_saved_model(self):
        model = MaskRCNN(
            num_classes=80,
            bounding_box_format="xyxy",
            backbone=keras_cv.models.ResNet18V2Backbone(
                input_shape=(32, 32, 3)
            ),
        )
        input_batch = ops.ones(shape=(1, 32, 32, 3))
        model_output = model(input_batch)
        save_path = os.path.join(self.get_temp_dir(), "mask_rcnn.keras")
        model.save(save_path)
        restored_model = keras.models.load_model(save_path)

        # Check we got the real object back.
        self.assertIsInstance(restored_model, MaskRCNN)

        # Check that output matches.
        restored_output = restored_model(input_batch)
        self.assertAllClose(
            tf.nest.map_structure(ops.convert_to_numpy, model_output),
            tf.nest.map_structure(ops.convert_to_numpy, restored_output),
        )

    @pytest.mark.large
    @pytest.mark.skipif(not keras_3(), reason="disabling test for Keras 2")
    def test_mask_rcnn_infer(self):
        model = MaskRCNN(
            num_classes=80,
            bounding_box_format="xyxy",
            backbone=keras_cv.models.ResNet18V2Backbone(
                input_shape=(128, 128, 3)
            ),
        )
        images = ops.ones((1, 128, 128, 3))
        outputs = model(images, training=False)
        # 1000 proposals in inference
        self.assertAllEqual([1, 1000, 81], outputs["classification"].shape)
        self.assertAllEqual([1, 1000, 4], outputs["box"].shape)
        self.assertAllEqual([1, 1000, 14, 14, 81], outputs["segmask"].shape)

    @pytest.mark.large
    @pytest.mark.skipif(not keras_3(), reason="disabling test for Keras 2")
    def test_mask_rcnn_train(self):
        model = MaskRCNN(
            num_classes=80,
            bounding_box_format="xyxy",
            backbone=keras_cv.models.ResNet18V2Backbone(
                input_shape=(128, 128, 3)
            ),
        )
        images = ops.ones((1, 128, 128, 3))
        outputs = model(images, training=True)
        self.assertAllEqual([1, 1000, 81], outputs["classification"].shape)
        self.assertAllEqual([1, 1000, 4], outputs["box"].shape)
        self.assertAllEqual([1, 1000, 14, 14, 81], outputs["segmask"].shape)

    @pytest.mark.skipif(not keras_3(), reason="disabling test for Keras 2")
    def test_invalid_compile(self):
        model = MaskRCNN(
            num_classes=80,
            bounding_box_format="yxyx",
            backbone=keras_cv.models.ResNet18V2Backbone(
                input_shape=(32, 32, 3)
            ),
            num_sampled_rois=256,
        )
        with self.assertRaisesRegex(ValueError, "expects"):
            model.compile(rpn_box_loss="binary_crossentropy")
        with self.assertRaisesRegex(ValueError, "from_logits"):
            model.compile(
                box_loss="Huber",
                classification_loss="CategoricalCrossentropy",
                rpn_box_loss="Huber",
                rpn_classification_loss=keras.losses.BinaryCrossentropy(
                    from_logits=False
                ),
                mask_loss="BinaryCrossentropy",
            )

    @pytest.mark.extra_large  # Fit is slow, so mark these large.
    @pytest.mark.skipif(not keras_3(), reason="disabling test for Keras 2")
    def test_mask_rcnn_with_dictionary_input_format(self):
        mask_rcnn = MaskRCNN(
            num_classes=3,
            bounding_box_format="xywh",
            backbone=keras_cv.models.ResNet18V2Backbone(
                input_shape=(32, 32, 3)
            ),
            num_sampled_rois=16,
        )

        images, boxes = _create_bounding_box_segmask_dataset(
            "xywh", image_shape=(32, 32, 3)
        )
        dataset = tf.data.Dataset.from_tensor_slices(
            {"images": images, "bounding_boxes": boxes}
        ).batch(1, drop_remainder=True)

        mask_rcnn.compile(
            optimizer=keras.optimizers.Adam(),
            box_loss="Huber",
            classification_loss="CategoricalCrossentropy",
            rpn_box_loss="Huber",
            rpn_classification_loss="BinaryCrossentropy",
            mask_loss="BinaryCrossentropy",
        )

        mask_rcnn.fit(dataset, epochs=1)

    @pytest.mark.extra_large  # Fit is slow, so mark these large.
    @pytest.mark.skipif(not keras_3(), reason="disabling test for Keras 2")
    def test_fit_with_no_valid_gt_bbox(self):
        bounding_box_format = "xywh"
        mask_rcnn = MaskRCNN(
            num_classes=2,
            bounding_box_format=bounding_box_format,
            backbone=keras_cv.models.ResNet18V2Backbone(
                input_shape=(32, 32, 3)
            ),
            num_sampled_rois=16,
        )

        mask_rcnn.compile(
            optimizer=keras.optimizers.Adam(),
            box_loss="Huber",
            classification_loss="CategoricalCrossentropy",
            rpn_box_loss="Huber",
            rpn_classification_loss="BinaryCrossentropy",
            mask_loss="BinaryCrossentropy",
        )
        xs, ys = _create_bounding_box_segmask_dataset(
            bounding_box_format, image_shape=(32, 32, 3)
        )
        xs = ops.convert_to_tensor(xs)
        # Make all bounding_boxes invalid and filter them out
        ys["classes"] = -ops.ones_like(ys["classes"])

        mask_rcnn.fit(x=xs, y=ys, epochs=1, batch_size=1)


# TODO: add presets test cases once model training is done.
