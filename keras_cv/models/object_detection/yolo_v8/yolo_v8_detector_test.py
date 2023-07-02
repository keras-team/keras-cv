# Copyright 2023 The KerasCV Authors
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

import pytest
import tensorflow as tf
from absl.testing import parameterized
from tensorflow import keras

import keras_cv
from keras_cv import bounding_box
from keras_cv.models.backbones.test_backbone_presets import (
    test_backbone_presets,
)
from keras_cv.models.object_detection.__test_utils__ import (
    _create_bounding_box_dataset,
)
from keras_cv.models.object_detection.yolo_v8.yolo_v8_detector_presets import (
    yolo_v8_detector_presets,
)


class YOLOV8DetectorTest(tf.test.TestCase, parameterized.TestCase):
    @pytest.mark.large  # Fit is slow, so mark these large.
    def test_fit(self):
        bounding_box_format = "xywh"
        yolo = keras_cv.models.YOLOV8Detector(
            num_classes=2,
            fpn_depth=1,
            bounding_box_format=bounding_box_format,
            backbone=keras_cv.models.YOLOV8Backbone.from_preset(
                "yolo_v8_xs_backbone"
            ),
        )

        yolo.compile(
            optimizer="adam",
            classification_loss="binary_crossentropy",
            box_loss="ciou",
        )
        xs, ys = _create_bounding_box_dataset(bounding_box_format)

        yolo.fit(x=xs, y=ys, epochs=1)

    @pytest.mark.large  # Fit is slow, so mark these large.
    def test_fit_with_ragged_tensors(self):
        bounding_box_format = "xywh"
        yolo = keras_cv.models.YOLOV8Detector(
            num_classes=2,
            fpn_depth=1,
            bounding_box_format=bounding_box_format,
            backbone=keras_cv.models.YOLOV8Backbone.from_preset(
                "yolo_v8_xs_backbone"
            ),
        )

        yolo.compile(
            optimizer="adam",
            classification_loss="binary_crossentropy",
            box_loss="ciou",
        )
        xs, ys = _create_bounding_box_dataset(bounding_box_format)
        ys = bounding_box.to_ragged(ys)

        yolo.fit(x=xs, y=ys, epochs=1)

    @pytest.mark.large  # Fit is slow, so mark these large.
    def test_fit_with_no_valid_gt_bbox(self):
        bounding_box_format = "xywh"
        yolo = keras_cv.models.YOLOV8Detector(
            num_classes=1,
            fpn_depth=1,
            bounding_box_format=bounding_box_format,
            backbone=keras_cv.models.YOLOV8Backbone.from_preset(
                "yolo_v8_xs_backbone"
            ),
        )

        yolo.compile(
            optimizer="adam",
            classification_loss="binary_crossentropy",
            box_loss="ciou",
        )
        xs, ys = _create_bounding_box_dataset(bounding_box_format)
        # Make all bounding_boxes invalid and filter out them
        ys["classes"] = -tf.ones_like(ys["classes"])
        ys = bounding_box.to_ragged(ys)

        yolo.fit(x=xs, y=ys, epochs=1)

    def test_trainable_weight_count(self):
        yolo = keras_cv.models.YOLOV8Detector(
            num_classes=2,
            fpn_depth=1,
            bounding_box_format="xywh",
            backbone=keras_cv.models.YOLOV8Backbone.from_preset(
                "yolo_v8_s_backbone"
            ),
        )

        self.assertEqual(len(yolo.trainable_weights), 195)

    def test_bad_loss(self):
        yolo = keras_cv.models.YOLOV8Detector(
            num_classes=2,
            fpn_depth=1,
            bounding_box_format="xywh",
            backbone=keras_cv.models.YOLOV8Backbone.from_preset(
                "yolo_v8_xs_backbone"
            ),
        )

        with self.assertRaisesRegex(
            ValueError,
            "Invalid box loss",
        ):
            yolo.compile(
                box_loss="bad_loss", classification_loss="binary_crossentropy"
            )

        with self.assertRaisesRegex(
            ValueError,
            "Invalid classification loss",
        ):
            yolo.compile(box_loss="ciou", classification_loss="bad_loss")

    @parameterized.named_parameters(
        ("tf_format", "tf", "model"),
        ("keras_format", "keras_v3", "model.keras"),
    )
    @pytest.mark.large  # Saving is slow, so mark these large.
    def test_saved_model(self, save_format, filename):
        model = keras_cv.models.YOLOV8Detector(
            num_classes=20,
            bounding_box_format="xywh",
            fpn_depth=1,
            backbone=keras_cv.models.YOLOV8Backbone.from_preset(
                "yolo_v8_xs_backbone"
            ),
        )
        xs, _ = _create_bounding_box_dataset("xywh")
        model_output = model(xs)
        save_path = os.path.join(self.get_temp_dir(), filename)
        model.save(save_path, save_format=save_format)
        restored_model = keras.models.load_model(save_path)

        # Check we got the real object back.
        self.assertIsInstance(restored_model, keras_cv.models.YOLOV8Detector)

        # Check that output matches.
        restored_output = restored_model(xs)
        self.assertAllClose(model_output, restored_output)

    def test_update_prediction_decoder(self):
        yolo = keras_cv.models.YOLOV8Detector(
            num_classes=2,
            fpn_depth=1,
            bounding_box_format="xywh",
            backbone=keras_cv.models.YOLOV8Backbone.from_preset(
                "yolo_v8_s_backbone"
            ),
            prediction_decoder=keras_cv.layers.MultiClassNonMaxSuppression(
                bounding_box_format="xywh",
                from_logits=False,
                confidence_threshold=0.0,
                iou_threshold=1.0,
            ),
        )

        image = tf.ones((1, 512, 512, 3))

        outputs = yolo.predict(image)
        # We predicted at least 1 box with confidence_threshold 0
        self.assertGreater(outputs["boxes"]._values.shape[0], 0)

        yolo.prediction_decoder = keras_cv.layers.MultiClassNonMaxSuppression(
            bounding_box_format="xywh",
            from_logits=False,
            confidence_threshold=1.0,
            iou_threshold=1.0,
        )

        outputs = yolo.predict(image)
        # We predicted no boxes with confidence threshold 1
        self.assertEqual(outputs["boxes"]._values.shape[0], 0)


@pytest.mark.large
class YOLOV8DetectorSmokeTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(
        *[(preset, preset) for preset in test_backbone_presets]
    )
    def test_backbone_preset(self, preset):
        model = keras_cv.models.YOLOV8Detector.from_preset(
            preset,
            num_classes=20,
            bounding_box_format="xywh",
        )
        xs, _ = _create_bounding_box_dataset(bounding_box_format="xywh")
        output = model(xs)

        # 64 represents number of parameters in a box
        # 5376 is the number of anchors for a 512x512 image
        self.assertEqual(output["boxes"].shape, (xs.shape[0], 5376, 64))

    def test_preset_with_forward_pass(self):
        model = keras_cv.models.YOLOV8Detector.from_preset(
            "yolo_v8_m_pascalvoc",
            bounding_box_format="xywh",
        )

        image = tf.ones((1, 512, 512, 3))
        encoded_predictions = model(image)

        self.assertAllClose(
            encoded_predictions["boxes"][0, 0:5, 0],
            [-0.8303556, 0.75213313, 1.809204, 1.6576759, 1.4134747],
        )
        self.assertAllClose(
            encoded_predictions["classes"][0, 0:5, 0],
            [
                7.6146556e-08,
                8.0103280e-07,
                9.7873999e-07,
                2.2314548e-06,
                2.5051115e-06,
            ],
        )


@pytest.mark.extra_large
class YOLOV8DetectorPresetFullTest(tf.test.TestCase):
    """
    Test the full enumeration of our presets.
    This every presets for YOLOV8Detector and is only run manually.
    Run with:
    `pytest keras_cv/models/object_detection/yolo_v8/yolo_v8_detector_test.py --run_extra_large`
    """  # noqa: E501

    def test_load_yolo_v8_detector(self):
        input_data = tf.ones(shape=(2, 224, 224, 3))
        for preset in yolo_v8_detector_presets:
            model = keras_cv.models.YOLOV8Detector.from_preset(
                preset, bounding_box_format="xywh"
            )
            model(input_data)
