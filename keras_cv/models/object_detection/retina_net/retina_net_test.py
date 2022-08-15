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

import pytest
import tensorflow as tf
from absl.testing import parameterized
from tensorflow.keras import optimizers
import statistics
import keras_cv


class RetinaNetTest(tf.test.TestCase, parameterized.TestCase):
    @pytest.fixture(autouse=True)
    def cleanup_global_session(self):
        # Code before yield runs before the test
        yield
        tf.keras.backend.clear_session()

    def test_retina_net_construction(self):
        retina_net = keras_cv.models.RetinaNet(
            classes=20,
            bounding_box_format="xywh",
            backbone="resnet50",
            backbone_weights=None,
            include_rescaling=True,
        )
        loss = keras_cv.losses.ObjectDetectionLoss(
            classes=20,
            classification_loss=keras_cv.losses.FocalLoss(
                from_logits=True, reduction="none"
            ),
            box_loss=keras_cv.losses.SmoothL1Loss(l1_cutoff=1.0, reduction="none"),
            reduction="auto",
        )
        retina_net.compile(
            loss=loss,
            optimizer="adam",
            metrics=[
                keras_cv.metrics.COCOMeanAveragePrecision(
                    class_ids=range(20),
                    bounding_box_format="xyxy",
                    name="Standard MaP",
                ),
            ],
        )

        # TODO(lukewood) uncomment when using keras_cv.models.ResNet50
        # self.assertIsNotNone(retina_net.backbone.get_layer(name="rescaling"))
        # TODO(lukewood): test compile with the FocalLoss class

    def test_raises_when_box_variances_dont_match(self):
        # TODO(lukewood): write test
        pass

    def test_retina_net_include_rescaling_required_with_default_backbone(self):
        with self.assertRaises(ValueError):
            _ = keras_cv.models.RetinaNet(
                classes=20,
                bounding_box_format="xywh",
                backbone="resnet50",
                backbone_weights=None,
                # Note no include_rescaling is provided
            )

    def test_retina_net_call(self):
        retina_net = keras_cv.models.RetinaNet(
            classes=20,
            bounding_box_format="xywh",
            backbone="resnet50",
            backbone_weights=None,
            include_rescaling=True,
        )
        images = tf.random.uniform((2, 512, 512, 3))
        outputs = retina_net(images)
        self.assertIn("inference", outputs)
        self.assertIn("train_predictions", outputs)

    def test_all_metric_formats_must_match(self):
        retina_net = keras_cv.models.RetinaNet(
            classes=20,
            bounding_box_format="xywh",
            backbone="resnet50",
            backbone_weights=None,
            include_rescaling=True,
        )

        # all metric formats must match
        with self.assertRaises(ValueError):
            retina_net.compile(
                optimizer="adam",
                metrics=[
                    keras_cv.metrics.COCOMeanAveragePrecision(
                        class_ids=range(20),
                        bounding_box_format="xyxy",
                        name="Standard MaP",
                    ),
                    keras_cv.metrics.COCOMeanAveragePrecision(
                        class_ids=range(20),
                        bounding_box_format="rel_xyxy",
                        name="Standard MaP",
                    ),
                ],
            )

    def test_fit_coco_metrics(self):
        retina_net = keras_cv.models.RetinaNet(
            classes=2,
            bounding_box_format="xywh",
            backbone="resnet50",
            backbone_weights=None,
            include_rescaling=False,
        )
        loss = keras_cv.losses.ObjectDetectionLoss(
            classes=1,
            classification_loss=keras_cv.losses.FocalLoss(
                from_logits=True, reduction="none"
            ),
            box_loss=keras_cv.losses.SmoothL1Loss(l1_cutoff=1.0, reduction="none"),
            reduction="sum",
        )

        with self.assertRaisesRegExp(
            ValueError,
            "Does your model's `classes` parameter match your losses `classes` parameter",
        ):
            retina_net.compile(
                optimizer=optimizers.SGD(learning_rate=0.25),
                loss=loss,
            )

    @pytest.mark.skipif(
        not "INTEGRATION" in os.environ,
        reason="Takes a long time to run, only runs when INTEGRATION "
        "environment variable is set.",
    )
    @parameterized.named_parameters(
        ("xywh", "xywh"),
        # ("xyxy", "xyxy"),
        # ("rel_xyxy", "rel_xyxy")
    )
    def test_fit_coco_metrics(self, bounding_box_format):
        retina_net = keras_cv.models.RetinaNet(
            classes=1,
            bounding_box_format=bounding_box_format,
            backbone="resnet50",
            backbone_weights=None,
            include_rescaling=False,
        )
        loss = keras_cv.losses.ObjectDetectionLoss(
            classes=1,
            classification_loss=keras_cv.losses.FocalLoss(
                from_logits=True, reduction="none"
            ),
            box_loss=keras_cv.losses.SmoothL1Loss(l1_cutoff=1.0, reduction="none"),
            reduction="sum",
        )

        retina_net.compile(
            optimizer=optimizers.SGD(),
            loss=loss,
            metrics=[
                keras_cv.metrics.COCOMeanAveragePrecision(
                    class_ids=range(1),
                    bounding_box_format=bounding_box_format,
                    name="MaP",
                ),
                keras_cv.metrics.COCORecall(
                    class_ids=range(1),
                    bounding_box_format=bounding_box_format,
                    name="Recall",
                ),
            ],
        )

        xs, ys = _create_bounding_box_dataset(bounding_box_format)

        for _ in range(50):
            history = retina_net.fit(x=xs, y=ys, epochs=1)
            metrics = history.history
            metrics = [metrics["loss"], metrics["Recall"], metrics["MaP"]]
            metrics = [statistics.mean(metric) for metric in metrics]
            nonzero = [x != 0.0 for x in metrics]
            if all(nonzero):
                return
        raise ValueError("Did not achieve better than 0.0 for all metrics in 50 epochs")


def _create_bounding_box_dataset(bounding_box_format):

    # Just about the easiest dataset you can have, all classes are 0, all boxes are
    # exactly the same.  [1, 1, 2, 2] are the coordinates in xyxy
    xs = tf.ones((10, 512, 512, 3), dtype=tf.float32)
    y_classes = tf.ones((10, 10, 1), dtype=tf.float32)

    ys = tf.constant([0.25, 0.25, 0.1, 0.1], dtype=tf.float32)
    ys = tf.expand_dims(ys, axis=0)
    ys = tf.expand_dims(ys, axis=0)
    ys = tf.tile(ys, [10, 10, 1])
    ys = tf.concat([ys, y_classes], axis=-1)

    ys = keras_cv.bounding_box.convert_format(
        ys, source="rel_xywh", target=bounding_box_format, images=xs, dtype=tf.float32
    )
    return xs, ys
