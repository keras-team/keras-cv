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
import keras_cv


class RetinaNetTest(tf.test.TestCase):
    @pytest.fixture(autouse=True)
    def cleanup_global_session(self):
        # Code before yield runs before the test
        yield
        tf.keras.backend.clear_session()

    def test_retina_net_construction(self):
        retina_net = keras_cv.applications.RetinaNet(
            num_classes=20,
            bounding_box_format="xywh",
            backbone="resnet50",
            backbone_weights=None,
            include_rescaling=True,
        )
        retina_net.compile(
            loss="mse",
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

    def test_retina_net_include_rescaling_required_with_default_backbone(self):
        with self.assertRaises(ValueError):
            _ = keras_cv.applications.RetinaNet(
                num_classes=20,
                bounding_box_format="xywh",
                backbone="resnet50",
                backbone_weights=None,
                # Note no include_rescaling is provided
            )
        # TODO(lukewood): test compile with the FocalLoss class

    def test_all_metric_formats_must_match(self):
        retina_net = keras_cv.applications.RetinaNet(
            num_classes=20,
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

#     #     TODO(lukewood): reintroduce this when FocalLoss is added.
#     def test_overfits_single_bounding_box(self):
#         bounding_box_format = "xywh"
#         retina_net = keras_cv.applications.RetinaNet(
#             num_classes=2,
#             bounding_box_format="xywh",
#             backbone='resnet50',
#             backbone_weights=None,
#             include_rescaling=False,
#         )
#         retina_net.compile(
#             optimizer="adam",
#             loss=keras_cv.applications.RetinaNetLoss(num_classes=2, reduction="sum"),
#             metrics=[
#                 keras_cv.metrics.COCOMeanAveragePrecision(
#                     class_ids=range(1),
#                     bounding_box_format="xywh",
#                     name="Standard MaP",
#                 )
#             ],
#         )
#
#         xs, ys = create_bounding_box_dataset(bounding_box_format)
#         for _ in range(50):
#             retina_net.fit(x=xs, y=ys, epochs=1)
#             map = retina_net.evaluate(x=xs, y=ys, return_dict=True)["Standard MaP"]
#             if map == 1.0:
#                 break
#
#         self.assertEqual(metric.result(), 1.0)
#
# def create_bounding_box_dataset(bounding_box_format):
#     xs = tf.ones((32, 512, 512, 3), tf.float32)
#     ys = tf.ones((32, 4, 6), tf.float32)
#     ys = keras_cv.bounding_box.convert_format(
#         ys, source="xywh", target=bounding_box_format, dtype=tf.float32
#     )
#     return xs, ys
