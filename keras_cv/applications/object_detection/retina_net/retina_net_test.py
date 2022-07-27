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
from absl.testing import parameterized

import keras_cv


class RetinaNetTest(tf.test.TestCase):
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

    def test_overfits_single_bounding_box(self):
        # TODO(lukewood): overfit a single stacked bounding box, make sure the format
        # is correct
        pass
