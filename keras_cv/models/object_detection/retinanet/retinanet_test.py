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

import copy
import os

import pytest
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers

import keras_cv
from keras_cv.models.object_detection.__test_utils__ import (
    _create_bounding_box_dataset,
)


class RetinaNetTest(tf.test.TestCase):
    @pytest.fixture(autouse=True)
    def cleanup_global_session(self):
        # Code before yield runs before the test
        tf.config.set_soft_device_placement(False)
        yield
        # Reset soft device placement to not interfere with other unit test
        # files
        tf.config.set_soft_device_placement(True)
        keras.backend.clear_session()

    def test_retinanet_construction(self):
        retinanet = keras_cv.models.RetinaNet(
            num_classes=20,
            bounding_box_format="xywh",
            backbone=keras_cv.models.ResNet50V2Backbone(),
        )
        retinanet.compile(
            classification_loss="focal",
            box_loss="smoothl1",
            optimizer="adam",
        )

        # TODO(lukewood) uncomment when using keras_cv.models.ResNet50
        # self.assertIsNotNone(retinanet.backbone.get_layer(name="rescaling"))
        # TODO(lukewood): test compile with the FocalLoss class

    @pytest.mark.skipif(
        "INTEGRATION" not in os.environ or os.environ["INTEGRATION"] != "true",
        reason="Takes a long time to run, only runs when INTEGRATION "
        "environment variable is set. To run the test please run: \n"
        "`INTEGRATION=true pytest keras_cv/",
    )
    def test_retinanet_call(self):
        retinanet = keras_cv.models.RetinaNet(
            num_classes=20,
            bounding_box_format="xywh",
            backbone=keras_cv.models.ResNet50V2Backbone(),
        )
        images = tf.random.uniform((2, 512, 512, 3))
        _ = retinanet(images)
        _ = retinanet.predict(images)

    def test_wrong_logits(self):
        retinanet = keras_cv.models.RetinaNet(
            num_classes=2,
            bounding_box_format="xywh",
            backbone=keras_cv.models.ResNet50V2Backbone(),
        )

        with self.assertRaisesRegex(
            ValueError,
            "from_logits",
        ):
            retinanet.compile(
                optimizer=optimizers.SGD(learning_rate=0.25),
                classification_loss=keras_cv.losses.FocalLoss(
                    from_logits=False, reduction="none"
                ),
                box_loss=keras_cv.losses.SmoothL1Loss(
                    l1_cutoff=1.0, reduction="none"
                ),
            )

    def test_no_metrics(self):
        retinanet = keras_cv.models.RetinaNet(
            num_classes=2,
            bounding_box_format="xywh",
            backbone=keras_cv.models.ResNet50V2Backbone(),
        )

        retinanet.compile(
            optimizer=optimizers.SGD(learning_rate=0.25),
            classification_loss=keras_cv.losses.FocalLoss(
                from_logits=True, reduction="none"
            ),
            box_loss=keras_cv.losses.SmoothL1Loss(
                l1_cutoff=1.0, reduction="none"
            ),
        )

    def test_weights_contained_in_trainable_variables(self):
        bounding_box_format = "xywh"
        retinanet = keras_cv.models.RetinaNet(
            num_classes=2,
            bounding_box_format=bounding_box_format,
            backbone=keras_cv.models.ResNet50V2Backbone(),
        )
        retinanet.backbone.trainable = False
        retinanet.compile(
            optimizer=optimizers.Adam(),
            classification_loss=keras_cv.losses.FocalLoss(
                from_logits=True, reduction="none"
            ),
            box_loss=keras_cv.losses.SmoothL1Loss(
                l1_cutoff=1.0, reduction="none"
            ),
        )
        xs, ys = _create_bounding_box_dataset(bounding_box_format)

        # call once
        _ = retinanet(xs)
        variable_names = [x.name for x in retinanet.trainable_variables]
        # classification_head
        self.assertIn("prediction_head/conv2d_8/kernel:0", variable_names)
        # box_head
        self.assertIn("prediction_head_1/conv2d_12/kernel:0", variable_names)

    def test_weights_change(self):
        bounding_box_format = "xywh"
        retinanet = keras_cv.models.RetinaNet(
            num_classes=2,
            bounding_box_format=bounding_box_format,
            backbone=keras_cv.models.ResNet50V2Backbone(),
        )

        retinanet.compile(
            optimizer=optimizers.Adam(),
            classification_loss=keras_cv.losses.FocalLoss(
                from_logits=True, reduction="none"
            ),
            box_loss=keras_cv.losses.SmoothL1Loss(
                l1_cutoff=1.0, reduction="none"
            ),
        )
        xs, ys = _create_bounding_box_dataset(bounding_box_format)

        # call once
        _ = retinanet(xs)
        original_fpn_weights = retinanet.feature_pyramid.get_weights()
        original_box_head_weights = retinanet.box_head.get_weights()
        original_classification_head_weights = (
            retinanet.classification_head.get_weights()
        )

        retinanet.fit(x=xs, y=ys, epochs=1)
        fpn_after_fit = retinanet.feature_pyramid.get_weights()
        box_head_after_fit_weights = retinanet.box_head.get_weights()
        classification_head_after_fit_weights = (
            retinanet.classification_head.get_weights()
        )

        for w1, w2 in zip(
            original_classification_head_weights,
            classification_head_after_fit_weights,
        ):
            self.assertNotAllClose(w1, w2)

        for w1, w2 in zip(
            original_box_head_weights, box_head_after_fit_weights
        ):
            self.assertNotAllClose(w1, w2)

        for w1, w2 in zip(original_fpn_weights, fpn_after_fit):
            self.assertNotAllClose(w1, w2)

    def test_serialization(self):
        # TODO(haifengj): Reuse test code from
        # ModelTest._test_model_serialization.
        model = keras_cv.models.RetinaNet(
            num_classes=20,
            bounding_box_format="xywh",
            backbone=keras_cv.models.ResNet50V2Backbone(),
        )
        serialized_1 = keras.utils.serialize_keras_object(model)
        restored = keras.utils.deserialize_keras_object(
            copy.deepcopy(serialized_1)
        )
        serialized_2 = keras.utils.serialize_keras_object(restored)
        self.assertEqual(serialized_1, serialized_2)


@pytest.mark.large
class RetinaNetSmokeTest(tf.test.TestCase):
    def test_backbone_preset_weight_loading(self):
        # Check that backbone preset weights loaded correctly
        # TODO(lukewood): need to forward pass test once proper weights are
        # implemented
        keras_cv.models.RetinaNet.from_preset(
            "resnet50_v2_imagenet",
            num_classes=20,
            bounding_box_format="xywh",
        )

    def test_full_preset_weight_loading(self):
        # Check that backbone preset weights loaded correctly
        # TODO(lukewood): need to forward pass test once proper weights are
        # implemented
        keras_cv.models.RetinaNet.from_preset(
            "retinanet_resnet50_pascalvoc",
            bounding_box_format="xywh",
        )
