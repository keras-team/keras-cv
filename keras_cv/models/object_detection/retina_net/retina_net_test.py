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
from tensorflow.keras import optimizers

import keras_cv
from keras_cv.models.object_detection.__test_utils__ import _create_bounding_box_dataset


class RetinaNetTest(tf.test.TestCase):
    @pytest.fixture(autouse=True)
    def cleanup_global_session(self):
        # Code before yield runs before the test
        tf.config.set_soft_device_placement(False)
        yield
        # Reset soft device placement to not interfere with other unit test files
        tf.config.set_soft_device_placement(True)
        tf.keras.backend.clear_session()

    def test_retina_net_construction(self):
        retina_net = keras_cv.models.RetinaNet(
            classes=20,
            bounding_box_format="xywh",
            backbone="resnet50",
            backbone_weights=None,
            include_rescaling=True,
        )
        retina_net.compile(
            classification_loss="focal",
            box_loss="smoothl1",
            optimizer="adam",
        )

        # TODO(lukewood) uncomment when using keras_cv.models.ResNet50
        # self.assertIsNotNone(retina_net.backbone.get_layer(name="rescaling"))
        # TODO(lukewood): test compile with the FocalLoss class

    def test_retina_net_include_rescaling_required_with_default_backbone(self):
        with self.assertRaises(ValueError):
            _ = keras_cv.models.RetinaNet(
                classes=20,
                bounding_box_format="xywh",
                backbone="resnet50",
                backbone_weights=None,
                # Note no include_rescaling is provided
            )

    @pytest.mark.skipif(
        "INTEGRATION" not in os.environ or os.environ["INTEGRATION"] != "true",
        reason="Takes a long time to run, only runs when INTEGRATION "
        "environment variable is set.  To run the test please run: \n"
        "`INTEGRATION=true pytest keras_cv/",
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
        _ = retina_net(images)
        _ = retina_net.predict(images)

    def test_loss_output_shape_error_messages(self):
        retina_net = keras_cv.models.RetinaNet(
            classes=20,
            bounding_box_format="xywh",
            backbone="resnet50",
            backbone_weights=None,
            include_rescaling=True,
        )
        xs, ys = _create_bounding_box_dataset("xywh")

        # all metric formats must match
        retina_net.compile(
            optimizer="adam",
            box_loss=keras_cv.losses.SmoothL1Loss(reduction="none"),
            classification_loss=keras_cv.losses.FocalLoss(
                from_logits=True, reduction="sum"
            ),
        )

        with self.assertRaisesRegex(
            ValueError, "output shape of `classification_loss`"
        ):
            retina_net.fit(x=xs, y=ys, epochs=1)

        # all metric formats must match
        retina_net.compile(
            optimizer="adam",
            box_loss=keras_cv.losses.SmoothL1Loss(reduction="sum"),
            classification_loss=keras_cv.losses.FocalLoss(
                from_logits=True, reduction="none"
            ),
        )
        with self.assertRaisesRegex(ValueError, "output shape of `box_loss`"):
            retina_net.fit(x=xs, y=ys, epochs=1)

    def test_wrong_logits(self):
        retina_net = keras_cv.models.RetinaNet(
            classes=2,
            bounding_box_format="xywh",
            backbone="resnet50",
            backbone_weights=None,
            include_rescaling=False,
        )

        with self.assertRaisesRegex(
            ValueError,
            "from_logits",
        ):
            retina_net.compile(
                optimizer=optimizers.SGD(learning_rate=0.25),
                classification_loss=keras_cv.losses.FocalLoss(
                    from_logits=False, reduction="none"
                ),
                box_loss=keras_cv.losses.SmoothL1Loss(l1_cutoff=1.0, reduction="none"),
            )

    def test_no_metrics(self):
        retina_net = keras_cv.models.RetinaNet(
            classes=2,
            bounding_box_format="xywh",
            backbone="resnet50",
            backbone_weights=None,
            include_rescaling=False,
        )

        retina_net.compile(
            optimizer=optimizers.SGD(learning_rate=0.25),
            classification_loss=keras_cv.losses.FocalLoss(
                from_logits=True, reduction="none"
            ),
            box_loss=keras_cv.losses.SmoothL1Loss(l1_cutoff=1.0, reduction="none"),
        )

    def test_weights_contained_in_trainable_variables(self):
        bounding_box_format = "xywh"
        retina_net = keras_cv.models.RetinaNet(
            classes=1,
            bounding_box_format=bounding_box_format,
            backbone="resnet50",
            backbone_weights=None,
            include_rescaling=False,
        )
        retina_net.backbone.trainable = False
        retina_net.compile(
            optimizer=optimizers.Adam(),
            classification_loss=keras_cv.losses.FocalLoss(
                from_logits=True, reduction="none"
            ),
            box_loss=keras_cv.losses.SmoothL1Loss(l1_cutoff=1.0, reduction="none"),
        )
        xs, ys = _create_bounding_box_dataset(bounding_box_format)

        # call once
        _ = retina_net(xs)
        variable_names = [x.name for x in retina_net.trainable_variables]
        # classification_head
        self.assertIn("RetinaNet/prediction_head/conv2d_8/kernel:0", variable_names)
        # box_head
        self.assertIn("RetinaNet/prediction_head_1/conv2d_12/kernel:0", variable_names)

    def test_weights_change(self):
        bounding_box_format = "xywh"
        retina_net = keras_cv.models.RetinaNet(
            classes=1,
            bounding_box_format=bounding_box_format,
            backbone="resnet50",
            backbone_weights=None,
            include_rescaling=False,
        )

        retina_net.compile(
            optimizer=optimizers.Adam(),
            classification_loss=keras_cv.losses.FocalLoss(
                from_logits=True, reduction="none"
            ),
            box_loss=keras_cv.losses.SmoothL1Loss(l1_cutoff=1.0, reduction="none"),
        )
        xs, ys = _create_bounding_box_dataset(bounding_box_format)

        # call once
        _ = retina_net(xs)
        original_fpn_weights = retina_net.feature_pyramid.get_weights()
        original_box_head_weights = retina_net.box_head.get_weights()
        original_classification_head_weights = (
            retina_net.classification_head.get_weights()
        )

        retina_net.fit(x=xs, y=ys, epochs=1)
        fpn_after_fit = retina_net.feature_pyramid.get_weights()
        box_head_after_fit_weights = retina_net.box_head.get_weights()
        classification_head_after_fit_weights = (
            retina_net.classification_head.get_weights()
        )

        # print('after_fit', after_fit)

        for w1, w2 in zip(
            original_classification_head_weights, classification_head_after_fit_weights
        ):
            self.assertNotAllClose(w1, w2)

        for w1, w2 in zip(original_box_head_weights, box_head_after_fit_weights):
            self.assertNotAllClose(w1, w2)

        for w1, w2 in zip(original_fpn_weights, fpn_after_fit):
            self.assertNotAllClose(w1, w2)

    # TODO(lukewood): configure for other coordinate systems.
    @pytest.mark.skipif(
        "INTEGRATION" not in os.environ or os.environ["INTEGRATION"] != "true",
        reason="Takes a long time to run, only runs when INTEGRATION "
        "environment variable is set.  To run the test please run: \n"
        "`INTEGRATION=true pytest "
        "keras_cv/models/object_detection/retina_net/retina_net_test.py -k "
        "test_fit_coco_metrics -s`",
    )
    def test_fit_coco_metrics(self):
        bounding_box_format = "xywh"
        retina_net = keras_cv.models.RetinaNet(
            classes=1,
            bounding_box_format=bounding_box_format,
            backbone="resnet50",
            backbone_weights=None,
            include_rescaling=False,
        )

        retina_net.compile(
            optimizer=optimizers.Adam(),
            classification_loss=keras_cv.losses.FocalLoss(
                from_logits=True, reduction="none"
            ),
            box_loss=keras_cv.losses.SmoothL1Loss(l1_cutoff=1.0, reduction="none"),
        )

        xs, ys = _create_bounding_box_dataset(bounding_box_format)
        retina_net.fit(x=xs, y=ys, epochs=10)
        _ = retina_net.predict(xs)
