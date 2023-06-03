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
from tensorflow import keras
from tensorflow.keras import optimizers

import keras_cv
from keras_cv.models.object_detection.__test_utils__ import (
    _create_bounding_box_dataset,
)


class RetinaNetTest(tf.test.TestCase, parameterized.TestCase):
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
            backbone=keras_cv.models.ResNet18V2Backbone(),
        )
        retinanet.compile(
            classification_loss="focal",
            box_loss="smoothl1",
            optimizer="adam",
        )

        # TODO(lukewood) uncomment when using keras_cv.models.ResNet18
        # self.assertIsNotNone(retinanet.backbone.get_layer(name="rescaling"))
        # TODO(lukewood): test compile with the FocalLoss class

    @pytest.mark.large  # Fit is slow, so mark these large.
    def test_retinanet_call(self):
        retinanet = keras_cv.models.RetinaNet(
            num_classes=20,
            bounding_box_format="xywh",
            backbone=keras_cv.models.ResNet18V2Backbone(),
        )
        images = tf.random.uniform((2, 512, 512, 3))
        _ = retinanet(images)
        _ = retinanet.predict(images)

    def test_wrong_logits(self):
        retinanet = keras_cv.models.RetinaNet(
            num_classes=2,
            bounding_box_format="xywh",
            backbone=keras_cv.models.ResNet18V2Backbone(),
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
            backbone=keras_cv.models.ResNet18V2Backbone(),
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
            backbone=keras_cv.models.ResNet18V2Backbone(),
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

    @pytest.mark.large  # Fit is slow, so mark these large.
    def test_no_nans(self):
        retina_net = keras_cv.models.RetinaNet(
            num_classes=2,
            bounding_box_format="xywh",
            backbone=keras_cv.models.ResNet18V2Backbone(),
        )

        retina_net.compile(
            optimizer=optimizers.Adam(),
            classification_loss="focal",
            box_loss="smoothl1",
        )

        # only a -1 box
        xs = tf.ones((1, 512, 512, 3), tf.float32)
        ys = {
            "classes": tf.constant([[-1]], tf.float32),
            "boxes": tf.constant([[[0, 0, 0, 0]]], tf.float32),
        }
        ds = tf.data.Dataset.from_tensor_slices((xs, ys))
        ds = ds.repeat(2)
        ds = ds.batch(2)
        retina_net.fit(ds, epochs=1)

        weights = retina_net.get_weights()
        for weight in weights:
            self.assertFalse(tf.math.reduce_any(tf.math.is_nan(weight)))

    @pytest.mark.large  # Fit is slow, so mark these large.
    def test_weights_change(self):
        bounding_box_format = "xywh"
        retinanet = keras_cv.models.RetinaNet(
            num_classes=2,
            bounding_box_format=bounding_box_format,
            backbone=keras_cv.models.ResNet18V2Backbone(),
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

    @parameterized.named_parameters(
        ("tf_format", "tf", "model"),
        ("keras_format", "keras_v3", "model.keras"),
    )
    @pytest.mark.large  # Saving is slow, so mark these large.
    def test_saved_model(self, save_format, filename):
        model = keras_cv.models.RetinaNet(
            num_classes=20,
            bounding_box_format="xywh",
            backbone=keras_cv.models.ResNet18V2Backbone(),
        )
        input_batch = tf.ones(shape=(2, 224, 224, 3))
        model_output = model(input_batch)
        save_path = os.path.join(self.get_temp_dir(), filename)
        model.save(save_path, save_format=save_format)
        restored_model = keras.models.load_model(save_path)

        # Check we got the real object back.
        self.assertIsInstance(restored_model, keras_cv.models.RetinaNet)

        # Check that output matches.
        restored_output = restored_model(input_batch)
        self.assertAllClose(model_output, restored_output)


@pytest.mark.large
class RetinaNetSmokeTest(tf.test.TestCase):
    def test_backbone_preset(self):
        weights = [
            "csp_darknet_tiny",
            "csp_darknet_s",
            "csp_darknet_m",
            "csp_darknet_l",
            "csp_darknet_xl",
            "efficientnetv2_s",
            "efficientnetv2_m",
            "efficientnetv2_l",
            "efficientnetv2_b0",
            "efficientnetv2_b1",
            "efficientnetv2_b2",
            "efficientnetv2_b3",
            "mobilenet_v3_small",
            "mobilenet_v3_small",
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
            "resnet152",
            "resnet18_v2",
            "resnet34_v2",
            "resnet50_v2",
            "resnet101_v2",
            "resnet152_v2",
        ]
        for weight in weights:
            model = keras_cv.models.RetinaNet.from_preset(
                weight,
                num_classes=20,
                bounding_box_format="xywh",
            )
            xs, _ = _create_bounding_box_dataset(bounding_box_format="xywh")
            output = model(xs)
            self.assertEqual(output["box"].shape, (xs.shape[0], 49104, 4))

    def test_full_preset_weight_loading(self):
        model = keras_cv.models.RetinaNet.from_preset(
            "retinanet_resnet50_pascalvoc",
            bounding_box_format="xywh",
        )
        xs, _ = _create_bounding_box_dataset(bounding_box_format="xywh")
        output = model(xs)

        expected_box = tf.constant(
            [-0.40364307, 0.15742484, -1.3398054, 0.38104224]
        )
        self.assertAllClose(output["box"][0, 123, :], expected_box, atol=0.2)

        expected_class = tf.constant(
            [
                -8.110557,
                -7.5584545,
                -7.9096074,
                -8.163499,
                -6.721413,
                -7.8551226,
                -6.9021387,
                -8.499062,
                -6.190583,
                -7.294302,
                -6.341216,
                -7.6847053,
                -7.8064694,
                -7.7984314,
                -6.5489616,
                -6.524928,
                -7.1218467,
                -7.070874,
                -8.457317,
                -6.167918,
            ]
        )
        expected_class = tf.reshape(expected_class, (20,))
        self.assertAllClose(
            output["classification"][0, 123], expected_class, atol=0.2
        )
