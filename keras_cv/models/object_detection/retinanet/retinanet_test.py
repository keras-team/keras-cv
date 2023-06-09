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
from keras_cv.models.object_detection.retinanet import RetinaNetLabelEncoder


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

    def test_call_with_custom_label_encoder(self):
        anchor_generator = (
            keras_cv.models.RetinaNet.default_anchor_generator("xywh"),
        )
        model = keras_cv.models.RetinaNet(
            num_classes=20,
            bounding_box_format="xywh",
            backbone=keras_cv.models.ResNet18V2Backbone(),
            label_encoder=RetinaNetLabelEncoder(
                bounding_box_format="xywh",
                anchor_generator=anchor_generator,
                box_variance=[0.1, 0.1, 0.2, 0.2],
            ),
        )
        model(tf.ones(shape=(2, 224, 224, 3)))


@pytest.mark.large
class RetinaNetSmokeTest(tf.test.TestCase):
    def test_backbone_preset(self):
        for preset in keras_cv.models.RetinaNet.presets_without_weights:
            model = keras_cv.models.RetinaNet.from_preset(
                preset,
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
        xs = tf.ones((1, 512, 512, 3), tf.float32)
        output = model(xs)

        expected_box = tf.constant(
            [-1.2427993, 0.05179548, -1.9953268, 0.32456252]
        )
        self.assertAllClose(output["box"][0, 123, :], expected_box, atol=1e-5)

        expected_class = tf.constant(
            [
                -8.387445,
                -7.891776,
                -8.14204,
                -8.117359,
                -7.2517176,
                -7.906804,
                -7.0910635,
                -8.295824,
                -6.5567474,
                -7.086027,
                -6.3826647,
                -7.960227,
                -7.556676,
                -8.28963,
                -6.526232,
                -7.071624,
                -6.9687414,
                -6.6398506,
                -8.598567,
                -6.484198,
            ]
        )
        expected_class = tf.reshape(expected_class, (20,))
        self.assertAllClose(
            output["classification"][0, 123], expected_class, atol=1e-5
        )
