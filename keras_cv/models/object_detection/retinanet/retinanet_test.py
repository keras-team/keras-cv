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
from keras_cv.models.backbones.test_backbone_presets import (
    test_backbone_presets,
)
from keras_cv.models.object_detection.__test_utils__ import (
    _create_bounding_box_dataset,
)
from keras_cv.models.object_detection.retinanet import RetinaNetLabelEncoder
from keras_cv.tests.test_case import TestCase

from keras_cv.backend.config import multi_backend

class RetinaNetTest(TestCase):
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

    def test_retinanet_recompilation_without_metrics(self):
        retinanet = keras_cv.models.RetinaNet(
            num_classes=20,
            bounding_box_format="xywh",
            backbone=keras_cv.models.ResNet18V2Backbone(),
        )
        retinanet.compile(
            classification_loss="focal",
            box_loss="smoothl1",
            optimizer="adam",
            metrics=[
                keras_cv.metrics.BoxCOCOMetrics(
                    bounding_box_format="center_xywh", evaluate_freq=20
                )
            ],
        )
        self.assertIsNotNone(retinanet._user_metrics)
        retinanet.compile(
            classification_loss="focal",
            box_loss="smoothl1",
            optimizer="adam",
            metrics=None,
        )

        self.assertIsNone(retinanet._user_metrics)

    @pytest.mark.large
    def test_cuda_device(self):
        if multi_backend() and keras.config.backend() == "torch":
            print(ops.ones((1)).device.type)
            self.assertTrue(ops.ones((1)).device.type == 'cuda')
        elif multi_backend() and keras.config.backend() == "jax":
            print(str(ops.ones((1)).device()))
            self.assertTrue(str(ops.ones((1)).device()) == "gpu:0")

    @pytest.mark.large  # Fit is slow, so mark these large.
    def test_retinanet_call(self):
        retinanet = keras_cv.models.RetinaNet(
            num_classes=20,
            bounding_box_format="xywh",
            backbone=keras_cv.models.ResNet18V2Backbone(),
        )
        images = np.random.uniform(size=(2, 512, 512, 3))
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
                optimizer=keras.optimizers.SGD(learning_rate=0.25),
                classification_loss=keras_cv.losses.FocalLoss(
                    from_logits=False, reduction="none"
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
            optimizer=keras.optimizers.Adam(),
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
        self.assertEqual(len(retinanet.trainable_variables), 32)

    @pytest.mark.large  # Fit is slow, so mark these large.
    def test_no_nans(self):
        retina_net = keras_cv.models.RetinaNet(
            num_classes=2,
            bounding_box_format="xywh",
            backbone=keras_cv.models.CSPDarkNetTinyBackbone(),
        )

        retina_net.compile(
            optimizer=keras.optimizers.Adam(),
            classification_loss="focal",
            box_loss="smoothl1",
        )

        # only a -1 box
        xs = ops.ones((1, 512, 512, 3), "float32")
        ys = {
            "classes": ops.array([[-1]], "float32"),
            "boxes": ops.array([[[0, 0, 0, 0]]], "float32"),
        }
        ds = tf.data.Dataset.from_tensor_slices((xs, ys))
        ds = ds.repeat(2)
        ds = ds.batch(2, drop_remainder=True)
        retina_net.fit(ds, epochs=1)

        weights = retina_net.get_weights()
        for weight in weights:
            self.assertFalse(ops.any(ops.isnan(weight)))

    @pytest.mark.large  # Fit is slow, so mark these large.
    def test_weights_change(self):
        bounding_box_format = "xywh"
        retinanet = keras_cv.models.RetinaNet(
            num_classes=2,
            bounding_box_format=bounding_box_format,
            backbone=keras_cv.models.CSPDarkNetTinyBackbone(),
        )

        retinanet.compile(
            optimizer=keras.optimizers.Adam(),
            classification_loss=keras_cv.losses.FocalLoss(
                from_logits=True, reduction="sum"
            ),
            box_loss=keras_cv.losses.SmoothL1Loss(
                l1_cutoff=1.0, reduction="sum"
            ),
        )
        ds = _create_bounding_box_dataset(
            bounding_box_format, use_dictionary_box_format=True
        )

        # call once
        _ = retinanet(ops.ones((1, 512, 512, 3)))
        original_fpn_weights = retinanet.feature_pyramid.get_weights()
        original_box_head_weights = retinanet.box_head.get_weights()
        original_classification_head_weights = (
            retinanet.classification_head.get_weights()
        )

        retinanet.fit(ds, epochs=1)
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

    @pytest.mark.large  # Saving is slow, so mark these large.
    def test_saved_model(self):
        model = keras_cv.models.RetinaNet(
            num_classes=20,
            bounding_box_format="xywh",
            backbone=keras_cv.models.CSPDarkNetTinyBackbone(),
        )
        input_batch = ops.ones(shape=(2, 224, 224, 3))
        model_output = model(input_batch)
        save_path = os.path.join(self.get_temp_dir(), "retinanet.keras")
        model.save(save_path)
        restored_model = keras.models.load_model(save_path)

        # Check we got the real object back.
        self.assertIsInstance(restored_model, keras_cv.models.RetinaNet)

        # Check that output matches.
        restored_output = restored_model(input_batch)
        self.assertAllClose(
            tf.nest.map_structure(ops.convert_to_numpy, model_output),
            tf.nest.map_structure(ops.convert_to_numpy, restored_output),
        )

    def test_call_with_custom_label_encoder(self):
        anchor_generator = keras_cv.models.RetinaNet.default_anchor_generator(
            "xywh"
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
        model(ops.ones(shape=(2, 224, 224, 3)))


@pytest.mark.large
class RetinaNetSmokeTest(TestCase):
    @parameterized.named_parameters(
        *[(preset, preset) for preset in test_backbone_presets]
    )
    def test_backbone_preset(self, preset):
        model = keras_cv.models.RetinaNet.from_preset(
            preset,
            num_classes=20,
            bounding_box_format="xywh",
        )
        xs, _ = _create_bounding_box_dataset(bounding_box_format="xywh")
        output = model(xs)

        # 4 represents number of parameters in a box
        # 49104 is the number of anchors for a 512x512 image
        self.assertEqual(output["box"].shape, (xs.shape[0], 49104, 4))

    def test_full_preset_weight_loading(self):
        model = keras_cv.models.RetinaNet.from_preset(
            "retinanet_resnet50_pascalvoc",
            bounding_box_format="xywh",
        )
        xs = ops.ones((1, 512, 512, 3))
        output = model(xs)

        expected_box = ops.array(
            [-1.2427993, 0.05179548, -1.9953268, 0.32456252]
        )
        self.assertAllClose(
            ops.convert_to_numpy(output["box"][0, 123, :]),
            expected_box,
            atol=1e-5,
        )

        expected_class = ops.array(
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
        expected_class = ops.reshape(expected_class, (20,))
        self.assertAllClose(
            ops.convert_to_numpy(output["classification"][0, 123]),
            expected_class,
            atol=1e-5,
        )
