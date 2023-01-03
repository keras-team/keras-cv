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
import tempfile

import pytest
import tensorflow as tf

import keras_cv


@pytest.mark.skipif(
    "INTEGRATION" not in os.environ or os.environ["INTEGRATION"] != "true",
    reason="Takes a long time to run, only runs when INTEGRATION "
    "environment variable is set.  To run the test please run: \n"
    "`INTEGRATION=true pytest keras_cv/",
)
class RetinaNetTest(tf.test.TestCase):
    @pytest.fixture(autouse=True)
    def cleanup_global_session(self):
        # Code before yield runs before the test
        tf.config.set_soft_device_placement(False)
        yield
        # Reset soft device placement to not interfere with other unit test files
        tf.config.set_soft_device_placement(True)
        # Clear global session to prevent OOM between test runs
        tf.keras.backend.clear_session()

    def test_weight_setting(self):
        x, y = _create_bounding_box_dataset(bounding_box_format="xywh")
        pretrained_retina_net, new_retina_net = _create_retina_nets(x, y, epochs=1)

        new_retina_net.set_weights(pretrained_retina_net.get_weights())

        # check if all weights that show up via `get_weights()` are loaded
        for retina_net_weight, post_load_weight in zip(
            pretrained_retina_net.get_weights(), new_retina_net.get_weights()
        ):
            self.assertAllEqual(retina_net_weight, post_load_weight)

        for layer_original, layer_new in zip(
            # manually check layers to make sure nothing is missed
            _get_retina_net_layers(pretrained_retina_net),
            _get_retina_net_layers(new_retina_net),
        ):
            for weight, weight_new in zip(
                layer_original.get_weights(), layer_new.get_weights()
            ):
                self.assertAllEqual(weight, weight_new)

    def test_decoder_doesnt_get_updated(self):
        x, y = _create_bounding_box_dataset(bounding_box_format="xywh")
        pretrained_retina_net, new_retina_net = _create_retina_nets(
            x, y, epochs=1, custom_decoder=True
        )
        new_retina_net.set_weights(pretrained_retina_net.get_weights())

        # check if all weights that show up via `get_weights()` are loaded
        for retina_net_weight, post_load_weight in zip(
            pretrained_retina_net.get_weights(), new_retina_net.get_weights()
        ):
            self.assertAllEqual(retina_net_weight, post_load_weight)

        pretrained_decoder = pretrained_retina_net.prediction_decoder
        new_decoder = new_retina_net.prediction_decoder
        self.assertEqual(new_decoder.suppression_layer.iou_threshold, 0.75)
        self.assertNotEqual(
            new_decoder.suppression_layer.iou_threshold,
            pretrained_decoder.suppression_layer.iou_threshold,
        )

    @pytest.mark.skipif(os.name == "nt", reason="tempfile does not work on windows")
    def test_savedmodel_creation(self):
        x, y = _create_bounding_box_dataset(bounding_box_format="xywh")
        pretrained_retina_net, new_retina_net = _create_retina_nets(x, y, epochs=1)

        tmp = tempfile.mkdtemp()
        pretrained_retina_net.save(f"{tmp}/checkpoint/")
        load_model = tf.saved_model.load(f"{tmp}/checkpoint/")
        _ = load_model(x)

    @pytest.mark.skipif(os.name == "nt", reason="tempfile does not work on windows")
    def test_savedmodel_format_weight_loading(self):
        x, y = _create_bounding_box_dataset(bounding_box_format="xywh")
        pretrained_retina_net, new_retina_net = _create_retina_nets(x, y, epochs=1)

        tmp = tempfile.mkdtemp()
        pretrained_retina_net.save_weights(f"{tmp}/checkpoint/")
        new_retina_net.load_weights(f"{tmp}/checkpoint/")
        for layer_original, layer_new in zip(
            pretrained_retina_net.layers, new_retina_net.layers
        ):
            for weight, weight_new in zip(
                layer_original.get_weights(), layer_new.get_weights()
            ):
                self.assertAllEqual(weight, weight_new)

    def test_set_prediction_decoder(self):
        x, y = _create_bounding_box_dataset(bounding_box_format="xywh")
        pretrained_retina_net, _ = _create_retina_nets(x, y, epochs=0)

        prediction_decoder = keras_cv.layers.NmsDecoder(
            bounding_box_format="xywh",
            anchor_generator=keras_cv.models.RetinaNet.default_anchor_generator(
                bounding_box_format="xywh"
            ),
            suppression_layer=keras_cv.layers.NonMaxSuppression(
                iou_threshold=0.75,
                bounding_box_format="xywh",
                classes=20,
                confidence_threshold=0.85,
            ),
        )
        pretrained_retina_net.prediction_decoder = prediction_decoder
        _ = pretrained_retina_net.predict(x)

    @pytest.mark.skipif(os.name == "nt", reason="tempfile does not work on windows")
    def test_weight_loading(self):
        x, y = _create_bounding_box_dataset(bounding_box_format="xywh")
        pretrained_retina_net, new_retina_net = _create_retina_nets(x, y, epochs=1)

        tmp = tempfile.mkdtemp()
        pretrained_retina_net.save_weights(f"{tmp}/checkpoint.h5")
        new_retina_net.load_weights(f"{tmp}/checkpoint.h5")

        # manually check layers to make sure nothing is missed
        for layer_original, layer_new in zip(
            _get_retina_net_layers(pretrained_retina_net),
            _get_retina_net_layers(new_retina_net),
        ):
            for weight, weight_new in zip(
                layer_original.get_weights(), layer_new.get_weights()
            ):
                self.assertAllEqual(weight, weight_new)

        # manually check layers to make sure nothing is missed in `get_weights()`
        for layer_original, layer_new in zip(
            pretrained_retina_net.layers, new_retina_net.layers
        ):
            for weight, weight_new in zip(
                layer_original.get_weights(), layer_new.get_weights()
            ):
                self.assertAllEqual(weight, weight_new)

    def test_weight_loading_via_metrics(self):
        x, y = _create_bounding_box_dataset(bounding_box_format="xywh")
        pretrained_retina_net, new_retina_net = _create_retina_nets(x, y, epochs=30)

        tmp = tempfile.mkdtemp()
        pretrained_retina_net.save_weights(f"{tmp}/checkpoint.h5")
        new_retina_net.load_weights(f"{tmp}/checkpoint.h5")

        metrics = pretrained_retina_net.evaluate(x, y, return_dict=True)
        new_metrics = new_retina_net.evaluate(x, y, return_dict=True)

        for key in metrics:
            self.assertEqual(metrics[key], new_metrics[key])


def _get_retina_net_layers(model):
    return [
        model.backbone,
        model.feature_pyramid,
        model.prediction_decoder,
        model.anchor_generator,
        model.label_encoder,
        model.classification_head,
        model.box_head,
    ]


def _create_retina_nets(x, y, epochs=1, custom_decoder=False):
    backbone = keras_cv.models.ResNet50(
        include_top=False, weights="imagenet", include_rescaling=False
    ).as_backbone()
    pretrained_retina_net = keras_cv.models.RetinaNet(
        classes=20,
        bounding_box_format="xywh",
        backbone=backbone,
        backbone_weights="imagenet",
        include_rescaling=True,
    )
    pretrained_retina_net.compile(
        classification_loss=keras_cv.losses.FocalLoss(
            from_logits=True,
            reduction="none",
        ),
        box_loss=keras_cv.losses.SmoothL1Loss(l1_cutoff=1.0, reduction="none"),
        optimizer="adam",
    )
    pretrained_retina_net.build((None, None, None, 3))
    # we need to fit the pretrained retina net to ensure the classification_head and
    # regression head get updated.
    if epochs != 0:
        pretrained_retina_net.fit(x, y, epochs=epochs)

    # New RetinaNet is constructed with a custom prediction decoder, and no
    # pretrained backbone weights
    prediction_decoder = None
    if custom_decoder:
        prediction_decoder = keras_cv.layers.NmsDecoder(
            bounding_box_format="xywh",
            anchor_generator=keras_cv.models.RetinaNet.default_anchor_generator(
                bounding_box_format="xywh"
            ),
            suppression_layer=keras_cv.layers.NonMaxSuppression(
                iou_threshold=0.75,
                bounding_box_format="xywh",
                classes=20,
                confidence_threshold=0.85,
            ),
        )
    new_retina_net = keras_cv.models.RetinaNet(
        classes=20,
        bounding_box_format="xywh",
        backbone=backbone,
        backbone_weights=None,
        include_rescaling=True,
        prediction_decoder=prediction_decoder,
    )
    new_retina_net.compile(
        classification_loss=keras_cv.losses.FocalLoss(
            from_logits=True,
            reduction="none",
        ),
        box_loss=keras_cv.losses.SmoothL1Loss(l1_cutoff=1.0, reduction="none"),
        optimizer="adam",
    )
    new_retina_net.build((None, None, None, 3))
    return pretrained_retina_net, new_retina_net


def _create_bounding_box_dataset(bounding_box_format):

    # Just about the easiest dataset you can have, all classes are 0, all boxes are
    # exactly the same.  [1, 1, 2, 2] are the coordinates in xyxy
    xs = tf.ones((10, 512, 512, 3), dtype=tf.float32)
    y_classes = tf.zeros((10, 10, 1), dtype=tf.float32)

    ys = tf.constant([0.25, 0.25, 0.1, 0.1], dtype=tf.float32)
    ys = tf.expand_dims(ys, axis=0)
    ys = tf.expand_dims(ys, axis=0)
    ys = tf.tile(ys, [10, 10, 1])

    ys = keras_cv.bounding_box.convert_format(
        ys, source="rel_xywh", target=bounding_box_format, images=xs, dtype=tf.float32
    )
    return xs, {"boxes": ys, "classes": y_classes}
