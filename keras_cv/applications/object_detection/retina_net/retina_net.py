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

import numpy as np
import tensorflow as tf
from tensorflow import keras

from keras_cv import bounding_box
from keras_cv.applications.object_detection.retina_net.__internal__ import (
    layers as layers_lib,
)
from keras_cv.applications.object_detection.retina_net.__internal__ import (
    utils as utils_lib,
)


# TODO(lukewood): update docstring to include documentation on creating a custom label
# decoder/etc.
# TODO(lukewood): link to keras.io guide on creating custom backbone and FPN.
class RetinaNet(keras.Model):
    """A Keras model implementing the RetinaNet architecture.

    Implements the RetinaNet architecture for object detection.  The constructor
    requires `classes`, `bounding_box_format` and a `backbone`.  Optionally, a
    custom label encoder, feature pyramid network, and prediction decoder may all be
    provided.

    Usage:
    ```python
    retina_net = keras_cv.applications.RetinaNet(
        classes=20,
        bounding_box_format="xywh",
        backbone="resnet50",
        backbone_weights="imagenet",
        include_rescaling=True,
    )
    ```

    Args:
        classes: the number of classes in your dataset excluding the background
            class.  Classes should be represented by integers in the range
            [0, classes).
        bounding_box_format: The format of bounding boxes of input dataset. Refer
            [to the keras.io docs](https://keras.io/api/keras_cv/bounding_box/formats/)
            for more details on supported bounding box formats.
        backbone: Either `"resnet50"` or a custom backbone model.
        include_rescaling: Required if provided backbone is a pre-configured model.
            If set to `True`, inputs will be passed through a `Rescaling(1/255.0)`
            layer.
        backbone_weights: (Optional) if using a KerasCV provided backbone, the
            underlying backbone model will be loaded using the weights provided in this
            argument.  Can be a model checkpoint path, or a string from the supported
            weight sets in the underlying model.
        label_encoder: (Optional) a keras.Layer that accepts an image Tensor and a
            bounding box Tensor to its `call()` method, and returns RetinaNet training
            targets.  By default, a KerasCV standard LabelEncoder is created and used.
            Results of this `call()` method are passed to the `loss` object passed into
            `compile()` as the `y_true` argument.
        feature_pyramid: (Optional) A `keras.Model` representing a feature pyramid
            network (FPN).  The feature pyramid network is called on the outputs of the
            `backbone`.  The KerasCV default backbones return three outputs in a list,
            but custom backbones may be written and used with custom feature pyramid
            networks.  If not provided, a default feature pyramid neetwork is produced
            by the library.  The default feature pyramid network is compatible with all
            standard keras_cv backbones.
        prediction_decoder: (Optional)  A `keras.layer` that is responsible for
            transforming RetinaNet predictions into usable bounding box Tensors.  If
            not provided, a default is provided.  The default `PredictionDecoder` layer
            operates using an AnchorBox matching algorithm and a `NonMaxSuppression`
            operation.
        name: (Optional) name for the model, defaults to `"RetinaNet"`.
    """

    def __init__(
        self,
        classes,
        bounding_box_format,
        backbone,
        include_rescaling=None,
        backbone_weights=None,
        label_encoder=None,
        feature_pyramid=None,
        prediction_decoder=None,
        name="RetinaNet",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.bounding_box_format = bounding_box_format
        self.classes = classes
        self.backbone = _parse_backbone(backbone, include_rescaling, backbone_weights)

        self.label_encoder = label_encoder or utils_lib.LabelEncoder(
            bounding_box_format=bounding_box_format
        )
        self.feature_pyramid = feature_pyramid or layers_lib.FeaturePyramid()

        prior_probability = tf.constant_initializer(-np.log((1 - 0.01) / 0.01))
        self.classification_head = layers_lib.PredictionHead(
            output_filters=9 * classes, bias_initializer=prior_probability
        )
        self.box_head = layers_lib.PredictionHead(
            output_filters=9 * 4, bias_initializer="zeros"
        )
        self.prediction_decoder = prediction_decoder or layers_lib.DecodePredictions(
            classes=classes, bounding_box_format=bounding_box_format
        )
        self._metrics_bounding_box_format = None

    def compile(self, metrics=None, **kwargs):
        metrics = metrics or []
        super().compile(metrics=metrics, **kwargs)

        all_have_format = any(
            [
                m.bounding_box_format != self._metrics_bounding_box_format
                for m in metrics
            ]
        )
        if not all_have_format:
            raise ValueError(
                "All metrics passed to RetinaNet.compile() must have "
                f"a `bounding_box_format` attribute.  Received metrics={metrics}"
            )

        if len(metrics) != 0:
            self._metrics_bounding_box_format = metrics[0].bounding_box_format
        else:
            self._metrics_bounding_box_format = self.bounding_box_format

        any_wrong_format = any(
            [
                m.bounding_box_format != self._metrics_bounding_box_format
                for m in metrics
            ]
        )
        if any_wrong_format:
            raise ValueError(
                "All metrics passed to RetinaNet.compile() must have "
                "the same `bounding_box_format` attribute.  For example, if one metric "
                "uses 'xyxy', all other metrics must use 'xyxy'.  Received "
                f"metrics={metrics}"
            )

    def call(self, x, training=False):
        backbone_outputs = self.backbone(x, training=training)
        features = self.feature_pyramid(backbone_outputs, training=training)

        N = tf.shape(x)[0]
        cls_outputs = []
        box_outputs = []
        for feature in features:
            box_outputs.append(tf.reshape(self.box_head(feature), [N, -1, 4]))
            cls_outputs.append(
                tf.reshape(self.classification_head(feature), [N, -1, self.classes])
            )

        cls_outputs = tf.concat(cls_outputs, axis=1)
        box_outputs = tf.concat(box_outputs, axis=1)
        train_preds = tf.concat([box_outputs, cls_outputs], axis=-1)

        # no-op if default decoder is used.
        pred_for_inference = bounding_box.convert_format(
            train_preds,
            source=self.bounding_box_format,
            target=self.prediction_decoder.bounding_box_format,
            images=x,
        )
        pred_for_inference = self.prediction_decoder(x, pred_for_inference)
        pred_for_inference = bounding_box.convert_format(
            pred_for_inference,
            source=self.prediction_decoder.bounding_box_format,
            target=self.bounding_box_format,
            images=x,
        )
        return {"train_predictions": train_preds, "inference": pred_for_inference}

    def _encode_data(self, x, y):
        y_for_metrics = y

        y = bounding_box.convert_format(
            y,
            source=self.bounding_box_format,
            target=self.label_encoder.bounding_box_format,
            images=x,
        )
        y_training_target = self.label_encoder(x, y)
        y_training_target = bounding_box.convert_format(
            y_training_target,
            source=self.label_encoder.bounding_box_format,
            target=self.bounding_box_format,
            images=x,
        )
        return y_for_metrics, y_training_target

    def train_step(self, data):
        x, y = data
        # y comes in in self.bounding_box_format
        y_for_metrics, y_training_target = self._encode_data(x, y)

        with tf.GradientTape() as tape:
            predictions = self(x, training=True)
            # predictions technically do not have a format
            # loss accepts

            # TODO(lukewood): allow distinct 'classification' and 'box' loss metrics
            loss = self.compiled_loss(
                y_training_target,
                predictions["train_predictions"],
                regularization_losses=self.losses,
            )

        # Training specific code
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # To minimize GPU transfers, we update metrics AFTER we take grades and apply
        # them.

        # TODO(lukewood): assert that all metric formats are the same
        self._update_metrics(y_for_metrics, predictions["inference"])
        return self._metrics_result(loss)

    def test_step(self, data):
        x, y = data
        y_for_metrics, y_training_target = self._encode_data(x, y)

        predictions = self(x)
        loss = self.compiled_loss(
            y_training_target,
            predictions["train_predictions"],
            regularization_losses=self.losses,
        )

        self._update_metrics(y_for_metrics, predictions["inference"])
        return self._metrics_result(loss)

    def _update_metrics(self, y_true, y_pred):
        y_true = bounding_box.convert_format(
            y_true,
            source=self.bounding_box_format,
            target=self._metrics_bounding_box_format,
        )
        y_pred = bounding_box.convert_format(
            y_pred,
            source=self.bounding_box_format,
            target=self._metrics_bounding_box_format,
        )
        self.compiled_metrics.update_state(y_true, y_pred)

    def _metrics_result(self, loss):
        metrics_result = {m.name: m.result() for m in self.metrics}
        metrics_result["loss"] = loss
        return metrics_result

    def inference(self, x):
        predictions = self.predict(x)
        return predictions["inference"]


def _parse_backbone(backbone, include_rescaling, backbone_weights):
    if isinstance(backbone, str) and include_rescaling is None:
        raise ValueError(
            "When using a preconfigured backbone, please do provide a "
            "`include_rescaling` parameter.  `include_rescaling` is passed to the "
            "Keras application constructor for the provided backbone.  When "
            "`include_rescaling=True`, image inputs are passed through a "
            "`layers.Rescaling(1/255.0)` layer. When `include_rescaling=False`, no "
            "downscaling is performed. "
            f"Received backbone={backbone}, include_rescaling={include_rescaling}."
        )

    if isinstance(backbone, str):
        if backbone == "resnet50":
            return _resnet50_backbone(include_rescaling, backbone_weights)
        else:
            raise ValueError(
                "backbone expected to be one of ['resnet50', keras.Model]. "
                f"Received backbone={backbone}."
            )
    if include_rescaling or backbone_weights:
        raise ValueError(
            "When a custom backbone is used, include_rescaling and "
            f"backbone_weights are not supported.  Received backbone={backbone}, "
            f"include_rescaling={include_rescaling}, and "
            f"backbone_weights={backbone_weights}."
        )
    if not isinstance(backbone, keras.Model):
        raise ValueError(
            "Custom backbones should be subclasses of a keras.Model. "
            f"Received backbone={backbone}."
        )
    return backbone


def _resnet50_backbone(include_rescaling, backbone_weights):
    inputs = keras.layers.Input(shape=(None, None, 3))
    x = inputs

    if include_rescaling:
        x = keras.applications.resnet.preprocess_input(x)

    # TODO(lukewood): this should really be calling keras_cv.models.ResNet50
    backbone = keras.applications.ResNet50(
        include_top=False, input_tensor=x, weights=backbone_weights
    )
    x = backbone(x)

    c3_output, c4_output, c5_output = [
        backbone.get_layer(layer_name).output
        for layer_name in ["conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]
    ]
    return keras.Model(inputs=inputs, outputs=[c3_output, c4_output, c5_output])
