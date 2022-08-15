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
from keras_cv import layers as cv_layers
from keras_cv.models.object_detection.retina_net.__internal__ import (
    layers as layers_lib,
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
    retina_net = keras_cv.models.RetinaNet(
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
        anchor_generator: (Optional) a `keras_cv.layers.AnchorGenerator`.  If provided,
            the anchor generator will be passed to both the `label_encoder` and the
            `prediction_decoder`.  Only to be used when both `label_encoder` and
            `prediction_decoder` are both `None`.
        label_encoder: (Optional) a keras.Layer that accepts an image Tensor and a
            bounding box Tensor to its `call()` method, and returns RetinaNet training
            targets.  By default, a KerasCV standard LabelEncoder is created and used.
            Results of this `call()` method are passed to the `loss` object passed into
            `compile()` as the `y_true` argument.
        prediction_decoder: (Optional)  A `keras.layer` that is responsible for
            transforming RetinaNet predictions into usable bounding box Tensors.  If
            not provided, a default is provided.  The default `prediction_decoder` layer
            uses a `NonMaxSuppression` operation for box pruning.
        feature_pyramid: (Optional) A `keras.Model` representing a feature pyramid
            network (FPN).  The feature pyramid network is called on the outputs of the
            `backbone`.  The KerasCV default backbones return three outputs in a list,
            but custom backbones may be written and used with custom feature pyramid
            networks.  If not provided, a default feature pyramid neetwork is produced
            by the library.  The default feature pyramid network is ompatible with all
            standard keras_cv backbones.
        name: (Optional) name for the model, defaults to `"RetinaNet"`.
    """

    def __init__(
        self,
        classes,
        bounding_box_format,
        backbone,
        include_rescaling=None,
        backbone_weights=None,
        anchor_generator=None,
        label_encoder=None,
        prediction_decoder=None,
        feature_pyramid=None,
        name="RetinaNet",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        if anchor_generator is not None and (prediction_decoder or label_encoder):
            raise ValueError(
                "`anchor_generator` is only to be provided when "
                "both `label_encoder` and `prediction_decoder` are both `None`. "
                f"Received `anchor_generator={anchor_generator}` "
                f"`label_encoder={label_encoder}`, "
                f"`prediction_decoder={prediction_decoder}`. To customize the behavior of "
                "the anchor_generator inside of a custom `label_encoder` or custom "
                "`prediction_decoder` you should provide both to `RetinaNet`, and ensure "
                "that the `anchor_generator` provided to both is identical"
            )

        self.bounding_box_format = bounding_box_format
        anchor_generator = anchor_generator or _default_anchor_generator(
            bounding_box_format
        )
        self.classes = classes
        self.backbone = _parse_backbone(backbone, include_rescaling, backbone_weights)

        self.label_encoder = label_encoder or cv_layers.ObjectDetectionLabelEncoder(
            bounding_box_format=bounding_box_format, anchor_generator=anchor_generator
        )
        self.prediction_decoder = (
            prediction_decoder
            or cv_layers.ObjectDetectionPredictionDecoder(
                bounding_box_format=bounding_box_format,
                anchor_generator=anchor_generator,
                classes=classes,
            )
        )

        # initialize trainable networks
        self.feature_pyramid = feature_pyramid or layers_lib.FeaturePyramid()
        prior_probability = tf.constant_initializer(-np.log((1 - 0.01) / 0.01))
        self.classification_head = layers_lib.PredictionHead(
            output_filters=9 * classes, bias_initializer=prior_probability
        )
        self.box_head = layers_lib.PredictionHead(
            output_filters=9 * 4, bias_initializer="zeros"
        )
        self._metrics_bounding_box_format = None

        # Construct should run in eager mode
        if any(
            self.prediction_decoder.box_variance.numpy()
            != self.label_encoder.box_variance.numpy()
        ):
            raise ValueError(
                "`prediction_decoder` and `label_encoder` must "
                "have matching `box_variance` arguments.  Did you customize the "
                "`box_variance` in either `prediction_decoder` or `label_encoder`? "
                "If so, please also customize the other.  Received: "
                f"`prediction_decoder.box_variance={prediction_decoder.box_variance}`, "
                f"`label_encoder.box_variance={label_encoder.box_variance}`."
            )

    def compile(self, metrics=None, loss=None, **kwargs):
        metrics = metrics or []
        super().compile(metrics=metrics, loss=loss, **kwargs)
        # if loss.classes != self.classes:
        #     raise ValueError("RetinaNet.classes != loss.classes, di")
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
            # predictions technically do not have a format, so loss accepts whatever
            # is output by the model.  This actually causes scaling issues if you use
            # a rel_ format, or a different format.
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

        # To minimize GPU transfers, we update metrics AFTER we take grads and apply
        # them.
        # TODO(lukewood): ensure this runs on TPU.
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


def _default_anchor_generator(bounding_box_format):
    strides = [2**i for i in range(3, 8)]
    scales = [2**x for x in [0, 1 / 3, 2 / 3]]
    sizes = [32.0, 64.0, 128.0, 256.0, 512.0]
    aspect_ratios = [0.5, 1.0, 2.0]
    return cv_layers.AnchorGenerator(
        bounding_box_format=bounding_box_format,
        anchor_sizes=sizes,
        aspect_ratios=aspect_ratios,
        scales=scales,
        strides=strides,
        clip_boxes=True,
    )
