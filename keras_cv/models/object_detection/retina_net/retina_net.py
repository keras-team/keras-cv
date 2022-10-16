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
from keras_cv.models.object_detection.object_detection_base_model import (
    ObjectDetectionBaseModel,
)
from keras_cv.models.object_detection.retina_net.__internal__ import (
    layers as layers_lib,
)


# TODO(lukewood): update docstring to include documentation on creating a custom label
# decoder/etc.
# TODO(lukewood): link to keras.io guide on creating custom backbone and FPN.
class RetinaNet(ObjectDetectionBaseModel):
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
            `prediction_decoder` are both `None`.  Defaults to an anchor generator with
            the parameterization: `strides=[2**i for i in range(3, 8)]`,
            `scales=[2**x for x in [0, 1 / 3, 2 / 3]]`,
            `sizes=[32.0, 64.0, 128.0, 256.0, 512.0]`,
            and `aspect_ratios=[0.5, 1.0, 2.0]`.
        label_encoder: (Optional) a keras.Layer that accepts an image Tensor and a
            bounding box Tensor to its `call()` method, and returns RetinaNet training
            targets.  By default, a KerasCV standard LabelEncoder is created and used.
            Results of this `call()` method are passed to the `loss` object passed into
            `compile()` as the `y_true` argument.
        prediction_decoder: (Optional)  A `keras.layer` that is responsible for
            transforming RetinaNet predictions into usable bounding box Tensors.  If
            not provided, a default is provided.  The default `prediction_decoder`
            layer uses a `NonMaxSuppression` operation for box pruning.
        feature_pyramid: (Optional) A `keras.Model` representing a feature pyramid
            network (FPN).  The feature pyramid network is called on the outputs of the
            `backbone`.  The KerasCV default backbones return three outputs in a list,
            but custom backbones may be written and used with custom feature pyramid
            networks.  If not provided, a default feature pyramid neetwork is produced
            by the library.  The default feature pyramid network is compatible with all
            standard keras_cv backbones.
        classification_head: (Optional) A `keras.Layer` that performs classification of
            the bounding boxes.  If not provided, a simple ConvNet with 1 layer will be
            used.
        box_head: (Optional) A `keras.Layer` that performs regression of
            the bounding boxes.  If not provided, a simple ConvNet with 1 layer will be
            used.
        evaluate_train_time_metrics: (Optional) whether or not to evaluate metrics
            passed in `compile()` inside of the `train_step()`.  This is NOT
            recommended, as it dramatically reduces performance due to the synchronous
            label decoding and COCO metric evaluation.  For example, on a single GPU on
            the PascalVOC dataset epoch time goes from 3 minutes to 30 minutes with this
            set to `True`. Defaults to `False`.
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
        classification_head=None,
        box_head=None,
        evaluate_train_time_metrics=False,
        name="RetinaNet",
        **kwargs,
    ):
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
        anchor_generator = anchor_generator or RetinaNet.default_anchor_generator(
            bounding_box_format
        )
        label_encoder = label_encoder or cv_layers.RetinaNetLabelEncoder(
            bounding_box_format=bounding_box_format, anchor_generator=anchor_generator
        )
        super().__init__(
            bounding_box_format=bounding_box_format,
            label_encoder=label_encoder,
            name=name,
            **kwargs,
        )
        self.evaluate_train_time_metrics = evaluate_train_time_metrics
        self.label_encoder = label_encoder
        self.anchor_generator = anchor_generator
        if bounding_box_format.lower() != "xywh":
            raise ValueError(
                "`keras_cv.models.RetinaNet` only supports the 'xywh' "
                "`bounding_box_format`.  In future releases, more formats will be "
                "supported.  For now, please pass `bounding_box_format='xywh'`. "
                f"Received `bounding_box_format={bounding_box_format}`"
            )

        self.bounding_box_format = bounding_box_format
        self.classes = classes
        self.backbone = _parse_backbone(backbone, include_rescaling, backbone_weights)

        self._prediction_decoder = prediction_decoder or cv_layers.NmsPredictionDecoder(
            bounding_box_format=bounding_box_format,
            anchor_generator=anchor_generator,
            classes=classes,
        )

        # initialize trainable networks
        self.feature_pyramid = feature_pyramid or layers_lib.FeaturePyramid()
        prior_probability = tf.constant_initializer(-np.log((1 - 0.01) / 0.01))

        self.classification_head = classification_head or layers_lib.PredictionHead(
            output_filters=9 * classes, bias_initializer=prior_probability
        )

        self.box_head = box_head or layers_lib.PredictionHead(
            output_filters=9 * 4, bias_initializer="zeros"
        )

        self._metrics_bounding_box_format = None
        self.loss_metric = tf.keras.metrics.Mean(name="loss")
        self.classification_loss_metric = tf.keras.metrics.Mean(
            name="classification_loss"
        )
        self.box_loss_metric = tf.keras.metrics.Mean(name="box_loss")
        self.regularization_loss_metric = tf.keras.metrics.Mean(
            name="regularization_loss"
        )
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

    @property
    def prediction_decoder(self):
        return self._prediction_decoder

    @prediction_decoder.setter
    def prediction_decoder(self, prediction_decoder):
        self._prediction_decoder = prediction_decoder
        self.make_predict_function(force=True)
        self.make_test_function(force=True)
        self.make_train_function(force=True)

    @staticmethod
    def default_anchor_generator(bounding_box_format):
        strides = [2**i for i in range(3, 8)]
        scales = [2**x for x in [0, 1 / 3, 2 / 3]]
        sizes = [32.0, 64.0, 128.0, 256.0, 512.0]
        aspect_ratios = [0.5, 1.0, 2.0]
        return cv_layers.AnchorGenerator(
            bounding_box_format=bounding_box_format,
            sizes=sizes,
            aspect_ratios=aspect_ratios,
            scales=scales,
            strides=strides,
            clip_boxes=True,
        )

    @property
    def metrics(self):
        return super().metrics + self.train_metrics

    @property
    def train_metrics(self):
        return [
            self.loss_metric,
            self.classification_loss_metric,
            self.regularization_loss_metric,
            self.box_loss_metric,
        ]

    def call(self, x, training=False):
        backbone_outputs = self.backbone(x, training=training)
        features = self.feature_pyramid(backbone_outputs, training=training)

        N = tf.shape(x)[0]
        cls_outputs = []
        box_outputs = []
        for feature in features:
            box_outputs.append(
                tf.reshape(self.box_head(feature, training=training), [N, -1, 4])
            )
            cls_outputs.append(
                tf.reshape(
                    self.classification_head(feature, training=training),
                    [N, -1, self.classes],
                )
            )

        cls_outputs = tf.concat(cls_outputs, axis=1)
        box_outputs = tf.concat(box_outputs, axis=1)
        return tf.concat([box_outputs, cls_outputs], axis=-1)

    def decode_training_predictions(self, x, train_predictions):
        # no-op if default decoder is used.
        pred_for_inference = bounding_box.convert_format(
            train_predictions,
            source=self.bounding_box_format,
            target=self.prediction_decoder.bounding_box_format,
            images=x,
        )
        pred_for_inference = self.prediction_decoder(x, pred_for_inference)
        return bounding_box.convert_format(
            pred_for_inference,
            source=self.prediction_decoder.bounding_box_format,
            target=self.bounding_box_format,
            images=x,
        )

    def compile(
        self, box_loss=None, classification_loss=None, loss=None, metrics=None, **kwargs
    ):
        super().compile(metrics=metrics, **kwargs)
        if loss is not None:
            raise ValueError(
                "`RetinaNet` does not accept a `loss` to `compile()`. "
                "Instead, please pass `box_loss` and `classification_loss`. "
                "`loss` will be ignored during training."
            )
        self.box_loss = box_loss
        self.classification_loss = classification_loss
        metrics = metrics or []

        if hasattr(classification_loss, "from_logits"):
            if not classification_loss.from_logits:
                raise ValueError(
                    "RetinaNet.compile() expects `from_logits` to be True for "
                    "`classification_loss`. Got "
                    "`classification_loss.from_logits="
                    f"{classification_loss.from_logits}`"
                )
        if hasattr(box_loss, "bounding_box_format"):
            if box_loss.bounding_box_format != self.bounding_box_format:
                raise ValueError(
                    "Wrong `bounding_box_format` passed to `box_loss` in "
                    "`RetinaNet.compile()`. "
                    f"Got `box_loss.bounding_box_format={box_loss.bounding_box_format}`, "
                    f"want `box_loss.bounding_box_format={self.bounding_box_format}`"
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
        if metrics and any_wrong_format:
            raise ValueError(
                "All metrics passed to RetinaNet.compile() must have "
                "the same `bounding_box_format` attribute.  For example, if one metric "
                "uses 'xyxy', all other metrics must use 'xyxy'.  Received "
                f"metrics={metrics}."
            )

    def compute_losses(self, y_true, y_pred):

        if y_true.shape[-1] != 5:
            raise ValueError(
                "y_true should have shape (None, None, 5).  Got "
                f"y_true.shape={tuple(y_true.shape)}"
            )

        if y_pred.shape[-1] != self.classes + 4:
            raise ValueError(
                "y_pred should have shape (None, None, classes + 4). "
                f"Got y_pred.shape={tuple(y_pred.shape)}.  Does your model's `classes` "
                "parameter match your losses `classes` parameter?"
            )

        box_labels = y_true[:, :, :4]
        box_predictions = y_pred[:, :, :4]

        cls_labels = tf.one_hot(
            tf.cast(y_true[:, :, 4], dtype=tf.int32),
            depth=self.classes,
            dtype=tf.float32,
        )
        cls_predictions = y_pred[:, :, 4:]

        positive_mask = tf.cast(tf.greater(y_true[:, :, 4], -1.0), dtype=tf.float32)
        ignore_mask = tf.cast(tf.equal(y_true[:, :, 4], -2.0), dtype=tf.float32)

        classification_loss = self.classification_loss(cls_labels, cls_predictions)
        box_loss = self.box_loss(box_labels, box_predictions)
        if len(classification_loss.shape) != 2:
            raise ValueError(
                "RetinaNet expects the output shape of `classification_loss` to be "
                "`(batch_size, num_anchor_boxes)`.  Expected "
                f"classification_loss(predictions)={box_predictions.shape[:2]}, got "
                f"classification_loss(predictions)={classification_loss.shape}. "
                "Try passing `reduction='none'` to your classification_loss's "
                "constructor."
            )
        if len(box_loss.shape) != 2:
            raise ValueError(
                "RetinaNet expects the output shape of `box_loss` to be "
                "`(batch_size, num_anchor_boxes)`.  Expected "
                f"box_loss(predictions)={box_predictions.shape[:2]}, got "
                f"box_loss(predictions)={box_loss.shape}. "
                "Try passing `reduction='none'` to your box_loss's "
                "constructor."
            )
        classification_loss = tf.where(
            tf.equal(ignore_mask, 1.0), 0.0, classification_loss
        )

        box_loss = tf.where(tf.equal(positive_mask, 1.0), box_loss, 0.0)

        normalizer = tf.reduce_sum(positive_mask, axis=-1)
        classification_loss = tf.math.divide_no_nan(
            tf.reduce_sum(classification_loss, axis=-1), normalizer
        )
        box_loss = tf.math.divide_no_nan(tf.reduce_sum(box_loss, axis=-1), normalizer)

        classification_loss = tf.reduce_sum(classification_loss, axis=-1)
        box_loss = tf.reduce_sum(box_loss, axis=-1)

        # ensure losses are scalars
        # only runs at trace time
        if tuple(classification_loss.shape) != ():
            raise ValueError(
                "Expected `classification_loss` to be a scalar by the "
                "end of `compute_losses()`, instead got "
                f"`classification_loss.shape={classification_loss.shape}`"
            )
        if tuple(box_loss.shape) != ():
            raise ValueError(
                "Expected `box_loss` to be a scalar by the "
                "end of `compute_losses()`, instead got "
                f"`box_loss.shape={box_loss.shape}`"
            )

        return classification_loss, box_loss

    def _backward(self, y_true, y_pred):
        classification_loss, box_loss = self.compute_losses(
            y_true,
            y_pred,
        )
        regularization_loss = 0.0
        for loss in self.losses:
            regularization_loss += tf.nn.scale_regularization_loss(loss)
        loss = classification_loss + box_loss + regularization_loss

        self.classification_loss_metric.update_state(classification_loss)
        self.box_loss_metric.update_state(box_loss)
        self.regularization_loss_metric.update_state(regularization_loss)
        self.loss_metric.update_state(loss)
        return loss

    def train_step(self, data):
        x, y = data
        y_for_metrics, y_training_target = y

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self._backward(y_training_target, y_pred)
        # Training specific code
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Early exit for no train time metrics
        if not self.evaluate_train_time_metrics:
            # To minimize GPU transfers, we update metrics AFTER we take grads and apply
            # them.
            return {m.name: m.result() for m in self.train_metrics}

        predictions = self.decode_training_predictions(x, y_pred)
        self._update_metrics(y_for_metrics, predictions)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        y_for_metrics, y_training_target = y
        y_pred = self(x, training=False)
        _ = self._backward(y_training_target, y_pred)

        predictions = self.decode_training_predictions(x, y_pred)
        self._update_metrics(y_for_metrics, predictions)
        return {m.name: m.result() for m in self.metrics}

    def predict_step(self, x):
        predictions = super().predict_step(x)
        return self.decode_training_predictions(x, predictions)

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

    c3_output, c4_output, c5_output = [
        backbone.get_layer(layer_name).output
        for layer_name in ["conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]
    ]
    return keras.Model(inputs=inputs, outputs=[c3_output, c4_output, c5_output])
