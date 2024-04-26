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

import numpy as np

from keras_cv.src import bounding_box
from keras_cv.src import layers as cv_layers
from keras_cv.src import losses
from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.backend import keras
from keras_cv.src.backend import ops
from keras_cv.src.bounding_box.converters import _decode_deltas_to_boxes
from keras_cv.src.models.backbones.backbone_presets import backbone_presets
from keras_cv.src.models.backbones.backbone_presets import (
    backbone_presets_with_weights,
)
from keras_cv.src.models.object_detection.__internal__ import unpack_input
from keras_cv.src.models.object_detection.retinanet import FeaturePyramid
from keras_cv.src.models.object_detection.retinanet import PredictionHead
from keras_cv.src.models.object_detection.retinanet import RetinaNetLabelEncoder
from keras_cv.src.models.object_detection.retinanet.retinanet_presets import (
    retinanet_presets,
)
from keras_cv.src.models.task import Task
from keras_cv.src.utils.python_utils import classproperty
from keras_cv.src.utils.train import get_feature_extractor

BOX_VARIANCE = [0.1, 0.1, 0.2, 0.2]


@keras_cv_export(
    ["keras_cv.models.RetinaNet", "keras_cv.models.object_detection.RetinaNet"]
)
class RetinaNet(Task):
    """A Keras model implementing the RetinaNet meta-architecture.

    Implements the RetinaNet architecture for object detection. The constructor
    requires `num_classes`, `bounding_box_format`, and a backbone. Optionally,
    a custom label encoder, and prediction decoder may be provided.

    Example:
    ```python
    images = np.ones((1, 512, 512, 3))
    labels = {
        "boxes": tf.cast([
            [
                [0, 0, 100, 100],
                [100, 100, 200, 200],
                [300, 300, 100, 100],
            ]
        ], dtype=tf.float32),
        "classes": tf.cast([[1, 1, 1]], dtype=tf.float32),
    }
    model = keras_cv.models.RetinaNet(
        num_classes=20,
        bounding_box_format="xywh",
        backbone=keras_cv.models.ResNet50Backbone.from_preset(
            "resnet50_imagenet"
        )
    )

    # Evaluate model without box decoding and NMS
    model(images)

    # Prediction with box decoding and NMS
    model.predict(images)

    # Train model
    model.compile(
        classification_loss='focal',
        box_loss='smoothl1',
        optimizer=keras.optimizers.SGD(global_clipnorm=10.0),
        jit_compile=False,
    )
    model.fit(images, labels)
    ```

    Args:
        num_classes: the number of classes in your dataset excluding the
            background class. Classes should be represented by integers in the
            range [0, num_classes).
        bounding_box_format: The format of bounding boxes of input dataset.
            Refer
            [to the keras.io docs](https://keras.io/api/keras_cv/bounding_box/formats/)
            for more details on supported bounding box formats.
        backbone: `keras.Model`. If the default `feature_pyramid` is used,
            must implement the `pyramid_level_inputs` property with keys "P3", "P4",
            and "P5" and layer names as values. A somewhat sensible backbone
            to use in many cases is the:
            `keras_cv.models.ResNetBackbone.from_preset("resnet50_imagenet")`
        anchor_generator: (Optional) a `keras_cv.layers.AnchorGenerator`. If
            provided, the anchor generator will be passed to both the
            `label_encoder` and the `prediction_decoder`. Only to be used when
            both `label_encoder` and `prediction_decoder` are both `None`.
            Defaults to an anchor generator with the parameterization:
            `strides=[2**i for i in range(3, 8)]`,
            `scales=[2**x for x in [0, 1 / 3, 2 / 3]]`,
            `sizes=[32.0, 64.0, 128.0, 256.0, 512.0]`,
            and `aspect_ratios=[0.5, 1.0, 2.0]`.
        label_encoder: (Optional) a keras.Layer that accepts an image Tensor, a
            bounding box Tensor and a bounding box class Tensor to its `call()`
            method, and returns RetinaNet training targets. By default, a
            KerasCV standard `RetinaNetLabelEncoder` is created and used.
            Results of this object's `call()` method are passed to the `loss`
            object for `box_loss` and `classification_loss` the `y_true`
            argument.
        prediction_decoder: (Optional)  A `keras.layers.Layer` that is
            responsible for transforming RetinaNet predictions into usable
            bounding box Tensors. If not provided, a default is provided. The
            default `prediction_decoder` layer is a
            `keras_cv.layers.MultiClassNonMaxSuppression` layer, which uses
            a Non-Max Suppression for box pruning.
        feature_pyramid: (Optional) A `keras.layers.Layer` that produces
            a list of 4D feature maps (batch dimension included)
            when called on the pyramid-level outputs of the `backbone`.
            If not provided, the reference implementation from the paper will be used.
        classification_head: (Optional) A `keras.Layer` that performs
            classification of the bounding boxes. If not provided, a simple
            ConvNet with 3 layers will be used.
        box_head: (Optional) A `keras.Layer` that performs regression of the
            bounding boxes. If not provided, a simple ConvNet with 3 layers
            will be used.
    """  # noqa: E501

    def __init__(
        self,
        backbone,
        num_classes,
        bounding_box_format,
        anchor_generator=None,
        label_encoder=None,
        prediction_decoder=None,
        feature_pyramid=None,
        classification_head=None,
        box_head=None,
        **kwargs,
    ):
        if anchor_generator is not None and label_encoder is not None:
            raise ValueError(
                "`anchor_generator` is only to be provided when "
                "`label_encoder` is `None`. Received `anchor_generator="
                f"{anchor_generator}`, label_encoder={label_encoder}`. To "
                "customize the behavior of the anchor_generator inside of a "
                "custom `label_encoder` you should provide both to `RetinaNet`"
                "provide both to `RetinaNet`, and ensure that the "
                "`anchor_generator` provided to both is identical"
            )

        if label_encoder is None:
            anchor_generator = (
                anchor_generator
                or RetinaNet.default_anchor_generator(bounding_box_format)
            )

            label_encoder = RetinaNetLabelEncoder(
                bounding_box_format=bounding_box_format,
                anchor_generator=anchor_generator,
                box_variance=BOX_VARIANCE,
            )

        extractor_levels = ["P3", "P4", "P5"]
        extractor_layer_names = [
            backbone.pyramid_level_inputs[i] for i in extractor_levels
        ]
        feature_extractor = get_feature_extractor(
            backbone, extractor_layer_names, extractor_levels
        )
        feature_pyramid = feature_pyramid or FeaturePyramid()

        prior_probability = keras.initializers.Constant(
            -np.log((1 - 0.01) / 0.01)
        )
        classification_head = classification_head or PredictionHead(
            output_filters=9 * num_classes,
            bias_initializer=prior_probability,
        )
        box_head = box_head or PredictionHead(
            output_filters=9 * 4, bias_initializer=keras.initializers.Zeros()
        )

        # Begin construction of forward pass
        images = keras.layers.Input(
            feature_extractor.input_shape[1:], name="images"
        )

        backbone_outputs = feature_extractor(images)
        features = feature_pyramid(backbone_outputs)

        cls_pred = []
        box_pred = []
        for feature in features:
            box_pred.append(keras.layers.Reshape((-1, 4))(box_head(feature)))
            cls_pred.append(
                keras.layers.Reshape((-1, num_classes))(
                    classification_head(feature)
                )
            )

        cls_pred = keras.layers.Concatenate(axis=1, name="classification")(
            cls_pred
        )
        box_pred = keras.layers.Concatenate(axis=1, name="box")(box_pred)
        # box_pred is always in "center_yxhw" delta-encoded no matter what
        # format you pass in.

        inputs = {"images": images}
        outputs = {"box": box_pred, "classification": cls_pred}

        super().__init__(
            inputs=inputs,
            outputs=outputs,
            **kwargs,
        )
        self.label_encoder = label_encoder
        self.anchor_generator = label_encoder.anchor_generator
        self.bounding_box_format = bounding_box_format
        self.num_classes = num_classes
        self.backbone = backbone

        self.feature_extractor = feature_extractor
        self._prediction_decoder = (
            prediction_decoder
            or cv_layers.NonMaxSuppression(
                bounding_box_format=bounding_box_format,
                from_logits=True,
            )
        )

        self.feature_pyramid = feature_pyramid
        self.classification_head = classification_head
        self.box_head = box_head
        self.build(backbone.input_shape)

    def predict_step(self, *args):
        outputs = super().predict_step(*args)
        if type(outputs) is tuple:
            return self.decode_predictions(outputs[0], args[-1]), outputs[1]
        else:
            return self.decode_predictions(outputs, args[-1])

    @property
    def prediction_decoder(self):
        return self._prediction_decoder

    @prediction_decoder.setter
    def prediction_decoder(self, prediction_decoder):
        if prediction_decoder.bounding_box_format != self.bounding_box_format:
            raise ValueError(
                "Expected `prediction_decoder` and RetinaNet to "
                "use the same `bounding_box_format`, but got "
                "`prediction_decoder.bounding_box_format="
                f"{prediction_decoder.bounding_box_format}`, and "
                "`self.bounding_box_format="
                f"{self.bounding_box_format}`."
            )
        self._prediction_decoder = prediction_decoder
        self.make_predict_function(force=True)
        self.make_train_function(force=True)
        self.make_test_function(force=True)

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

    def decode_predictions(self, predictions, images):
        box_pred, cls_pred = predictions["box"], predictions["classification"]
        # box_pred is on "center_yxhw" format, convert to target format.
        image_shape = tuple(images[0].shape)
        anchors = self.anchor_generator(image_shape=image_shape)
        anchors = ops.concatenate([a for a in anchors.values()], axis=0)

        box_pred = _decode_deltas_to_boxes(
            anchors=anchors,
            boxes_delta=box_pred,
            anchor_format=self.anchor_generator.bounding_box_format,
            box_format=self.bounding_box_format,
            variance=BOX_VARIANCE,
            image_shape=image_shape,
        )
        # box_pred is now in "self.bounding_box_format" format
        box_pred = bounding_box.convert_format(
            box_pred,
            source=self.bounding_box_format,
            target=self.prediction_decoder.bounding_box_format,
            image_shape=image_shape,
        )
        y_pred = self.prediction_decoder(
            box_pred, cls_pred, image_shape=image_shape
        )
        y_pred["boxes"] = bounding_box.convert_format(
            y_pred["boxes"],
            source=self.prediction_decoder.bounding_box_format,
            target=self.bounding_box_format,
            image_shape=image_shape,
        )
        return y_pred

    def compile(
        self,
        box_loss=None,
        classification_loss=None,
        loss=None,
        metrics=None,
        **kwargs,
    ):
        """compiles the RetinaNet.

        compile() mirrors the standard Keras compile() method, but has a few key
        distinctions. Primarily, all metrics must support bounding boxes, and
        two losses must be provided: `box_loss` and `classification_loss`.

        Args:
            box_loss: a Keras loss to use for box offset regression.
                Preconfigured losses are provided when the string "huber" or
                "smoothl1" are passed.
            classification_loss: a Keras loss to use for box classification.
                A preconfigured `FocalLoss` is provided when the string "focal"
                is passed.
            weight_decay: a float for variable weight decay.
            metrics: KerasCV object detection metrics that accept decoded
                bounding boxes as their inputs. Examples of this metric type
                are `keras_cv.metrics.BoxRecall()` and
                `keras_cv.metrics.BoxMeanAveragePrecision()`. When `metrics` are
                included in the call to `compile()`, the RetinaNet will perform
                non-max suppression decoding during the forward pass. By
                default, the RetinaNet uses a
                `keras_cv.layers.MultiClassNonMaxSuppression()` layer to
                perform decoding. This behavior can be customized by passing in
                a `prediction_decoder` to the constructor or by modifying the
                `prediction_decoder` attribute on the model. It should be noted
                that the default non-max suppression operation does not have
                TPU support, and thus when training on TPU metrics must be
                evaluated in a `keras.utils.SidecarEvaluator` or a
                `keras.callbacks.Callback`.
            kwargs: most other `keras.Model.compile()` arguments are supported
                and propagated to the `keras.Model` class.
        """
        if loss is not None:
            raise ValueError(
                "`RetinaNet` does not accept a `loss` to `compile()`. "
                "Instead, please pass `box_loss` and `classification_loss`. "
                "`loss` will be ignored during training."
            )
        box_loss = _parse_box_loss(box_loss)
        classification_loss = _parse_classification_loss(classification_loss)

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
                    "`RetinaNet.compile()`. Got "
                    "`box_loss.bounding_box_format="
                    f"{box_loss.bounding_box_format}`, want "
                    "`box_loss.bounding_box_format="
                    f"{self.bounding_box_format}`"
                )

        self.box_loss = box_loss
        self.classification_loss = classification_loss
        losses = {
            "box": self.box_loss,
            "classification": self.classification_loss,
        }
        self._has_user_metrics = metrics is not None and len(metrics) != 0
        self._user_metrics = metrics
        super().compile(loss=losses, **kwargs)

    def compute_loss(self, x, y, y_pred, sample_weight, **kwargs):
        y_for_label_encoder = bounding_box.convert_format(
            y,
            source=self.bounding_box_format,
            target=self.label_encoder.bounding_box_format,
            images=x,
        )

        boxes, classes = self.label_encoder(x, y_for_label_encoder)

        box_pred = y_pred["box"]
        cls_pred = y_pred["classification"]

        if boxes.shape[-1] != 4:
            raise ValueError(
                "boxes should have shape (None, None, 4). Got "
                f"boxes.shape={tuple(boxes.shape)}"
            )

        if box_pred.shape[-1] != 4:
            raise ValueError(
                "box_pred should have shape (None, None, 4). Got "
                f"box_pred.shape={tuple(box_pred.shape)}. Does your model's "
                "`num_classes` parameter match your losses `num_classes` "
                "parameter?"
            )
        if cls_pred.shape[-1] != self.num_classes:
            raise ValueError(
                "cls_pred should have shape (None, None, 4). Got "
                f"cls_pred.shape={tuple(cls_pred.shape)}. Does your model's "
                "`num_classes` parameter match your losses `num_classes` "
                "parameter?"
            )

        cls_labels = ops.one_hot(
            ops.cast(classes, "int32"), self.num_classes, dtype="float32"
        )
        positive_mask = ops.cast(ops.greater(classes, -1.0), dtype="float32")
        normalizer = ops.sum(positive_mask)
        cls_weights = ops.cast(ops.not_equal(classes, -2.0), dtype="float32")
        cls_weights /= normalizer
        box_weights = positive_mask / normalizer
        y_true = {
            "box": boxes,
            "classification": cls_labels,
        }
        sample_weights = {
            "box": box_weights,
            "classification": cls_weights,
        }
        zero_weight = {
            "box": ops.zeros_like(box_weights),
            "classification": ops.zeros_like(cls_weights),
        }

        sample_weights = ops.cond(
            normalizer == 0,
            lambda: zero_weight,
            lambda: sample_weights,
        )
        return super().compute_loss(
            x=x, y=y_true, y_pred=y_pred, sample_weight=sample_weights
        )

    def train_step(self, *args):
        data = args[-1]
        args = args[:-1]
        x, y = unpack_input(data)
        return super().train_step(*args, (x, y))

    def test_step(self, *args):
        data = args[-1]
        args = args[:-1]
        x, y = unpack_input(data)
        return super().test_step(*args, (x, y))

    def compute_metrics(self, x, y, y_pred, sample_weight):
        metrics = {}
        metrics.update(super().compute_metrics(x, {}, {}, sample_weight={}))

        if not self._has_user_metrics:
            return metrics

        y_pred = self.decode_predictions(y_pred, x)

        for metric in self._user_metrics:
            metric.update_state(y, y_pred, sample_weight=sample_weight)

        for metric in self._user_metrics:
            result = metric.result()
            if isinstance(result, dict):
                metrics.update(result)
            else:
                metrics[metric.name] = result
        return metrics

    def get_config(self):
        return {
            "num_classes": self.num_classes,
            "bounding_box_format": self.bounding_box_format,
            "backbone": keras.saving.serialize_keras_object(self.backbone),
            "label_encoder": keras.saving.serialize_keras_object(
                self.label_encoder
            ),
            "prediction_decoder": self._prediction_decoder,
            "classification_head": keras.saving.serialize_keras_object(
                self.classification_head
            ),
            "box_head": keras.saving.serialize_keras_object(self.box_head),
        }

    @classmethod
    def from_config(cls, config):
        if "box_head" in config and isinstance(config["box_head"], dict):
            config["box_head"] = keras.layers.deserialize(config["box_head"])
        if "classification_head" in config and isinstance(
            config["classification_head"], dict
        ):
            config["classification_head"] = keras.layers.deserialize(
                config["classification_head"]
            )
        if "label_encoder" in config and isinstance(
            config["label_encoder"], dict
        ):
            config["label_encoder"] = keras.layers.deserialize(
                config["label_encoder"]
            )
        if "prediction_decoder" in config and isinstance(
            config["prediction_decoder"], dict
        ):
            config["prediction_decoder"] = keras.layers.deserialize(
                config["prediction_decoder"]
            )
        return super().from_config(config)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return copy.deepcopy({**backbone_presets, **retinanet_presets})

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return copy.deepcopy(
            {**backbone_presets_with_weights, **retinanet_presets}
        )

    @classproperty
    def backbone_presets(cls):
        """Dictionary of preset names and configurations of compatible
        backbones."""
        return copy.deepcopy(backbone_presets)


def _parse_box_loss(loss):
    if not isinstance(loss, str):
        # support arbitrary callables
        return loss

    # case insensitive comparison
    if loss.lower() == "smoothl1":
        return losses.SmoothL1Loss(l1_cutoff=1.0, reduction="sum")
    if loss.lower() == "huber":
        return keras.losses.Huber(reduction="sum")

    raise ValueError(
        "Expected `box_loss` to be either a Keras Loss, "
        f"callable, or the string 'SmoothL1'. Got loss={loss}."
    )


def _parse_classification_loss(loss):
    if not isinstance(loss, str):
        # support arbitrary callables
        return loss

    # case insensitive comparison
    if loss.lower() == "focal":
        return losses.FocalLoss(from_logits=True, reduction="sum")

    raise ValueError(
        "Expected `classification_loss` to be either a Keras Loss, "
        f"callable, or the string 'Focal'. Got loss={loss}."
    )
