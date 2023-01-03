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

import keras_cv
from keras_cv import bounding_box
from keras_cv import layers as cv_layers
from keras_cv.bounding_box.converters import _decode_deltas_to_boxes
from keras_cv.models.object_detection import predict_utils
from keras_cv.models.object_detection.retina_net.__internal__ import (
    layers as layers_lib,
)

BOX_VARIANCE = [0.1, 0.1, 0.2, 0.2]


# TODO(lukewood): update docstring to include documentation on creating a custom label
# decoder/etc.
# TODO(lukewood): link to keras.io guide on creating custom backbone and FPN.
@keras.utils.register_keras_serializable(package="keras_cv")
class RetinaNet(tf.keras.Model):
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
        backbone=backbone,
    )
    ```

    Args:
        classes: the number of classes in your dataset excluding the background
            class.  Classes should be represented by integers in the range
            [0, classes).
        bounding_box_format: The format of bounding boxes of input dataset. Refer
            [to the keras.io docs](https://keras.io/api/keras_cv/bounding_box/formats/)
            for more details on supported bounding box formats.
        backbone: backbone: an optional `tf.keras.Model` custom backbone model. Defaults
            to a keras_cv.models.ResNet50 with include_rescaling=True
        anchor_generator: (Optional) a `keras_cv.layers.AnchorGenerator`.  If provided,
            the anchor generator will be passed to both the `label_encoder` and the
            `prediction_decoder`.  Only to be used when both `label_encoder` and
            `prediction_decoder` are both `None`.  Defaults to an anchor generator with
            the parameterization: `strides=[2**i for i in range(3, 8)]`,
            `scales=[2**x for x in [0, 1 / 3, 2 / 3]]`,
            `sizes=[32.0, 64.0, 128.0, 256.0, 512.0]`,
            and `aspect_ratios=[0.5, 1.0, 2.0]`.
        label_encoder: (Optional) a keras.Layer that accepts an image Tensor, a
            bounding box Tensor and a bounding box class Tensor to its `call()` method,
            and returns RetinaNet training targets.  By default, a KerasCV standard LabelEncoder is created and used.
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
    """

    def __init__(
        self,
        classes,
        bounding_box_format,
        backbone=None,
        anchor_generator=None,
        label_encoder=None,
        prediction_decoder=None,
        feature_pyramid=None,
        classification_head=None,
        box_head=None,
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
            bounding_box_format=bounding_box_format,
            anchor_generator=anchor_generator,
            box_variance=BOX_VARIANCE,
        )
        super().__init__(
            name=name,
            **kwargs,
        )
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
        self.backbone = (
            backbone
            or keras_cv.models.ResNet50(
                include_top=False, include_rescaling=True
            ).as_backbone()
        )

        self._prediction_decoder = prediction_decoder or cv_layers.NmsDecoder(
            bounding_box_format=bounding_box_format,
            from_logits=True,
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

    def make_predict_function(self, force=False):
        return predict_utils.make_predict_function(self, force=force)

    @property
    def prediction_decoder(self):
        return self._prediction_decoder

    @prediction_decoder.setter
    def prediction_decoder(self, prediction_decoder):
        self._prediction_decoder = prediction_decoder
        self.make_predict_function(force=True)

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

    def _forward(self, images, training=None):
        backbone_outputs = self.backbone(images, training=training)
        features = self.feature_pyramid(backbone_outputs, training=training)

        N = tf.shape(images)[0]
        cls_pred = []
        box_pred = []
        for feature in features:
            box_pred.append(
                tf.reshape(self.box_head(feature, training=training), [N, -1, 4])
            )
            cls_pred.append(
                tf.reshape(
                    self.classification_head(feature, training=training),
                    [N, -1, self.classes],
                )
            )

        cls_pred = tf.concat(cls_pred, axis=1)
        box_pred = tf.concat(box_pred, axis=1)

        return box_pred, cls_pred

    def call(self, images, training=None):
        box_pred, cls_pred = self._forward(images, training=training)
        if not training:
            # box_pred is on "center_yxhw" format, convert to target format.
            anchors = self.anchor_generator(images[0])
            anchors = tf.concat(tf.nest.flatten(anchors), axis=0)
            box_pred = _decode_deltas_to_boxes(
                anchors=anchors,
                boxes_delta=box_pred,
                anchor_format=self.anchor_generator.bounding_box_format,
                box_format=self.bounding_box_format,
                variance=BOX_VARIANCE,
            )
        return box_pred, cls_pred

    def decode_predictions(self, predictions, images):
        # no-op if default decoder is used.
        box_pred, cls_pred = predictions
        box_pred = bounding_box.convert_format(
            box_pred,
            source=self.bounding_box_format,
            target=self.prediction_decoder.bounding_box_format,
            images=images,
        )
        y_pred = self.prediction_decoder(box_pred, cls_pred)
        box_pred = bounding_box.convert_format(
            y_pred["boxes"],
            source=self.prediction_decoder.bounding_box_format,
            target=self.bounding_box_format,
            images=images,
        )
        y_pred["boxes"] = box_pred
        return y_pred

    def compile(
        self,
        box_loss=None,
        classification_loss=None,
        weight_decay=0.0001,
        loss=None,
        **kwargs,
    ):
        """compiles the RetinaNet.

        compile() mirrors the standard Keras compile() method, but has a few key
        distinctions.  Primarily, all metrics must support bounding boxes, and
        two losses must be provided: `box_loss` and `classification_loss`.

        Args:
            box_loss: a Keras loss to use for box offset regression.  Preconfigured
                losses are provided when the string "huber" or "smoothl1" are passed.
            classification_loss: a Keras loss to use for box classification.
                A preconfigured `FocalLoss` is provided when the string "focal" is
                passed.
            weight_decay: a float for variable weight decay.
            kwargs: most other `keras.Model.compile()` arguments are supported and
                propagated to the `keras.Model` class.
        """
        if "metrics" in kwargs.keys():
            raise ValueError("currently metrics support is not supported intentionally")
        super().compile(**kwargs)
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
                    "`RetinaNet.compile()`. "
                    f"Got `box_loss.bounding_box_format={box_loss.bounding_box_format}`, "
                    f"want `box_loss.bounding_box_format={self.bounding_box_format}`"
                )

        self.box_loss = box_loss
        self.classification_loss = classification_loss
        self.weight_decay = weight_decay
        losses = {
            "box": self.box_loss,
            "cls": self.classification_loss,
        }
        super().compile(loss=losses, **kwargs)

    def compute_loss(self, images, gt_boxes, gt_classes, training):
        box_pred, cls_pred = self._forward(images, training=training)
        if gt_boxes.shape[-1] != 4:
            raise ValueError(
                "gt_boxes should have shape (None, None, 4).  Got "
                f"gt_boxes.shape={tuple(gt_boxes.shape)}"
            )

        if box_pred.shape[-1] != 4:
            raise ValueError(
                "box_pred should have shape (None, None, 4). "
                f"Got box_pred.shape={tuple(box_pred.shape)}.  Does your model's `classes` "
                "parameter match your losses `classes` parameter?"
            )
        if cls_pred.shape[-1] != self.classes:
            raise ValueError(
                "cls_pred should have shape (None, None, 4). "
                f"Got cls_pred.shape={tuple(cls_pred.shape)}.  Does your model's `classes` "
                "parameter match your losses `classes` parameter?"
            )

        cls_labels = tf.one_hot(
            tf.cast(gt_classes, dtype=tf.int32),
            depth=self.classes,
            dtype=tf.float32,
        )

        positive_mask = tf.cast(tf.greater(gt_classes, -1.0), dtype=tf.float32)
        normalizer = tf.reduce_sum(positive_mask)
        cls_weights = tf.cast(tf.math.not_equal(gt_classes, -2.0), dtype=tf.float32)
        cls_weights /= normalizer
        box_weights = positive_mask / normalizer
        y_true = {
            "box": gt_boxes,
            "cls": cls_labels,
        }
        y_pred = {
            "box": box_pred,
            "cls": cls_pred,
        }
        sample_weights = {
            "box": box_weights,
            "cls": cls_weights,
        }
        return super().compute_loss(
            x=images, y=y_true, y_pred=y_pred, sample_weight=sample_weights
        )

    def train_step(self, data):
        x, y = data
        gt_boxes = y["boxes"]
        gt_classes = y["classes"]
        gt_boxes = bounding_box.convert_format(
            gt_boxes,
            source=self.bounding_box_format,
            target=self.label_encoder.bounding_box_format,
            images=x,
        )
        gt_boxes, gt_classes = self.label_encoder(x, gt_boxes, gt_classes)
        gt_boxes = bounding_box.convert_format(
            gt_boxes,
            source=self.label_encoder.bounding_box_format,
            target=self.bounding_box_format,
            images=x,
        )

        with tf.GradientTape() as tape:
            total_loss = self.compute_loss(x, gt_boxes, gt_classes, training=True)

            reg_losses = []
            if self.weight_decay:
                for var in self.trainable_variables:
                    if "bn" not in var.name:
                        reg_losses.append(self.weight_decay * tf.nn.l2_loss(var))
                l2_loss = tf.math.add_n(reg_losses)
            total_loss += l2_loss
        # Training specific code
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return self.compute_metrics(x, {}, {}, sample_weight={})

    def test_step(self, data):
        x, y = data
        gt_boxes = y["boxes"]
        gt_classes = y["classes"]
        gt_boxes = bounding_box.convert_format(
            gt_boxes,
            source=self.bounding_box_format,
            target=self.label_encoder.bounding_box_format,
            images=x,
        )
        gt_boxes, gt_classes = self.label_encoder(x, gt_boxes, gt_classes)
        gt_boxes = bounding_box.convert_format(
            gt_boxes,
            source=self.label_encoder.bounding_box_format,
            target=self.bounding_box_format,
            images=x,
        )
        _ = self.compute_loss(x, gt_boxes, gt_classes, training=False)

        return self.compute_metrics(x, {}, {}, sample_weight={})

    def get_config(self):
        return {
            "classes": self.classes,
            "bounding_box_format": self.bounding_box_format,
            "backbone": self.backbone,
            "anchor_generator": self.anchor_generator,
            "label_encoder": self.label_encoder,
            "prediction_decoder": self._prediction_decoder,
            "feature_pyramid": self.feature_pyramid,
            "classification_head": self.classification_head,
            "box_head": self.box_head,
        }


def _parse_box_loss(loss):
    if not isinstance(loss, str):
        # support arbitrary callables
        return loss

    # case insensitive comparison
    if loss.lower() == "smoothl1":
        return keras_cv.losses.SmoothL1Loss(
            l1_cutoff=1.0, reduction=tf.keras.losses.Reduction.SUM
        )
    if loss.lower() == "huber":
        return keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

    raise ValueError(
        "Expected `box_loss` to be either a Keras Loss, "
        f"callable, or the string 'SmoothL1'.  Got loss={loss}."
    )


def _parse_classification_loss(loss):
    if not isinstance(loss, str):
        # support arbitrary callables
        return loss

    # case insensitive comparison
    if loss.lower() == "focal":
        return keras_cv.losses.FocalLoss(
            from_logits=True, reduction=tf.keras.losses.Reduction.SUM
        )

    raise ValueError(
        "Expected `classification_loss` to be either a Keras Loss, "
        f"callable, or the string 'Focal'.  Got loss={loss}."
    )
