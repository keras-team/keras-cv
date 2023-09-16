# Copyright 2023 The KerasCV Authors
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

import tree
from absl import logging

from keras_cv import bounding_box
from keras_cv import layers as cv_layers
from keras_cv import models
from keras_cv.api_export import keras_cv_export
from keras_cv.backend import keras
from keras_cv.backend import ops
from keras_cv.bounding_box.converters import _decode_deltas_to_boxes

# from keras_cv.models.backbones.backbone_presets import backbone_presets
# from keras_cv.models.backbones.backbone_presets import (
#     backbone_presets_with_weights
# )
from keras_cv.bounding_box.utils import _clip_boxes
from keras_cv.layers.object_detection.anchor_generator import AnchorGenerator
from keras_cv.layers.object_detection.box_matcher import BoxMatcher
from keras_cv.layers.object_detection.roi_align import _ROIAligner
from keras_cv.layers.object_detection.roi_generator import ROIGenerator
from keras_cv.layers.object_detection.roi_sampler import _ROISampler
from keras_cv.layers.object_detection.rpn_label_encoder import _RpnLabelEncoder
from keras_cv.models.backbones.backbone_presets import backbone_presets
from keras_cv.models.backbones.backbone_presets import (
    backbone_presets_with_weights,
)
from keras_cv.models.object_detection.__internal__ import unpack_input
from keras_cv.models.object_detection.faster_rcnn import FeaturePyramid
from keras_cv.models.object_detection.faster_rcnn import RCNNHead
from keras_cv.models.object_detection.faster_rcnn import RPNHead

# from keras_cv.models.object_detection.faster_rcnn.faster_rcnn_presets import faster_rcnn_presets
from keras_cv.models.task import Task
from keras_cv.utils.python_utils import classproperty
from keras_cv.utils.train import get_feature_extractor

BOX_VARIANCE = [0.1, 0.1, 0.2, 0.2]


# TODO(tanzheny): add more configurations
@keras_cv_export("keras_cv.models.FasterRCNN")
class FasterRCNN(Task):
    """A Keras model implementing the FasterRCNN architecture.

    Implements the FasterRCNN architecture for object detection. The constructor
    requires `backbone`, `num_classes`, and a `bounding_box_format`.

    References:
        - [FasterRCNN](https://arxiv.org/abs/1506.01497)

    Args:
        backbone: `keras.Model`. Must implement the
            `pyramid_level_inputs` property with keys "P2", "P3", "P4", and "P5"
            and layer names as values.
        num_classes: the number of classes in your dataset excluding the
            background class. classes should be represented by integers in the
            range [0, num_classes).
        bounding_box_format: The format of bounding boxes of model output. Refer
            [to the keras.io docs](https://keras.io/api/keras_cv/bounding_box/formats/)
            for more details on supported bounding box formats.
        anchor_generator: (Optional) a `keras_cv.layers.AnchorGenerator`. It is
            used in the model to match ground truth boxes and labels with
            anchors, or with region proposals. By default it uses the sizes and
            ratios from the paper, that is optimized for image size between
            [640, 800]. The users should pass their own anchor generator if the
            input image size differs from paper. For now, only anchor generator
            with per level dict output is supported,
        label_encoder: (Optional) a keras.Layer that accepts an anchors Tensor,
            a bounding box Tensor and a bounding box class Tensor to its
            `call()` method, and returns RetinaNet training targets. It returns
            box and class targets as well as sample weights.
        rcnn_head: (Optional) a `keras.layers.Layer` that takes input feature
            map and returns a box delta prediction (in reference to rois) and
            multi-class prediction (all foreground classes + one background
            class). By default it uses the rcnn head from paper, which is 2 FC
            layer with 1024 dimension, 1 box regressor and 1 softmax classifier.
        prediction_decoder: (Optional) a `keras.layers.Layer` that takes input
            box prediction and softmaxed score prediction, and returns NMSed box
            prediction, NMSed softmaxed score prediction, NMSed class
            prediction, and NMSed valid detection.

    Examples:

    ```python
    images = np.ones((1, 512, 512, 3))
    labels = {
        "boxes": [
            [
                [0, 0, 100, 100],
                [100, 100, 200, 200],
                [300, 300, 100, 100],
            ]
        ],
        "classes": [[1, 1, 1]],
    }
    model = keras_cv.models.FasterRCNN(
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
    """  # noqa: E501

    def __init__(
        self,
        backbone,
        num_classes,
        bounding_box_format,
        anchor_generator=None,
        label_encoder=None,
        rcnn_head=None,
        prediction_decoder=None,
        feature_pyramid=None,
        **kwargs,
    ):
        self.num_classes = num_classes
        self.bounding_box_format = bounding_box_format
        super().__init__(**kwargs)
        scales = [2**x for x in [0]]
        aspect_ratios = [0.5, 1.0, 2.0]
        self.anchor_generator = anchor_generator or AnchorGenerator(
            bounding_box_format="yxyx",
            sizes={
                "P2": 32.0,
                "P3": 64.0,
                "P4": 128.0,
                "P5": 256.0,
                "P6": 512.0,
            },
            scales=scales,
            aspect_ratios=aspect_ratios,
            strides={f"P{i}": 2**i for i in range(2, 7)},
            clip_boxes=True,
        )
        self.rpn_head = RPNHead(
            num_anchors_per_location=len(scales) * len(aspect_ratios)
        )
        self.roi_generator = ROIGenerator(
            bounding_box_format="yxyx",
            nms_score_threshold_train=float("-inf"),
            nms_score_threshold_test=float("-inf"),
        )
        self.box_matcher = BoxMatcher(
            thresholds=[0.0, 0.5], match_values=[-2, -1, 1]
        )
        self.roi_sampler = _ROISampler(
            bounding_box_format="yxyx",
            roi_matcher=self.box_matcher,
            background_class=num_classes,
            num_sampled_rois=512,
        )
        self.roi_pooler = _ROIAligner(bounding_box_format="yxyx")
        self.rcnn_head = rcnn_head or RCNNHead(num_classes)
        self.backbone = backbone or models.ResNet50Backbone()
        extractor_levels = ["P2", "P3", "P4", "P5"]
        extractor_layer_names = [
            self.backbone.pyramid_level_inputs[i] for i in extractor_levels
        ]
        self.feature_extractor = get_feature_extractor(
            self.backbone, extractor_layer_names, extractor_levels
        )
        self.feature_pyramid = feature_pyramid or FeaturePyramid()
        self.rpn_labeler = label_encoder or _RpnLabelEncoder(
            anchor_format="yxyx",
            ground_truth_box_format="yxyx",
            positive_threshold=0.7,
            negative_threshold=0.3,
            samples_per_image=256,
            positive_fraction=0.5,
            box_variance=BOX_VARIANCE,
        )
        self._prediction_decoder = (
            prediction_decoder
            or cv_layers.NonMaxSuppression(
                bounding_box_format=bounding_box_format,
                from_logits=False,
                iou_threshold=0.5,
                confidence_threshold=0.5,
                max_detections=100,
            )
        )

    def _call_rpn(self, images, anchors, training=None):
        image_shape = ops.shape(images[0])
        backbone_outputs = self.feature_extractor(images, training=training)
        feature_map = self.feature_pyramid(backbone_outputs, training=training)
        # [BS, num_anchors, 4], [BS, num_anchors, 1]
        rpn_boxes, rpn_scores = self.rpn_head(feature_map, training=training)
        # the decoded format is center_xywh, convert to yxyx
        decoded_rpn_boxes = _decode_deltas_to_boxes(
            anchors=anchors,
            boxes_delta=rpn_boxes,
            anchor_format="yxyx",
            box_format="yxyx",
            variance=BOX_VARIANCE,
        )
        rois, _ = self.roi_generator(
            decoded_rpn_boxes, rpn_scores, training=training
        )
        rois = _clip_boxes(rois, "yxyx", image_shape)
        rpn_boxes = ops.concatenate(tree.flatten(rpn_boxes), axis=1)
        rpn_scores = ops.concatenate(tree.flatten(rpn_scores), axis=1)
        return rois, feature_map, rpn_boxes, rpn_scores

    def _call_rcnn(self, rois, feature_map, training=None):
        feature_map = self.roi_pooler(feature_map, rois)
        # [BS, H*W*K, pool_shape*C]
        feature_map = ops.reshape(
            feature_map, ops.concatenate([ops.shape(rois)[:2], [-1]], axis=0)
        )
        # [BS, H*W*K, 4], [BS, H*W*K, num_classes + 1]
        rcnn_box_pred, rcnn_cls_pred = self.rcnn_head(
            feature_map, training=training
        )
        return rcnn_box_pred, rcnn_cls_pred

    def call(self, images, training=None):
        image_shape = ops.shape(images[0])
        anchors = self.anchor_generator(image_shape=image_shape)
        rois, feature_map, _, _ = self._call_rpn(
            images, anchors, training=training
        )
        box_pred, cls_pred = self._call_rcnn(
            rois, feature_map, training=training
        )
        if not training:
            # box_pred is on "center_yxhw" format, convert to target format.
            box_pred = _decode_deltas_to_boxes(
                anchors=rois,
                boxes_delta=box_pred,
                anchor_format="yxyx",
                box_format=self.bounding_box_format,
                variance=[0.1, 0.1, 0.2, 0.2],
            )
        outputs = {
            "boxes": box_pred,
            "classes": cls_pred,
        }
        return outputs

    # TODO(tanzhenyu): Support compile with metrics.
    def compile(
        self,
        box_loss=None,
        classification_loss=None,
        rpn_box_loss=None,
        rpn_classification_loss=None,
        weight_decay=0.0001,
        loss=None,
        **kwargs,
    ):
        # TODO(tanzhenyu): Add metrics support once COCOMap issue is addressed.
        # https://github.com/keras-team/keras-cv/issues/915
        if "metrics" in kwargs.keys():
            raise ValueError(
                "`FasterRCNN` does not currently support the use of "
                "`metrics` due to performance and distribution concerns. "
                "Please use the `PyCOCOCallback` to evaluate COCO metrics."
            )
        if loss is not None:
            raise ValueError(
                "`FasterRCNN` does not accept a `loss` to `compile()`. "
                "Instead, please pass `box_loss` and `classification_loss`. "
                "`loss` will be ignored during training."
            )
        box_loss = _validate_and_get_loss(box_loss, "box_loss")
        classification_loss = _validate_and_get_loss(
            classification_loss, "classification_loss"
        )
        rpn_box_loss = _validate_and_get_loss(rpn_box_loss, "rpn_box_loss")
        if rpn_classification_loss == "BinaryCrossentropy":
            rpn_classification_loss = keras.losses.BinaryCrossentropy(
                from_logits=True, reduction=keras.losses.Reduction.SUM
            )
        rpn_classification_loss = _validate_and_get_loss(
            rpn_classification_loss, "rpn_cls_loss"
        )
        if not rpn_classification_loss.from_logits:
            raise ValueError(
                "`rpn_classification_loss` must come with `from_logits`=True"
            )

        self.rpn_box_loss = rpn_box_loss
        self.rpn_cls_loss = rpn_classification_loss
        self.box_loss = box_loss
        self.cls_loss = classification_loss
        self.weight_decay = weight_decay
        losses = {
            "box": self.box_loss,
            "classification": self.cls_loss,
            "rpn_box": self.rpn_box_loss,
            "rpn_classification": self.rpn_cls_loss,
        }
        super().compile(loss=losses, **kwargs)

    def compute_loss(self, x, y, y_pred, sample_weight, **kwargs):
        images = x
        boxes = y["boxes"]
        if len(y["classes"].shape) != 2:
            raise ValueError(
                "Expected 'classes' to be a tf.Tensor of rank 2. "
                f"Got y['classes'].shape={y['classes'].shape}."
            )
        # TODO(tanzhenyu): remove this hack and perform broadcasting elsewhere
        classes = ops.expand_dims(y["classes"], axis=-1)

        local_batch = images.get_shape().as_list()[0]
        anchors = self.anchor_generator(image_shape=tuple(images[0].shape))
        (
            rpn_box_targets,
            rpn_box_weights,
            rpn_cls_targets,
            rpn_cls_weights,
        ) = self.rpn_labeler(
            ops.concatenate(tree.flatten(anchors), axis=0), boxes, classes
        )
        rpn_box_weights /= (
            self.rpn_labeler.samples_per_image * local_batch * 0.25
        )
        rpn_cls_weights /= self.rpn_labeler.samples_per_image * local_batch
        rois, feature_map, rpn_box_pred, rpn_cls_pred = self._call_rpn(
            images,
            anchors,
        )
        rois = ops.stop_gradient(rois)
        (
            rois,
            box_targets,
            box_weights,
            cls_targets,
            cls_weights,
        ) = self.roi_sampler(rois, boxes, classes)
        box_weights /= self.roi_sampler.num_sampled_rois * local_batch * 0.25
        cls_weights /= self.roi_sampler.num_sampled_rois * local_batch
        box_pred, cls_pred = self._call_rcnn(
            rois,
            feature_map,
        )
        y_true = {
            "rpn_box": rpn_box_targets,
            "rpn_classification": rpn_cls_targets,
            "box": box_targets,
            "classification": cls_targets,
        }
        y_pred = {
            "rpn_box": rpn_box_pred,
            "rpn_classification": rpn_cls_pred,
            "box": box_pred,
            "classification": cls_pred,
        }
        weights = {
            "rpn_box": rpn_box_weights,
            "rpn_classification": rpn_cls_weights,
            "box": box_weights,
            "classification": cls_weights,
        }
        return super().compute_loss(
            x=images, y=y_true, y_pred=y_pred, sample_weight=weights
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
        self._prediction_decoder = prediction_decoder
        self.make_predict_function(force=True)

    def decode_predictions(self, predictions, images):
        # no-op if default decoder is used.
        box_pred, scores_pred = predictions["boxes"], predictions["classes"]
        box_pred = bounding_box.convert_format(
            box_pred,
            source=self.bounding_box_format,
            target=self.prediction_decoder.bounding_box_format,
            images=images,
        )
        y_pred = self.prediction_decoder(box_pred, scores_pred[..., :-1])
        box_pred = bounding_box.convert_format(
            y_pred["boxes"],
            source=self.prediction_decoder.bounding_box_format,
            target=self.bounding_box_format,
            images=images,
        )
        y_pred["boxes"] = box_pred
        return y_pred

    def get_config(self):
        return {
            "num_classes": self.num_classes,
            "bounding_box_format": self.bounding_box_format,
            "backbone": keras.saving.serialize_keras_object(self.backbone),
            "anchor_generator": keras.saving.serialize_keras_object(
                self.anchor_generator
            ),
            "label_encoder": keras.saving.serialize_keras_object(
                self.rpn_labeler
            ),
            "prediction_decoder": keras.saving.serialize_keras_object(
                self._prediction_decoder
            ),
            "feature_pyramid": keras.saving.serialize_keras_object(
                self.feature_pyramid
            ),
            "rcnn_head": keras.saving.serialize_keras_object(self.rcnn_head),
        }

    @classmethod
    def from_config(cls, config):
        if "rcnn_head" in config and isinstance(config["rcnn_head"], dict):
            config["rcnn_head"] = keras.layers.deserialize(config["rcnn_head"])
        if "feature_pyramid" in config and isinstance(
            config["feature_pyramid"], dict
        ):
            config["feature_pyramid"] = keras.layers.deserialize(
                config["feature_pyramid"]
            )
        if "prediction_decoder" in config and isinstance(
            config["prediction_decoder"], dict
        ):
            config["prediction_decoder"] = keras.layers.deserialize(
                config["prediction_decoder"]
            )
        if "label_encoder" in config and isinstance(
            config["label_encoder"], dict
        ):
            config["label_encoder"] = keras.layers.deserialize(
                config["label_encoder"]
            )
        if "anchor_generator" in config and isinstance(
            config["anchor_generator"], dict
        ):
            config["anchor_generator"] = keras.layers.deserialize(
                config["anchor_generator"]
            )
        if "backbone" in config and isinstance(config["backbone"], dict):
            config["backbone"] = keras.layers.deserialize(config["backbone"])
        return super().from_config(config)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        # return copy.deepcopy({**backbone_presets, **fasterrcnn_presets})
        return copy.deepcopy({**backbone_presets})

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return copy.deepcopy(
            # {**backbone_presets_with_weights, **fasterrcnn_presets}
            {
                **backbone_presets_with_weights,
            }
        )

    @classproperty
    def backbone_presets(cls):
        """Dictionary of preset names and configurations of compatible
        backbones."""
        return copy.deepcopy(backbone_presets)


def _validate_and_get_loss(loss, loss_name):
    if isinstance(loss, str):
        loss = keras.losses.get(loss)
    if loss is None or not isinstance(loss, keras.losses.Loss):
        raise ValueError(
            f"FasterRCNN only accepts `keras.losses.Loss` for {loss_name}, "
            f"got {loss}"
        )
    if loss.reduction != keras.losses.Reduction.SUM:
        logging.info(
            f"FasterRCNN only accepts `SUM` reduction, got {loss.reduction}, "
            "automatically converted."
        )
        loss.reduction = keras.losses.Reduction.SUM
    return loss
