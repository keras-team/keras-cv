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

import tree

from keras_cv.src import losses
from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.backend import keras
from keras_cv.src.backend import ops
from keras_cv.src.bounding_box import convert_format
from keras_cv.src.bounding_box.converters import _decode_deltas_to_boxes
from keras_cv.src.bounding_box.utils import _clip_boxes
from keras_cv.src.layers.object_detection.anchor_generator import (
    AnchorGenerator,
)
from keras_cv.src.layers.object_detection.box_matcher import BoxMatcher
from keras_cv.src.layers.object_detection.non_max_suppression import (
    NonMaxSuppression,
)
from keras_cv.src.layers.object_detection.roi_align import ROIAligner
from keras_cv.src.layers.object_detection.roi_generator import ROIGenerator
from keras_cv.src.layers.object_detection.roi_sampler import ROISampler
from keras_cv.src.layers.object_detection.rpn_label_encoder import (
    RpnLabelEncoder,
)
from keras_cv.src.models.object_detection.faster_rcnn import FeaturePyramid
from keras_cv.src.models.object_detection.faster_rcnn import RCNNHead
from keras_cv.src.models.object_detection.faster_rcnn import RPNHead
from keras_cv.src.models.task import Task
from keras_cv.src.utils.train import get_feature_extractor

BOX_VARIANCE = [0.1, 0.1, 0.2, 0.2]


@keras_cv_export(
    [
        "keras_cv.models.FasterRCNN",
        "keras_cv.models.object_detection.FasterRCNN",
    ]
)
class FasterRCNN(Task):
    """A Keras model implementing the Faster R-CNN architecture.

    This model is compatible with Keras 3 only. Implements the Faster R-CNN architecture
    for object detection. The constructor requires `num_classes`, `bounding_box_format`,
    and a backbone. Optionally, a custom label encoder, and prediction decoder
    may be provided.

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
    model = FasterRCNN(
        num_classes=80,
        bounding_box_format="xyxy",
        backbone=keras_cv.models.ResNet18V2Backbone(
            input_shape=(512, 512, 3)
        ),
    )

    # Evaluate model without box decoding and NMS
    model(images)

    # Prediction with box decoding and NMS
    model.predict(images)

    # Train model
    model.compile(
        optimizer=keras.optimizers.SGD(),
        box_loss="Huber",
        classification_loss="CategoricalCrossentropy",
        rpn_box_loss="Huber",
        rpn_classification_loss="BinaryCrossentropy",
    )
    model.fit(images, labels, batch_size=1)
    ```

    Args:
         backbone: `keras.Model`. If the default `feature_pyramid` is used,
            must implement the `pyramid_level_inputs` property with keys "P3", "P4",
            and "P5" and layer names as values. A somewhat sensible backbone
            to use in many cases is the:
            `keras_cv.models.ResNetBackbone.from_preset("resnet50_imagenet")`
        num_classes: the number of classes in your dataset excluding the
            background class. Classes should be represented by integers in the
            range [1, num_classes].
        bounding_box_format: The format of bounding boxes of input dataset.
            Refer
            [to the keras.io docs](https://keras.io/api/keras_cv/bounding_box/formats/)
            for more details on supported bounding box formats.
        anchor_generator: (Optional) a `keras_cv.layers.AnchorGenerator`. If
            provided, the anchor generator will be passed to both the
            `label_encoder` and the `prediction_decoder`. Only to be used when
            both `label_encoder` and `prediction_decoder` are both `None`.
            Defaults to an anchor generator with the parameterization:
            `strides=[2**i for i in range(3, 8)]`,
            `scales=[2**x for x in [0, 1 / 3, 2 / 3]]`,
            `sizes=[32.0, 64.0, 128.0, 256.0, 512.0]`,
            and `aspect_ratios=[0.5, 1.0, 2.0]`.
        anchor_scales: (Optional) list of anchor scales for
            default anchor generator.
        anchor_aspect_ratios: (Optional) list of anchor aspect ratios for
            default anchor generator.
        feature_pyramid: (Optional) A `keras.layers.Layer` that produces
            a list of 4D feature maps (batch dimension included)
            when called on the pyramid-level outputs of the `backbone`.
            If not provided, the reference implementation from the paper will be used.
        fpn_min_level: (Optional) the minimum level of the feature pyramid.
        fpn_max_level: (Optional) the maximum level of the feature pyramid.
        rpn_head: (Optional) A `keras.Layer` that performs regression and
            classification(background or foreground) of the bounding boxes.
            If not provided, a simple ConvNet with 3 layers will be used.
        rpn_label_encoder_posistive_threshold: (Optional) the float threshold to set an
            anchor to positive match to gt box. Values above it are positive matches.
        rpn_label_encoder_negative_threshold: (Optional) the float threshold to set an
            anchor to negative matchto gt box. Values below it are negative matches.
        rpn_label_encoder_samples_per_image: (Optional) for each image, the number of
            positive and negative samples to generate.
        rpn_label_encoder_positive_fraction: (Optional) the fraction of positive samples to the total samples.
        rcnn_head: (Optional) A `keras.Layer` that performs regression and
            classification(final prediction) of the bounding boxes.
            If not provided, a simple network with 2 dense layers with
            box head and regression head will be used.
        label_encoder: (Optional) a keras.Layer that accepts an image Tensor, a
            bounding box Tensor and a bounding box class Tensor to its `call()`
            method, and returns RetinaNet training targets. By default, a
            KerasCV standard `RpnLabelEncoder` is created and used.
            Results of this object's `call()` method are passed to the `loss`
            object for `rpn_box_loss` and `rpn_classification_loss` the `y_true`
            argument.
        prediction_decoder: (Optional)  A `keras.layers.Layer` that is
            responsible for transforming RetinaNet predictions into usable
            bounding box Tensors. If not provided, a default is provided. The
            default `prediction_decoder` layer is a
            `keras_cv.layers.MultiClassNonMaxSuppression` layer, which uses
            a Non-Max Suppression for box pruning.
        num_max_detections: the maximum detections to consider after nms is applied. A
            large number may trigger significant memory overhead, defaults to 100.
    """  # noqa: E501

    def __init__(
        self,
        backbone,
        num_classes,
        bounding_box_format,
        anchor_generator=None,
        anchor_scales=[1],
        anchor_aspect_ratios=[0.5, 1.0, 2.0],
        feature_pyramid=None,
        fpn_min_level=2,
        fpn_max_level=5,
        rpn_head=None,
        rpn_filters=256,
        rpn_kernel_size=3,
        rpn_label_encoder_posistive_threshold=0.7,
        rpn_label_encoder_negative_threshold=0.3,
        rpn_label_encoder_samples_per_image=256,
        rpn_label_encoder_positive_fraction=0.5,
        rcnn_head=None,
        num_sampled_rois=512,
        label_encoder=None,
        prediction_decoder=None,
        num_max_decoder_detections=100,
        *args,
        **kwargs,
    ):
        # Backbone
        extractor_levels = [
            f"P{level}" for level in range(fpn_min_level, fpn_max_level + 1)
        ]
        extractor_layer_names = [
            backbone.pyramid_level_inputs[i] for i in extractor_levels
        ]
        feature_extractor = get_feature_extractor(
            backbone, extractor_layer_names, extractor_levels
        )

        # Feature Pyramid
        feature_pyramid = feature_pyramid or FeaturePyramid(
            min_level=fpn_min_level, max_level=fpn_max_level
        )

        # Anchors
        anchor_generator = (
            anchor_generator
            or FasterRCNN.default_anchor_generator(
                fpn_min_level,
                fpn_max_level + 1,
                anchor_scales,
                anchor_aspect_ratios,
                "yxyx",
            )
        )

        # RPN Head
        num_anchors_per_location = len(anchor_scales) * len(
            anchor_aspect_ratios
        )
        rpn_head = rpn_head or RPNHead(
            num_anchors_per_location=num_anchors_per_location,
            num_filters=rpn_filters,
            kernel_size=rpn_kernel_size,
        )

        # RoI Generator
        roi_generator = ROIGenerator(
            bounding_box_format="yxyx",
            nms_score_threshold_train=float("-inf"),
            nms_score_threshold_test=float("-inf"),
            nms_from_logits=True,
            name="roi_generator",
        )

        # RoI Align
        roi_aligner = ROIAligner(bounding_box_format="yxyx", name="roi_align")

        # R-CNN Head
        rcnn_head = rcnn_head or RCNNHead(num_classes, name="rcnn_head")

        # Begin construction of forward pass
        image_shape = feature_extractor.input_shape[1:]
        if None in image_shape:
            raise ValueError(
                "Found `None` in image_shape, to build anchors `image_shape`"
                "is required without any `None`. Make sure to pass "
                "`image_shape` to the backbone preset while passing to"
                "the Faster R-CNN detector."
            )

        images = keras.layers.Input(
            image_shape,
            name="images",
        )

        # Forward through backbone
        backbone_outputs = feature_extractor(images)

        # Forward through FPN decoder
        feature_map = feature_pyramid(backbone_outputs)

        # [P2, P3, P4, P5, P6] -> ([BS, num_anchors, 4], [BS, num_anchors, 1])
        # Pass through RPN Head
        rpn_boxes, rpn_scores = rpn_head(feature_map)

        # Reshape and Concatenate all the output boxes of all levels
        for lvl in rpn_boxes:
            rpn_boxes[lvl] = keras.layers.Reshape(target_shape=(-1, 4))(
                rpn_boxes[lvl]
            )

        for lvl in rpn_scores:
            rpn_scores[lvl] = keras.layers.Reshape(target_shape=(-1, 1))(
                rpn_scores[lvl]
            )
        rpn_cls_pred = keras.layers.Concatenate(
            axis=1, name="rpn_classification"
        )(tree.flatten(rpn_scores))
        rpn_box_pred = keras.layers.Concatenate(axis=1, name="rpn_box")(
            tree.flatten(rpn_boxes)
        )

        anchors = anchor_generator(image_shape=image_shape)
        decoded_rpn_boxes = _decode_deltas_to_boxes(
            anchors=anchors,
            boxes_delta=rpn_boxes,
            anchor_format="yxyx",
            box_format="yxyx",
            variance=BOX_VARIANCE,
        )

        rois, roi_scores = roi_generator(decoded_rpn_boxes, rpn_scores)
        rois = _clip_boxes(rois, "yxyx", image_shape)

        feature_map = roi_aligner(features=feature_map, boxes=rois)

        # Reshape the feature map [BS, H*W*K]
        feature_map = keras.layers.Reshape(
            target_shape=(
                rois.shape[1],
                (roi_aligner.target_size**2) * rpn_head.num_filters,
            )
        )(feature_map)

        # Pass final feature map to RCNN Head for predictions
        box_pred, cls_pred = rcnn_head(feature_map=feature_map)

        box_pred = keras.layers.Concatenate(axis=1, name="box")([box_pred])
        cls_pred = keras.layers.Concatenate(axis=1, name="classification")(
            [cls_pred]
        )

        inputs = {"images": images}
        outputs = {
            "rpn_box": rpn_box_pred,
            "rpn_classification": rpn_cls_pred,
            "box": box_pred,
            "classification": cls_pred,
        }

        super().__init__(
            inputs=inputs,
            outputs=outputs,
            **kwargs,
        )

        self.bounding_box_format = bounding_box_format
        self.anchor_generator = anchor_generator
        self.num_classes = num_classes
        self.feature_extractor = feature_extractor
        self.backbone = backbone
        self.feature_pyramid = feature_pyramid
        self.rpn_head = rpn_head
        self.label_encoder = label_encoder or RpnLabelEncoder(
            anchor_format="yxyx",
            ground_truth_box_format=bounding_box_format,
            positive_threshold=rpn_label_encoder_posistive_threshold,
            negative_threshold=rpn_label_encoder_negative_threshold,
            samples_per_image=rpn_label_encoder_samples_per_image,
            positive_fraction=rpn_label_encoder_positive_fraction,
            box_variance=BOX_VARIANCE,
        )
        self.roi_generator = roi_generator
        self.box_matcher = BoxMatcher(
            thresholds=[0.0, 0.5], match_values=[-2, -1, 1]
        )
        self.roi_sampler = ROISampler(
            roi_bounding_box_format="yxyx",
            gt_bounding_box_format=bounding_box_format,
            roi_matcher=self.box_matcher,
            num_sampled_rois=num_sampled_rois,
        )

        self.roi_aligner = roi_aligner
        self.rcnn_head = rcnn_head
        self._prediction_decoder = prediction_decoder or NonMaxSuppression(
            bounding_box_format=bounding_box_format,
            from_logits=False,
            max_detections=num_max_decoder_detections,
        )
        self.build(backbone.input_shape)

    def compile(
        self,
        rpn_box_loss=None,
        rpn_classification_loss=None,
        box_loss=None,
        classification_loss=None,
        weight_decay=0.0001,
        loss=None,
        metrics=None,
        **kwargs,
    ):
        if loss is not None:
            raise ValueError(
                "`FasterRCNN` does not accept a `loss` to `compile()`. "
                "Instead, please pass `box_loss` and `classification_loss`. "
                "`loss` will be ignored during training."
            )
        if (
            rpn_box_loss is None
            or rpn_classification_loss is None
            or box_loss is None
            or classification_loss is None
        ):
            raise ValueError(
                "`FasterRCNN` expects all of `rpn_box_loss`, "
                "`rpn_classification_loss`,"
                "`box_loss`, and "
                "`classification_loss` to be not `None`."
            )

        rpn_box_loss = _parse_box_loss(rpn_box_loss)
        rpn_classification_loss = _parse_rpn_classification_loss(
            rpn_classification_loss
        )

        if hasattr(rpn_classification_loss, "from_logits"):
            if not rpn_classification_loss.from_logits:
                raise ValueError(
                    "FasterRCNN.compile() expects `from_logits` to be True for "
                    "`rpn_classification_loss`. Got "
                    "`rpn_classification_loss.from_logits="
                    f"{rpn_classification_loss.from_logits}`"
                )
        box_loss = _parse_box_loss(box_loss)
        classification_loss = _parse_classification_loss(classification_loss)

        if hasattr(classification_loss, "from_logits"):
            if not classification_loss.from_logits:
                raise ValueError(
                    "FasterRCNN.compile() expects `from_logits` to be True for "
                    "`classification_loss`. Got "
                    "`classification_loss.from_logits="
                    f"{classification_loss.from_logits}`"
                )
        if hasattr(box_loss, "bounding_box_format"):
            if box_loss.bounding_box_format != self.bounding_box_format:
                raise ValueError(
                    "Wrong `bounding_box_format` passed to `box_loss` in "
                    "`FasterRCNN.compile()`. Got "
                    "`box_loss.bounding_box_format="
                    f"{box_loss.bounding_box_format}`, want "
                    "`box_loss.bounding_box_format="
                    f"{self.bounding_box_format}`"
                )

        self.rpn_box_loss = rpn_box_loss
        self.rpn_cls_loss = rpn_classification_loss
        self.box_loss = box_loss
        self.cls_loss = classification_loss
        self.weight_decay = weight_decay
        losses = {
            "rpn_box": self.rpn_box_loss,
            "rpn_classification": self.rpn_cls_loss,
            "box": self.box_loss,
            "classification": self.cls_loss,
        }
        self._has_user_metrics = metrics is not None and len(metrics) != 0
        self._user_metrics = metrics
        super().compile(loss=losses, **kwargs)

    def compute_loss(
        self, x, y, y_pred, sample_weight, training=True, **kwargs
    ):

        # 1. Unpack the inputs
        images = x
        gt_boxes = y["boxes"]
        if ops.ndim(y["classes"]) != 2:
            raise ValueError(
                "Expected 'classes' to be a Tensor of rank 2. "
                f"Got y['classes'].shape={ops.shape(y['classes'])}."
            )

        gt_classes = y["classes"]
        gt_classes = ops.expand_dims(gt_classes, axis=-1)

        # Generate  Anchors and Generate RPN Targets
        local_batch = ops.shape(images)[0]
        image_shape = ops.shape(images)[1:]
        anchors = self.anchor_generator(image_shape=image_shape)

        # Label with the anchors -- exclusive to compute_loss
        (
            rpn_box_targets,
            rpn_box_weights,
            rpn_cls_targets,
            rpn_cls_weights,
        ) = self.label_encoder(
            anchors_dict=ops.concatenate(
                tree.flatten(anchors),
                axis=0,
            ),
            gt_boxes=gt_boxes,
            gt_classes=gt_classes,
        )

        # Computing the weights
        rpn_box_weights /= (
            self.label_encoder.samples_per_image * local_batch * 0.25
        )
        rpn_cls_weights /= self.label_encoder.samples_per_image * local_batch

        # Call Backbone, FPN and RPN Head
        backbone_outputs = self.feature_extractor(images)
        feature_map = self.feature_pyramid(backbone_outputs)
        rpn_boxes, rpn_scores = self.rpn_head(feature_map)

        for lvl in rpn_boxes:
            rpn_boxes[lvl] = keras.layers.Reshape(target_shape=(-1, 4))(
                rpn_boxes[lvl]
            )

        for lvl in rpn_scores:
            rpn_scores[lvl] = keras.layers.Reshape(target_shape=(-1, 1))(
                rpn_scores[lvl]
            )

        # [BS, num_anchors, 4], [BS, num_anchors, 1]
        rpn_cls_pred = keras.layers.Concatenate(
            axis=1, name="rpn_classification"
        )(tree.flatten(rpn_scores))
        rpn_box_pred = keras.layers.Concatenate(axis=1, name="rpn_box")(
            tree.flatten(rpn_boxes)
        )

        # Generate RoI's and RoI Sampling
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

        # Stop gradient from flowing into the ROI
        # -- exclusive to compute_loss
        rois = ops.stop_gradient(rois)

        # Sample the ROIS -- exclusive to compute_loss
        (
            rois,
            box_targets,
            box_weights,
            cls_targets,
            cls_weights,
        ) = self.roi_sampler(rois, gt_boxes, gt_classes)

        cls_targets = ops.squeeze(cls_targets, axis=-1)
        cls_weights = ops.squeeze(cls_weights, axis=-1)

        # Box and class weights -- exclusive to compute loss
        box_weights /= self.roi_sampler.num_sampled_rois * local_batch * 0.25
        cls_weights /= self.roi_sampler.num_sampled_rois * local_batch
        cls_targets = ops.one_hot(cls_targets, num_classes=self.num_classes + 1)

        # Call RoI Aligner and RCNN Head
        feature_map = self.roi_aligner(features=feature_map, boxes=rois)

        # [BS, H*W*K]
        feature_map = ops.reshape(
            feature_map,
            newshape=ops.shape(rois)[:2] + (-1,),
        )

        # [BS, H*W*K, 4], [BS, H*W*K, num_classes + 1]
        box_pred, cls_pred = self.rcnn_head(feature_map=feature_map)

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
            x=x, y=y_true, y_pred=y_pred, sample_weight=weights, **kwargs
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
        if prediction_decoder.bounding_box_format != self.bounding_box_format:
            raise ValueError(
                "Expected `prediction_decoder` and FasterRCNN to "
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

    def decode_predictions(self, predictions, images):
        image_shape = ops.shape(images)[1:]
        anchors = self.anchor_generator(image_shape=image_shape)
        rpn_boxes, rpn_scores = (
            predictions["rpn_box"],
            predictions["rpn_classification"],
        )
        decoded_rpn_boxes = _decode_deltas_to_boxes(
            anchors=ops.concatenate(
                tree.flatten(anchors),
                axis=0,
            ),
            boxes_delta=rpn_boxes,
            anchor_format="yxyx",
            box_format="yxyx",
            variance=BOX_VARIANCE,
        )

        rois, _ = self.roi_generator(
            decoded_rpn_boxes, rpn_scores, training=False
        )
        rois = _clip_boxes(rois, "yxyx", image_shape)
        box_pred, cls_pred = predictions["box"], predictions["classification"]

        # box_pred is on "center_yxhw" format, convert to target format.
        box_pred = _decode_deltas_to_boxes(
            anchors=rois,
            boxes_delta=box_pred,
            anchor_format=self.roi_aligner.bounding_box_format,
            box_format=self.bounding_box_format,
            variance=BOX_VARIANCE,
            image_shape=image_shape,
        )

        box_pred = convert_format(
            box_pred,
            source=self.bounding_box_format,
            target=self.prediction_decoder.bounding_box_format,
            image_shape=image_shape,
        )
        cls_pred = ops.softmax(cls_pred)
        cls_pred = ops.slice(
            cls_pred,
            start_indices=[0, 0, 1],
            shape=[cls_pred.shape[0], cls_pred.shape[1], cls_pred.shape[2] - 1],
        )

        y_pred = self.prediction_decoder(
            box_pred, cls_pred, image_shape=image_shape
        )

        y_pred["classes"] = ops.where(
            y_pred["classes"] == -1, -1, y_pred["classes"] + 1
        )

        y_pred["boxes"] = convert_format(
            y_pred["boxes"],
            source=self.prediction_decoder.bounding_box_format,
            target=self.bounding_box_format,
            image_shape=image_shape,
        )
        return y_pred

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

    @staticmethod
    def default_anchor_generator(
        min_level, max_level, scales, aspect_ratios, bounding_box_format
    ):
        strides = {f"P{i}": 2**i for i in range(min_level, max_level + 1)}
        sizes = {f"P{i}": 2 ** (3 + i) for i in range(min_level, max_level + 1)}
        return AnchorGenerator(
            bounding_box_format=bounding_box_format,
            sizes=sizes,
            aspect_ratios=aspect_ratios,
            scales=scales,
            strides=strides,
            clip_boxes=True,
            name="anchor_generator",
        )

    def get_config(self):
        return {
            "num_classes": self.num_classes,
            "bounding_box_format": self.bounding_box_format,
            "backbone": keras.saving.serialize_keras_object(self.backbone),
            "label_encoder": keras.saving.serialize_keras_object(
                self.label_encoder
            ),
            "rpn_head": keras.saving.serialize_keras_object(self.rpn_head),
            "prediction_decoder": self._prediction_decoder,
            "rcnn_head": self.rcnn_head,
        }

    @classmethod
    def from_config(cls, config):
        if "rpn_head" in config and isinstance(config["rpn_head"], dict):
            config["rpn_head"] = keras.layers.deserialize(config["rpn_head"])
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
        if "rcnn_head" in config and isinstance(config["rcnn_head"], dict):
            config["rcnn_head"] = keras.layers.deserialize(config["rcnn_head"])

        return super().from_config(config)


def _parse_box_loss(loss):
    # support arbitrary callables
    if not isinstance(loss, str):
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


def _parse_rpn_classification_loss(loss):
    # support arbitrary callables
    if not isinstance(loss, str):
        return loss

    if loss.lower() == "binarycrossentropy":
        return keras.losses.BinaryCrossentropy(
            reduction="sum", from_logits=True
        )

    raise ValueError(
        f"Expected `rpn_classification_loss` to be either BinaryCrossentropy"
        f" loss callable, or the string 'BinaryCrossentropy'. Got loss={loss}."
    )


def _parse_classification_loss(loss):
    # support arbitrary callables
    if not isinstance(loss, str):
        return loss

    # case insensitive comparison
    if loss.lower() == "focal":
        return losses.FocalLoss(reduction="sum", from_logits=True)
    if loss.lower() == "categoricalcrossentropy":
        return keras.losses.CategoricalCrossentropy(
            reduction="sum", from_logits=True
        )

    raise ValueError(
        f"Expected `classification_loss` to be either a Keras Loss, "
        f"callable, or the string 'Focal', CategoricalCrossentropy'. "
        f"Got loss={loss}."
    )


def unpack_input(data):
    if type(data) is dict:
        return data["images"], data["bounding_boxes"]
    else:
        return data
