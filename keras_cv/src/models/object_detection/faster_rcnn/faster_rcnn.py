import tree

from keras_cv.src import losses
from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.backend import keras
from keras_cv.src.backend import ops
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
from keras_cv.src.models.object_detection.__internal__ import unpack_input
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
    def __init__(
        self,
        backbone,
        num_classes,
        bounding_box_format,
        anchor_generator=None,
        feature_pyramid=None,
        fpn_min_level=2,
        fpn_max_level=5,
        rpn_head=None,
        rpn_filters=256,
        rpn_kernel_size=3,
        rpn_label_en_pos_th=0.7,
        rpn_label_en_neg_th=0.3,
        rpn_label_en_samples_per_image=256,
        rpn_label_en_pos_frac=0.5,
        rcnn_head=None,
        label_encoder=None,
        prediction_decoder=None,
        *args,
        **kwargs,
    ):
        # 1. Backbone
        extractor_levels = [
            f"P{level}" for level in range(fpn_min_level, fpn_max_level + 1)
        ]
        extractor_layer_names = [
            backbone.pyramid_level_inputs[i] for i in extractor_levels
        ]
        feature_extractor = get_feature_extractor(
            backbone, extractor_layer_names, extractor_levels
        )

        # 2. Feature Pyramid
        feature_pyramid = feature_pyramid or FeaturePyramid(
            min_level=fpn_min_level, max_level=fpn_max_level
        )

        # 3. Anchors
        scales = [2**x for x in [0]]
        aspect_ratios = [0.5, 1.0, 2.0]
        anchor_generator = (
            anchor_generator
            or FasterRCNN.default_anchor_generator(
                scales,
                aspect_ratios,
                "yxyx",
            )
        )

        # 4. RPN Head
        num_anchors_per_location = len(scales) * len(aspect_ratios)
        rpn_head = rpn_head or RPNHead(
            num_anchors_per_location=num_anchors_per_location,
            num_filters=rpn_filters,
            kernel_size=rpn_kernel_size,
        )

        # 5. ROI Generator
        roi_generator = ROIGenerator(
            bounding_box_format="yxyx",
            nms_score_threshold_train=float("-inf"),
            nms_score_threshold_test=float("-inf"),
            name="roi_generator",
        )

        # 6. ROI Pooler
        roi_pooler = ROIAligner(bounding_box_format="yxyx", name="roi_pooler")

        # 7. RCNN Head
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
        backbone_outputs = feature_extractor(images)
        feature_map = feature_pyramid(backbone_outputs)

        # [P2, P3, P4, P5, P6] -> ([BS, num_anchors, 4], [BS, num_anchors, 1])
        rpn_boxes, rpn_scores = rpn_head(feature_map)

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

        rois, _ = roi_generator(decoded_rpn_boxes, rpn_scores)
        rois = _clip_boxes(rois, "yxyx", image_shape)

        feature_map = roi_pooler(features=feature_map, boxes=rois)

        # Reshape the feature map [BS, H*W*K]
        feature_map = keras.layers.Reshape(
            target_shape=(
                rois.shape[1],
                (roi_pooler.target_size**2) * rpn_head.num_filters,
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

        # Define the model parameters
        self.bounding_box_format = bounding_box_format
        self.anchor_generator = anchor_generator
        self.num_classes = num_classes
        self.label_encoder = label_encoder or RpnLabelEncoder(
            anchor_format="yxyx",
            ground_truth_box_format=bounding_box_format,
            positive_threshold=rpn_label_en_pos_th,
            negative_threshold=rpn_label_en_neg_th,
            samples_per_image=rpn_label_en_samples_per_image,
            positive_fraction=rpn_label_en_pos_frac,
            box_variance=BOX_VARIANCE,
        )
        self.backbone = backbone
        self.feature_extractor = feature_extractor
        self.feature_pyramid = feature_pyramid
        self.rpn_head = rpn_head
        self.roi_generator = roi_generator
        self.box_matcher = BoxMatcher(
            thresholds=[0.0, 0.5], match_values=[-2, -1, 1]
        )
        self.roi_sampler = ROISampler(
            roi_bounding_box_format="yxyx",
            gt_bounding_box_format=bounding_box_format,
            roi_matcher=self.box_matcher,
        )
        self.roi_pooler = roi_pooler
        self.rcnn_head = rcnn_head
        self._prediction_decoder = (
            prediction_decoder
            or NonMaxSuppression(
                bounding_box_format=bounding_box_format,
                from_logits=True,
                max_detections=100,
            )
        )

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

        # Generate anchors
        # image shape must not contain the batch size
        local_batch = ops.shape(images)[0]
        image_shape = ops.shape(images)[1:]
        anchors = self.anchor_generator(image_shape=image_shape)

        # 2. Label with the anchors -- exclusive to compute_loss
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
        # 3. Computing the weights
        rpn_box_weights /= (
            self.label_encoder.samples_per_image * local_batch * 0.25
        )
        rpn_cls_weights /= self.label_encoder.samples_per_image * local_batch

        #######################################################################
        # Call RPN
        #######################################################################

        backbone_outputs = self.feature_extractor(images)
        feature_map = self.feature_pyramid(backbone_outputs)

        # [BS, num_anchors, 4], [BS, num_anchors, 1]
        rpn_boxes, rpn_scores = self.rpn_head(feature_map)
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

        # print(f"ROI's Generated from RPN Network: {rois}")

        # 4. Stop gradient from flowing into the ROI
        # -- exclusive to compute_loss
        rois = ops.stop_gradient(rois)
        # 5. Sample the ROIS -- exclusive to compute_loss
        (
            rois,
            box_targets,
            box_weights,
            cls_targets,
            cls_weights,
        ) = self.roi_sampler(rois, gt_boxes, gt_classes)

        # to apply one hot encoding
        cls_targets = ops.squeeze(cls_targets, axis=-1)
        cls_weights = ops.squeeze(cls_weights, axis=-1)

        # 6. Box and class weights -- exclusive to compute loss
        box_weights /= self.roi_sampler.num_sampled_rois * local_batch * 0.25
        cls_weights /= self.roi_sampler.num_sampled_rois * local_batch
        cls_targets = ops.one_hot(cls_targets, num_classes=self.num_classes+1)

        # print(f"Box Targets Shape: {box_targets.shape}")
        # print(f"Box Weights Shape: {box_weights.shape}")
        # print(f"Cls Targets Shape: {cls_targets.shape}")
        # print(f"Cls Weights Shape: {cls_weights.shape}")
        # print(f"RPN Box Targets Shape: {rpn_box_targets.shape}")
        # print(f"RPN Box Weights Shape: {rpn_box_weights.shape}")
        # print(f"RPN Cls Targets Shape: {rpn_cls_targets.shape}")
        # print(f"RPN Cls Weights Shape: {rpn_cls_weights.shape}")
        # print(f"Cls Weights: {cls_weights}")
        # print(f"Box Weights: {box_weights}")

        # print(f"Cls Targets: {cls_targets}")
        # print(f"Box Targets: {box_targets}")

        #######################################################################
        # Call RCNN
        #######################################################################

        feature_map = self.roi_pooler(features=feature_map, boxes=rois)

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

    @staticmethod
    def default_anchor_generator(scales, aspect_ratios, bounding_box_format):
        strides = {f"P{i}": 2**i for i in range(2, 7)}
        sizes = {
            "P2": 32.0,
            "P3": 64.0,
            "P4": 128.0,
            "P5": 256.0,
            "P6": 512.0,
        }
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
            "rpn_head": keras.saving.serialize_keras_object(
                self.rpn_head
            ),
            "prediction_decoder": self._prediction_decoder,
            "rcnn_head": self.rcnn_head, 
        }

    @classmethod
    def from_config(cls, config):
        if "rpn_head" in config and isinstance(
            config["rpn_head"], dict
        ):
            config["rpn_head"] = keras.layers.deserialize(
                config["rpn_head"]
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
        if "rcnn_head" in config and isinstance(
            config["rcnn_head"], dict
        ):
            config["rcnn_head"] = keras.layers.deserialize(
                config["rcnn_head"]
            )

        return super().from_config(config)


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


def _parse_rpn_classification_loss(loss):
    if not isinstance(loss, str):
        # support arbitrary callables
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
    if not isinstance(loss, str):
        # support arbitrary callables
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
