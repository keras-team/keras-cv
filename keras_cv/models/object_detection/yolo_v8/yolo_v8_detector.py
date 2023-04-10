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

import tensorflow as tf
from keras import layers
from tensorflow import keras

import keras_cv
from keras_cv import bounding_box
from keras_cv.models.backbones.backbone_presets import backbone_presets
from keras_cv.models.backbones.backbone_presets import (
    backbone_presets_with_weights,
)
from keras_cv.models.object_detection import predict_utils
from keras_cv.models.object_detection.yolo_v8.compat_anchor_generation import (
    get_anchors,
)
from keras_cv.models.object_detection.yolo_v8.yolo_v8_detector_presets import (
    yolo_v8_detector_presets,
)
from keras_cv.models.object_detection.yolo_v8.yolo_v8_layers import conv_bn
from keras_cv.models.object_detection.yolo_v8.yolo_v8_layers import (
    csp_with_2_conv,
)
from keras_cv.models.task import Task
from keras_cv.utils.python_utils import classproperty
from keras_cv.utils.train import get_feature_extractor


def path_aggregation_fpn(features, depth=3, name=None):
    # yolov8
    # 9: p5 1024 ---+----------------------+-> 21: out2 1024
    #               v [up 1024 -> concat]  ^ [down 512 -> concat]
    # 6: p4 512 --> 12: p4p5 512 --------> 18: out1 512
    #               v [up 512 -> concat]   ^ [down 256 -> concat]
    # 4: p3 256 --> 15: p3p4p5 256 --------+--> 15: out0 128
    # features: [p3, p4, p5]
    channel_axis = -1
    upsamples = [features[-1]]
    # upsamples: [p5], features[:-1][::-1]: [p4, p3] -> [p5, p4p5, p3p4p5]
    for id, feature in enumerate(features[:-1][::-1]):
        size = tf.shape(feature)[1:-1]
        nn = tf.image.resize(upsamples[-1], size, method="nearest")
        nn = tf.concat([nn, feature], axis=channel_axis)

        out_channel = feature.shape[channel_axis]
        nn = csp_with_2_conv(
            nn,
            channels=out_channel,
            depth=depth,
            shortcut=False,
            activation="swish",
            name=f"{name}_p{len(features) + 1 - id}",
        )
        upsamples.append(nn)

    downsamples = [upsamples[-1]]
    # downsamples: [p3p4p5], upsamples[:-1][::-1]:
    # [p4p5, p5] -> [p3p4p5, p3p4p5 + p4p5, p3p4p5 + p4p5 + p5]
    for id, ii in enumerate(upsamples[:-1][::-1]):
        cur_name = f"{name}_c3n{id + 3}"
        nn = conv_bn(
            downsamples[-1],
            downsamples[-1].shape[channel_axis],
            kernel_size=3,
            strides=2,
            activation="swish",
            name=f"{cur_name}_down",
        )
        nn = tf.concat([nn, ii], axis=channel_axis)

        out_channel = ii.shape[channel_axis]
        nn = csp_with_2_conv(
            nn,
            channels=out_channel,
            depth=depth,
            shortcut=False,
            activation="swish",
            name=cur_name,
        )
        downsamples.append(nn)
    return downsamples


def yolov8_head(
    inputs,
    num_classes=80,
    bbox_len=64,
    name="yolov8_head",
):
    outputs = []
    reg_channels = max(16, bbox_len, inputs[0].shape[-1] // 4)
    cls_channels = max(num_classes, inputs[0].shape[-1])
    for id, feature in enumerate(inputs):
        cur_name = f"{name}_{id+1}"

        reg_nn = conv_bn(
            feature,
            reg_channels,
            3,
            activation="swish",
            name=f"{cur_name}_reg_1",
        )
        reg_nn = conv_bn(
            reg_nn,
            reg_channels,
            3,
            activation="swish",
            name=f"{cur_name}_reg_2",
        )
        reg_out = layers.Conv2D(
            filters=bbox_len,
            kernel_size=1,
            name=f"{cur_name}_reg_3_conv",
        )(reg_nn)

        cls_nn = conv_bn(
            feature,
            cls_channels,
            3,
            activation="swish",
            name=f"{cur_name}_cls_1",
        )
        cls_nn = conv_bn(
            cls_nn,
            cls_channels,
            3,
            activation="swish",
            name=f"{cur_name}_cls_2",
        )
        cls_out = layers.Conv2D(
            filters=num_classes,
            kernel_size=1,
            name=f"{cur_name}_cls_3_conv",
        )(cls_nn)
        cls_out = layers.Activation("sigmoid", name=f"{cur_name}_classifier")(
            cls_out
        )

        out = tf.concat([reg_out, cls_out], axis=-1)
        out = layers.Reshape(
            [-1, out.shape[-1]], name=f"{cur_name}_output_reshape"
        )(out)
        outputs.append(out)

    outputs = tf.concat(outputs, axis=1)
    return outputs


def decode_regression_to_boxes(preds, regression_max=16):
    preds_bbox = tf.reshape(preds, (-1, preds.shape[1], 4, regression_max))
    preds_bbox = tf.nn.softmax(preds_bbox, axis=-1) * tf.range(
        regression_max, dtype="float32"
    )
    return tf.reduce_sum(preds_bbox, axis=-1)


def decode_boxes(preds, anchors):
    # Boxes expected to be in encoded format
    preds_top_left, preds_bottom_right = tf.split(preds, [2, 2], axis=-1)

    # Converts rel_yxyx anchors to rel_center_yxhw
    anchors_hw = anchors[:, 2:] - anchors[:, :2]
    anchors_center = (anchors[:, :2] + anchors[:, 2:]) * 0.5

    pred_sum = preds_bottom_right + preds_top_left
    pred_hw_half = (preds_bottom_right - preds_top_left) / 2

    bboxes_center = pred_hw_half * anchors_hw + anchors_center
    bboxes_hw = pred_sum * anchors_hw

    # Preds in rel_yxyx
    preds_top_left = bboxes_center - 0.5 * bboxes_hw
    pred_bottom_right = preds_top_left + bboxes_hw

    # Returns results in rel_yxyx
    return tf.concat([preds_top_left, pred_bottom_right], axis=-1)


@keras.utils.register_keras_serializable(package="keras_cv")
class YOLOV8Detector(Task):
    """
    Implements the YOLOV8 architecture for object detection.

    Note: this implementation **does not yet support training**, and is
    for presets only.

    Examples:
    ```python
    images = tf.ones(shape=(1, 512, 512, 3))
    model = keras_cv.models.YOLOV8Detector.from_preset("yolov8_n_coco", bounding_box_format="xywh")

    predictions = model.predict(images)
    ```

    Args:
        num_classes: the number of classes in your dataset excluding the
            background class. Classes should be represented by integers in the
            range [0, num_classes).
        bounding_box_format: The format of bounding boxes of input dataset.
            Refer
            [to the keras.io docs](https://keras.io/api/keras_cv/bounding_box/formats/)
            for more details on supported bounding box formats.
        backbone: `keras.Model`. Must implement the `pyramid_level_inputs`
            property with keys 3, 4, and 5 and layer names as values. A
            sensible backbone to use in many cases is the
            `keras_cv.models.YOLOV8Backbone`
        fpn_depth: integer, a specification of the depth for the Feature
            Pyramid Network. This is usually 1, 2, or 3, depending on the
            size of your YOLOV8Detector model.
        prediction_decoder: (Optional)  A `keras.layers.Layer` that is
            responsible for transforming RetinaNet predictions into usable
            bounding box Tensors. If not provided, a default is provided. The
            default `prediction_decoder` layer is a
            `keras_cv.layers.MultiClassNonMaxSuppression` layer, which uses
            a Non-Max Suppression for box pruning.
    """  # noqa: E501

    def __init__(
        self,
        bounding_box_format,
        backbone,
        fpn_depth,
        num_classes,
        prediction_decoder=None,
        **kwargs,
    ):
        extractor_levels = [2, 3, 4]
        if 5 in backbone.pyramid_level_inputs.keys():
            extractor_levels.append(5)
        extractor_layer_names = [
            backbone.pyramid_level_inputs[i] for i in extractor_levels
        ]
        feature_extractor = get_feature_extractor(
            backbone, extractor_layer_names, extractor_levels
        )

        images = layers.Input(feature_extractor.input_shape[1:])
        features = list(feature_extractor(images).values())

        # Apply the FPN
        fpn_features = path_aggregation_fpn(
            features, depth=fpn_depth, name="pa_fpn"
        )

        outputs = yolov8_head(
            fpn_features,
            num_classes,
            64,  # bbox_len
        )
        outputs = layers.Activation(
            "linear", dtype="float32", name="outputs_fp32"
        )(outputs)
        boxes, scores = outputs[:, :, :64], outputs[:, :, 64:]

        outputs = {"boxes": boxes, "classes": scores}
        super().__init__(inputs=images, outputs=outputs, **kwargs)

        self.bounding_box_format = bounding_box_format
        self.prediction_decoder = (
            prediction_decoder
            or keras_cv.layers.MultiClassNonMaxSuppression(
                bounding_box_format=bounding_box_format,
                from_logits=False,
                confidence_threshold=0.3,
                iou_threshold=0.5,
            )
        )
        self.backbone = backbone
        self.num_classes = num_classes

    def decode_predictions(
        self,
        pred,
        images,
    ):
        boxes = pred["boxes"]
        scores = pred["classes"]

        boxes = decode_regression_to_boxes(boxes, 64 // 4)

        anchors = get_anchors(image_shape=images.shape[1:])
        anchors = tf.concat(tf.nest.flatten(anchors), axis=0)

        decoded_boxes = decode_boxes(boxes, anchors)
        decoded_boxes = bounding_box.convert_format(
            decoded_boxes,
            source="rel_yxyx",
            target=self.bounding_box_format,
            images=images,
        )

        return self.prediction_decoder(decoded_boxes, scores)

    def make_predict_function(self, force=False):
        return predict_utils.make_predict_function(self, force=force)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return copy.deepcopy({**backbone_presets, **yolo_v8_detector_presets})

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return copy.deepcopy(
            {**backbone_presets_with_weights, **yolo_v8_detector_presets}
        )

    @classproperty
    def backbone_presets(cls):
        """Dictionary of preset names and configurations of compatible
        backbones."""
        return copy.deepcopy(backbone_presets)
