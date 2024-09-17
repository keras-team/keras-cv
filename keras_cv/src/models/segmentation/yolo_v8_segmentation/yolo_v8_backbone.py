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
from keras_cv.src.backend import keras
from keras_cv.src.backend import ops

BATCH_NORM_EPSILON = 1e-3
BATCH_NORM_MOMENTUM = 0.97
BOX_REGRESSION_CHANNELS = 64


def apply_conv_bn(
    inputs,
    output_channel,
    kernel_size=1,
    strides=1,
    activation="swish",
    name="conv_bn",
):
    if kernel_size > 1:
        inputs = keras.layers.ZeroPadding2D(
            padding=kernel_size // 2, name=f"{name}_pad"
        )(inputs)

    x = keras.layers.Conv2D(
        filters=output_channel,
        kernel_size=kernel_size,
        strides=strides,
        padding="valid",
        use_bias=False,
        name=f"{name}_conv",
    )(inputs)
    x = keras.layers.BatchNormalization(
        momentum=BATCH_NORM_MOMENTUM,
        epsilon=BATCH_NORM_EPSILON,
        name=f"{name}_bn",
    )(x)
    x = keras.layers.Activation(activation, name=name)(x)
    return x


# TODO(ianstenbit): Remove this method once we're using CSPDarkNet backbone
# Calls to it should instead call the CSP block from the DarkNet implementation.
def apply_csp_block(
    inputs,
    channels=-1,
    depth=2,
    shortcut=True,
    expansion=0.5,
    activation="swish",
    name="csp_block",
):
    channel_axis = -1
    channels = channels if channels > 0 else inputs.shape[channel_axis]
    hidden_channels = int(channels * expansion)

    pre = apply_conv_bn(
        inputs,
        hidden_channels * 2,
        kernel_size=1,
        activation=activation,
        name=f"{name}_pre",
    )
    short, deep = ops.split(pre, 2, axis=channel_axis)

    out = [short, deep]
    for id in range(depth):
        deep = apply_conv_bn(
            deep,
            hidden_channels,
            kernel_size=3,
            activation=activation,
            name=f"{name}_pre_{id}_1",
        )
        deep = apply_conv_bn(
            deep,
            hidden_channels,
            kernel_size=3,
            activation=activation,
            name=f"{name}_pre_{id}_2",
        )
        deep = (out[-1] + deep) if shortcut else deep
        out.append(deep)
    out = ops.concatenate(out, axis=channel_axis)
    out = apply_conv_bn(
        out,
        channels,
        kernel_size=1,
        activation=activation,
        name=f"{name}_output",
    )
    return out


def get_anchors(
    image_shape,
    strides=[8, 16, 32],
    base_anchors=[0.5, 0.5],
):
    """Gets anchor points for YOLOV8.

    YOLOV8 uses anchor points representing the center of proposed boxes, and
    matches ground truth boxes to anchors based on center points.

    Args:
        image_shape: tuple or list of two integers representing the height and
            width of input images, respectively.
        strides: tuple of list of integers, the size of the strides across the
            image size that should be used to create anchors.
        base_anchors: tuple or list of two integers representing the offset from
            (0,0) to start creating the center of anchor boxes, relative to the
            stride. For example, using the default (0.5, 0.5) creates the first
            anchor box for each stride such that its center is half of a stride
            from the edge of the image.

    Returns:
        A tuple of anchor centerpoints and anchor strides. Multiplying the
        two together will yield the centerpoints in absolute x,y format.

    """
    base_anchors = ops.array(base_anchors, dtype="float32")

    all_anchors = []
    all_strides = []
    for stride in strides:
        hh_centers = ops.arange(0, image_shape[0], stride)
        ww_centers = ops.arange(0, image_shape[1], stride)
        ww_grid, hh_grid = ops.meshgrid(ww_centers, hh_centers)
        grid = ops.cast(
            ops.reshape(ops.stack([hh_grid, ww_grid], 2), [-1, 1, 2]),
            "float32",
        )
        anchors = (
            ops.expand_dims(
                base_anchors * ops.array([stride, stride], "float32"), 0
            )
            + grid
        )
        anchors = ops.reshape(anchors, [-1, 2])
        all_anchors.append(anchors)
        all_strides.append(ops.repeat(stride, anchors.shape[0]))

    all_anchors = ops.cast(ops.concatenate(all_anchors, axis=0), "float32")
    all_strides = ops.cast(ops.concatenate(all_strides, axis=0), "float32")

    all_anchors = all_anchors / all_strides[:, None]

    # Swap the x and y coordinates of the anchors.
    all_anchors = ops.concatenate(
        [all_anchors[:, 1, None], all_anchors[:, 0, None]], axis=-1
    )
    return all_anchors, all_strides


def apply_path_aggregation_fpn(features, depth=3, name="fpn"):
    """Applies the Feature Pyramid Network (FPN) to the outputs of a backbone.

    Args:
        features: list of tensors representing the P3, P4, and P5 outputs of the
            backbone.
        depth: integer, the depth of the CSP blocks used in the FPN.
        name: string, a prefix for names of layers used by the FPN.

    Returns:
        A list of three tensors whose shapes are the same as the three inputs,
        but which are dependent on each of the three inputs to combine the high
        resolution of the P3 inputs with the strong feature representations of
        the P5 inputs.

    """
    p3, p4, p5 = features

    # Upsample P5 and concatenate with P4, then apply a CSPBlock.
    p5_upsampled = ops.repeat(ops.repeat(p5, 2, axis=1), 2, axis=2)
    p4p5 = ops.concatenate([p5_upsampled, p4], axis=-1)
    p4p5 = apply_csp_block(
        p4p5,
        channels=p4.shape[-1],
        depth=depth,
        shortcut=False,
        activation="swish",
        name=f"{name}_p4p5",
    )

    # Upsample P4P5 and concatenate with P3, then apply a CSPBlock.
    p4p5_upsampled = ops.repeat(ops.repeat(p4p5, 2, axis=1), 2, axis=2)
    p3p4p5 = ops.concatenate([p4p5_upsampled, p3], axis=-1)
    p3p4p5 = apply_csp_block(
        p3p4p5,
        channels=p3.shape[-1],
        depth=depth,
        shortcut=False,
        activation="swish",
        name=f"{name}_p3p4p5",
    )

    # Downsample P3P4P5, concatenate with P4P5, and apply a CSP Block.
    p3p4p5_d1 = apply_conv_bn(
        p3p4p5,
        p3p4p5.shape[-1],
        kernel_size=3,
        strides=2,
        activation="swish",
        name=f"{name}_p3p4p5_downsample1",
    )
    p3p4p5_d1 = ops.concatenate([p3p4p5_d1, p4p5], axis=-1)
    p3p4p5_d1 = apply_csp_block(
        p3p4p5_d1,
        channels=p4p5.shape[-1],
        shortcut=False,
        activation="swish",
        name=f"{name}_p3p4p5_downsample1_block",
    )

    # Downsample the resulting P3P4P5 again, concatenate with P5, and apply
    # another CSP Block.
    p3p4p5_d2 = apply_conv_bn(
        p3p4p5_d1,
        p3p4p5_d1.shape[-1],
        kernel_size=3,
        strides=2,
        activation="swish",
        name=f"{name}_p3p4p5_downsample2",
    )
    p3p4p5_d2 = ops.concatenate([p3p4p5_d2, p5], axis=-1)
    p3p4p5_d2 = apply_csp_block(
        p3p4p5_d2,
        channels=p5.shape[-1],
        shortcut=False,
        activation="swish",
        name=f"{name}_p3p4p5_downsample2_block",
    )

    return [p3p4p5, p3p4p5_d1, p3p4p5_d2]


def decode_regression_to_boxes(preds):
    """Decodes the results of the YOLOV8Detector forward-pass into boxes.

    Returns left / top / right / bottom predictions with respect to anchor
    points.

    Each coordinate is encoded with 16 predicted values. Those predictions are
    softmaxed and multiplied by [0..15] to make predictions. The resulting
    predictions are relative to the stride of an anchor box (and correspondingly
    relative to the scale of the feature map from which the predictions came).
    """
    preds_bbox = keras.layers.Reshape((-1, 4, BOX_REGRESSION_CHANNELS // 4))(
        preds
    )
    preds_bbox = ops.nn.softmax(preds_bbox, axis=-1) * ops.arange(
        BOX_REGRESSION_CHANNELS // 4, dtype="float32"
    )
    return ops.sum(preds_bbox, axis=-1)


def dist2bbox(distance, anchor_points):
    """Decodes distance predictions into xyxy boxes.

    Input left / top / right / bottom predictions are transformed into xyxy box
    predictions based on anchor points.

    The resulting xyxy predictions must be scaled by the stride of their
    corresponding anchor points to yield an absolute xyxy box.
    """
    left_top, right_bottom = ops.split(distance, 2, axis=-1)
    x1y1 = anchor_points - left_top
    x2y2 = anchor_points + right_bottom
    return ops.concatenate((x1y1, x2y2), axis=-1)  # xyxy bbox


def apply_yolo_v8_head(
    inputs,
    num_classes,
    name="yolo_v8_head",
):
    """Applies a YOLOV8 head.

    Makes box and class predictions based on the output of a feature pyramid
    network.

    Args:
        inputs: list of tensors output by the Feature Pyramid Network, should
            have the same shape as the P3, P4, and P5 outputs of the backbone.
        num_classes: integer, the number of classes that a bounding box could
            possibly be assigned to.
        name: string, a prefix for names of layers used by the head.

    Returns: A dictionary with two entries. The "boxes" entry contains box
        regression predictions, while the "classes" entry contains class
        predictions.
    """
    # 64 is the default number of channels, as 16 components are used to predict
    # each of the 4 offsets for corner points of a bounding box with respect
    # to the center point. In cases where the input has much higher resolution
    # (e.g. the P3 input has >256 channels), we use additional channels for
    # the intermediate conv layers. This is only true for very large backbones.
    box_channels = max(BOX_REGRESSION_CHANNELS, inputs[0].shape[-1] // 4)

    # We use at least num_classes channels for intermediate conv layer for class
    # predictions. In most cases, the P3 input has many more channels than the
    # number of classes, so we preserve those channels until the final layer.
    class_channels = max(num_classes, inputs[0].shape[-1])

    # We compute box and class predictions for each of the feature maps from
    # the FPN and then combine them.
    outputs = []
    for id, feature in enumerate(inputs):
        cur_name = f"{name}_{id+1}"

        box_predictions = apply_conv_bn(
            feature,
            box_channels,
            kernel_size=3,
            activation="swish",
            name=f"{cur_name}_box_1",
        )
        box_predictions = apply_conv_bn(
            box_predictions,
            box_channels,
            kernel_size=3,
            activation="swish",
            name=f"{cur_name}_box_2",
        )
        box_predictions = keras.layers.Conv2D(
            filters=BOX_REGRESSION_CHANNELS,
            kernel_size=1,
            name=f"{cur_name}_box_3_conv",
        )(box_predictions)

        class_predictions = apply_conv_bn(
            feature,
            class_channels,
            kernel_size=3,
            activation="swish",
            name=f"{cur_name}_class_1",
        )
        class_predictions = apply_conv_bn(
            class_predictions,
            class_channels,
            kernel_size=3,
            activation="swish",
            name=f"{cur_name}_class_2",
        )
        class_predictions = keras.layers.Conv2D(
            filters=num_classes,
            kernel_size=1,
            name=f"{cur_name}_class_3_conv",
        )(class_predictions)
        class_predictions = keras.layers.Activation(
            "sigmoid", name=f"{cur_name}_classifier"
        )(class_predictions)

        out = ops.concatenate([box_predictions, class_predictions], axis=-1)
        out = keras.layers.Reshape(
            [-1, out.shape[-1]], name=f"{cur_name}_output_reshape"
        )(out)
        outputs.append(out)

    outputs = ops.concatenate(outputs, axis=1)
    outputs = keras.layers.Activation(
        "linear", dtype="float32", name="box_outputs"
    )(outputs)

    return {
        "boxes": outputs[:, :, :BOX_REGRESSION_CHANNELS],
        "classes": outputs[:, :, BOX_REGRESSION_CHANNELS:],
    }
