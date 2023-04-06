import math

import tensorflow as tf
from keras import initializers
from keras import layers
from keras import models
from keras.utils.data_utils import get_file
from keras_cv_attention_models.coco import anchors_func

from keras_cv.utils.train import get_feature_extractor

""" Yolov8Backbone """
BATCH_NORM_EPSILON = 1e-3
BATCH_NORM_MOMENTUM = 0.97


def conv_bn(
    inputs,
    output_channel,
    kernel_size=1,
    strides=1,
    activation="swish",
    name="",
):
    if kernel_size > 1:
        inputs = layers.ZeroPadding2D(
            padding=kernel_size // 2, name=name and name + "pad"
        )(inputs)

    nn = layers.Conv2D(
        filters=output_channel,
        kernel_size=kernel_size,
        strides=strides,
        padding="VALID",
        use_bias=False,
        name=name and name + "conv",
    )(inputs)
    nn = layers.BatchNormalization(
        momentum=BATCH_NORM_MOMENTUM,
        epsilon=BATCH_NORM_EPSILON,
        name=name and name + "bn",
    )(nn)
    nn = layers.Activation(activation, name=name)(nn)
    return nn


def csp_with_2_conv(
    inputs,
    channels=-1,
    depth=2,
    shortcut=True,
    expansion=0.5,
    activation="swish",
    name="",
):
    channel_axis = -1
    channels = channels if channels > 0 else inputs.shape[channel_axis]
    hidden_channels = int(channels * expansion)

    pre = conv_bn(
        inputs,
        hidden_channels * 2,
        kernel_size=1,
        activation=activation,
        name=name + "pre_",
    )
    short, deep = tf.split(pre, 2, axis=channel_axis)

    out = [short, deep]
    for id in range(depth):
        deep = conv_bn(
            deep,
            hidden_channels,
            kernel_size=3,
            activation=activation,
            name=name + "pre_{}_1_".format(id),
        )
        deep = conv_bn(
            deep,
            hidden_channels,
            kernel_size=3,
            activation=activation,
            name=name + "pre_{}_2_".format(id),
        )
        deep = (out[-1] + deep) if shortcut else deep
        out.append(deep)
    out = tf.concat(out, axis=channel_axis)
    out = conv_bn(
        out,
        channels,
        kernel_size=1,
        activation=activation,
        name=name + "output_",
    )
    return out


def spatial_pyramid_pooling_fast(
    inputs, pool_size=5, activation="swish", name=""
):
    channel_axis = -1
    input_channels = inputs.shape[channel_axis]
    hidden_channels = int(input_channels // 2)

    nn = conv_bn(
        inputs,
        hidden_channels,
        kernel_size=1,
        activation=activation,
        name=name + "pre_",
    )
    pool_1 = layers.MaxPool2D(pool_size=pool_size, strides=1, padding="SAME")(
        nn
    )
    pool_2 = layers.MaxPool2D(pool_size=pool_size, strides=1, padding="SAME")(
        pool_1
    )
    pool_3 = layers.MaxPool2D(pool_size=pool_size, strides=1, padding="SAME")(
        pool_2
    )

    out = tf.concat([nn, pool_1, pool_2, pool_3], axis=channel_axis)
    out = conv_bn(
        out,
        input_channels,
        kernel_size=1,
        activation=activation,
        name=name + "output_",
    )
    return out


def YOLOV8Backbone(
    include_rescaling,
    channels=[128, 256, 512, 1024],
    depths=[3, 6, 6, 3],
    input_shape=(512, 512, 3),
    activation="swish",
    model_name="yolov8_backbone",
):
    inputs = layers.Input(input_shape)

    x = inputs
    if include_rescaling:
        x = layers.Rescaling(1 / 255.0)(x)

    """ Stem """
    # stem_width = stem_width if stem_width > 0 else channels[0]
    stem_width = channels[0]
    nn = conv_bn(
        x,
        stem_width // 2,
        kernel_size=3,
        strides=2,
        activation=activation,
        name="stem_1_",
    )
    nn = conv_bn(
        nn,
        stem_width,
        kernel_size=3,
        strides=2,
        activation=activation,
        name="stem_2_",
    )

    """ blocks """
    pyramid_level_inputs = {0: nn.node.layer.name}
    features = {0: nn}
    for stack_id, (channel, depth) in enumerate(zip(channels, depths)):
        stack_name = "stack{}_".format(stack_id + 1)
        if stack_id >= 1:
            nn = conv_bn(
                nn,
                channel,
                kernel_size=3,
                strides=2,
                activation=activation,
                name=stack_name + "downsample_",
            )
        nn = csp_with_2_conv(
            nn,
            depth=depth,
            expansion=0.5,
            activation=activation,
            name=stack_name + "c2f_",
        )

        if stack_id == len(depths) - 1:
            nn = spatial_pyramid_pooling_fast(
                nn,
                pool_size=5,
                activation=activation,
                name=stack_name + "spp_fast_",
            )
        pyramid_level_inputs[stack_id + 1] = nn.node.layer.name
        features[stack_id + 1] = nn

    model = models.Model(inputs, features, name=model_name)
    model.pyramid_level_inputs = pyramid_level_inputs
    return model

def path_aggregation_fpn(features, depth=3, name=""):
    # yolov8
    # 9: p5 1024 ---+----------------------+-> 21: out2 1024
    #               v [up 1024 -> concat]  ^ [down 512 -> concat]
    # 6: p4 512 --> 12: p4p5 512 --------> 18: out1 512
    #               v [up 512 -> concat]   ^ [down 256 -> concat]
    # 4: p3 256 --> 15: p3p4p5 256 --------+--> 15: out0 128
    # features: [p3, p4, p5]
    channel_axis = -1
    upsamples = [features[-1]]
    p_name = "p{}_".format(len(features) + 2)
    # upsamples: [p5], features[:-1][::-1]: [p4, p3] -> [p5, p4p5, p3p4p5]
    for id, feature in enumerate(features[:-1][::-1]):
        cur_p_name = "p{}".format(len(features) + 1 - id)
        p_name = cur_p_name + p_name
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
            name=name + p_name,
        )
        upsamples.append(nn)

    downsamples = [upsamples[-1]]
    # downsamples: [p3p4p5], upsamples[:-1][::-1]: [p4p5, p5] -> [p3p4p5, p3p4p5 + p4p5, p3p4p5 + p4p5 + p5]
    for id, ii in enumerate(upsamples[:-1][::-1]):
        cur_name = name + "c3n{}_".format(id + 3)
        nn = conv_bn(
            downsamples[-1],
            downsamples[-1].shape[channel_axis],
            kernel_size=3,
            strides=2,
            activation="swish",
            name=cur_name + "down_",
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
    num_anchors=1,
    use_object_scores=False,
    name="",
):
    outputs = []
    reg_channels = max(16, bbox_len, inputs[0].shape[-1] // 4)
    cls_channels = max(num_classes, inputs[0].shape[-1])
    for id, feature in enumerate(inputs):
        cur_name = name + "{}_".format(id + 1)

        reg_nn = conv_bn(
            feature,
            reg_channels,
            3,
            activation="swish",
            name=cur_name + "reg_1_",
        )
        reg_nn = conv_bn(
            reg_nn,
            reg_channels,
            3,
            activation="swish",
            name=cur_name + "reg_2_",
        )
        reg_out = layers.Conv2D(
            filters=bbox_len * num_anchors,
            kernel_size=1,
            name=cur_name + "reg_3_" + "conv",
        )(reg_nn)

        cls_nn = conv_bn(
            feature,
            cls_channels,
            3,
            activation="swish",
            name=cur_name + "cls_1_",
        )
        cls_nn = conv_bn(
            cls_nn,
            cls_channels,
            3,
            activation="swish",
            name=cur_name + "cls_2_",
        )
        cls_out = layers.Conv2D(
            filters=num_classes * num_anchors,
            kernel_size=1,
            name=cur_name + "cls_3_" + "conv",
        )(cls_nn)
        cls_out = layers.Activation("sigmoid", name=cur_name + "classifier_")(
            cls_out
        )

        # obj_preds
        if use_object_scores:
            obj_out = layers.Conv2D(
                filters=num_anchors,
                kernel_size=1,
                bias_initializer=initializers.constant(
                    -math.log((1 - 0.01) / 0.01)
                ),
                name=cur_name + "object_" + "conv",
            )(reg_nn)
            obj_out = layers.Activation("sigmoid", name=name + "object_out_")(
                obj_out
            )
            out = tf.concat([reg_out, cls_out, obj_out], axis=-1)
        else:
            out = tf.concat([reg_out, cls_out], axis=-1)
        out = layers.Reshape(
            [-1, out.shape[-1]], name=cur_name + "output_reshape"
        )(out)
        outputs.append(out)

    outputs = tf.concat(outputs, axis=1)
    return outputs


def __decode_single__(pred, score_threshold=0.3, iou_or_sigma=0.5, max_output_size=100, method="hard", mode="global", topk=0, input_shape=None):
    # https://github.com/google/automl/tree/master/efficientdet/tf2/postprocess.py#L159
    pred = tf.cast(pred, tf.float32)

    boxes, scores = pred[:, :64], pred[:, 64:]
    ccs, labels = tf.reduce_max(scores, axis=-1), tf.argmax(scores, axis=-1)

    anchors = anchors_func.get_anchor_free_anchors((640, 640, 3), pyramid_levels=[3, 5], grid_zero_start=False)

    decoded_boxes = anchors_func.yolov8_decode_bboxes(boxes, anchors, regression_max=64 // 4)

    iou_threshold, soft_nms_sigma = (1.0, iou_or_sigma / 2) if method.lower() == "gaussian" else (iou_or_sigma, 0.0)

    rr, nms_scores = tf.image.non_max_suppression_with_scores(decoded_boxes, ccs, max_output_size, iou_threshold, score_threshold, soft_nms_sigma)
    decoded_boxes, labels, confidence = tf.gather(decoded_boxes, rr), tf.gather(labels, rr), nms_scores

    return decoded_boxes, labels, confidence

""" YOLOV8 models """


def YOLOV8(
    backbone,
    depths=[1, 2, 2, 1],
    extractor_levels=[2, 3, 4],  # [Detector parameters]
    bbox_len=64,  # Typical value is 4, for yolov8 reg_max=16 -> bbox_len = 16 * 4 == 64
    num_anchors="auto",  # "auto" means: anchors_mode=="anchor_free" -> 1, anchors_mode=="yolor" -> 3, else 9
    use_object_scores="auto",  # "auto" means: True if anchors_mode=="anchor_free" or anchors_mode=="yolor", else False
    num_classes=80,
    model_name="yolov8",
    pyramid_levels_min=3,  # Init anchors for model prediction.
    anchor_scale="auto",  # Init anchors for model prediction. "auto" means 1 if (anchors_mode=="anchor_free" or anchors_mode=="yolor"), else 4
    rescale_mode="raw01",  # For decode predictions, raw01 means input value in range [0, 1].
):
    extractor_layer_names = [
        backbone.pyramid_level_inputs[i] for i in extractor_levels
    ]
    feature_extractor = get_feature_extractor(
        backbone, extractor_layer_names, extractor_levels
    )

    images = layers.Input(feature_extractor.input_shape[1:])
    features = list(feature_extractor(images).values())

    (
        use_object_scores,
        num_anchors,
        anchor_scale,
    ) = anchors_func.get_anchors_mode_parameters(
        "yolov8", use_object_scores, num_anchors, anchor_scale
    )

    # fpn = FeaturePyramid(depth=depths[-1])
    # fpn_features = fpn(features)
    fpn_features = path_aggregation_fpn(
        features, depth=depths[-1], name="pafpn_"
    )

    outputs = yolov8_head(
        fpn_features,
        num_classes,
        bbox_len,
        num_anchors,
        use_object_scores,
        name="head_",
    )
    outputs = layers.Activation("linear", dtype="float32", name="outputs_fp32")(
        outputs
    )
    model = models.Model(images, outputs, name=model_name)
    model.load_weights("kcv-converted.h5")

    model.decode = __decode_single__
    return model


def YOLOv8_N(
    input_shape=(640, 640, 3),
    num_classes=80,
):
    return YOLOV8(
        backbone=YOLOV8Backbone(
            include_rescaling=True,
            input_shape=input_shape,
            channels=[32, 64, 128, 256],
            depths=[1, 2, 2, 1],
        ),
        num_classes=num_classes,
        model_name="yolov8_n",
    )


def YOLOV8_S(
    input_shape=(640, 640, 3),
    num_classes=80,
    backbone=None,
):
    csp_channels = [64, 128, 256, 512]
    return YOLOV8(**locals(), model_name="yolov8_s")


def YOLOV8_M(
    input_shape=(640, 640, 3),
    num_classes=80,
    backbone=None,
):
    csp_channels = [96, 192, 384, 768]
    depths = [2, 4, 4, 2]
    return YOLOV8(**locals(), model_name="yolov8_m")


def YOLOV8_L(
    input_shape=(640, 640, 3),
    num_classes=80,
    backbone=None,
):
    csp_channels = [128, 256, 512, 512]
    depths = [3, 6, 6, 3]
    return YOLOV8(**locals(), model_name="yolov8_l")


def YOLOV8_X(
    input_shape=(640, 640, 3),
    num_classes=80,
    backbone=None,
):
    csp_channels = [160, 320, 640, 640]
    depths = [3, 6, 6, 3]
    return YOLOV8(**locals(), model_name="yolov8_x")


def YOLOV8_X6(
    input_shape=(640, 640, 3),
    num_classes=80,
    backbone=None,
):
    csp_channels = [160, 320, 640, 640, 640]
    depths = [3, 6, 6, 3, 3]
    extractor_levels = [2, 3, 4, 5]
    return YOLOV8(**locals(), model_name="yolov8_x6")
