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

import tensorflow as tf
from tensorflow.keras import layers

from keras_cv.layers.spatial_pyramid import SpatialPyramidPooling
from keras_cv.models import utils
from keras_cv.models.resnet_v2 import ResNetV2
from keras_cv.models.segmentation.__internal__ import SegmentationHead
from keras_cv.models.weights import parse_weights

BACKBONE_CONFIG = {
    "ResNet50V2": {
        "stackwise_filters": [64, 128, 256, 512],
        "stackwise_blocks": [3, 4, 6, 3],
        "stackwise_strides": [1, 2, 2, 2],
        "stackwise_dilations": [1, 1, 1, 2],
    }
}


def DeepLabV3(
    classes,
    include_rescaling,
    backbone,
    backbone_weights=None,
    spatial_pyramid_pooling=None,
    segmentation_head=None,
    segmentation_head_activation="softmax",
    name=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    **kwargs,
):

    """
    A segmentation model based on the DeepLab v3.

    Args:
        classes: int, the number of classes for the detection model. Note that
            the classes doesn't contain the background class, and the classes
            from the data should be represented by integers with range
            [0, classes).
        include_rescaling: boolean, whether to Rescale the inputs. If set to True,
            inputs will be passed through a `Rescaling(1/255.0)` layer.
        backbone: an optional backbone network for the model. Can be a `tf.keras.layers.Layer`
            instance. The supported pre-defined backbone models are:
            1. "resnet50_v2", a ResNet50 V2 model
            Default to 'resnet50_v2'.
        backbone_weights: weights for the backbone model. one of `None` (random
            initialization), a pretrained weight file path, or a reference to
            pre-trained weights (e.g. 'imagenet/classification') (see available
            pre-trained weights in weights.py)
        weights: weights for the complete DeepLabV3 model. one of `None` (random
            initialization), a pretrained weight file path, or a reference to
            pre-trained weights (e.g. 'imagenet/classification') (see available
            pre-trained weights in weights.py)
        decoder: an optional decoder network for segmentation model, e.g. FPN. The
            supported premade decoder is: "fpn". The decoder is called on
            the output of the backbone network to up-sample the feature output.
            Default to 'fpn'.
        segmentation_head: an optional `tf.keras.Layer` that predict the segmentation
            mask based on feature from backbone and feature from decoder.
    """

    if backbone_weights and not tf.io.gfile.exists(backbone_weights):
        raise ValueError(
            "The `weights` argument should be either `None` or the path to the "
            "weights file to be loaded. Weights file not found at location: {weights}"
        )

    inputs = utils.parse_model_inputs(input_shape, input_tensor)
    height = input_shape[0]
    width = input_shape[1]

    x = inputs

    if include_rescaling:
        x = layers.Rescaling(1 / 255.0)(x)

    if isinstance(backbone, str):
        supported_premade_backbone = [
            "resnet50_v2",
        ]
        if backbone not in supported_premade_backbone:
            raise ValueError(
                "Supported premade backbones are: "
                f'{supported_premade_backbone}, received "{backbone}"'
            )

        if backbone == "resnet50_v2":
            backbone = get_resnet_backbone(
                backbone_weights, include_rescaling, **kwargs
            )

    else:
        # TODO(scottzhu): Might need to do more assertion about the model
        # What else do we want to test for? Shapes? This feels like too little, but
        # more assertions feel like they'd be limiting.
        if not isinstance(backbone, tf.keras.layers.Layer):
            raise ValueError(
                "Backbone need to be a `tf.keras.layers.Layer`, " f"received {backbone}"
            )

    feature_map = backbone(x)
    output = SpatialPyramidPooling(dilation_rates=[6, 12, 18])(feature_map)
    output = tf.keras.layers.UpSampling2D(
        size=(height // feature_map.shape[1], width // feature_map.shape[2]),
        interpolation="bilinear",
    )(output)

    if segmentation_head is None:
        segmentation_head = SegmentationHead(classes=classes, name="segmentation_head")

    # Segmentation head expects a multiple-level output dictionary
    output = segmentation_head({1: output})
    if segmentation_head_activation is not None:
        output = layers.Activation(segmentation_head_activation, name="top_activation")(
            output
        )

    model = tf.keras.Model(inputs, output, name=name, **kwargs)

    if backbone_weights is not None:
        backbone.load_weights(backbone_weights)

    return model


def get_resnet_backbone(backbone_weights, include_rescaling, **kwargs):
    return ResNetV2(
        stackwise_filters=BACKBONE_CONFIG["ResNet50V2"]["stackwise_filters"],
        stackwise_blocks=BACKBONE_CONFIG["ResNet50V2"]["stackwise_blocks"],
        stackwise_strides=BACKBONE_CONFIG["ResNet50V2"]["stackwise_strides"],
        stackwise_dilations=BACKBONE_CONFIG["ResNet50V2"]["stackwise_dilations"],
        include_rescaling=include_rescaling,
        include_top=False,
        name="resnet50v2",
        weights=parse_weights(backbone_weights, False, "resnet50v2"),
        pooling=None,
        **kwargs,
    )
