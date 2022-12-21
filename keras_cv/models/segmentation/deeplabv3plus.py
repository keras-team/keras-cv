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
from tensorflow import keras
from tensorflow.keras import layers

from keras_cv.layers.spatial_pyramid import SpatialPyramidPooling
from keras_cv.models import utils
from keras_cv.models.resnet_v2 import ResNetV2
from keras_cv.models.segmentation.__internal__ import SegmentationHead
from keras_cv.models.weights import parse_weights

BACKBONE_CONFIG = {
    "ResNet101V2": {
        "stackwise_filters": [64, 128, 256, 512],
        "stackwise_blocks": [3, 4, 23, 3],
        "stackwise_strides": [1, 2, 2, 2],
        "stackwise_dilations": [1, 1, 1, 2],
    }
}


@keras.utils.register_keras_serializable(package="keras_cv")
class DeepLabV3Plus(keras.Model):
    """
    A segmentation model based on the DeepLabV3Plus model.

    Args:
        classes: int, the number of classes for the detection model. Note that
            the classes doesn't contain the background class, and the classes
            from the data should be represented by integers with range
            [0, classes).
        include_rescaling: boolean, whether to Rescale the inputs. If set to True,
            inputs will be passed through a `Rescaling(1/255.0)` layer.
        backbone: an optional backbone network for the model. Can be a `tf.keras.layers.Layer`
            instance. The supported pre-defined backbone models are:
            1. "resnet101_v2", a ResNet101 V2 model
            Default to 'resnet101_v2'.
        backbone_weights: weights for the backbone model. one of `None` (random
            initialization), a pretrained weight file path, or a reference to
            pre-trained weights (e.g. 'imagenet/classification') (see available
            pre-trained weights in weights.py)
        weights: weights for the complete DeepLabV3Plus model. one of `None` (random
            initialization), a pretrained weight file path, or a reference to
            pre-trained weights (e.g. 'imagenet/classification') (see available
            pre-trained weights in weights.py)
        spatial_pyramid_pooling: also known as Atrous Spatial Pyramid Pooling (ASPP).
            Performs spatial pooling on different spatial levels in the pyramid, with
            dilation.
        segmentation_head: an optional `tf.keras.Layer` that predict the segmentation
            mask based on feature from backbone and feature from decoder.
        segmentation_head_activation: default 'softmax', the activation layer to apply after
            the segmentation head. Should be synchronized with the backbone's final activation.
        feature_layers: the layer names for the low-level features and high-level features
            to use for encoding/decoding spatial information for the supplied backbone.
    """

    def __init__(
        self,
        classes,
        include_rescaling,
        backbone,
        backbone_weights=None,
        spatial_pyramid_pooling=None,
        segmentation_head=None,
        segmentation_head_activation="softmax",
        input_shape=(None, None, 3),
        input_tensor=None,
        feature_layers=(None, None),
        **kwargs,
    ):

        inputs = utils.parse_model_inputs(input_shape, input_tensor)
        x = inputs

        if include_rescaling:
            x = layers.Rescaling(1 / 255.0)(x)

        if isinstance(backbone, str):
            supported_premade_backbone = [
                "resnet101_v2",
            ]
            if backbone not in supported_premade_backbone:
                raise ValueError(
                    "Supported premade backbones are: "
                    f'{supported_premade_backbone}, received "{backbone}"'
                )

            if backbone == "resnet101_v2":
                backbone = get_resnet_backbone(
                    backbone_weights, include_rescaling, input_tensor=x, **kwargs
                )
                if feature_layers != (None, None):
                    raise ValueError(
                        "When using a predefined backbone, you cannot set custom feature layers."
                        f"received {feature_layers}"
                    )
                low_level_features = backbone.get_layer(
                    "v2_stack_1_block4_1_relu"
                ).output
                high_level_features = backbone.get_layer(
                    "v2_stack_3_block2_2_relu"
                ).output

        else:
            # TODO(scottzhu): Might need to do more assertion about the model
            # What else do we want to test for? Shapes? This feels like too little, but
            # more assertions feel like they'd be limiting.
            if not isinstance(backbone, tf.keras.layers.Layer):
                raise ValueError(
                    "Backbone need to be a `tf.keras.layers.Layer`, "
                    f"received {backbone}"
                )
            low_level_features = backbone.get_layer(feature_layers[0]).output
            high_level_features = backbone.get_layer(feature_layers[1]).output

        if spatial_pyramid_pooling is None:
            spatial_pyramid_pooling = SpatialPyramidPooling(dilation_rates=[6, 12, 18])

        output = spatial_pyramid_pooling(high_level_features)
        output = tf.keras.layers.UpSampling2D(
            size=(4, 4),
            interpolation="bilinear",
        )(output)

        output = layers.Concatenate()([output, low_level_features])

        if segmentation_head is None:
            segmentation_head = SegmentationHead(
                classes=classes,
                name="segmentation_head",
                output_scale_factor=4,
                convs=1,
                dropout=0.2,
                kernel_size=1,
            )

        # Segmentation head expects a multiple-level output dictionary
        output = segmentation_head({1: output})

        if segmentation_head_activation is not None:
            # Force float32 output to avoid NaN issues with mixed-precision training
            output = layers.Activation(
                segmentation_head_activation, dtype=tf.float32, name="top_activation"
            )(output)

        super().__init__(
            inputs={
                "inputs": inputs,
            },
            outputs={
                "output": output,
            },
            **kwargs,
        )

        # All references to `self` below this line
        self.classes = classes
        self.include_rescaling = include_rescaling
        self.backbone = backbone
        self.backbone_weights = backbone_weights
        self.spatial_pyramid_pooling = spatial_pyramid_pooling
        self.segmentation_head = segmentation_head
        self.segmentation_head_activation = segmentation_head_activation
        self.feature_layers = feature_layers

        def get_config(self):
            return {
                "classes": self.classes,
                "include_rescaling": self.include_rescaling,
                "backbone": self.backbone,
                "backbone_weights": self.backbone_weights,
                "spatial_pyramid_pooling": self.spatial_pyramid_pooling,
                "segmentation_head": self.segmentation_head,
                "segmentation_head_activation": self.segmentation_head_activation,
                "feature_layers": self.feature_layers,
            }


def get_resnet_backbone(backbone_weights, include_rescaling, **kwargs):
    return ResNetV2(
        stackwise_filters=BACKBONE_CONFIG["ResNet101V2"]["stackwise_filters"],
        stackwise_blocks=BACKBONE_CONFIG["ResNet101V2"]["stackwise_blocks"],
        stackwise_strides=BACKBONE_CONFIG["ResNet101V2"]["stackwise_strides"],
        stackwise_dilations=BACKBONE_CONFIG["ResNet101V2"]["stackwise_dilations"],
        include_rescaling=include_rescaling,
        include_top=False,
        name="resnet101v2",
        weights=parse_weights(backbone_weights, False, "resnet101v2"),
        pooling=None,
        **kwargs,
    )
