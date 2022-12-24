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
    "ResNet50V2": {
        "stackwise_filters": [64, 128, 256, 512],
        "stackwise_blocks": [3, 4, 6, 3],
        "stackwise_strides": [1, 2, 2, 2],
        "stackwise_dilations": [1, 1, 1, 2],
    }
}


@keras.utils.register_keras_serializable(package="keras_cv")
class DeepLabV3(keras.Model):
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
        spatial_pyramid_pooling: also known as Atrous Spatial Pyramid Pooling (ASPP).
            Performs spatial pooling on different spatial levels in the pyramid, with
            dilation.
        segmentation_head: an optional `tf.keras.Layer` that predict the segmentation
            mask based on feature from backbone and feature from decoder.
    """

    def __init__(
        self,
        classes,
        include_rescaling,
        backbone,
        backbone_weights=None,
        spatial_pyramid_pooling=None,
        segmentation_head=None,
        segmentation_head_activation=None,
        input_shape=(None, None, 3),
        input_tensor=None,
        **kwargs,
    ):

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
                if backbone_weights and input_shape[-1] == 1:
                    raise ValueError(
                        "The input shape is set up for greyscale images with one channel, but backbone weights are trained on colored images and cannot be loaded."
                    )
                backbone = get_resnet_backbone(
                    backbone_weights, include_rescaling, input_shape, **kwargs
                )

        else:
            if not isinstance(backbone, tf.keras.layers.Layer):
                raise ValueError(
                    "Backbone need to be a `tf.keras.layers.Layer`, "
                    f"received {backbone}"
                )

        feature_map = backbone(x)
        if spatial_pyramid_pooling is None:
            spatial_pyramid_pooling = SpatialPyramidPooling(dilation_rates=[6, 12, 18])

        output = spatial_pyramid_pooling(feature_map)
        output = tf.keras.layers.UpSampling2D(
            size=(height // feature_map.shape[1], width // feature_map.shape[2]),
            interpolation="bilinear",
        )(output)

        if segmentation_head is None:
            segmentation_head = SegmentationHead(
                classes=classes,
                name="segmentation_head",
                convs=1,
                dropout=0.2,
                kernel_size=1,
                output_activation=segmentation_head_activation,
            )

        # Segmentation head expects a multiple-level output dictionary
        output = segmentation_head({1: output})

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

    def get_config(self):
        return {
            "classes": self.classes,
            "include_rescaling": self.include_rescaling,
            "backbone": self.backbone,
            "backbone_weights": self.backbone_weights,
            "spatial_pyramid_pooling": self.spatial_pyramid_pooling,
            "segmentation_head": self.segmentation_head,
            "segmentation_head_activation": self.segmentation_head_activation,
        }


def get_resnet_backbone(backbone_weights, include_rescaling, input_shape, **kwargs):
    return ResNetV2(
        stackwise_filters=BACKBONE_CONFIG["ResNet50V2"]["stackwise_filters"],
        stackwise_blocks=BACKBONE_CONFIG["ResNet50V2"]["stackwise_blocks"],
        stackwise_strides=BACKBONE_CONFIG["ResNet50V2"]["stackwise_strides"],
        stackwise_dilations=BACKBONE_CONFIG["ResNet50V2"]["stackwise_dilations"],
        include_rescaling=include_rescaling,
        include_top=False,
        name="resnet50v2",
        input_shape=input_shape,
        weights=parse_weights(backbone_weights, False, "resnet50v2"),
        pooling=None,
        **kwargs,
    )
