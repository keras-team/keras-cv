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
from keras_cv.models.segmentation.segmentation_head import SegmentationHead
from keras_cv.models.weights import parse_weights


@keras.utils.register_keras_serializable(package="keras_cv")
class DeepLabV3(keras.Model):
    """
    A segmentation model based on the DeepLabV3
        References:
        - [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587)

    Args:
        classes: int, the number of classes for the detection model. Note that
            the classes doesn't contain the background class, and the classes
            from the data should be represented by integers with range
            [0, classes).
        backbone: an optional backbone network for the model. Should be a KerasCV model.
        weights: weights for the complete DeepLabV3 model. one of `None` (random
            initialization), a pretrained weight file path, or a reference to
            pre-trained weights (e.g. 'imagenet/classification' or 'voc/segmentation') (see available
            pre-trained weights in weights.py)
        spatial_pyramid_pooling: also known as Atrous Spatial Pyramid Pooling (ASPP).
            Performs spatial pooling on different spatial levels in the pyramid, with
            dilation.
        segmentation_head: an optional `tf.keras.Layer` that predict the segmentation
            mask based on feature from backbone and feature from decoder.
    """

    def build(self, input_shape):
        height = input_shape[1]
        width = input_shape[2]
        feature_map_shape = self.backbone.compute_output_shape(input_shape)
        self.up_layer = tf.keras.layers.UpSampling2D(
            size=(height // feature_map_shape[1], width // feature_map_shape[2]),
            interpolation="bilinear",
        )

    def __init__(
        self,
        classes,
        backbone,
        spatial_pyramid_pooling=None,
        segmentation_head=None,
        segmentation_head_activation="softmax",
        input_shape=(None, None, 3),
        input_tensor=None,
        weights=None,
        **kwargs,
    ):

        if not isinstance(backbone, tf.keras.layers.Layer):
            raise ValueError(
                "Backbone need to be a `tf.keras.layers.Layer`, " f"received {backbone}"
            )

        if weights and not tf.io.gfile.exists(
            parse_weights(weights, True, "deeplabv3")
        ):
            raise ValueError(
                "The `weights` argument should be either `None` or the path to the "
                "weights file to be loaded. Weights file not found at location: {weights}"
            )

        inputs = utils.parse_model_inputs(input_shape, input_tensor)

        if input_shape[0] is None and input_shape[1] is None:
            input_shape = backbone.input_shape[1:]
            inputs = layers.Input(tensor=input_tensor, shape=input_shape)

        if input_shape[0] is None and input_shape[1] is None:
            raise ValueError(
                "Input shapes for both the backbone and DeepLabV3 are `None`."
            )

        height = input_shape[0]
        width = input_shape[1]

        x = inputs

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
                conv_type="conv2d",
                output_scale_factor=None,
                convs=1,
                dropout=0.2,
                kernel_size=1,
                activation=segmentation_head_activation,
            )

        # Segmentation head expects a multiple-level output dictionary
        output = segmentation_head(output)

        super().__init__(
            inputs={
                "inputs": inputs,
            },
            outputs={
                "output": output,
            },
            **kwargs,
        )

        if weights is not None:
            self.load_weights(parse_weights(weights, True, "deeplabv3"))

        # All references to `self` below this line
        self.classes = classes
        self.backbone = backbone
        self.spatial_pyramid_pooling = spatial_pyramid_pooling
        self.segmentation_head = segmentation_head
        self.segmentation_head_activation = segmentation_head_activation

    # TODO(tanzhenyu): consolidate how regularization should be applied to KerasCV.
    def compile(self, weight_decay=0.0001, **kwargs):
        self.weight_decay = weight_decay
        super().compile(**kwargs)

    def train_step(self, data):
        images, y_true, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            y_pred = self(images, training=True)
            total_loss = self.compute_loss(images, y_true, y_pred, sample_weight)
            reg_losses = []
            if self.weight_decay:
                for var in self.trainable_variables:
                    if "bn" not in var.name:
                        reg_losses.append(self.weight_decay * tf.nn.l2_loss(var))
                l2_loss = tf.math.add_n(reg_losses)
                total_loss += l2_loss
        self.optimizer.minimize(total_loss, self.trainable_variables, tape=tape)
        return self.compute_metrics(images, y_true, y_pred, sample_weight=sample_weight)

    def get_config(self):
        return {
            "classes": self.classes,
            "backbone": self.backbone,
            "spatial_pyramid_pooling": self.spatial_pyramid_pooling,
            "segmentation_head": self.segmentation_head,
            "segmentation_head_activation": self.segmentation_head_activation,
        }
