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

from keras_cv.layers.segmentation.segmentation_head import SegmentationHead
from keras_cv.layers.spatial_pyramid import SpatialPyramidPooling
from keras_cv.models.legacy import utils
from keras_cv.models.task import Task


@keras.utils.register_keras_serializable(package="keras_cv")
class DeepLabV3(Task):
    """A Keras model implementing the DeepLabV3 architecture for semantic
    segmentation.

    References:
        - [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587)
        (CVPR 2017)

    Examples:
    ```python
    import tensorflow as tf
    import keras_cv

    images = tf.ones(shape=(1, 96, 96, 3))
    backbone = keras_cv.models.ResNet50V2Backbone(input_shape=[96, 96, 3])
    model = keras_cv.model.segmentation.DeepLabV3(
        num_classes=1, backbone=backbone
    )

    # Evaluate model
    model(images)

    # Train model
    model.compile(
        weight_decay=0.0001,
        optimizer="adam",
        loss=keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )
    model.fit(training_dataset.take(10), epochs=3)
    ```

    Args:
        num_classes: int, the number of classes for the detection model. Note
            that the num_classes doesn't contain the background class, and the
            classes from the data should be represented by integers with range
            [0, num_classes).
        backbone: Backbone network for the model. Should be a KerasCV model.
        spatial_pyramid_pooling: Also known as Atrous Spatial Pyramid Pooling
            (ASPP). Performs spatial pooling on different spatial levels in the
            pyramid, with dilation.
        segmentation_head: Optional `keras.Layer` that predict the segmentation
            mask based on feature from backbone and feature from decoder.
        segmentation_head_activation: Optional `str` or function, activation
            functions between the `keras.layers.Conv2D` layers and the final
            classification layer, defaults to `"relu"`.
        input_shape: optional shape tuple, defaults to (None, None, 3).
        input_tensor: optional Keras tensor (i.e., output of `layers.Input()`)
            to use as image input for the model.
    """

    def __init__(
        self,
        num_classes,
        backbone,
        spatial_pyramid_pooling=None,
        segmentation_head=None,
        segmentation_head_activation="softmax",
        input_shape=(None, None, 3),
        input_tensor=None,
        **kwargs,
    ):
        if not isinstance(backbone, keras.layers.Layer) or not isinstance(
            backbone, keras.Model
        ):
            raise ValueError(
                "Argument `backbone` must be a `keras.layers.Layer` instance "
                f" or `keras.Model`. Received instead "
                f"backbone={backbone} (of type {type(backbone)})."
            )

        inputs = utils.parse_model_inputs(input_shape, input_tensor)

        if input_shape[0] is None and input_shape[1] is None:
            input_shape = backbone.input_shape[1:]
            inputs = layers.Input(tensor=input_tensor, shape=input_shape)

        if input_shape[0] is None and input_shape[1] is None:
            raise ValueError(
                "Input shapes for both the backbone and DeepLabV3 cannot be "
                "`None`. Received: input_shape={input_shape} and "
                "backbone.input_shape={backbone.input_shape[1:]}"
            )

        height = input_shape[0]
        width = input_shape[1]

        feature_map = backbone(inputs)
        if spatial_pyramid_pooling is None:
            spatial_pyramid_pooling = SpatialPyramidPooling(
                dilation_rates=[6, 12, 18]
            )

        outputs = spatial_pyramid_pooling(feature_map)
        outputs = keras.layers.UpSampling2D(
            size=(
                height // feature_map.shape[1],
                width // feature_map.shape[2],
            ),
            interpolation="bilinear",
        )(outputs)

        if segmentation_head is None:
            segmentation_head = SegmentationHead(
                num_classes=num_classes,
                name="segmentation_head",
                convolutions=1,
                dropout=0.2,
                kernel_size=1,
                activation=segmentation_head_activation,
            )

        # Segmentation head expects a multiple-level output dictionary
        outputs = segmentation_head({1: outputs})

        super().__init__(inputs=inputs, outputs=outputs, **kwargs)

        self.num_classes = num_classes
        self.backbone = backbone
        self.spatial_pyramid_pooling = spatial_pyramid_pooling
        self.segmentation_head = segmentation_head
        self.segmentation_head_activation = segmentation_head_activation

    def get_config(self):
        return {
            "num_classes": self.num_classes,
            "backbone": self.backbone,
            "spatial_pyramid_pooling": self.spatial_pyramid_pooling,
            "segmentation_head": self.segmentation_head,
            "segmentation_head_activation": self.segmentation_head_activation,
        }

    def build(self, input_shape):
        height = input_shape[1]
        width = input_shape[2]
        feature_map_shape = self.backbone.compute_output_shape(input_shape)
        self.up_layer = keras.layers.UpSampling2D(
            size=(
                height // feature_map_shape[1],
                width // feature_map_shape[2],
            ),
            interpolation="bilinear",
        )

    def compile(self, weight_decay=0.0001, **kwargs):
        """compiles the DeepLabV3 model.

        Args:
            weight_decay: Optional float, factor of weight decay applied during
                training, defaults to `0.0001`.
        """
        self.weight_decay = weight_decay
        super().compile(**kwargs)

    def train_step(self, data):
        images, y_true, sample_weight = keras.utils.unpack_x_y_sample_weight(
            data
        )
        with tf.GradientTape() as tape:
            y_pred = self(images, training=True)
            total_loss = self.compute_loss(
                images, y_true, y_pred, sample_weight
            )
            reg_losses = []
            if self.weight_decay:
                for var in self.trainable_variables:
                    if "bn" not in var.name:
                        reg_losses.append(
                            self.weight_decay * tf.nn.l2_loss(var)
                        )
                l2_loss = tf.math.add_n(reg_losses)
                total_loss += l2_loss
        self.optimizer.minimize(total_loss, self.trainable_variables, tape=tape)
        return self.compute_metrics(
            images, y_true, y_pred, sample_weight=sample_weight
        )
