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

import tensorflow as tf
from tensorflow import keras

from keras_cv.layers.spatial_pyramid import SpatialPyramidPooling
from keras_cv.models.legacy import utils
from keras_cv.models.task import Task


@keras.utils.register_keras_serializable(package="keras_cv")
class DeepLabV3(Task):
    """A Keras model implementing the DeepLabV3 and DeepLabV3+ architectures
    for semantic segmentation.

    References:
        - [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611)
        (ECCV 2018)
        - [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587)
        (CVPR 2017)

    Args:
        num_classes: int, the number of classes for the detection model. Note
            that the `num_classes` doesn't contain the background class, and the
            classes from the data should be represented by integers with range
            [0, `num_classes`).
        backbone: `keras.Model`. The backbone network for the model that is
            used as a feature extractor for the DeepLabV3 Encoder. Should be a
            `keras_cv.models.backbones.backbone.Backbone`. A somewhat sensible
            backbone to use in many cases is the:
            `keras_cv.models.ResNet50V2Backbone.from_preset("resnet50_v2_imagenet")`.
        use_low_level_features: bool, whether to combine low-level features from
            the backbone with the output of the encoder or not. If set to `True`,
            the model is DeepLabV3+, otherwise it is DeepLabV3.
        projection_filters: int, number of filters in the
            convolution layer projecting low-level features from the `backbone`.
            The default value is set to `48`, as per the
            [TensorFlow implementation of DeepLab](https://github.com/tensorflow/models/blob/master/research/deeplab/model.py#L676).
            This parameter is only relevant if `use_low_level_features` is also
            specified.
        spatial_pyramid_pooling: (Optional) a `keras.layers.Layer`. Also known
            as Atrous Spatial Pyramid Pooling (ASPP). Performs spatial pooling
            on different spatial levels in the pyramid, with dilation. If
            provided, the feature map from the backbone is passed to it inside
            the DeepLabV3 Encoder, otherwise
            `keras_cv.layers.spatial_pyramid.SpatialPyramidPooling` is used.
        segmentation_head: (Optional) a `keras.layers.Layer`. If provided, the
            outputs of the DeepLabV3 encoder is passed to this layer and it
            should predict the segmentation mask based on feature from backbone
            and feature from decoder, otherwise a similar architecture is used.

    Examples:
    ```python
    import tensorflow as tf
    import keras_cv

    images = tf.ones(shape=(1, 96, 96, 3))
    labels = tf.zeros(shape=(1, 96, 96, 1))
    backbone = keras_cv.models.ResNet50V2Backbone(input_shape=[96, 96, 3])
    model = keras_cv.models.segmentation.DeepLabV3(
        num_classes=1, backbone=backbone,
    )

    # Evaluate model
    model(images)

    # Train model
    model.compile(
        optimizer="adam",
        loss=keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )
    model.fit(images, labels, epochs=3)
    ```
    """

    def __init__(
        self,
        num_classes,
        backbone,
        use_low_level_features=False,
        projection_filters=48,
        spatial_pyramid_pooling=None,
        segmentation_head=None,
        dropout=0.0,
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

        inputs = backbone.input
        input_shape = backbone.input_shape[1:]
        input_tensor = backbone.input_tensor

        if input_shape[0] is None and input_shape[1] is None:
            input_shape = backbone.input_shape[1:]
            inputs = keras.Input(tensor=input_tensor, shape=input_shape)

        if input_shape[0] is None and input_shape[1] is None:
            raise ValueError(
                "Input shapes for both the backbone and DeepLabV3 cannot be "
                "`None`. Received: input_shape={input_shape} and "
                "backbone.input_shape={backbone.input_shape[1:]}"
            )

        height = input_shape[0]
        width = input_shape[1]

        feature_extractor = keras.Model(
            inputs=backbone.input,
            outputs=backbone.get_layer(
                list(backbone.pyramid_level_inputs.values())[-1]
            ).output,
        )
        feature_map = feature_extractor(inputs)
        if spatial_pyramid_pooling is None:
            spatial_pyramid_pooling = SpatialPyramidPooling(
                dilation_rates=[6, 12, 18]
            )

        spp_outputs = spatial_pyramid_pooling(feature_map)

        encoder_outputs = keras.layers.UpSampling2D(
            size=(
                height // 4 // spp_outputs.shape[1],
                width // 4 // spp_outputs.shape[2],
            ),
            interpolation="bilinear",
        )(spp_outputs)

        if use_low_level_features:
            low_level_feature_extractor = keras.Model(
                inputs=backbone.input,
                outputs=backbone.get_layer(
                    backbone.pyramid_level_inputs["P2"]
                ).output,
            )
            low_level_feature_projector = keras.Sequential(
                [
                    keras.layers.Conv2D(
                        name="low_level_feature_conv",
                        filters=projection_filters,
                        kernel_size=1,
                        padding="same",
                        use_bias=False,
                    ),
                    keras.layers.BatchNormalization(
                        name="low_level_feature_norm"
                    ),
                    keras.layers.ReLU(name="low_level_feature_relu"),
                ]
            )

            low_level_features = low_level_feature_extractor(inputs)
            low_level_projected_features = low_level_feature_projector(
                low_level_features
            )
            combined_encoder_outputs = keras.layers.Concatenate(axis=-1)(
                [encoder_outputs, low_level_projected_features]
            )
        else:
            combined_encoder_outputs = encoder_outputs

        if segmentation_head is None:
            segmentation_head = keras.Sequential(
                [
                    keras.layers.Conv2D(
                        name="segmentation_head_conv",
                        filters=256,
                        kernel_size=1,
                        padding="same",
                        use_bias=False,
                    ),
                    keras.layers.BatchNormalization(
                        name="segmentation_head_norm"
                    ),
                    keras.layers.ReLU(name="segmentation_head_relu"),
                    keras.layers.UpSampling2D(
                        size=(4, 4), interpolation="bilinear"
                    ),
                ]
            )

            if dropout:
                segmentation_head.add(
                    keras.layers.Dropout(
                        dropout, name="segmentation_head_dropout"
                    )
                )

            # Classification layer
            segmentation_head.add(
                keras.layers.Conv2D(
                    name="segmentation_output",
                    filters=num_classes,
                    kernel_size=1,
                    use_bias=False,
                    padding="same",
                    activation="softmax",
                    # Force the dtype of the classification layer to float32
                    # to avoid the NAN loss issue when used with mixed
                    # precision API.
                    dtype=tf.float32,
                )
            )

        outputs = segmentation_head(combined_encoder_outputs)

        super().__init__(inputs=inputs, outputs=outputs, **kwargs)

        self.num_classes = num_classes
        self.backbone = backbone
        self.use_low_level_features = use_low_level_features
        self.spatial_pyramid_pooling = spatial_pyramid_pooling
        self.projection_filters = projection_filters
        self.segmentation_head = segmentation_head

    def get_config(self):
        return {
            "num_classes": self.num_classes,
            "backbone": self.backbone,
            "use_low_level_features": self.use_low_level_features,
            "spatial_pyramid_pooling": self.spatial_pyramid_pooling,
            "projection_filters": self.projection_filters,
            "segmentation_head": self.segmentation_head,
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
