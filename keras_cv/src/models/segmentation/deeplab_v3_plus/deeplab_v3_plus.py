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

from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.backend import keras
from keras_cv.src.backend.config import keras_3
from keras_cv.src.layers.spatial_pyramid import SpatialPyramidPooling
from keras_cv.src.models.backbones.backbone_presets import backbone_presets
from keras_cv.src.models.backbones.backbone_presets import (
    backbone_presets_with_weights,
)
from keras_cv.src.models.segmentation.deeplab_v3_plus.deeplab_v3_plus_presets import (  # noqa: E501
    deeplab_v3_plus_presets,
)
from keras_cv.src.models.task import Task
from keras_cv.src.utils.python_utils import classproperty
from keras_cv.src.utils.train import get_feature_extractor


@keras_cv_export(
    [
        "keras_cv.models.DeepLabV3Plus",
        "keras_cv.models.segmentation.DeepLabV3Plus",
    ]
)
class DeepLabV3Plus(Task):
    """A Keras model implementing the DeepLabV3+ architecture for semantic
    segmentation.

    References:
        - [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611)  # noqa: E501
        (ECCV 2018)
        - [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587)  # noqa: E501
        (CVPR 2017)

    Args:
        backbone: `keras.Model`. The backbone network for the model that is
            used as a feature extractor for the DeepLabV3+ Encoder. Should
            either be a `keras_cv.models.backbones.backbone.Backbone` or a
            `keras.Model` that implements the `pyramid_level_inputs`
            property with keys "P2" and "P5" and layer names as values. A
            somewhat sensible backbone to use in many cases is the
            `keras_cv.models.ResNet50V2Backbone.from_preset("resnet50_v2_imagenet")`.
        num_classes: int, the number of classes for the detection model. Note
            that the `num_classes` contains the background class, and the
            classes from the data should be represented by integers with range
            [0, `num_classes`).
        projection_filters: int, number of filters in the convolution layer
            projecting low-level features from the `backbone`. The default
            value is set to `48`, as per the
            [TensorFlow implementation of DeepLab](https://github.com/tensorflow/models/blob/master/research/deeplab/model.py#L676).  # noqa: E501
        spatial_pyramid_pooling: (Optional) a `keras.layers.Layer`. Also known
            as Atrous Spatial Pyramid Pooling (ASPP). Performs spatial pooling
            on different spatial levels in the pyramid, with dilation. If
            provided, the feature map from the backbone is passed to it inside
            the DeepLabV3 Encoder, otherwise
            `keras_cv.layers.spatial_pyramid.SpatialPyramidPooling` is used.
        segmentation_head: (Optional) a `keras.layers.Layer`. If provided, the
            outputs of the DeepLabV3 encoder is passed to this layer and it
            should predict the segmentation mask based on feature from backbone
            and feature from decoder, otherwise a default DeepLabV3
            convolutional head is used.

    Example:
    ```python
    import keras_cv

    images = np.ones(shape=(1, 96, 96, 3))
    labels = np.zeros(shape=(1, 96, 96, 1))
    backbone = keras_cv.models.ResNet50V2Backbone(input_shape=[96, 96, 3])
    model = keras_cv.models.segmentation.DeepLabV3Plus(
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
        backbone,
        num_classes,
        projection_filters=48,
        spatial_pyramid_pooling=None,
        segmentation_head=None,
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

        extractor_levels = ["P2", "P5"]
        extractor_layer_names = [
            backbone.pyramid_level_inputs[i] for i in extractor_levels
        ]
        feature_extractor = get_feature_extractor(
            backbone, extractor_layer_names, extractor_levels
        )
        backbone_features = feature_extractor(inputs)

        if spatial_pyramid_pooling is None:
            spatial_pyramid_pooling = SpatialPyramidPooling(
                dilation_rates=[6, 12, 18]
            )
        spp_outputs = spatial_pyramid_pooling(backbone_features["P5"])

        low_level_feature_projector = keras.Sequential(
            [
                keras.layers.Conv2D(
                    name="low_level_feature_conv",
                    filters=projection_filters,
                    kernel_size=1,
                    padding="same",
                    use_bias=False,
                ),
                keras.layers.BatchNormalization(name="low_level_feature_norm"),
                keras.layers.ReLU(name="low_level_feature_relu"),
            ]
        )

        low_level_projected_features = low_level_feature_projector(
            backbone_features["P2"]
        )

        encoder_outputs = keras.layers.UpSampling2D(
            size=(8, 8),
            interpolation="bilinear",
            name="encoder_output_upsampling",
        )(spp_outputs)

        combined_encoder_outputs = keras.layers.Concatenate(axis=-1)(
            [encoder_outputs, low_level_projected_features]
        )

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
                    # Classification layer
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
                        dtype="float32",
                    ),
                ]
            )

        outputs = segmentation_head(combined_encoder_outputs)

        super().__init__(inputs=inputs, outputs=outputs, **kwargs)

        self.num_classes = num_classes
        self.backbone = backbone
        self.spatial_pyramid_pooling = spatial_pyramid_pooling
        self.projection_filters = projection_filters
        self.segmentation_head = segmentation_head

    def get_config(self):
        return {
            "num_classes": self.num_classes,
            "backbone": keras.saving.serialize_keras_object(self.backbone),
            "spatial_pyramid_pooling": keras.saving.serialize_keras_object(
                self.spatial_pyramid_pooling
            ),
            "projection_filters": self.projection_filters,
            "segmentation_head": keras.saving.serialize_keras_object(
                self.segmentation_head
            ),
        }

    @classmethod
    def from_config(cls, config):
        if "backbone" in config and isinstance(config["backbone"], dict):
            config["backbone"] = keras.layers.deserialize(config["backbone"])
        if "spatial_pyramid_pooling" in config and isinstance(
            config["spatial_pyramid_pooling"], dict
        ):
            config["spatial_pyramid_pooling"] = keras.layers.deserialize(
                config["spatial_pyramid_pooling"]
            )
        if "segmentation_head" in config and isinstance(
            config["segmentation_head"], dict
        ):
            config["segmentation_head"] = keras.layers.deserialize(
                config["segmentation_head"]
            )
        return super().from_config(config)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        if keras_3():
            return copy.deepcopy(
                {**backbone_presets, **deeplab_v3_plus_presets}
            )
        else:
            # TODO: #2246 Deeplab V3 presets don't work in Keras 2
            return copy.deepcopy({**backbone_presets})

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return copy.deepcopy(
            {**backbone_presets_with_weights, **deeplab_v3_plus_presets}
        )

    @classproperty
    def backbone_presets(cls):
        """Dictionary of preset names and configurations of compatible
        backbones."""
        return copy.deepcopy(backbone_presets)
