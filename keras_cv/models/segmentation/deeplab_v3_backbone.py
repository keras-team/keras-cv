import copy

from tensorflow import keras
from tensorflow.keras import layers

from keras_cv.layers.spatial_pyramid import SpatialPyramidPooling
from keras_cv.models import utils
from keras_cv.models.backbones.backbone import Backbone
from keras_cv.models.segmentation.deeplab_v3_backbone_presets import (
    backbone_presets,
)
from keras_cv.models.segmentation.deeplab_v3_backbone_presets import (
    backbone_presets_with_weights,
)
from keras_cv.utils.python_utils import classproperty
from keras_cv.utils.train import get_feature_extractor


@keras.utils.register_keras_serializable(package="keras_cv.models")
class DeepLabV3Backbone(Backbone):
    def __init__(
        self,
        feature_extractor,
        spatial_pyramid_pooling=None,
        input_shape=(None, None, 3),
        input_tensor=None,
        **kwargs,
    ):
        if not isinstance(feature_extractor, keras.layers.Layer):
            raise ValueError(
                "Argument `backbone` must be a `keras.layers.Layer` instance. "
                f"Received instead backbone={feature_extractor} (of type "
                f"{type(feature_extractor)})."
            )

        inputs = utils.parse_model_inputs(input_shape, input_tensor)

        if input_shape[0] is None and input_shape[1] is None:
            input_shape = feature_extractor.input_shape[1:]
            inputs = layers.Input(tensor=input_tensor, shape=input_shape)

        if input_shape[0] is None and input_shape[1] is None:
            raise ValueError(
                "Input shapes for both the backbone and DeepLabV3 cannot be "
                "`None`. Received: input_shape={input_shape} and "
                "backbone.input_shape={backbone.input_shape[1:]}"
            )

        height = input_shape[0]
        width = input_shape[1]

        feature_map = feature_extractor(inputs)
        if spatial_pyramid_pooling is None:
            spatial_pyramid_pooling = SpatialPyramidPooling(
                dilation_rates=[6, 12, 18]
            )

        output = spatial_pyramid_pooling(feature_map)
        output = keras.layers.UpSampling2D(
            size=(
                height // feature_map.shape[1],
                width // feature_map.shape[2],
            ),
            interpolation="bilinear",
        )(output)

        super().__init__(inputs=inputs, outputs=output, **kwargs)

        self.feature_extractor = feature_extractor
        self.spatial_pyramid_pooling = spatial_pyramid_pooling
        self.input_shape = input_shape
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "feature_extractor": self.feature_extractor,
                "spatial_pyramid_pooling": self.spatial_pyramid_pooling,
                "input_shape": self.input_shape[1:],
                "input_tensor": self.input_tensor,
            }
        )
        return config

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return copy.deepcopy(backbone_presets)

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return copy.deepcopy(backbone_presets_with_weights)
