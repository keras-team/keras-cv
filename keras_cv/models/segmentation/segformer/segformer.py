import tensorflow as tf

from keras_cv.backend import keras
from keras_cv.models.task import Task
from keras_cv.utils.train import get_feature_extractor


@keras.utils.register_keras_serializable(package="keras_cv")
class SegFormer(Task):
    def __init__(
        self,
        num_classes=None,
        backbone=None,
        embed_dim=None,
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

        feature_extractor = get_feature_extractor(
            backbone, list(backbone.pyramid_level_inputs.values())
        )
        # Multi-level dictionary
        features = list(feature_extractor(inputs).values())

        # Get H and W of level one output
        _, H, W, _ = features[0].shape
        # Project all multi-level outputs onto the same dimensionality
        # and feature map shape
        multi_layer_outs = []
        for feature_dim, feature in zip(backbone.output_channels, features):
            out = keras.layers.Dense(embed_dim, name=f"linear_{feature_dim}")(
                feature
            )
            out = keras.layers.Resizing(H, W, interpolation="bilinear")(out)
            multi_layer_outs.append(out)

        # Concat now-equal feature maps
        concatenated_outs = keras.layers.Concatenate(axis=3)(
            multi_layer_outs[::-1]
        )

        # Fuse multi-channel segmentation map into a single-channel segmentation map
        seg = keras.Sequential(
            [
                keras.layers.Conv2D(
                    filters=embed_dim, kernel_size=1, use_bias=False
                ),
                keras.layers.BatchNormalization(),
                keras.layers.Activation("relu"),
            ]
        )(concatenated_outs)

        seg = keras.layers.Dropout(0.1)(seg)
        seg = keras.layers.Conv2D(filters=num_classes, kernel_size=1)(seg)

        output = keras.layers.Resizing(
            height=inputs.shape[1],
            width=inputs.shape[2],
            interpolation="bilinear",
        )(seg)

        super().__init__(
            inputs=inputs,
            outputs=output,
            **kwargs,
        )

        self.num_classes = num_classes
        self.embed_dim = embed_dim

    def get_config(self):
        return {
            "num_classes": self.num_classes,
            "backbone": self.backbone,
            "embed_dim": self.embed_dim,
        }
