from keras_cv.backend import keras
from keras_cv.models import utils
from keras_cv.models.task import Task


@keras.utils.register_keras_serializable(package="keras_cv")
class SegFormer(Task):
    def __init__(
        self,
        num_classes=None,
        backbone=None,
        embed_dim=None,
        input_shape=None,
        input_tensor=None,
        softmax_output=None,
        **kwargs,
    ):
        inputs = utils.parse_model_inputs(input_shape, input_tensor)
        x = inputs
        y = backbone(x)
        y = SegFormerHead(
            in_dims=backbone.output_channels,
            embed_dim=embed_dim,
            num_classes=num_classes,
            name="segformer_head",
        )(y)
        output = keras.layers.Resizing(
            height=x.shape[1], width=x.shape[2], interpolation="bilinear"
        )(y)
        if softmax_output:
            output = keras.layers.Activation(
                "softmax", name="output_activation"
            )(output)

        super().__init__(
            inputs=inputs,
            outputs=output,
            **kwargs,
        )

        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.softmax_output = softmax_output


class SegFormerHead(keras.layers.Layer):
    def __init__(self, in_dims, embed_dim=256, num_classes=19, **kwargs):
        super().__init__(**kwargs)
        self.linear_layers = []

        for i in in_dims:
            self.linear_layers.append(
                keras.layers.Dense(embed_dim, name=f"linear_{i}")
            )

        # To fuse multiple layer outputs into a single feature map using a Conv2d
        self.linear_fuse = keras.Sequential(
            [
                keras.layers.Conv2D(
                    filters=embed_dim, kernel_size=1, use_bias=False
                ),
                keras.layers.BatchNormalization(),
                keras.layers.Activation("relu"),
            ]
        )
        self.dropout = keras.layers.Dropout(0.1)
        # Final segmentation output
        self.seg_out = keras.layers.Conv2D(filters=num_classes, kernel_size=1)

    def call(self, features):
        B, H, W, _ = features[0].shape
        outs = []

        for feature, layer in zip(features, self.linear_layers):
            feature = layer(feature)
            feature = keras.image.resize(
                feature, size=(H, W), method="bilinear"
            )
            outs.append(feature)

        seg = self.linear_fuse(keras.ops.concat(outs[::-1], axis=3))
        seg = self.dropout(seg)
        seg = self.seg_out(seg)

        return seg
