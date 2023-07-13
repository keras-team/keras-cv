import tensorflow as tf

from deepvision.layers.segformer_segmentation_head import SegFormerHead
from deepvision.utils.utils import parse_model_inputs


class __SegFormerTF(tf.keras.Model):
    def __init__(
        self,
        num_classes=None,
        backbone=None,
        embed_dim=None,
        input_shape=None,
        input_tensor=None,
        softmax_output=None,
        **kwargs
    ):
        inputs = parse_model_inputs("tensorflow", input_shape, input_tensor)
        x = inputs
        y = backbone(x)
        y = SegFormerHead(
            in_dims=backbone.output_channels,
            embed_dim=embed_dim,
            num_classes=num_classes,
            name="segformer_head",
            backend="tensorflow",
        )(y)
        output = tf.keras.layers.Resizing(
            height=x.shape[1], width=x.shape[2], interpolation="bilinear"
        )(y)
        if softmax_output:
            output = tf.keras.layers.Activation("softmax", name="output_activation")(
                output
            )

        super().__init__(
            inputs=inputs,
            outputs=output,
            **kwargs,
        )

        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.softmax_output = softmax_output