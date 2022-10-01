import tensorflow as tf
from tensorflow import keras

from keras_cv.models.generative.stable_diffusion.__internal__.layers.attention_block import (
    AttentionBlock,
)
from keras_cv.models.generative.stable_diffusion.__internal__.layers.group_normalization import (
    GroupNormalization,
)
from keras_cv.models.generative.stable_diffusion.__internal__.layers.padded_conv2d import (
    PaddedConv2D,
)
from keras_cv.models.generative.stable_diffusion.__internal__.layers.resnet_block import (
    ResnetBlock,
)


class VAEEncoder(keras.Sequential):
    def __init__(self):
        super().__init__(
            [
                PaddedConv2D(128, 3, padding=1),
                ResnetBlock(128),
                ResnetBlock(128),
                PaddedConv2D(128, 3, padding=1, strides=2),
                ResnetBlock(256),
                ResnetBlock(256),
                PaddedConv2D(256, 3, padding=1, strides=2),
                ResnetBlock(512),
                ResnetBlock(512),
                PaddedConv2D(512, 3, padding=1, strides=2),
                ResnetBlock(512),
                ResnetBlock(512),
                ResnetBlock(512),
                AttentionBlock(512),
                ResnetBlock(512),
                GroupNormalization(epsilon=1e-5),
                keras.layers.Activation("swish"),
                PaddedConv2D(8, 3, padding=1),
                PaddedConv2D(8, 1),
                # TODO(lukewood): can this be refactored to be a Rescaling layer?
                # Perhaps some sort of rescale and gather?
                # why is it the case we only take four dimensions anyways?
                # shouldn't we have 8 dimensions, or bottleneck the representation in a
                # smaller space instead of discarding?  This just seems very strange to
                # me.
                #
                # ... But, Divam has done this in his implementation.  I'll ask him what
                # the reason for this is, and look around other implementations a bit.
                #
                # Come to think of it, does anyone know why the latent spaces has a
                # final dimension of 4 channels instead of 3 channels?  How does that
                # work with a diffusion model anyways...
                keras.layers.Lambda(lambda x: x[..., :4] * 0.18215),
            ]
        )
