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

from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.backend import keras
from keras_cv.src.backend import ops
from keras_cv.src.backend import random
from keras_cv.src.utils import conv_utils


@keras_cv_export("keras_cv.layers.DropBlock2D")
class DropBlock2D(keras.layers.Layer):
    """Applies DropBlock regularization to input features.

    DropBlock is a form of structured dropout, where units in a contiguous
    region of a feature map are dropped together. DropBlock works better than
    dropout on convolutional layers due to the fact that activation units in
    convolutional layers are spatially correlated.

    It is advised to use DropBlock after activation in Conv -> BatchNorm ->
    Activation block in further layers of the network. For example, the paper
    mentions using DropBlock in 3rd and 4th group of ResNet blocks.

    Reference:
    - [DropBlock: A regularization method for convolutional networks](https://arxiv.org/abs/1810.12890)

    Args:
        rate: float. Probability of dropping a unit. Must be between 0 and 1.
            For best results, the value should be between 0.05-0.25.
        block_size: integer, or tuple of integers. The size of the block to be
            dropped. In case of an integer a square block will be dropped. In
            case of a tuple, the numbers are block's (height, width). Must be
            bigger than 0, and should not be bigger than the input feature map
            size. The paper authors use `block_size=7` for input feature's of
            size `14x14xchannels`. If this value is greater or equal to the
            input feature map size you will encounter `nan` values.
        seed: integer. To use as random seed.
        name: string. The name of the layer.

    Examples:
    DropBlock2D can be used inside a `keras.Model`:
    ```python
    # (...)
    x = Conv2D(32, (1, 1))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = DropBlock2D(0.1, block_size=7)(x)
    # (...)
    ```
    When used directly, the layer will zero-out some inputs in a contiguous
    region and normalize the remaining values.

    ```python
    # Small feature map shape for demonstration purposes:
    features = tf.random.stateless_uniform((1, 4, 4, 1), seed=[0, 1])

    # Preview the feature map
    print(features[..., 0])
    # tf.Tensor(
    # [[[0.08216608 0.40928006 0.39318466 0.3162533 ]
    #   [0.34717774 0.73199546 0.56369007 0.9769211 ]
    #   [0.55243933 0.13101244 0.2941643  0.5130266 ]
    #   [0.38977218 0.80855536 0.6040567  0.10502195]]], shape=(1, 4, 4),
    # dtype=float32)

    layer = DropBlock2D(0.1, block_size=2, seed=1234) # Small size for
        demonstration
    output = layer(features, training=True)

    # Preview the feature map after dropblock:
    print(output[..., 0])
    # tf.Tensor(
    #     [[[0.10955477 0.54570675 0.5242462  0.42167106]
    #       [0.46290365 0.97599393 0.         0.        ]
    #       [0.7365858  0.17468326 0.         0.        ]
    #       [0.51969624 1.0780739  0.80540895 0.14002927]]],
    #     shape=(1, 4, 4),
    #     dtype=float32)

    # We can observe two things:
    # 1. A 2x2 block has been dropped
    # 2. The inputs have been slightly scaled to account for missing values.

    # The number of blocks dropped can vary, between the channels - sometimes no
    # blocks will be dropped, and sometimes there will be multiple overlapping
    # blocks. Let's present on a larger feature map:

    features = tf.random.stateless_uniform((1, 4, 4, 36), seed=[0, 1])
    layer = DropBlock2D(0.1, (2, 2), seed=123)
    output = layer(features, training=True)

    print(output[..., 0])  # no drop
    # tf.Tensor(
    #     [[[0.09136613 0.98085546 0.15265216 0.19690938]
    #       [0.48835075 0.52433217 0.1661478  0.7067729 ]
    #       [0.07383626 0.9938906  0.14309917 0.06882786]
    #       [0.43242374 0.04158871 0.24213943 0.1903095 ]]],
    #     shape=(1, 4, 4),
    #     dtype=float32)

    print(output[..., 9])  # drop single block
    # tf.Tensor(
    #     [[[0.14568178 0.01571623 0.9082305  1.0545396 ]
    #       [0.24126057 0.86874676 0.         0.        ]
    #       [0.44101703 0.29805306 0.         0.        ]
    #       [0.56835717 0.04925899 0.6745584  0.20550345]]],
    #     shape=(1, 4, 4),
    #     dtype=float32)

    print(output[..., 22])  # drop two blocks
    # tf.Tensor(
    #     [[[0.69479376 0.49463132 1.0627024  0.58349967]
    #       [0.         0.         0.36143216 0.58699244]
    #       [0.         0.         0.         0.        ]
    #       [0.0315055  1.0117861  0.         0.        ]]],
    #     shape=(1, 4, 4),
    #     dtype=float32)

    print(output[..., 29])  # drop two blocks with overlap
    # tf.Tensor(
    #     [[[0.2137237  0.9120104  0.9963533  0.33937347]
    #       [0.21868704 0.44030213 0.5068906  0.20034194]
    #       [0.         0.         0.         0.5915383 ]
    #       [0.         0.         0.         0.9526224 ]]],
    #     shape=(1, 4, 4),
    #     dtype=float32)
    ```
    """  # noqa: E501

    def __init__(
        self,
        rate,
        block_size,
        seed=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not 0.0 <= rate <= 1.0:
            raise ValueError(
                f"rate must be a number between 0 and 1. " f"Received: {rate}"
            )

        self._rate = rate
        (
            self._dropblock_height,
            self._dropblock_width,
        ) = conv_utils.normalize_tuple(
            value=block_size, n=2, name="block_size", allow_zero=False
        )
        self.seed = seed
        self._random_generator = random.SeedGenerator(self.seed)

    def call(self, x, training=None):
        if not training or self._rate == 0.0:
            return x

        _, height, width, _ = ops.split(ops.shape(x), 4)

        # Unnest scalar values
        height = ops.squeeze(height)
        width = ops.squeeze(width)

        dropblock_height = ops.minimum(self._dropblock_height, height)
        dropblock_width = ops.minimum(self._dropblock_width, width)

        gamma = (
            self._rate
            * ops.cast(width * height, dtype="float32")
            / ops.cast(dropblock_height * dropblock_width, dtype="float32")
            / ops.cast(
                (width - self._dropblock_width + 1)
                * (height - self._dropblock_height + 1),
                "float32",
            )
        )

        # Forces the block to be inside the feature map.
        w_i, h_i = ops.meshgrid(ops.arange(width), ops.arange(height))
        valid_block = ops.logical_and(
            ops.logical_and(
                w_i >= int(dropblock_width // 2),
                w_i < width - (dropblock_width - 1) // 2,
            ),
            ops.logical_and(
                h_i >= int(dropblock_height // 2),
                h_i < width - (dropblock_height - 1) // 2,
            ),
        )

        valid_block = ops.reshape(valid_block, [1, height, width, 1])

        random_noise = random.uniform(
            ops.shape(x), seed=self._random_generator, dtype="float32"
        )
        valid_block = ops.cast(valid_block, dtype="float32")
        seed_keep_rate = ops.cast(1 - gamma, dtype="float32")
        block_pattern = (1 - valid_block + seed_keep_rate + random_noise) >= 1
        block_pattern = ops.cast(block_pattern, dtype="float32")

        window_size = [1, self._dropblock_height, self._dropblock_width, 1]

        # Double negative and max_pool is essentially min_pooling
        block_pattern = -ops.max_pool(
            -block_pattern,
            pool_size=window_size,
            strides=[1, 1, 1, 1],
            padding="SAME",
        )

        # Slightly scale the values, to account for magnitude change
        percent_ones = ops.cast(ops.sum(block_pattern), "float32") / ops.cast(
            ops.size(block_pattern), "float32"
        )
        return (
            x
            / ops.cast(percent_ones, x.dtype)
            * ops.cast(block_pattern, x.dtype)
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "rate": self._rate,
                "block_size": (self._dropblock_height, self._dropblock_width),
                "seed": self.seed,
            }
        )
        return config
