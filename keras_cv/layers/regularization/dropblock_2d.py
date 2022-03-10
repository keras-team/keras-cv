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
from tensorflow.keras.__internal__.layers import BaseRandomLayer

from keras_cv.utils import conv_utils


class DropBlock2D(BaseRandomLayer):
    """Applies DropBlock regularization to input features.

    DropBlock is a form of structured dropout, where units in a contiguous
    region of a feature map are dropped together. DropBlock works better than
    dropout on convolutional layers due to the fact that activation units in
    convolutional layers are spatially correlated.

    It is advised to use DropBlock after activation in Conv -> BatchNorm -> Activation
    block in further layers of the network. For example, the paper mentions using
    DropBlock in 3rd and 4th group of ResNet blocks.

    Reference:
    - [DropBlock: A regularization method for convolutional networks](
        https://arxiv.org/abs/1810.12890
    )

    Args:
        keep_probability: float. Probability of keeping a unit. Defaults to 0.9.
            Must be between 0 and 1. For best results, the value should be between
            0.75-0.95
        dropblock_size: integer. The size of the block to be dropped. Defaults to 7.
            Must be bigger than 0, and should not be bigger than the input feature map
            size. If this value is greater by 1 from the input feature map size you will
            encounter `ZeroDivisionError`.
        data_format: string. One of channels_last (default) or channels_first. The
            ordering of the dimensions in the inputs. channels_last corresponds to
            inputs with shape (batch_size, height, width, channels) while channels_first
            corresponds to inputs with shape (batch_size, channels,height, width). It
            defaults to the image_data_format value found in your Keras config file at
            ~/.keras/keras.json. If you never set it, then it will be channels_last.
        seed: integer. To use as random seed.
        name: string. The name of the layer.

    Usage:
    DropBlock2D can be used inside a `tf.keras.Model`:
    ```python
    # (...)
    x = Conv2D(32, (1, 1))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = DropBlock2D()(x)
    # (...)
    ```
    When used directly, the layer will zero-out some inputs in a contiguous region and
    normalize the remaining values.

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

    layer = DropBlock2D(dropblock_size=2, seed=1234)  # Small size for demonstration
    output = layer(features, training=True)

    # Preview the feature map after dropblock:
    print(output[..., 0])
    # tf.Tensor(
    # [[[0.10955477 0.54570675 0.5242462  0.42167106]
    #   [0.46290365 0.97599393 0.75158674 1.3025614 ]
    #   [0.         0.         0.39221907 0.6840355 ]
    #   [0.         0.         0.80540895 0.14002927]]], shape=(1, 4, 4),
    # dtype=float32)

    ```
    We can observe two things:
    1. A 2x2 block has been set to zero.
    2. The inputs have been normalized.

    One must remember, that DropBlock operation is random, so bigger or smaller
    patches can be dropped.
    """

    def __init__(
        self,
        keep_probability=0.9,
        dropblock_size=7,
        data_format=None,
        seed=None,
        name=None,
    ):
        super().__init__(seed=seed, name=name, force_generator=True)
        if not dropblock_size > 0:
            raise ValueError(
                f"dropblock_size must be greater than 0. Received: {dropblock_size}"
            )
        if not 0.0 <= keep_probability <= 1.0:
            raise ValueError(
                f"keep_probability must be a number between 0 and 1. "
                f"Received: {keep_probability}"
            )

        self._keep_probability = keep_probability
        self._dropblock_size = dropblock_size
        self._data_format = conv_utils.normalize_data_format(data_format)

    def call(self, x, training=False):
        if not training or self._keep_probability == 1.0:
            return x

        if self._data_format == "channels_last":
            _, height, width, _ = x.get_shape().as_list()
        else:
            _, _, height, width = x.get_shape().as_list()

        total_size = width * height
        dropblock_size = min(self._dropblock_size, width, height)

        # Seed_drop_rate is the gamma parameter of DropBlock.
        seed_drop_rate = (
            (1.0 - self._keep_probability)
            * total_size
            / dropblock_size**2
            / ((width - self._dropblock_size + 1) * (height - self._dropblock_size + 1))
        )

        # Forces the block to be inside the feature map.
        w_i, h_i = tf.meshgrid(tf.range(width), tf.range(height))
        valid_block = tf.logical_and(
            tf.logical_and(
                w_i >= int(dropblock_size // 2), w_i < width - (dropblock_size - 1) // 2
            ),
            tf.logical_and(
                h_i >= int(dropblock_size // 2), h_i < width - (dropblock_size - 1) // 2
            ),
        )

        if self._data_format == "channels_last":
            valid_block = tf.reshape(valid_block, [1, height, width, 1])
        else:
            valid_block = tf.reshape(valid_block, [1, 1, height, width])

        random_noise = self._random_generator.random_uniform(
            tf.shape(x), dtype=tf.float32
        )
        valid_block = tf.cast(valid_block, dtype=tf.float32)
        seed_keep_rate = tf.cast(1 - seed_drop_rate, dtype=tf.float32)
        block_pattern = (1 - valid_block + seed_keep_rate + random_noise) >= 1
        block_pattern = tf.cast(block_pattern, dtype=tf.float32)

        if self._data_format == "channels_last":
            window_size = [1, self._dropblock_size, self._dropblock_size, 1]
        else:
            window_size = [1, 1, self._dropblock_size, self._dropblock_size]

        # Double negative and max_pool is essentially min_pooling
        block_pattern = -tf.nn.max_pool(
            -block_pattern,
            ksize=window_size,
            strides=[1, 1, 1, 1],
            padding="SAME",
            data_format="NHWC" if self._data_format == "channels_last" else "NCHW",
        )

        percent_ones = tf.cast(tf.reduce_sum(block_pattern), tf.float32) / tf.cast(
            tf.size(block_pattern), tf.float32
        )

        return x / tf.cast(percent_ones, x.dtype) * tf.cast(block_pattern, x.dtype)

    def get_config(self):
        config = {
            "keep_probability": self._keep_probability,
            "dropblock_size": self._dropblock_size,
            "data_format": self._data_format,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
