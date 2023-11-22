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
from keras_cv.backend import config

if config.keras_3():
    from keras.src.backend.tensorflow import *  # noqa: F403, F401
    from keras.src.backend.tensorflow import (  # noqa: F403, F401
        convert_to_numpy,
    )
    from keras.src.backend.tensorflow.core import *  # noqa: F403, F401
    from keras.src.backend.tensorflow.math import *  # noqa: F403, F401
    from keras.src.backend.tensorflow.nn import *  # noqa: F403, F401
    from keras.src.backend.tensorflow.numpy import *  # noqa: F403, F401
else:
    # isort: off
    from keras_core.src.backend.tensorflow import *  # noqa: F403, F401
    from keras_core.src.backend.tensorflow import (  # noqa: F403, F401
        convert_to_numpy,
    )
    from keras_core.src.backend.tensorflow.core import *  # noqa: F403, F401
    from keras_core.src.backend.tensorflow.math import *  # noqa: F403, F401
    from keras_core.src.backend.tensorflow.nn import *  # noqa: F403, F401
    from keras_core.src.backend.tensorflow.numpy import *  # noqa: F403, F401, E501


# Some TF APIs where the numpy API doesn't support raggeds that we need
from tensorflow import broadcast_to  # noqa: F403, F401
from tensorflow import concat as concatenate  # noqa: F403, F401
from tensorflow import repeat  # noqa: F403, F401
from tensorflow import reshape  # noqa: F403, F401

from tensorflow import range as arange  # noqa: F403, F401
from tensorflow import reduce_all as all  # noqa: F403, F401
from tensorflow import reduce_max as max  # noqa: F403, F401
from tensorflow import split  # noqa: F403, F401

import numpy as np
import tensorflow as tf


def smart_resize(x, size, interpolation="bilinear"):
    """Resize images to a target size without aspect ratio distortion.

    Warning: `tf.keras.preprocessing.image.smart_resize` is not recommended for
    new code. Prefer `tf.keras.layers.Resizing`, which provides the same
    functionality as a preprocessing layer and adds `tf.RaggedTensor` support.
    See the [preprocessing layer guide](
    https://www.tensorflow.org/guide/tf_keras/preprocessing_layers)
    for an overview of preprocessing layers.

    TensorFlow image datasets typically yield images that have each a different
    size. However, these images need to be batched before they can be
    processed by TF-Keras layers. To be batched, images need to share the same
    height and width.

    You could simply do:

    ```python
    size = (200, 200)
    ds = ds.map(lambda img: tf.image.resize(img, size))
    ```

    However, if you do this, you distort the aspect ratio of your images, since
    in general they do not all have the same aspect ratio as `size`. This is
    fine in many cases, but not always (e.g. for GANs this can be a problem).

    Note that passing the argument `preserve_aspect_ratio=True` to `resize`
    will preserve the aspect ratio, but at the cost of no longer respecting the
    provided target size. Because `tf.image.resize` doesn't crop images,
    your output images will still have different sizes.

    This calls for:

    ```python
    size = (200, 200)
    ds = ds.map(lambda img: smart_resize(img, size))
    ```

    Your output images will actually be `(200, 200)`, and will not be distorted.
    Instead, the parts of the image that do not fit within the target size
    get cropped out.

    The resizing process is:

    1. Take the largest centered crop of the image that has the same aspect
    ratio as the target size. For instance, if `size=(200, 200)` and the input
    image has size `(340, 500)`, we take a crop of `(340, 340)` centered along
    the width.
    2. Resize the cropped image to the target size. In the example above,
    we resize the `(340, 340)` crop to `(200, 200)`.

    Args:
      x: Input image or batch of images (as a tensor or NumPy array). Must be in
        format `(height, width, channels)` or `(batch_size, height, width,
        channels)`.
      size: Tuple of `(height, width)` integer. Target size.
      interpolation: String, interpolation to use for resizing. Supports
        `bilinear`, `nearest`, `bicubic`, `area`, `lanczos3`, `lanczos5`,
        `gaussian`, `mitchellcubic`. Defaults to `'bilinear'`.

    Returns:
      Array with shape `(size[0], size[1], channels)`. If the input image was a
      NumPy array, the output is a NumPy array, and if it was a TF tensor,
      the output is a TF tensor.
    """
    if len(size) != 2:
        raise ValueError(
            f"Expected `size` to be a tuple of 2 integers, but got: {size}."
        )
    img = tf.convert_to_tensor(x)
    if img.shape.rank is not None:
        if img.shape.rank < 3 or img.shape.rank > 4:
            raise ValueError(
                "Expected an image array with shape `(height, width, "
                "channels)`, or `(batch_size, height, width, channels)`, but "
                f"got input with incorrect rank, of shape {img.shape}."
            )
    shape = tf.shape(img)
    height, width = shape[-3], shape[-2]
    target_height, target_width = size
    if img.shape.rank is not None:
        static_num_channels = img.shape[-1]
    else:
        static_num_channels = None

    crop_height = tf.cast(
        tf.cast(width * target_height, "float32") / target_width, "int32"
    )
    crop_width = tf.cast(
        tf.cast(height * target_width, "float32") / target_height, "int32"
    )

    # Set back to input height / width if crop_height / crop_width is not
    # smaller.
    crop_height = tf.minimum(height, crop_height)
    crop_width = tf.minimum(width, crop_width)

    crop_box_hstart = tf.cast(
        tf.cast(height - crop_height, "float32") / 2, "int32"
    )
    crop_box_wstart = tf.cast(
        tf.cast(width - crop_width, "float32") / 2, "int32"
    )

    if img.shape.rank == 4:
        crop_box_start = tf.stack([0, crop_box_hstart, crop_box_wstart, 0])
        crop_box_size = tf.stack([-1, crop_height, crop_width, -1])
    else:
        crop_box_start = tf.stack([crop_box_hstart, crop_box_wstart, 0])
        crop_box_size = tf.stack([crop_height, crop_width, -1])

    img = tf.slice(img, crop_box_start, crop_box_size)
    img = tf.image.resize(images=img, size=size, method=interpolation)
    # Apparent bug in resize_images_v2 may cause shape to be lost
    if img.shape.rank is not None:
        if img.shape.rank == 4:
            img.set_shape((None, None, None, static_num_channels))
        if img.shape.rank == 3:
            img.set_shape((None, None, static_num_channels))
    if isinstance(x, np.ndarray):
        return img.numpy()
    return img
