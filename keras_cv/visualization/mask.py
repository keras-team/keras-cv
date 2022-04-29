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
"""Utility function to help visualize binary mask on image tensor."""
from collections import defaultdict
import tensorflow as tf
from keras_cv.visualization.colors import colors
from keras_cv.utils.preprocessing import blend

def _check_rgb_tuple(color):
    assert all(
        isinstance(c, int) for c in color
    ), f"Only integers are support for color tuple."
    assert all(
        c >= 0 and c <= 255 for c in rgb
    ), f"all values in color should be in range [0, 255].  got color={rgb}"
    assert len(rgb) == 3, f"Only RBG is supported but {rgb} passed."


def _map_color_on_mask(masks, color):
    def _raise_color_not_found_error(color):
        raise KeyError(
            f"{color} is not supported yet,"
            "please check supported colors at `keras_cv.visualization.colors`"
        )

    def _create_color_rgb_mapping():
        nonlocal color
        color_rgb = defaultdict(lambda: colors["red"])

        # map color code with RGB
        for code, name in color.items():
            if isinstance(name, (tuple, list)):
                _check_rgb_tuple(name)
                color_rgb.update({code: name})
            elif isinstance(name, str):
                try:
                    color_rgb.update({code: colors[name]})
                except KeyError:
                    _raise_color_not_found_error(name)
            else:
                raise TypeError(f"{type(name)} is not supported in color mapping.")
        return color_rgb

    # check distinct mask color codes with color mapping.
    distinct_mask_code = tf.unique(tf.reshape(masks, -1)).y

    if isinstance(color, str):
        try:
            color_rgb = defaultdict(lambda: colors[color])
        except KeyError:
            _raise_color_not_found_error(color)

    elif isinstance(color, (tuple, list)):
        _check_rgb_tuple(color)
        color_rgb = defaultdict(lambda: color)

    else:
        color_rgb = _create_color_rgb_mapping()

    keys_tensor = tf.cast(distinct_mask_code, tf.int32)
    colored_masks = tf.TensorArray(
        tf.int32, size=0, dynamic_size=True, clear_after_read=True
    )

    for c in range(3):
        vals_tensor = tf.map_fn(lambda color: color_rgb[color.numpy()][c], keys_tensor)
        table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor),
            default_value=0,
        )
        colored_masks.write(c, table[tf.cast(masks, tf.int32)]).mark_used()

    if masks.ndim == 2:
        return tf.einsum("chw->hwc", colored_masks.stack())
    return tf.einsum("cnhw->nhwc", colored_masks.stack())


def draw_map(image, mask, color='red', alpha=0.4):
    """Draws segmentation masks on images with desired color
    and transparency.
    Colors supported are standard `X11 Color set`.

    Args:
        image: 3D (unbatched) or 4D (batched) tensor with shape:
              `(N, height, width, 3)` or (height, width, 3) in `"channels_last"` format.
        mask: 2D (unbatched) or 3D (batched) tensor with shape:
              `(N, height, width)` or (height, width) in `"channels_last"` format.
        color: The color or colors to draw the segmentation map(s) in.
               This can either be a single color,
               or a dictionary mapping from class IDs to colors.
               Supported colors are `X11 Color Set` and formats
               are RGB tuples or strings.  A full list of color
               strings is available in The`KerasCV `colors.py` file.
               Defaults to 'red'.
        alpha: Alpha value between 0 and 1. (default: 0.4)
    Returns:
        Masks overlaid on images.

    Raises:
        ValueError: On incorrect data type and shapes for images or masks.
        KeyError: On incorrect color string in `color`, or a class ID missing from
        the color map.
        TypeError: On incorrect color type.

    References:
    - [KerasCV Colors](https://tinyurl.com/34bm28zu)
    - [X11 Color Set](https://www.w3.org/TR/css-color-4/#named-colors)

    Usage:
    ```python
    # Example1
    color = {1:(255, 0, 0), 2:(0, 255, 0)}
    images = keras_cv.visualization.draw_map(
        images, # 4D Batched images or 3D images
        masks, # 4D Batch masks or 3D masks
        color=color
    )

    # Example2
    color = "cyan"
    images = keras_cv.visualization.draw_map(
        images,
        masks,
        color=color
    )
    ```
    """

    # TODO: add support for regression format masks
    tf.debugging.assert_integer(
        mask, message="Only integer dtypes supported for `mask`."
    )

    if not isinstance(color, (dict, str, tuple, list)):
        raise TypeError(
            f"Want type(color) in [dict, str, tuple, list]. "
            "got type(color)={type(color)}."
        )

    if (image.ndim == 4) and (image.shape[:3] != mask.shape):
        raise ValueError(
            f"image.shape[:3] == mask.shape should be true, "
            f"got {image.shape[:3]} != {mask.shape}"
        )
    elif (image.ndim == 3) and (image.shape[:2] != mask.shape):
        raise ValueError(
            f"image.shape[:2] == mask.shape should be true, "
            f"got {image.shape[:2]} != {mask.shape}"
        )

    if alpha < 0 or alpha > 1:
        raise ValueError(f"alpha should be in the range [0, 1], got alpha={alpha}")

    _input_image_dtype = image.dtype

    # compute colored mask
    colored_mask = _map_color_on_mask(mask, color)

    # blend mask with image
    image = tf.cast(image, tf.float32)
    colored_mask = tf.cast(colored_mask, tf.float32)
    masked_image = blend(image, colored_mask, alpha)

    # stack masks along channel.
    mask_3d = tf.stack([mask] * 3, axis=-1)
    masked_image = tf.where(tf.cast(mask_3d, tf.bool), masked_image, image)
    masked_image = tf.cast(masked_image, dtype=_input_image_dtype)
    return masked_image
