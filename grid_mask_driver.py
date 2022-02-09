"""gridmask_demo.py shows how to use the GridMask preprocessing layer.

Operates on the oxford_flowers102 dataset.  In this script the flowers
are loaded, then are passed through the preprocessing layers.
Finally, they are shown using matplotlib.
"""

import matplotlib.pyplot as plt
import tensorflow as tf

from keras_cv.utils import fill_utils

RATIO = 0.6


def _compute_masks(inputs):
    """Computes grid masks for all inputs"""
    input_shape = tf.shape(inputs)
    batch_size = input_shape[0]
    height = tf.cast(input_shape[1], tf.float32)
    width = tf.cast(input_shape[2], tf.float32)

    # masks side length
    squared_w = tf.square(width)
    squared_h = tf.square(height)
    mask_hw = tf.math.ceil(tf.sqrt(squared_w + squared_h))
    mask_hw = tf.cast(mask_hw, tf.int32)

    # grid unit sizes
    unit_sizes = tf.random.uniform(
        shape=[batch_size],
        minval=tf.math.minimum(height * 0.5, width * 0.3),
        maxval=tf.math.maximum(height * 0.5, width * 0.3) + 1,
    )
    rectangle_lengths = tf.cast((1 - RATIO) * unit_sizes, tf.int32)

    # x and y offsets for grid units
    delta_x = tf.random.uniform([batch_size], minval=0, maxval=1, dtype=tf.float32)
    delta_y = tf.random.uniform([batch_size], minval=0, maxval=1, dtype=tf.float32)
    delta_x = tf.cast(delta_x * unit_sizes, tf.int32)
    delta_y = tf.cast(delta_y * unit_sizes, tf.int32)

    # number of diagonal units per grid (grid size)
    unit_sizes = tf.cast(unit_sizes, tf.int32)
    grid_sizes = mask_hw // unit_sizes + 1
    max_grid_size = tf.reduce_max(grid_sizes)

    # diagonal range per image
    diag_range = tf.range(1, max_grid_size + 1)
    diag_range = tf.tile(tf.expand_dims(diag_range, 0), [batch_size, 1])

    # add broadcasting axis for diagonal ranges
    delta_x = tf.expand_dims(delta_x, 1)
    delta_y = tf.expand_dims(delta_y, 1)
    unit_sizes = tf.expand_dims(unit_sizes, 1)
    rectangle_lengths = tf.expand_dims(rectangle_lengths, 1)

    # diagonal corner coordinates
    d_range = diag_range * unit_sizes
    x1 = d_range - delta_x
    x0 = x1 - rectangle_lengths
    y1 = d_range - delta_y
    y0 = y1 - rectangle_lengths

    # mask coordinates by grid ranges
    d_range_mask = tf.sequence_mask(
        lengths=grid_sizes, maxlen=max_grid_size, dtype=tf.int32
    )
    x1 = x1 * d_range_mask
    x0 = x0 * d_range_mask
    y1 = y1 * d_range_mask
    y0 = y0 * d_range_mask

    # expand diagonal top left corner coordinates into a mesh plane
    x0 = tf.tile(tf.expand_dims(x0, 1), [1, max_grid_size, 1])
    y0 = tf.tile(tf.expand_dims(y0, 1), [1, max_grid_size, 1])
    y0 = tf.transpose(y0, [0, 2, 1])

    # expand diagonal bottom right corner coordinates into a mesh plane
    x1 = tf.tile(tf.expand_dims(x1, 1), [1, max_grid_size, 1])
    y1 = tf.tile(tf.expand_dims(y1, 1), [1, max_grid_size, 1])
    y1 = tf.transpose(y1, [0, 2, 1])

    # flatten mesh planes to mesh grids
    x0 = tf.reshape(x0, [-1, max_grid_size])
    y0 = tf.reshape(y0, [-1, max_grid_size])
    x1 = tf.reshape(x1, [-1, max_grid_size])
    y1 = tf.reshape(y1, [-1, max_grid_size])

    # combine coordinates to (x0, y0, x1, y1) with shape (num_rectangles_in_batch, 4)
    corners0 = tf.stack([x0, y0], axis=-1)
    corners1 = tf.stack([x1, y1], axis=-1)
    corners0 = tf.reshape(corners0, [-1, 2])
    corners1 = tf.reshape(corners1, [-1, 2])
    corners = tf.concat([corners0, corners1], axis=1)

    # make mask for each rectangle
    mask_shape = (tf.shape(corners)[0], mask_hw, mask_hw)
    masks = fill_utils.rectangle_masks(mask_shape, corners)

    # reshape masks into shape (batch_size, rectangles_per_image, mask_height, mask_width)
    masks = tf.reshape(masks, [-1, max_grid_size * max_grid_size, mask_hw, mask_hw])

    # combine rectangle masks per image
    masks = tf.reduce_any(masks, axis=1)

    return masks


# %%
IMG_SHAPE = (5, 224, 224)
img = tf.ones(IMG_SHAPE)
masks = _compute_masks(img)

masks = tf.expand_dims(tf.cast(masks, tf.uint8), -1)

rotate = tf.keras.layers.RandomRotation(
    factor=1.0, fill_mode="constant", fill_value=0.0
)
masks = rotate(masks)

center_crop = tf.keras.layers.CenterCrop(224, 224)
masks = center_crop(masks)

masks = tf.cast(masks, tf.bool)

for m in masks:
    plt.imshow(m)
    plt.show()
