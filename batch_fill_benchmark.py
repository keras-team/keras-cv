import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


@tf.function
def rectangle_masks(images, x0, y0, x1, y1):
    images_shape = tf.shape(images)
    batch_size, image_height, image_width = (
        images_shape[0],
        images_shape[1],
        images_shape[2],
    )

    # add broadcasting axes
    x0 = x0[:, tf.newaxis, tf.newaxis]
    y0 = y0[:, tf.newaxis, tf.newaxis]
    x1 = x1[:, tf.newaxis, tf.newaxis]
    y1 = y1[:, tf.newaxis, tf.newaxis]

    # repeat for height and width
    x0_rep = tf.repeat(x0, image_height, axis=1)
    y0_rep = tf.repeat(y0, image_width, axis=2)
    x1_rep = tf.repeat(x1, image_height, axis=1)
    y1_rep = tf.repeat(y1, image_width, axis=2)

    # range grid
    range_row = tf.range(0, image_height)
    range_col = tf.range(0, image_width)
    range_row = tf.repeat(range_row[tf.newaxis, :, tf.newaxis], batch_size, 0)
    range_col = tf.repeat(range_col[tf.newaxis, tf.newaxis, :], batch_size, 0)

    # boolean masks
    mask_x0 = tf.less_equal(x0_rep, range_col)
    mask_y0 = tf.less_equal(y0_rep, range_row)
    mask_x1 = tf.less(range_col, x1_rep)
    mask_y1 = tf.less(range_row, y1_rep)

    masks = mask_x0 & mask_y0 & mask_x1 & mask_y1

    return masks


@tf.function
def fill_rectangle(image, center_x, center_y, height, width, fill=None):
    """Fill a rectangle in a given image using the value provided in replace.

    Args:
        image: the starting image to fill the rectangle on.
        center_x: the X center of the rectangle to fill
        center_y: the Y center of the rectangle to fill
        width: 1/2 the width of the resulting rectangle
        height: 1/2 the height of the resulting rectangle
        fill: A tensor with same shape as image. Values at rectangle
         position are used as fill.
    Returns:
        image: the modified image with the chosen rectangle filled.
    """
    image_shape = tf.shape(image)
    image_height = image_shape[0]
    image_width = image_shape[1]

    half_height = tf.cast(tf.math.ceil(height / 2), tf.int32)
    half_width = tf.cast(tf.math.ceil(width / 2), tf.int32)

    lower_pad = tf.maximum(0, center_y - half_height)
    upper_pad = tf.maximum(0, image_height - center_y - half_height)
    left_pad = tf.maximum(0, center_x - half_width)
    right_pad = tf.maximum(0, image_width - center_x - half_width)

    shape = [
        image_height - (lower_pad + upper_pad),
        image_width - (left_pad + right_pad),
    ]
    padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]
    mask = tf.pad(
        tf.zeros(shape, dtype=image.dtype), padding_dims, constant_values=1
    )
    mask = tf.expand_dims(mask, -1)

    image = tf.where(tf.equal(mask, 0), fill, image)
    return image


@tf.function
def batch_fill_rectangle(images, center_x, center_y, height, width, fill):
    """Fill a rectangle in a given image using the value provided in replace.

    Args:
        images: the starting image to fill the rectangle on.
        center_x: the X center of the rectangle to fill
        center_y: the Y center of the rectangle to fill
        height: the height of the resulting rectangle
        width:  width of the resulting rectangle
        fill: A tensor with same shape as image. Values at rectangle
         position are used as fill.
    Returns:
        image: the modified image with the chosen rectangle filled.
    """
    images_shape = tf.shape(images)
    image_height = images_shape[1]
    image_width = images_shape[2]

    half_height = tf.cast(tf.math.ceil(height / 2), tf.int32)
    half_width = tf.cast(tf.math.ceil(width / 2), tf.int32)

    x0 = tf.maximum(0, center_x - half_width)
    x1 = tf.minimum(image_width, center_x + half_width)
    y0 = tf.maximum(0, center_y - half_height)
    y1 = tf.minimum(image_height, center_y + half_height)

    is_patch_mask = rectangle_masks(images, x0, y0, x1, y1)
    is_patch_mask = tf.expand_dims(is_patch_mask, -1)

    images = tf.where(tf.equal(is_patch_mask, True), fill, images)
    return images


def time_batch_fill(images, cx, cy, height, width, fill, n):
    times = []
    for _ in range(n):
        st = time.time()
        batch_fill_rectangle(images, cx, cy, height, width, fill)
        total = time.time() - st
        times.append(total)
    return times


def time_map_fill(images, cx, cy, height, width, fill, n):
    times = []
    for _ in range(n):
        st = time.time()
        tf.map_fn(
            lambda x: fill_rectangle(*x),
            (
                images,
                cx,
                cy,
                height,
                width,
                fill,
            ),
            fn_output_signature=tf.TensorSpec.from_tensor(images[0]),
        )
        total = time.time() - st
        times.append(total)
    return times


#%%
n = 5
batch_sizes = [2 ** i for i in range(5)]

batch_times = []
map_times = []
for batch_size in batch_sizes:
    h, w = 32, 32
    batch_shape = (batch_size, h, w, 1)
    images = tf.ones(batch_shape)

    cx = tf.fill([batch_size], w // 2)
    cy = tf.fill([batch_size], h // 2)
    height = tf.fill([batch_size], h // 2)
    width = tf.fill([batch_size], w // 2)
    fill = tf.zeros_like(images)

    time_batch_fill(images, cx, cy, height, width, fill, 1)
    time_map_fill(images, cx, cy, height, width, fill, 1)

    batch_times_i = time_batch_fill(images, cx, cy, height, width, fill, n=n)
    single_times_i = time_map_fill(images, cx, cy, height, width, fill, n=n)
    batch_times.append(batch_times_i)
    map_times.append(single_times_i)

# batch_sizes x n
batch_times = np.array(batch_times)
map_times = np.array(map_times)

plt.plot(batch_sizes, batch_times.mean(1), label="batch")
plt.plot(batch_sizes, map_times.mean(1), label="map")
plt.xlabel("Batch size")
plt.ylabel("ylabel")
plt.legend()
plt.show()
