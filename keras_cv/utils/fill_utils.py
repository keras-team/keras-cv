import tensorflow as tf


def fill_rectangle(
    image, center_width, center_height, half_width, half_height, replace=None
):
    """Fill a rectangle in a given image using the value provided in replace.

    Args:
        image: the starting image to fill the rectangle on.
        center_width: the X center of the rectangle to fill
        center_height: the Y center of the rectangle to fill
        half_width: 1/2 the width of the resulting rectangle
        half_height: 1/2 the height of the resulting rectangle
        replace: The value to fill the rectangle with.  Accepts a Tensor,
            Constant, or None.
    Returns:
        image: the modified image with the chosen rectangle filled.
    """
    image_shape = tf.shape(image)
    image_height = image_shape[0]
    image_width = image_shape[1]

    lower_pad = tf.maximum(0, center_height - half_height)
    upper_pad = tf.maximum(0, image_height - center_height - half_height)
    left_pad = tf.maximum(0, center_width - half_width)
    right_pad = tf.maximum(0, image_width - center_width - half_width)

    shape = [
        image_height - (lower_pad + upper_pad),
        image_width - (left_pad + right_pad),
    ]
    padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]
    mask = tf.pad(tf.zeros(shape, dtype=image.dtype), padding_dims, constant_values=1)
    mask = tf.expand_dims(mask, -1)

    if replace is None:
        fill = tf.random.normal(image_shape, dtype=image.dtype)
    elif isinstance(replace, tf.Tensor):
        fill = replace
    else:
        fill = tf.ones_like(image, dtype=image.dtype) * replace
    image = tf.where(tf.equal(mask, 0), fill, image)
    return image
