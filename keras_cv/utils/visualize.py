import tensorflow as tf
import PIL.ImageColor as ImageColor


def draw_masks_on_images(image, mask, color="red", alpha=0.4):
    """Draws masks on images.

    Args:
        image: an uint8 tensor with shape (N, img_height, img_height, 3)
        mask: an uint8 tensor of shape (N, img_height, img_height) with values
            between either 0 or 1.
        color: color to draw the keypoints with. Default is red.
        alpha: transparency value between 0 and 1. (default: 0.4)
    Returns:
        Masks overlaid on images.

    Raises:
        ValueError: On incorrect data type for images or masks.
    """

    def _blend(image1, image2, factor):
        difference = image2 - image1
        scaled = factor * difference
        # Do addition in float.
        blended = image1 + scaled

        # Interpolate
        if factor > 0.0 and factor < 1.0:
            # Interpolation means we always stay within 0 and 255.
            return tf.round(blended)
        return blended

    if image.dtype != tf.uint8:
        raise ValueError("`image` not of type tf.uint8")
    if mask.dtype != tf.uint8:
        raise ValueError("`mask` not of type tf.uint8")
    if tf.math.reduce_any(tf.math.logical_and(mask != 1, mask != 0)):
        raise ValueError("`mask` elements should be in [0, 1]")
    if image.shape[:3] != mask.shape:
        raise ValueError(
            "The image has spatial dimensions %s but the mask has "
            "dimensions %s" % (image.shape[:2], mask.shape)
        )
    if alpha <= 0.0:
        return tf.cast(image, dtype=tf.float32)

    # compute colored mask
    rgb = ImageColor.getrgb(color)
    solid_color = tf.expand_dims(tf.ones_like(mask), axis=-1)
    color = tf.cast(tf.reshape(list(rgb), [1, 1, 1, 3]), dtype=tf.uint8)
    solid_color *= color
    colored_mask = tf.expand_dims(mask, axis=-1) * solid_color

    # blend mask with image
    image = tf.cast(image, tf.float32)
    colored_mask = tf.cast(colored_mask, tf.float32)
    if alpha >= 1.0:
        masked_image = colored_mask
    else:
        masked_image = _blend(image, colored_mask, alpha)

    # stack masks along channel.
    mask_3d = tf.stack([mask] * 3, axis=-1)

    # exclude non positive area on image
    masked_image = tf.where(tf.cast(mask_3d, tf.bool), masked_image, image)
    return masked_image
