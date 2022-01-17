import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.python.platform import tf_logging as logging


class CutOut(layers.Layer):
    """CutOut implements the CutOut data augmentation technique.

    Args:
        rate: Float between 0 and 1.  The fraction of samples to augment.
        length: Integer.  Inverse scale parameter for the gamma distribution.
            This controls the shape of the distribution from which the smoothing values are
            sampled.  Defaults 1.0, which is a recommended value when training an imagenet1k
            classification model.
        patches: Integer. When > 0, label values are smoothed, meaning the
            confidence on label values are relaxed. e.g. label_smoothing=0.2 means that we
            will use a value of 0.1 for label 0 and 0.9 for label 1.  Defaults 1.
        patch_value: Float. ..... Defaults to 0.0.
    References:
       [CutOut paper](https://arxiv.org/abs/1708.04552).

    Sample usage:
    ```python
    (images, labels), _ = tf.keras.datasets.cifar10.load_data()
    cutout = keras_cv.layers.preprocessing.cut_mix.CutOut(20)
    augmented_images, labels = cutmix(images, labels)
    ```
    """

    def __init__(self, rate, length, patches=1, patch_value=0.0, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate
        self.patches = patches
        self.length = length
        self.patch_value = patch_value
        self.seed = seed

    def call(self, images, labels):
        """call method for the CutOut layer.

        Args:
            images: Tensor representing images of shape [batch_size, width, height, channels], with dtype tf.float32.
            labels: One hot encoded tensor of labels for the images, with dtype tf.float32.
        Returns:
            images: augmented images, same shape as input.
            labels: updated labels with both label smoothing and the cutmix updates applied.
        """

        if tf.shape(images)[0] == 1:
            logging.warning(
                "CutMix received a single image to `call`.  The layer relies on combining multiple examples, "
                "and as such will not behave as expected.  Please call the layer with 2 or more samples."
            )

        augment_cond = tf.less(
            tf.random.uniform(shape=[], minval=0.0, maxval=1.0), self.rate
        )
        # pylint: disable=g-long-lambda
        cutout_augment = lambda: self._cutout(images, labels)
        no_augment = lambda: (images, labels)
        return tf.cond(augment_cond, cutout_augment, no_augment)

    def _cutout(self, images, labels):
        """Apply cutout."""
        input_shape = tf.shape(images)
        batch_size, image_height, image_width = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
        )

        cut_height, cut_width = self._compute_cut_size(image_height, image_width)

        cut_height = tf.cast(cut_height, dtype=tf.int32)
        cut_width = tf.cast(cut_width, dtype=tf.int32)

        random_center_height = tf.random.uniform(
            shape=[batch_size], minval=0, maxval=image_height, dtype=tf.int32
        )
        random_center_width = tf.random.uniform(
            shape=[batch_size], minval=0, maxval=image_width, dtype=tf.int32
        )

        images = tf.map_fn(
            lambda x: _fill_rectangle(*x),
            (
                images,
                random_center_width,
                random_center_height,
                cut_width // 2,
                cut_height // 2,
                # self.patch_value,
            ),
            fn_output_signature=tf.TensorSpec.from_tensor(images[0]),
        )

        return images, labels

    def _compute_cut_size(self, image_height, image_width):
        return self.length, self.length


def _fill_rectangle(
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

    cutout_shape = [
        image_height - (lower_pad + upper_pad),
        image_width - (left_pad + right_pad),
    ]
    padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]
    mask = tf.pad(
        tf.zeros(cutout_shape, dtype=image.dtype), padding_dims, constant_values=1
    )
    mask = tf.expand_dims(mask, -1)

    if replace is None:
        fill = tf.random.normal(image_shape, dtype=image.dtype)
    elif isinstance(replace, tf.Tensor):
        fill = replace
    else:
        fill = tf.ones_like(image, dtype=image.dtype) * replace
    image = tf.where(tf.equal(mask, 0), fill, image)
    return image
