import abc

import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.python.platform import tf_logging as logging


class BaseErase(layers.Layer, abc.ABC):
    """This can be inherited by layers that wants to implement erasing of patches.

    Args:
        rate: Float between 0 and 1.  The fraction of samples to augment.
        patch_value: Float. The value to fill in the patches. If None, will
            patches with gaussian noise. Defaults to 0.0.
    """

    def __init__(self, rate, patch_value=None, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate
        self.patch_value = patch_value
        self.seed = seed

    def call(self, images, labels):
        """call method for the RandomErase layer.

        Args:
            images: Tensor representing images of shape [batch_size, width, height, channels], with dtype tf.float32.
            labels: original labels.
        Returns:
            images: augmented images, same shape as input.
            labels: orignal labels.
        """

        if tf.shape(images)[0] == 1:
            logging.warning(
                "RandomErase received a single image to `call`.  The layer relies on combining multiple examples, "
                "and as such will not behave as expected.  Please call the layer with 2 or more samples."
            )

        augment_cond = tf.less(
            tf.random.uniform(shape=[], minval=0.0, maxval=1.0), self.rate
        )
        # pylint: disable=g-long-lambda
        random_erase_augment = lambda: self._erase(images, labels)
        no_augment = lambda: (images, labels)
        return tf.cond(augment_cond, random_erase_augment, no_augment)

    def _erase(self, images, labels):
        """Apply random erase."""
        input_shape = tf.shape(images)
        batch_size, image_height, image_width = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
        )

        cut_height, cut_width = self._compute_cut_size(
            batch_size, image_height, image_width
        )

        random_center_height = tf.random.uniform(
            shape=[batch_size], minval=0, maxval=image_height, dtype=tf.int32
        )
        random_center_width = tf.random.uniform(
            shape=[batch_size], minval=0, maxval=image_width, dtype=tf.int32
        )

        args = [
            images,
            random_center_width,
            random_center_height,
            cut_width // 2,
            cut_height // 2,
        ]
        if self.patch_value is not None:
            patch_value = tf.fill([batch_size], self.patch_value)
            args.append(patch_value)

        images = tf.map_fn(
            lambda x: _fill_rectangle(*x),
            args,
            fn_output_signature=tf.TensorSpec.from_tensor(images[0]),
        )

        return images, labels

    @abc.abstractmethod
    def _compute_cut_size(self, batch_size, image_height, image_width):
        pass


class RandomErase(BaseErase):
    """RandomErase implements the RandomErase data augmentation technique.

    Args:
        rate: Float between 0 and 1.  The fraction of samples to augment.
        scale: Tuple of float. Area ratio range (min, max) of erasing patch.
        ratio: Tuple of float. Aspect ratio range (min, max) of erasing patch.
        patch_value: Float. The value to fill in the patches. If None, will
            patches with gaussian noise. Defaults to 0.0.
    References:
       [RandomErase paper](https://arxiv.org/abs/1708.04896).

    Sample usage:
    ```python
    (images, labels), _ = tf.keras.datasets.cifar10.load_data()
    random_erase = keras_cv.layers.preprocessing.cut_mix.RandomErase(1.0)
    augmented_images, labels = random_erase(images, labels)
    ```
    """

    def __init__(
        self,
        rate,
        scale=(0.02, 0.33),
        ratio=(0.3, 3.3),
        patch_value=None,
        seed=None,
        **kwargs
    ):
        super().__init__(rate, patch_value, seed, **kwargs)
        self.scale = scale
        self.ratio = ratio

    def _compute_cut_size(self, batch_size, image_height, image_width):
        area = tf.cast(image_height * image_width, tf.float32)
        for _ in range(10):
            erase_area = area * tf.random.uniform(
                [batch_size], minval=self.scale[0], maxval=self.scale[1]
            )
            aspect_ratio = tf.random.uniform(
                [batch_size], minval=self.ratio[0], maxval=self.ratio[1]
            )
            h = tf.cast(tf.round(tf.sqrt(erase_area * aspect_ratio)), tf.int32)
            w = tf.cast(tf.round(tf.sqrt(erase_area / aspect_ratio)), tf.int32)

        return h, w


class CutOut(BaseErase):
    """CutOut implements the CutOut data augmentation technique.

    Args:
        rate: Float between 0 and 1.  The fraction of samples to augment.
        length: Integer. The side length of the square patches to cut out.
        patch_value: Float. The value to fill in the patches. If None, will
            patches with gaussian noise. Defaults to 0.0.
    References:
       [CutOut paper](https://arxiv.org/abs/1708.04552).

    Sample usage:
    ```python
    (images, labels), _ = tf.keras.datasets.cifar10.load_data()
    cutout = keras_cv.layers.preprocessing.cut_mix.CutOut(1.0, 50)
    augmented_images, labels = cutout(images, labels)
    ```
    """

    def __init__(self, rate, length, patch_value=0.0, seed=None, **kwargs):
        super().__init__(rate, patch_value, seed, **kwargs)
        self.length = length

    def _compute_cut_size(self, batch_size, image_height, image_width):
        length = tf.fill([batch_size], self.length)
        return length, length


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
