import abc

import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.python.keras.utils import layer_utils

from keras_cv.utils.fill_utils import fill_rectangle


class BaseErasing(layers.Layer, abc.ABC):
    """This can be inherited by layers that wants to implement erasing of patches.

    Args:
        rate: Float between 0 and 1.  The fraction of samples to augment.
        patch_value: Float or string. The value to fill in the patches. If "gaussian", will
            fill patches with gaussian noise. Defaults to "gaussian".
    """

    def __init__(self, rate, patch_value="gaussian", seed=None, **kwargs):
        super().__init__(**kwargs)

        if isinstance(patch_value, str):
            layer_utils.validate_string_arg(
                patch_value,
                allowable_strings=["gaussian"],
                layer_name=self.__class__.__name__,
                arg_name="patch_value",
                allow_none=False,
                allow_callables=False,
            )

        self.rate = rate
        self.patch_value = patch_value
        self.seed = seed

    def call(self, images, labels):
        """call method for the layer.

        Args:
            images: Tensor representing images of shape [batch_size, width, height, channels], with dtype tf.float32.
            labels: original labels.
        Returns:
            images: augmented images, same shape as input.
            labels: orignal labels.
        """

        augment_cond = tf.less(
            tf.random.uniform(shape=[], minval=0.0, maxval=1.0), self.rate
        )
        # pylint: disable=g-long-lambda
        random_erase_augment = lambda: self._erase(images, labels)
        no_augment = lambda: (images, labels)
        return tf.cond(augment_cond, random_erase_augment, no_augment)

    def _erase(self, images, labels):
        """Apply erasing."""
        input_shape = tf.shape(images)
        batch_size, image_height, image_width = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
        )

        patch_height, patch_width = self._compute_patch_size(
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
            patch_width // 2,
            patch_height // 2,
        ]
        if not isinstance(self.patch_value, str):
            patch_value = tf.fill([batch_size], self.patch_value)
            args.append(patch_value)

        images = tf.map_fn(
            lambda x: fill_rectangle(*x),
            args,
            fn_output_signature=tf.TensorSpec.from_tensor(images[0]),
        )

        return images, labels

    def get_config(self):
        config = {
            "rate": self.rate,
            "patch_value": self.patch_value,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @abc.abstractmethod
    def _compute_patch_size(self, batch_size, image_height, image_width):
        pass


class RandomErasing(BaseErasing):
    """RandomErasing implements the RandomErasing data augmentation technique.

    Args:
        rate: Float between 0 and 1.  The fraction of samples to augment.
        scale: Tuple of float. Area scale range (min, max) of erasing patch.
        ratio: Tuple of float. Aspect ratio range (min, max) of erasing patch.
        patch_value: Float or string. The value to fill in the patches. If "gaussian", will
            fill patches with gaussian noise. Defaults to "gaussian".
    References:
       [RandomErasing paper](https://arxiv.org/abs/1708.04896).

    Sample usage:
    ```python
    (images, labels), _ = tf.keras.datasets.cifar10.load_data()
    random_erase = keras_cv.layers.preprocessing.erasing.RandomErasing(1.0)
    augmented_images, labels = random_erase(images, labels)
    ```
    """

    def __init__(
        self,
        rate,
        scale=(0.02, 0.33),
        ratio=(0.3, 3.3),
        patch_value="gaussian",
        seed=None,
        **kwargs
    ):
        super().__init__(rate, patch_value, seed, **kwargs)
        self.scale = scale
        self.ratio = ratio

    def _compute_patch_size(self, batch_size, image_height, image_width):
        area = tf.cast(image_height * image_width, tf.float32)
        erase_area = area * tf.random.uniform(
            [batch_size], minval=self.scale[0], maxval=self.scale[1]
        )
        aspect_ratio = tf.random.uniform(
            [batch_size], minval=self.ratio[0], maxval=self.ratio[1]
        )
        h = tf.cast(tf.round(tf.sqrt(erase_area * aspect_ratio)), tf.int32)
        w = tf.cast(tf.round(tf.sqrt(erase_area / aspect_ratio)), tf.int32)

        h = tf.minimum(h, image_height - 1)
        w = tf.minimum(w, image_width - 1)

        return h, w

    def get_config(self):
        config = {
            "scale": self.scale,
            "ratio": self.ratio,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class CutOut(BaseErasing):
    """CutOut implements the CutOut data augmentation technique.

    Args:
        rate: Float between 0 and 1.  The fraction of samples to augment.
        length: Integer. The side length of the square patches to cut out.
        patch_value: Float or string. The value to fill in the patches. If "gaussian", will
            fill patches with gaussian noise. Defaults to 0.0.
    References:
       [CutOut paper](https://arxiv.org/abs/1708.04552).

    Sample usage:
    ```python
    (images, labels), _ = tf.keras.datasets.cifar10.load_data()
    cutout = keras_cv.layers.preprocessing.erasing.CutOut(1.0, 50)
    augmented_images, labels = cutout(images, labels)
    ```
    """

    def __init__(self, rate, length, patch_value=0.0, seed=None, **kwargs):
        super().__init__(rate, patch_value, seed, **kwargs)
        self.length = length

    def _compute_patch_size(self, batch_size, image_height, image_width):
        length = tf.fill([batch_size], self.length)
        return length, length

    def get_config(self):
        config = {"length": self.length}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
