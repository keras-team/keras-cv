import tensorflow as tf

from keras_cv.layers.preprocessing.base_image_augmentation_layer import (
    BaseImageAugmentationLayer,
)
from keras_cv.utils import get_interpolation

HEIGHT_AXIS = -3
WIDTH_AXIS = -2
IMAGES = "images"
BOUNDING_BOXES = "bounding_boxes"


class RandomHeight(BaseImageAugmentationLayer):
    """A preprocessing layer which randomly varies image height during training.

    This layer adjusts the height of a batch of images by a random factor.
    The input should be a 3D (unbatched) or 4D (batched) tensor in the
    `"channels_last"` image data format. Input pixel values can be of any range
    (e.g. `[0., 1.)` or `[0, 255]`) and of interger or floating point dtype. By
    default, the layer will output floats.

    By default, this layer is inactive during inference.

    For an overview and full list of preprocessing layers, see the preprocessing
    [guide](https://www.tensorflow.org/guide/keras/preprocessing_layers).
    Args:
      factor: A positive float (fraction of original height), or a tuple of size
        2 representing lower and upper bound for resizing vertically. When
        represented as a single float, this value is used for both the upper and
        lower bound. For instance, `factor=(0.2, 0.3)` results in an output with
        height changed by a random amount in the range `[20%, 30%]`.
        `factor=(-0.2, 0.3)` results in an output with height changed by a
        random amount in the range `[-20%, +30%]`. `factor=0.2` results in an
        output with height changed by a random amount in the range
        `[-20%, +20%]`.
      interpolation: String, the interpolation method. Defaults to `"bilinear"`.
        Supports `"bilinear"`, `"nearest"`, `"bicubic"`, `"area"`,
        `"lanczos3"`, `"lanczos5"`, `"gaussian"`, `"mitchellcubic"`.
      seed: Integer. Used to create a random seed.
    Input shape:
      3D (unbatched) or 4D (batched) tensor with shape:
      `(..., height, width, channels)`, in `"channels_last"` format.
    Output shape:
      3D (unbatched) or 4D (batched) tensor with shape:
      `(..., random_height, width, channels)`.
    """

    def __init__(self, factor, interpolation="bilinear", seed=None, **kwargs):
        super().__init__(seed=seed, force_generator=True, **kwargs)
        self.factor = factor
        if isinstance(factor, (tuple, list)):
            self.height_lower = factor[0]
            self.height_upper = factor[1]
        else:
            self.height_lower = -factor
            self.height_upper = factor

        if self.height_upper < self.height_lower:
            raise ValueError(
                "`factor` cannot have upper bound less than "
                "lower bound, got {}".format(factor)
            )
        if self.height_lower < -1.0 or self.height_upper < -1.0:
            raise ValueError(
                "`factor` must have values larger than -1, " "got {}".format(factor)
            )
        self.interpolation = interpolation
        self._interpolation_method = get_interpolation(interpolation)
        self.seed = seed

    def get_random_transformation(
        self,
        image=None,
        label=None,
        bounding_boxes=None,
        keypoints=None,
        segmentation_mask=None,
    ):
        height_factor = self._random_generator.random_uniform(
            shape=[],
            minval=(1.0 + self.height_lower),
            maxval=(1.0 + self.height_upper),
        )
        inputs_shape = tf.shape(image)
        img_hd = tf.cast(inputs_shape[HEIGHT_AXIS], tf.float32)
        adjusted_height = tf.cast(height_factor * img_hd, tf.int32)
        return {"height": adjusted_height}

    def _batch_augment(self, inputs):
        images = self.augment_image(
            inputs[IMAGES],
            transformation=self.get_random_transformation(image=inputs[IMAGES]),
        )
        result = {IMAGES: images}
        return result

    def augment_image(self, image, transformation, bounding_boxes=None, label=None):
        # The batch dimension of the input=image is not modified. The output
        # would be accurate for both unbatched and batched input
        inputs_shape = tf.shape(image)
        img_wd = inputs_shape[WIDTH_AXIS]
        adjusted_height = transformation["height"]
        adjusted_size = tf.stack([adjusted_height, img_wd])
        output = tf.image.resize(
            images=image, size=adjusted_size, method=self._interpolation_method
        )
        # tf.resize will output float32 in many cases regardless of input type.
        output = tf.cast(output, self.compute_dtype)
        output_shape = list(image.shape)
        output_shape[HEIGHT_AXIS] = None
        output.set_shape(output_shape)
        return output

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        input_shape[HEIGHT_AXIS] = None
        return tf.TensorShape(input_shape)

    def get_config(self):
        config = {
            "factor": self.factor,
            "interpolation": self.interpolation,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
