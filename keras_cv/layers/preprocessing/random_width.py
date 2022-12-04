import tensorflow as tf

from keras_cv.layers.preprocessing.base_image_augmentation_layer import (
    BaseImageAugmentationLayer,
)
from keras_cv.utils import preprocessing

HEIGHT_AXIS = -3
WIDTH_AXIS = -2
IMAGES = "images"
BOUNDING_BOXES = "bounding_boxes"


class RandomWidth(BaseImageAugmentationLayer):
    """A preprocessing layer which randomly varies image width during training.

    This layer will randomly adjusts the width of a batch of images of a
    batch of images by a random factor. The input should be a 3D (unbatched) or
    4D (batched) tensor in the `"channels_last"` image data format. Input pixel
    values can be of any range (e.g. `[0., 1.)` or `[0, 255]`) and of interger
    or floating point dtype. By default, the layer will output floats.

    By default, this layer is inactive during inference.

    For an overview and full list of preprocessing layers, see the preprocessing
    [guide](https://www.tensorflow.org/guide/keras/preprocessing_layers).

    Args:
      factor: A positive float (fraction of original width), or a tuple of size
        2 representing lower and upper bound for resizing vertically. When
        represented as a single float, this value is used for both the upper and
        lower bound. For instance, `factor=(0.2, 0.3)` results in an output with
        width changed by a random amount in the range `[20%, 30%]`.
        `factor=(-0.2, 0.3)` results in an output with width changed by a random
        amount in the range `[-20%, +30%]`. `factor=0.2` results in an output
        with width changed by a random amount in the range `[-20%, +20%]`.
      interpolation: String, the interpolation method. Defaults to `bilinear`.
        Supports `"bilinear"`, `"nearest"`, `"bicubic"`, `"area"`, `"lanczos3"`,
        `"lanczos5"`, `"gaussian"`, `"mitchellcubic"`.
      seed: Integer. Used to create a random seed.

    Input shape:
      3D (unbatched) or 4D (batched) tensor with shape:
      `(..., height, width, channels)`, in `"channels_last"` format.

    Output shape:
      3D (unbatched) or 4D (batched) tensor with shape:
      `(..., height, random_width, channels)`.
    """

    def __init__(self, factor, interpolation=tf.image.ResizeMethod.BILINEAR, seed=None, **kwargs):
        super().__init__(seed=seed, force_generator=True, **kwargs)
        self.factor = factor
        if isinstance(factor, (tuple, list)):
            self.width_lower = factor[0]
            self.width_upper = factor[1]
        else:
            self.width_lower = -factor
            self.width_upper = factor
        if self.width_upper < self.width_lower:
            raise ValueError(
                "`factor` cannot have upper bound less than "
                "lower bound, got {}".format(factor)
            )
        if self.width_lower < -1.0 or self.width_upper < -1.0:
            raise ValueError(
                "`factor` must have values larger than -1, " "got {}".format(factor)
            )
        self.interpolation = interpolation
        self._interpolation_method = preprocessing.get_interpolation(
            interpolation=interpolation
        )
        self.seed = seed
        self.auto_vectorize = False

    def augment_label(self, label, transformation=None, **kwargs):
        return label

    def _batch_augment(self, inputs):
        images = self.augment_image(
            inputs[IMAGES],
            transformation=self.get_random_transformation(image=inputs[IMAGES]),
        )
        result = {IMAGES: images}
        return result

    def augment_image(self, image, transformation, bounding_boxes=None, label=None):
        # The batch dimension of the input=image is not modified. The output
        # should be accurate for both unbatched and batched input
        inputs = preprocessing.ensure_tensor(image)
        inputs_shape = tf.shape(inputs)
        img_height_dim = inputs_shape[HEIGHT_AXIS]
        adjusted_width = transformation["width"]
        adjusted_size = tf.stack([img_height_dim, adjusted_width])
        output = tf.image.resize(
            images=inputs, size=adjusted_size, method=self._interpolation_method
        )
        # tf.resize will output float32 in many cases regardless of input type.
        output = tf.cast(output, self.compute_dtype)
        output_shape = inputs.shape.as_list()
        output_shape[WIDTH_AXIS] = None
        output.set_shape(output_shape)
        return output

    def get_random_transformation(
        self,
        image=None,
        label=None,
        bounding_boxes=None,
        keypoints=None,
        segmentation_mask=None,
    ):
        inputs_shape = tf.shape(image)
        img_weight_dim = tf.cast(inputs_shape[WIDTH_AXIS], tf.float32)
        width_factor = self._random_generator.random_uniform(
            shape=[], minval=(1.0 + self.width_lower), maxval=(1.0 + self.width_upper)
        )
        adjusted_width = tf.cast(width_factor * img_weight_dim, tf.int32)
        return {"width": adjusted_width}

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        input_shape[WIDTH_AXIS] = None
        return tf.TensorShape(input_shape)

    def get_config(self):
        config = {
            "factor": self.factor,
            "interpolation": self.interpolation,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
