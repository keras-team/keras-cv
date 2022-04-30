import tensorflow as tf
from keras.layers.preprocessing import preprocessing_utils as utils
from keras.utils import image_utils

@tf.keras.utils.register_keras_serializable(package="keras_cv")
class RandomResizedCrop(tf.keras.internal.layers.BaseImageAugmentationLayer):
    '''
    This layer is used to
    Crop a random portion of image and resize it to a given size.
    A crop of the original image is made: the crop has a random area (H * W) and a random aspect ratio.
    This crop is finally resized to the given size.
    This is popularly used to train the Inception networks.
    '''

    def __init__(self, height, width, seed=None, **kwargs):
        super().__init__(seed=seed,**kwargs)
        self.height = height
        self.width = width
        self.seed = seed

    #Method for taking the inputs
    def augment_image(self, inputs, training=True):
        inputs = utils.ensure_tensor(inputs, dtype=self.compute_dtype)
        if training:
            input_shape = tf.shape(inputs)
            h_diff = input_shape[-3] - self.height
            w_diff = input_shape[-2] - self.width
            return tf.cond(
                tf.reduce_all((h_diff >= 0, w_diff >= 0)),
                lambda: self._random_crop(inputs),
                lambda: self._resize(inputs))
        else:
            return self._resize(inputs)

    #Method for the random crop
    def _random_crop(self, inputs):
        input_shape = tf.shape(inputs)
        h_diff = input_shape[-3] - self.height
        w_diff = input_shape[-2] - self.width
        dtype = input_shape.dtype
        rands = self._random_generator.random_uniform([2], 0, dtype.max, dtype)
        h_start = rands[0] % (h_diff + 1)
        w_start = rands[1] % (w_diff + 1)
        return tf.image.crop_to_bounding_box(inputs, h_start, w_start,
                                             self.height, self.width)

    # method for resizing the images
    def _resize(self, inputs):
        self.new_height,self.new_width=self.get_random_transformation()
        outputs = image_utils.smart_resize(inputs, [self.new_height, self.new_width])
        # smart_resize will always output float32, so we need to re-cast.
        return tf.cast(outputs, self.compute_dtype)

    # Method for computing the output shape
    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        input_shape[-3] = self.height
        input_shape[-2] = self.width
        return tf.TensorShape(input_shape)

    #Configuration Method
    def get_config(self):
        config = {
            'height': self.height,
            'width': self.width,
            'seed': self.seed,
        }
        base_config = super(RandomResizedCrop, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_random_transformation(self, image=None, label=None, bounding_box=None):
        x = self._get_shear_amount(self.height)

        y = self._get_shear_amount(self.width)

        return (x, y)

    def _get_shear_amount(self, constraint):
        if constraint is None:
            return None

        negate = tf.random_uniform((), 0, self.constraint, dtype=tf.float32) > 0.5
        negate = tf.cond(negate, lambda: -1.0, lambda: 1.0)

        return negate * self._random_generator.random_uniform(
            (), constraint[0], constraint[1]
        )

    # Method for Random Resizing of the inputs
    def _random_resized_crop(self,inputs):
        return self._resize(self,self._random_crop(self,inputs))


