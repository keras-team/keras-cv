from keras import backend
from keras.engine import base_layer
from keras.engine import base_preprocessing_layer
from keras.layers.preprocessing import preprocessing_utils as utils
from keras.utils import image_utils
from keras.utils import tf_utils
import numpy as np
import tensorflow as tf
import random
from tensorflow.python.util.tf_export import keras_export
from tensorflow.tools.docs import doc_controls

H_AXIS = -3
W_AXIS = -2


class RandomResizedCrop(tf.keras.internal.layers.BaseImageAugmentationLayer):
    '''
    This layer is used to
    Crop a random portion of image and resize it to a given size.
    A crop of the original image is made: the crop has a random area (H * W) and a random aspect ratio.
    This crop is finally resized to the given size.
    This is popularly used to train the Inception networks.
    '''

    def __init__(self, height, width, seed=None, **kwargs):
        super().__init__
        # Attribute to crop image
        self.height = height
        # Attribute to take the custom height
        self.width = width
        # Attribute to take the custom width
        self.seed = seed
        # Attribute to take random seed

    #Methoid for taking the inputs
    def call(self, inputs, training=True):
        inputs = utils.ensure_tensor(inputs, dtype=self.compute_dtype)
        if training:
            input_shape = tf.shape(inputs)
            h_diff = input_shape[H_AXIS] - self.height
            w_diff = input_shape[W_AXIS] - self.width
            return tf.cond(
                tf.reduce_all((h_diff >= 0, w_diff >= 0)),
                lambda: self._random_crop(inputs),
                lambda: self._resize(inputs))
        else:
            return self._resize(inputs)

    #Method for the random crop
    def _random_crop(self, inputs):
        input_shape = tf.shape(inputs)
        h_diff = input_shape[H_AXIS] - self.height
        w_diff = input_shape[W_AXIS] - self.width
        dtype = input_shape.dtype
        rands = self._random_generator.random_uniform([2], 0, dtype.max, dtype)
        h_start = rands[0] % (h_diff + 1)
        w_start = rands[1] % (w_diff + 1)
        return tf.image.crop_to_bounding_box(inputs, h_start, w_start,
                                             self.height, self.width)

    # method for resizing the images
    def _resize(self, inputs):
        self.new_height=self.random.randint(0,self.height)
        self.new_width=random.randint(0,self.width)
        outputs = image_utils.smart_resize(inputs, [self.new_height, self.new_width])
        # smart_resize will always output float32, so we need to re-cast.
        return tf.cast(outputs, self.compute_dtype)

    # Method for computing the output shape
    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        input_shape[H_AXIS] = self.height
        input_shape[W_AXIS] = self.width
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
        x = self._get_shear_amount(self.x)

        y = self._get_shear_amount(self.y)
        return (x, y)

    def _get_shear_amount(self, constraint):
        if constraint is None:
            return None

        negate = self._random_generator.random_uniform((), 0, 1, dtype=tf.float32) > 0.5
        negate = tf.cond(negate, lambda: -1.0, lambda: 1.0)

        return negate * self._random_generator.random_uniform(
            (), constraint[0], constraint[1]
        )

    # Method for Random Resizing of the inputs
    def _random_resized_crop(self,inputs):
        return self._resize(self,self._random_crop(self,inputs))


