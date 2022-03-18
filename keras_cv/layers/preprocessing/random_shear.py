import tensorflow as tf
from tensorflow import keras


class RandomShear(tf.keras.__internal__.layers.BaseImageAugmentationLayer):

    def __init__(self, x_range=None, y_range=None, **kwargs):
        super(**kwargs)
        pass

    def get_random_transformation(self):
        

    def augment_image(self, image, transformation=None):
        return image
