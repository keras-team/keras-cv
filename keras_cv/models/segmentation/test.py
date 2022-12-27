import os

import pytest
import tensorflow as tf
import tensorflow_datasets as tfds

from keras_cv import models
from keras_cv.models.segmentation.fcn import FCN8S, FCN16S, FCN32S

vgg16 = models.VGG16(include_rescaling=False, include_top=False, input_shape=(64, 64, 3))
model_fcn8s = FCN8S(
            classes=11,
            backbone=vgg16,
        )

input_image = tf.random.uniform(shape=[2, 64, 64, 3])
output_fcn8s = model_fcn8s(input_image)
print(output_fcn8s['output_tensor'].shape)