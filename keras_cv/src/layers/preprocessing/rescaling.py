# Copyright 2022 The KerasCV Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf

from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.layers.preprocessing.base_image_augmentation_layer import (
    BaseImageAugmentationLayer,
)

# In order to support both unbatched and batched inputs, the horizontal
# and vertical axis is reverse indexed
H_AXIS = -3
W_AXIS = -2


@keras_cv_export("keras_cv.layers.Rescaling")
class Rescaling(BaseImageAugmentationLayer):
    """A preprocessing layer which rescales input values to a new range.

    This layer rescales every value of an input (often an image) by multiplying
    by `scale` and adding `offset`.

    For instance:

    1. To rescale an input in the ``[0, 255]`` range
    to be in the `[0, 1]` range, you would pass `scale=1./255`.

    2. To rescale an input in the ``[0, 255]`` range to be in the `[-1, 1]`
    range, you would pass `scale=1./127.5, offset=-1`.

    Inputs can be of integer or floating point dtype, and by default the layer
    will output floats.

    Input shape:
      Arbitrary.

    Output shape:
      Same as input.

    Args:
      scale: Float, the scale to apply to the inputs.
      offset: Float, the offset to apply to the inputs.
    """

    def __init__(self, scale, offset=0.0, **kwargs):
        super().__init__(**kwargs)

        self.scale = scale
        self.offset = offset

    def augment_image(self, image, transformation, **kwargs):
        dtype = self.compute_dtype
        scale = tf.cast(self.scale, dtype)
        offset = tf.cast(self.offset, dtype)
        return tf.cast(image, dtype) * scale + offset

    def augment_label(self, label, transformation, **kwargs):
        return label

    def augment_segmentation_mask(
        self, segmentation_mask, transformation, **kwargs
    ):
        return segmentation_mask

    def augment_bounding_boxes(
        self, bounding_boxes, transformation=None, **kwargs
    ):
        return bounding_boxes

    def get_config(self):
        config = {
            "scale": self.scale,
            "offset": self.offset,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
