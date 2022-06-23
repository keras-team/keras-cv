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

from keras_cv.layers.preprocessing.base_image_augmentation_layer import (
    BaseImageAugmentationLayer,
)


class RandomBrightness(BaseImageAugmentationLayer):
    """A preprocessing layer which randomly adjusts brightness during training.

    This layer will randomly increase/reduce the brightness for the input RGB
    images. At inference time, the output will be identical to the input.
    Call the layer with `training=True` to adjust the brightness of the input.

    Note that different brightness adjustment factors
    will be apply to each the images in the batch.

    For an overview and full list of preprocessing layers, see the preprocessing
    [guide](https://www.tensorflow.org/guide/keras/preprocessing_layers).

    Args:
      factor: Float or a list/tuple of 2 floats between -1.0 and 1.0. The
        factor is used to determine the lower bound and upper bound of the
        brightness adjustment. A float value will be chosen randomly between
        the limits. When -1.0 is chosen, the output image will be black, and
        when 1.0 is chosen, the image will be fully white. When only one float
        is provided, eg, 0.2, then -0.2 will be used for lower bound and 0.2
        will be used for upper bound.
      value_range: Optional list/tuple of 2 floats for the lower and upper limit
        of the values of the input data. Defaults to [0.0, 255.0]. Can be changed
        to e.g. [0.0, 1.0] if the image input has been scaled before this layer.
        The brightness adjustment will be scaled to this range, and the
        output values will be clipped to this range.
      seed: optional integer, for fixed RNG behavior.

    Inputs: 3D (HWC) or 4D (NHWC) tensor, with float or int dtype. Input pixel
      values can be of any range (e.g. `[0., 1.)` or `[0, 255]`)

    Output: 3D (HWC) or 4D (NHWC) tensor with brightness adjusted based on the
      `factor`. By default, the layer will output floats. The output value will
      be clipped to the range `[0, 255]`, the valid range of RGB colors, and
      rescaled based on the `value_range` if needed.

    Sample usage:

    ```python
    random_bright = tf.keras.layers.RandomBrightness(factor=0.2)
    # An image with shape [2, 2, 3]
    image = [[[1, 2, 3], [4 ,5 ,6]], [[7, 8, 9], [10, 11, 12]]]
    # Assume we randomly select the factor to be 0.1, then it will apply
    # 0.1 * 255 to all the channel
    output = random_bright(image, training=True)
    # output will be int64 with 25.5 added to each channel and round down.
    tf.Tensor([[[26.5, 27.5, 28.5]
              [29.5, 30.5, 31.5]]
             [[32.5, 33.5, 34.5]
              [35.5, 36.5, 37.5]]],
            shape=(2, 2, 3), dtype=int64)
    ```
    """
    def __init__(self, factor, value_range=(0, 255), seed=None, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self.base = tf.keras.layers.RandomBrightness(
            factor=factor, value_range=value_range, seed=seed, **kwargs
        )

    def get_config(self):
        config = super().get_config()
        config.update(self.base.get_config())
        return config

    def get_random_transformation(
        self, image=None, label=None, bounding_box=None, **kwargs
    ):
        return self.base.get_random_transformation(
            image=image, label=label, bounding_box=bounding_box
        )

    def augment_image(self, image, transformation=None, **kwargs):
        return self.base.augment_image(image=image, transformation=transformation)

    def augment_label(self, labels, transformation=None, **kwargs):
        return labels

    def augment_bounding_boxes(self, bounding_boxes, **kwargs):
        return bounding_boxes

    def augment_keypoints(self, keypoints, **kwargs):
        return keypoints
