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
from keras_cv.src.utils import preprocessing


@keras_cv_export("keras_cv.layers.RandomJpegQuality")
class RandomJpegQuality(BaseImageAugmentationLayer):
    """Applies Random Jpeg compression artifacts to an image.

    Performs the jpeg compression algorithm on the image. This layer can be used
    in order to ensure your model is robust to artifacts introduced by JPEG
    compression.

    Args:
        factor: 2 element tuple or 2 element list. During augmentation, a random
        number is drawn from the factor distribution. This value is passed to
        `tf.image.adjust_jpeg_quality()`.
        seed: Integer. Used to create a random seed.

    Example:
    ```python
    layer = keras_cv.RandomJpegQuality(factor=(75, 100)))
    (images, labels), _ = keras.datasets.cifar10.load_data()
    augmented_images = layer(images)
    ```
    """

    def __init__(self, factor, seed=None, **kwargs):
        super().__init__(**kwargs)
        if isinstance(factor, (float, int)):
            raise ValueError(
                "RandomJpegQuality() expects factor to be a 2 element "
                "tuple, list or a `keras_cv.FactorSampler`. "
                "RandomJpegQuality() received `factor={factor}`."
            )
        self.seed = seed
        self.factor = preprocessing.parse_factor(
            factor,
            min_value=0,
            max_value=100,
            param_name="factor",
            seed=self.seed,
        )

    def get_random_transformation(self, **kwargs):
        return self.factor(dtype=tf.int32)

    def augment_image(self, image, transformation=None, **kwargs):
        jpeg_quality = transformation
        return tf.image.adjust_jpeg_quality(image, jpeg_quality)

    def augment_bounding_boxes(self, bounding_boxes, **kwargs):
        return bounding_boxes

    def augment_label(self, label, transformation=None, **kwargs):
        return label

    def augment_segmentation_mask(
        self, segmentation_mask, transformation, **kwargs
    ):
        return segmentation_mask

    def get_config(self):
        config = super().get_config()
        config.update({"factor": self.factor, "seed": self.seed})
        return config
