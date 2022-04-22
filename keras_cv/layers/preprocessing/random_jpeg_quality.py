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

from keras_cv.utils import preprocessing


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class RandomJpegQuality(tf.keras.__internal__.layers.BaseImageAugmentationLayer):
    """Applies Random Jpeg compression artifacts to an image

    Args:
        factor: int, 2 element tuple or 2 element list. If int, this is the maximum
        compression that can be applied. If list or tuple, the first and second
        element represent the lower and upper bound respectively.
    """

    def __init__(self, factor, **kwargs):
        super().__init__(**kwargs)

        self.factor = preprocessing.parse_factor(
            factor, min_value=0, max_value=100, param_name="factor"
        )

    def get_random_transformation(self, image=None, label=None, bounding_box=None):
        return self.factor(dtype=tf.int32)

    def augment_image(self, image, transformation=None):
        jpeg_quality = transformation
        return tf.image.adjust_jpeg_quality(image, jpeg_quality)

    def get_config(self):
        config = super().get_config()
        config.update({"jpeg_quality": self.factor})
        return config
