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
import warnings

import tensorflow as tf

from keras_cv.layers import BaseImageAugmentationLayer
from keras_cv.utils import preprocessing


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class RandomResizedCrop(BaseImageAugmentationLayer):
    """
    Randomly crops a part of an image and resizes it to provided size. 

    Args:
        target_size: A tuple of two integers used as the target size to crop
            images to.
<<<<<<< HEAD
        aspect_ratio_factor: (Optional) A tuple of two floats, a single float or
            `keras_cv.FactorSampler`. Aspect ratio means the ratio of width to
            height of the cropped image. Represents the lower and upper bound
            for the aspect ratio of the cropped image before resizing it to
            `target_size`. Defaults to (3./4., 4./3.).
=======
        aspect_ratio_factor: (Optional) A tuple of two floats. Represents the
            lower and upper bounds for the aspect ratio of the cropped image
            before resizing it to `target_size`. Defaults to (3./4., 4./3.).
>>>>>>> 8967911 (Update keras_cv/layers/preprocessing/random_resized_crop.py)
        area_factor: (Optional) A tuple of two floats, a single float or
            `keras_cv.FactorSampler`. Represents the lower and upper bounds for
            the area relative to the original image of the cropped image before
            resizing it to `target_size`. Defaults to (0.08, 1.0).
        seed: (Optional) Integer. Used to create a random seed.
    """
    def __init__(self,
                 target_size,
                 aspect_ratio_factor=(3. / 4., 4. / 3.),
                 area_factor=(0.08, 1.0),
                 seed=None,
                 **kwargs):
        super(RandomResizedCrop, self).__init__(seed=seed, **kwargs)

        self.target_size = target_size
        self.aspect_ratio_factor = aspect_ratio_factor
        self.area_factor = preprocessing.parse_factor(area_factor,
                                                      param_name="area_factor",
                                                      seed=seed)

        self.seed = seed

        if area_factor == 0.0 and aspect_ratio_factor == 0.0:
            warnings.warn(
                "RandomResizedCrop received both `area_factor=0.0` and "
                "`aspect_ratio_factor=0.0`. As a result, the layer will perform no "
                "augmentation.")

    def get_random_transformation(self,
                                  image=None,
                                  label=None,
                                  bounding_box=None):
        area_factor = self.area_factor()
        aspect_ratio = tf.random.uniform((),
                                         minval=self.aspect_ratio_factor[0],
                                         maxval=self.aspect_ratio_factor[1],
                                         dtype=tf.float32)

        new_height = tf.clip_by_value(
            tf.sqrt(area_factor / aspect_ratio), 0.0,
            1.0)  # to avoid unwanted/unintuitive effects
        new_width = tf.clip_by_value(tf.sqrt(area_factor * aspect_ratio), 0.0,
                                     1.0)

        height_offset = tf.random.uniform(
            (),
            minval=tf.minimum(0.0, 1.0 - new_height),
            maxval=tf.maximum(0.0, 1.0 - new_height),
            dtype=tf.float32,
        )

        width_offset = tf.random.uniform(
            (),
            minval=tf.minimum(0.0, 1.0 - new_width),
            maxval=tf.maximum(0.0, 1.0 - new_width),
            dtype=tf.float32,
        )

        y1 = height_offset
        y2 = height_offset + new_height
        x1 = width_offset
        x2 = width_offset + new_width

        return [[y1, x1, y2, x2]]

    def augment_image(self, image, transformation):
        image = tf.expand_dims(image, axis=0)
        boxes = transformation

        # See bit.ly/tf_crop_resize for more details
        augmented_image = tf.image.crop_and_resize(
            image,  # image shape: [B, H, W, C]
            boxes,  # boxes: (1, 4) in this case; represents area
            # to be cropped from the original image
            [0],  # box_indices: maps boxes to images along batch axis
            # [0] since there is only one image
            self.target_size,  # output size
        )

        return tf.squeeze(augmented_image, axis=0)

    def get_config(self):
        config = super().get_config()
        config.update({
            "target_size": self.target_size,
            "area_factor": self.area_factor,
            "aspect_ratio_factor": self.aspect_ratio_factor,
            "seed": self.seed,
        })
        return config
