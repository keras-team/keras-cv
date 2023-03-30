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
from tensorflow import keras

from keras_cv.layers.preprocessing.base_image_augmentation_layer import (
    BaseImageAugmentationLayer,
)
from keras_cv.utils import preprocessing


@keras.utils.register_keras_serializable(package="keras_cv")
class FiveCrop(BaseImageAugmentationLayer):
    """ Crops image into four courners and one center crop.

    Args:
        size: a tuple or a int. Desired output size to crop.
            If it is a int, will be made a tuple.

    Usage:
    ```python
    SIZE = (1200, 1200)
    elephants = tf.keras.utils.get_file(
        "african_elephant.jpg", "https://i.imgur.com/Bvro0YD.png"
    )
    elephants = tf.keras.utils.load_img(elephants, target_size=SIZE)
    fivecrop = keras_cv.layers.preprocessing.FiveCrop(size=(300,300))
    top_left, top_right, bottom_left, bottom_right, center = fivecrop(elephants)
    tf.keras.preprocessing.image.array_to_img(center)
    ```

    Call arguments:
        images: Tensor of type int or float, with pixels in
            range [0, 255], [0, 1] and shape [batch, height, width, channels]
            or [height, width, channels].
    """

    def __init__(
        self,
        size,
        **kwargs
    ):
        super().__init__(**kwargs)
        if isinstance(size, int):
            size = (int(size), int(size))
        elif isinstance(size, tuple):
            size = (size[0], size[0])
        self.size=size

    def augment_image(self, image, transformation, **kwargs):
        return self._five_crop(image, transformation)

    def _five_crop(self, image, transformation):
        image_height, image_width = image.shape[0], image.shape[1]
        crop_height, crop_width = self.size

        if crop_width > image_width or crop_height > image_height:
            msg = "Requested crop size {} is bigger than input size {}"
            raise ValueError(msg.format(self.size, (image_height, image_width)))

        tl = tf.image.crop_to_bounding_box(image, 0, 0, crop_height, crop_width)
        tr = tf.image.crop_to_bounding_box(image, 0, image_width-crop_width, crop_height, crop_width)
        bl = tf.image.crop_to_bounding_box(image, image_height - crop_height, 0, crop_height, crop_width)
        br = tf.image.crop_to_bounding_box(image, image_height - crop_height, image_width - crop_width, crop_height, crop_width)

        center = tf.keras.layers.CenterCrop(crop_height, crop_width)(image)
        
        return tl, tr, bl, br, center

    def augment_labels(self, labels, transformations, **kwargs):
        return labels

    def augment_segmentation_masks(
        self, segmentation_masks, transformations, **kwargs
    ):
        return segmentation_masks

    def augment_bounding_boxes(self, bounding_boxes, transformations, **kwargs):
        return bounding_boxes

    def get_config(self):
        config = {
            "size": self.size
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        if isinstance(config["size"], dict):
            config["size"] = keras.utils.deserialize_keras_object(
                config["size"]
            )
        return cls(**config)