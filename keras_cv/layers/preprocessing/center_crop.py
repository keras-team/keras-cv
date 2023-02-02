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


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class CenterCrop(BaseImageAugmentationLayer):
    """This layers crops the central portion of the images to a target size.
    If an image is smaller than the target size, it will be resized and cropped
    so as to return the largest possible window in the image that matches the target aspect ratio.

    The input images should have values in the `[0-255]` or `[0-1]` range.

    Input shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format

    Output shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format

    Args:
        height: Integer, the height of the output shape.
        width: Integer, the width of the output shape.

    Call arguments:
        inputs: Tensor representing images of shape
            `(batch_size, width, height, channels)`, with dtype tf.float32 / tf.uint8,
            ` or (width, height, channels)`, with dtype tf.float32 / tf.uint8
        training: A boolean argument that determines whether the call should be run
            in inference mode or training mode. Default: True.

    Usage:
    ```python
    (images, labels), _ = tf.keras.datasets.cifar10.load_data()
    channel_shuffle = keras_cv.layers.CenterCrop(10, 10)
    augmented_images = channel_shuffle(images)
    ```
    """

    def __init__(
        self, height: int, width: int, bounding_box_format=None, seed=None, **kwargs
    ):
        super().__init__(seed=seed, **kwargs)
        self.height = height
        self.width = width
        self.bounding_box_format = bounding_box_format
        self.base_layer_ = keras.layers.CenterCrop(self.height, self.width)

        if self.bounding_box_format is not None:
            raise ValueError(
                "CenterCrop() will drop boxes outside of the specified crop region.  "
                "Due to this, KerasCV does not currently support the passing of bounding box inputs to CenterCrop.  "
                "It is recommended that you use `layers.Resizing(pad_to_aspect_ratio=True)` instead."
            )

    def compute_image_signature(self, images):
        return tf.TensorSpec(
            (self.height, self.width, images.shape[-1]), self.compute_dtype
        )

    def augment_image(self, image, transformation=None, **kwargs):
        image = self.base_layer_(image)
        return image

    def get_random_transformation(
        self,
        image=None,
        label=None,
        bounding_boxes=None,
        keypoints=None,
        segmentation_mask=None,
    ):
        image_shape = tf.shape(image)
        return image_shape[-3], image_shape[-2]

    def augment_bounding_boxes(
        self, bounding_boxes, transformation=None, image=None, **kwargs
    ):
        raise ValueError(
            "CenterCrop() will drop boxes outside of the specified crop region.  "
            "Due to this, KerasCV does not currently support the passing of bounding box inputs to CenterCrop.  "
            "It is recommended that you use `layers.Resizing(pad_to_aspect_ratio=True)` instead."
        )

    def augment_label(self, label, transformation=None, **kwargs):
        return label

    def augment_segmentation_mask(self, segmentation_mask, transformation, **kwargs):
        segmentation_mask = self.base_layer_(segmentation_mask)
        return segmentation_mask

    def get_config(self):
        config = super().get_config()
        config.update({"height": self.height, "width": self.width})
        return config

    def compute_output_shape(self, input_shape):
        return self.base_layer_.compute_output_shape(input_shape)
