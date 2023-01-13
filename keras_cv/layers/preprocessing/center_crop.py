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
import warnings

from keras_cv import bounding_box
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
            warnings.warn(
                "Using CenterCrop with bounding boxes can cause many boxes to be dropped."
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

    def _transform_bounding_boxes(self, bounding_boxes, transformation):
        original_height, original_width = transformation
        h_diff = original_height - self.height
        w_diff = original_width - self.width
        do_crop = tf.logical_and(h_diff >= 0, w_diff >= 0)

        def upsample_target_dims():
            target_height_ = tf.cast(
                tf.cast(original_width * self.height, "float32") / self.width, "int32"
            )
            target_width_ = tf.cast(
                tf.cast(original_height * self.width, "float32") / self.height, "int32"
            )
            target_height_ = tf.minimum(original_height, target_height_)
            target_width_ = tf.minimum(original_width, target_width_)
            return target_height_, target_width_

        target_height, target_width = tf.cond(
            do_crop, lambda: (self.height, self.width), upsample_target_dims
        )

        h_perc = keras.backend.cast(
            target_height / original_height, bounding_boxes.dtype
        )
        w_perc = keras.backend.cast(target_width / original_width, bounding_boxes.dtype)
        h0 = 0.5 - 0.5 * h_perc
        w0 = 0.5 - 0.5 * w_perc
        x1, y1, x2, y2, rest = tf.split(
            bounding_boxes, [1, 1, 1, 1, bounding_boxes.shape[-1] - 4], axis=-1
        )
        bounding_boxes = tf.concat(
            [
                (x1 - w0) / w_perc,
                (y1 - h0) / h_perc,
                (x2 - w0) / w_perc,
                (y2 - h0) / h_perc,
                rest,
            ],
            axis=-1,
        )
        bounding_boxes = bounding_box.filter_sentinels(bounding_boxes)
        return bounding_boxes

    def augment_bounding_boxes(
            self, bounding_boxes, transformation=None, image=None, **kwargs
    ):
        if self.bounding_box_format is None:
            raise ValueError(
                "`CenterCrop()` was called with bounding boxes,"
                "but no `bounding_box_format` was specified in the constructor."
                "Please specify a bounding box format in the constructor. i.e."
                "`CenterCrop(bounding_box_format='xyxy')`"
            )

        input_shape = (transformation[0], transformation[1], 3)
        output_shape = (self.height, self.width, 3)

        bounding_boxes = bounding_box.convert_format(
            bounding_boxes,
            source=self.bounding_box_format,
            target="rel_xyxy",
            image_shape=input_shape,
        )

        bounding_boxes = self._transform_bounding_boxes(bounding_boxes, transformation)

        bounding_boxes = bounding_box.clip_to_image(
            bounding_boxes,
            bounding_box_format="rel_xyxy",
            images=None,
            image_shape=output_shape,
        )
        bounding_boxes = bounding_box.convert_format(
            bounding_boxes,
            source="rel_xyxy",
            target=self.bounding_box_format,
            dtype=self.compute_dtype,
            image_shape=output_shape,
        )
        return bounding_boxes

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
