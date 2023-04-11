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

from keras_cv import bounding_box
from keras_cv.layers.preprocessing.base_image_augmentation_layer import (
    BaseImageAugmentationLayer,
)
from keras_cv.layers.preprocessing.random_affine_transf import process_segmentation_masks

# In order to support both unbatched and batched inputs, the horizontal
# and verticle axis is reverse indexed
H_AXIS = -3
W_AXIS = -2


@keras.utils.register_keras_serializable(package="keras_cv")
class RandomCrop(BaseImageAugmentationLayer):
    """A preprocessing layer which randomly crops images during training.
    During training, this layer will randomly choose a location to crop images
    down to a target size. The layer will crop all the images in the same batch
    to the same cropping location.
    At inference time, and during training if an input image is smaller than the
    target size, the input will be resized and cropped so as to return the
    largest possible window in the image that matches the target aspect ratio.
    If you need to apply random cropping at inference time, set `training` to
    True when calling the layer.
    Input pixel values can be of any range (e.g. `[0., 1.)` or `[0, 255]`) and
    of interger or floating point dtype. By default, the layer will output
    floats.
    Input shape:
      3D (unbatched) or 4D (batched) tensor with shape:
      `(..., height, width, channels)`, in `"channels_last"` format.
    Output shape:
      3D (unbatched) or 4D (batched) tensor with shape:
      `(..., target_height, target_width, channels)`.
    Args:
      height: Integer, the height of the output shape.
      width: Integer, the width of the output shape.
      seed: Integer. Used to create a random seed.
      bounding_box_format: The format of bounding boxes of input dataset. Refer
        https://github.com/keras-team/keras-cv/blob/master/keras_cv/bounding_box/converters.py
        for more details on supported bounding box formats.
      segmentation_classes: an optional integer with the number of classes in
        the input segmentation mask. Required iff augmenting data with sparse
        (non one-hot) segmentation masks. Include the background class in this
        count (e.g. for segmenting dog vs background, this should be set to 2).
    """

    def __init__(
        self, height, width, seed=None, bounding_box_format=None, segmentation_classes=None, **kwargs
    ):
        super().__init__(
            **kwargs, autocast=False, seed=seed, force_generator=True
        )
        self.height = height
        self.width = width
        self.seed = seed
        self.auto_vectorize = False
        self.bounding_box_format = bounding_box_format
        self.segmentation_classes = segmentation_classes

    def get_random_transformation(self, image=None, **kwargs):
        image_shape = tf.shape(image)
        h_diff = image_shape[H_AXIS] - self.height
        w_diff = image_shape[W_AXIS] - self.width
        dtype = image_shape.dtype
        rands = self._random_generator.random_uniform([2], 0, dtype.max, dtype)
        h_start = rands[0] % (h_diff + 1)
        w_start = rands[1] % (w_diff + 1)
        return {"top": h_start, "left": w_start}

    def augment_image(self, image, transformation, **kwargs):
        return self._crop_image(image, transformation)
    
    def _crop_image(self, image, transformation):
        image_shape = tf.shape(image)
        h_diff = image_shape[H_AXIS] - self.height
        w_diff = image_shape[W_AXIS] - self.width
        return tf.cond(
            tf.reduce_all((h_diff >= 0, w_diff >= 0)),
            lambda: self._crop(image, transformation),
            lambda: self._resize(image),
        )

    def compute_image_signature(self, images):
        return tf.TensorSpec(
            shape=(self.height, self.width, images.shape[-1]),
            dtype=self.compute_dtype,
        )

    def augment_bounding_boxes(
        self, bounding_boxes, transformation, image=None, **kwargs
    ):
        if self.bounding_box_format is None:
            raise ValueError(
                "`RandomCrop()` was called with bounding boxes,"
                "but no `bounding_box_format` was specified in the constructor."
                "Please specify a bounding box format in the constructor. i.e."
                "`RandomCrop(bounding_box_format='xyxy')`"
            )
        bounding_boxes = bounding_box.convert_format(
            bounding_boxes,
            source=self.bounding_box_format,
            target="xyxy",
            images=image,
        )
        image_shape = tf.shape(image)
        h_diff = image_shape[H_AXIS] - self.height
        w_diff = image_shape[W_AXIS] - self.width
        bounding_boxes = tf.cond(
            tf.reduce_all((h_diff >= 0, w_diff >= 0)),
            lambda: self._crop_bounding_boxes(
                image, bounding_boxes, transformation
            ),
            lambda: self._resize_bounding_boxes(
                image,
                bounding_boxes,
            ),
        )
        bounding_boxes = bounding_box.clip_to_image(
            bounding_boxes,
            bounding_box_format="xyxy",
            image_shape=(self.height, self.width, image_shape[-1]),
        )
        bounding_boxes = bounding_box.convert_format(
            bounding_boxes,
            source="xyxy",
            target=self.bounding_box_format,
            dtype=self.compute_dtype,
            images=image,
        )
        return bounding_boxes

    def augment_segmentation_mask(
        self, segmentation_mask, transformation, **kwargs
    ):
        return process_segmentation_masks(segmentation_mask, self.segmentation_classes,
                                   self._crop_image, transformation=transformation)
        
    def _crop(self, image, transformation):
        top = transformation["top"]
        left = transformation["left"]
        return tf.image.crop_to_bounding_box(
            image, top, left, self.height, self.width
        )

    def _resize(self, image):
        resizing_layer = keras.layers.Resizing(self.height, self.width)
        outputs = resizing_layer(image)
        # smart_resize will always output float32, so we need to re-cast.
        return tf.cast(outputs, self.compute_dtype)

    def augment_label(self, label, transformation, **kwargs):
        return label

    def get_config(self):
        config = {
            "height": self.height,
            "width": self.width,
            "seed": self.seed,
            "bounding_box_format": self.bounding_box_format,
            "segmentation_classes": self.segmentation_classes,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _crop_bounding_boxes(self, image, bounding_boxes, transformation):
        top = tf.cast(transformation["top"], dtype=self.compute_dtype)
        left = tf.cast(transformation["left"], dtype=self.compute_dtype)
        output = bounding_boxes.copy()
        x1, y1, x2, y2 = tf.split(
            bounding_boxes["boxes"], [1, 1, 1, 1], axis=-1
        )
        output["boxes"] = tf.concat(
            [
                x1 - left,
                y1 - top,
                x2 - left,
                y2 - top,
            ],
            axis=-1,
        )
        return output

    def _resize_bounding_boxes(self, image, bounding_boxes):
        output = bounding_boxes.copy()
        image_shape = tf.shape(image)
        x_scale = tf.cast(
            self.width / image_shape[W_AXIS], dtype=self.compute_dtype
        )
        y_scale = tf.cast(
            self.height / image_shape[H_AXIS], dtype=self.compute_dtype
        )
        x1, y1, x2, y2 = tf.split(
            bounding_boxes["boxes"], [1, 1, 1, 1], axis=-1
        )
        output["boxes"] = tf.concat(
            [
                x1 * x_scale,
                y1 * y_scale,
                x2 * x_scale,
                y2 * y_scale,
            ],
            axis=-1,
        )

        return output
