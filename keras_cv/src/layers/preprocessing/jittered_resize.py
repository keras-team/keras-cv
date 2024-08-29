# Copyright 2023 The KerasCV Authors
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
#
# Some code in this file was inspired & adapted from `tensorflow_models`.
# Reference:
# https://github.com/tensorflow/models/blob/master/official/vision/ops/preprocess_ops.py

import tensorflow as tf

from keras_cv.src import bounding_box
from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.layers.preprocessing.vectorized_base_image_augmentation_layer import (  # noqa: E501
    VectorizedBaseImageAugmentationLayer,
)
from keras_cv.src.utils import preprocessing as preprocessing_utils

H_AXIS = -3
W_AXIS = -2


@keras_cv_export("keras_cv.layers.JitteredResize")
class JitteredResize(VectorizedBaseImageAugmentationLayer):
    """JitteredResize implements resize with scale distortion.

    JitteredResize takes a three-step approach to size-distortion based image
    augmentation. This technique is specifically tuned for object detection
    pipelines. The layer takes an input of images and bounding boxes, both of
    which may be ragged. It outputs a dense image tensor, ready to feed to a
    model for training. As such this layer will commonly be the final step in an
    augmentation pipeline.

    The augmentation process is as follows:

    The image is first scaled according to a randomly sampled scale factor. The
    width and height of the image are then resized according to the sampled
    scale. This is done to introduce noise into the local scale of features in
    the image. A subset of the image is then cropped randomly according to
    `crop_size`. This crop is then padded to be `target_size`. Bounding boxes
    are translated and scaled according to the random scaling and random
    cropping.

    Args:
        target_size: A tuple representing the output size of images.
        scale_factor: A tuple of two floats or a `keras_cv.FactorSampler`. For
            each augmented image a value is sampled from the provided range.
            This factor is used to scale the input image.
            To replicate the results of the MaskRCNN paper pass `(0.8, 1.25)`.
        crop_size: (Optional) the size of the image to crop from the scaled
            image, defaults to `target_size` when not provided.
        bounding_box_format: The format of bounding boxes of input boxes.
            Refer to
            https://github.com/keras-team/keras-cv/blob/master/keras_cv/bounding_box/converters.py
            for more details on supported bounding box formats.
        interpolation: String, the interpolation method, defaults to
            `"bilinear"`. Supports `"bilinear"`, `"nearest"`, `"bicubic"`,
            `"area"`, `"lanczos3"`, `"lanczos5"`, `"gaussian"`,
            `"mitchellcubic"`.
        seed: (Optional) integer to use as the random seed.

    Example:
    ```python
    train_ds = load_object_detection_dataset()
    jittered_resize = layers.JitteredResize(
        target_size=(640, 640),
        scale_factor=(0.8, 1.25),
        bounding_box_format="xywh",
    )
    train_ds = train_ds.map(
        jittered_resize, num_parallel_calls=tf.data.AUTOTUNE
    )
    # images now are (640, 640, 3)

    # an example using crop size
    train_ds = load_object_detection_dataset()
    jittered_resize = layers.JitteredResize(
        target_size=(640, 640),
        crop_size=(250, 250),
        scale_factor=(0.8, 1.25),
        bounding_box_format="xywh",
    )
    train_ds = train_ds.map(
        jittered_resize, num_parallel_calls=tf.data.AUTOTUNE
    )
    # images now are (640, 640, 3), but they were resized from a 250x250 crop.
    ```
    """  # noqa: E501

    def __init__(
        self,
        target_size,
        scale_factor,
        crop_size=None,
        bounding_box_format=None,
        interpolation="bilinear",
        minimum_box_area_ratio=0.0,
        seed=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not isinstance(target_size, tuple) or len(target_size) != 2:
            raise ValueError(
                "JitteredResize() expects `target_size` to be a tuple of two "
                f"integers. Received `target_size={target_size}`"
            )

        crop_size = crop_size or target_size
        self.interpolation = preprocessing_utils.get_interpolation(
            interpolation
        )
        self.scale_factor = preprocessing_utils.parse_factor(
            scale_factor,
            min_value=0.0,
            max_value=None,
            param_name="scale_factor",
            seed=seed,
        )
        self.crop_size = crop_size
        self.target_size = target_size
        self.bounding_box_format = bounding_box_format
        self.seed = seed

        self.force_output_dense_images = True
        
        self.minimum_box_area_ratio = minimum_box_area_ratio

    def compute_ragged_image_signature(self, images):
        ragged_spec = tf.RaggedTensorSpec(
            shape=list(self.target_size) + [images.shape[-1]],
            ragged_rank=1,
            dtype=self.compute_dtype,
        )
        return ragged_spec

    def get_random_transformation_batch(
        self, batch_size, images=None, **kwargs
    ):
        heights, widths = self._get_image_shape(images)
        image_shapes = tf.cast(
            tf.concat((heights, widths), axis=-1), dtype=tf.float32
        )

        scaled_sizes = tf.round(
            image_shapes * self.scale_factor(shape=(batch_size, 1))
        )
        scales = tf.where(
            tf.less(
                scaled_sizes[..., 0] / image_shapes[..., 0],
                scaled_sizes[..., 1] / image_shapes[..., 1],
            ),
            scaled_sizes[..., 0] / image_shapes[..., 0],
            scaled_sizes[..., 1] / image_shapes[..., 1],
        )

        scaled_sizes = tf.round(image_shapes * scales[..., tf.newaxis])
        image_scales = scaled_sizes / image_shapes

        max_offsets = scaled_sizes - self.crop_size
        max_offsets = tf.where(
            tf.less(max_offsets, 0), tf.zeros_like(max_offsets), max_offsets
        )
        offsets = max_offsets * self._random_generator.uniform(
            shape=(batch_size, 2), minval=0, maxval=1, dtype=tf.float32
        )
        offsets = tf.cast(offsets, tf.int32)
        return {
            "image_scales": image_scales,
            "scaled_sizes": scaled_sizes,
            "offsets": offsets,
        }

    def augment_ragged_image(self, image, transformation, **kwargs):
        image = tf.expand_dims(image, axis=0)
        scaled_sizes = transformation["scaled_sizes"]
        offsets = transformation["offsets"]
        transformation = {
            "scaled_sizes": tf.expand_dims(scaled_sizes, axis=0),
            "offsets": tf.expand_dims(offsets, axis=0),
        }
        image = self.augment_images(
            images=image, transformations=transformation, **kwargs
        )
        return tf.squeeze(image, axis=0)

    def augment_images(
        self, images, transformations, resize_method="bilinear", **kwargs
    ):
        # unpackage augmentation arguments
        scaled_sizes = transformations["scaled_sizes"]
        offsets = transformations["offsets"]
        inputs_for_resize_and_crop_single_image = {
            "images": images,
            "scaled_sizes": scaled_sizes,
            "offsets": offsets,
        }
        scaled_images = tf.map_fn(
            lambda x: self.resize_and_crop_single_image(
                x, resize_method=resize_method
            ),
            inputs_for_resize_and_crop_single_image,
            fn_output_signature=tf.float32,
        )
        return tf.cast(scaled_images, self.compute_dtype)

    def augment_segmentation_masks(
        self, segmentation_masks, transformations, **kwargs
    ):
        return self.augment_images(
            segmentation_masks, transformations, resize_method="nearest"
        )

    def augment_labels(self, labels, transformations, **kwargs):
        return labels

    def augment_bounding_boxes(
        self, bounding_boxes, transformations, raw_images=None, **kwargs
    ):
        if self.bounding_box_format is None:
            raise ValueError(
                "Please provide a `bounding_box_format` when augmenting "
                "bounding boxes with `JitteredResize()`."
            )
        if isinstance(bounding_boxes["boxes"], tf.RaggedTensor):
            bounding_boxes = bounding_box.to_dense(bounding_boxes)
        result = bounding_boxes.copy()
        image_scales = tf.cast(
            transformations["image_scales"], self.compute_dtype
        )
        offsets = tf.cast(transformations["offsets"], self.compute_dtype)

        bounding_boxes = bounding_box.convert_format(
            bounding_boxes,
            images=raw_images,
            source=self.bounding_box_format,
            target="yxyx",
        )

        # Adjusts box coordinates based on image_scale and offset.
        yxyx = bounding_boxes["boxes"]
        yxyx *= tf.tile(image_scales, [1, 2])[..., tf.newaxis, :]
        yxyx -= tf.tile(offsets, [1, 2])[..., tf.newaxis, :]

        result["boxes"] = yxyx
        result = bounding_box.clip_to_image(
            result,
            image_shape=self.target_size + (3,),
            bounding_box_format="yxyx",
            minimum_box_area_ratio=self.minimum_box_area_ratio
        )
        result = bounding_box.convert_format(
            result,
            image_shape=self.target_size + (3,),
            source="yxyx",
            target=self.bounding_box_format,
        )
        return result

    def _get_image_shape(self, images):
        if isinstance(images, tf.RaggedTensor):
            heights = tf.reshape(images.row_lengths(), (-1, 1))
            widths = tf.reshape(
                tf.reduce_max(images.row_lengths(axis=2), 1), (-1, 1)
            )
        else:
            batch_size = tf.shape(images)[0]
            heights = tf.repeat(tf.shape(images)[H_AXIS], repeats=[batch_size])
            heights = tf.reshape(heights, shape=(-1, 1))
            widths = tf.repeat(tf.shape(images)[W_AXIS], repeats=[batch_size])
            widths = tf.reshape(widths, shape=(-1, 1))
        return tf.cast(heights, dtype=tf.int32), tf.cast(widths, dtype=tf.int32)

    def resize_and_crop_single_image(self, inputs, resize_method="bilinear"):
        image = inputs.get("images", None)
        scaled_size = inputs.get("scaled_sizes", None)
        offset = inputs.get("offsets", None)

        scaled_image = tf.image.resize(
            image, tf.cast(scaled_size, tf.int32), method=resize_method
        )
        scaled_image = scaled_image[
            offset[0] : offset[0] + self.crop_size[0],
            offset[1] : offset[1] + self.crop_size[1],
            :,
        ]
        scaled_image = tf.image.pad_to_bounding_box(
            scaled_image, 0, 0, self.target_size[0], self.target_size[1]
        )
        return scaled_image

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "target_size": self.target_size,
                "scale_factor": self.scale_factor,
                "crop_size": self.crop_size,
                "bounding_box_format": self.bounding_box_format,
                "interpolation": self.interpolation,
                "seed": self.seed,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
