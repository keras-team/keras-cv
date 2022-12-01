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
#
# Some code in this file was inspired & adapted from `tensorflow_models`.
# Reference:
# https://github.com/tensorflow/models/blob/master/official/vision/ops/preprocess_ops.py

import tensorflow as tf

import keras_cv
from keras_cv import layers
from keras_cv.layers.preprocessing.base_image_augmentation_layer import (
    BaseImageAugmentationLayer,
)


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class JitteredResize(BaseImageAugmentationLayer):
    """JitteredResize implements resize with scale distortion.

    JitteredResize takes a three step approach to size-distortion based image
    augmentation.  This technique is specifically tuned for object detection pipelines.
    The layer takes an input of images and bounding boxes, both of which may be ragged.
    It outputs a dense image tensor, ready to feed to a model for training.
    As such this layer will commonly be the final step in an augmentation
    pipeline.

    The augmentation process is as follows:

    The image is first scaled according to a randomly sampled scale factor.  The width
    and height of the image are then resized according to the sampled scale.  This is
    done to introduce noise into the local scale of features in the image. A subset of
    the image is then cropped randomly according to `crop_size`.  This crop is then
    padded to be `target_size`.  Bounding boxes are translated and scaled according to
    the random scaling and random cropping.

    Usage:
    ```python
    train_ds = load_object_detection_dataset()
    jittered_resize = layers.JitteredResize(
        target_size=(640, 640),
        scale_factor=(0.8, 1.25),
        bounding_box_format="xywh",
    )
    train_ds = train_ds.map(jittered_resize, num_parallel_calls=tf.data.AUTOTUNE)
    # images now are (640, 640, 3)

    # an example using crop size
    train_ds = load_object_detection_dataset()
    jittered_resize = layers.JitteredResize(
        target_size=(640, 640),
        crop_size=(250, 250),
        scale_factor=(0.8, 1.25),
        bounding_box_format="xywh",
    )
    train_ds = train_ds.map(jittered_resize, num_parallel_calls=tf.data.AUTOTUNE)
    # images now are (640, 640, 3), but they were resized from a 250x250 crop.
    ```

    Args:
        target_size: A tuple repesenting the output size of images.
        scale_factor: A tuple of two floats or a `keras_cv.FactorSampler`. For each
            augmented image a value is sampled from the provided range.
            This factor is used to scale the input image.
            To replicate the results of the MaskRCNN paper pass `(0.8, 1.25)`.
        crop_size: (Optional) the size of the image to crop from the scaled image.
            Defaults to `target_size` when not provided.
        bounding_box_format: The format of bounding boxes of input boxes. Refer
            to https://github.com/keras-team/keras-cv/blob/master/keras_cv/bounding_box/converters.py
            for more details on supported bounding box formats.
        interpolation: String, the interpolation method. Defaults to `"bilinear"`.
            Supports `"bilinear"`, `"nearest"`, `"bicubic"`, `"area"`, `"lanczos3"`,
            `"lanczos5"`, `"gaussian"`, `"mitchellcubic"`.
        seed: (Optional) integer to use as the random seed.
    """

    def __init__(
        self,
        target_size,
        scale_factor,
        crop_size=None,
        bounding_box_format=None,
        interpolation="bilinear",
        seed=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not isinstance(target_size, tuple) or len(target_size) != 2:
            raise ValueError(
                "JitteredResize() expects `target_size` to be "
                f"a tuple of two integers.  Received `target_size={target_size}`"
            )

        crop_size = crop_size or target_size
        self.interpolation = keras_cv.utils.get_interpolation(interpolation)
        self.scale_factor = keras_cv.utils.parse_factor(
            scale_factor,
            min_value=0.0,
            max_value=None,
            param_name="scale_factor",
            seed=seed,
        )
        self.crop_size = crop_size
        self.target_size = target_size
        self.inference_resizing = layers.Resizing(
            target_size[0],
            target_size[1],
            pad_to_aspect_ratio=True,
            interpolation=interpolation,
            bounding_box_format=bounding_box_format,
        )
        self.bounding_box_format = bounding_box_format
        self.seed = seed
        self.force_output_dense_images = True
        self.auto_vectorize = False

    def call(self, inputs, training=True):
        if training:
            return super().call(inputs, training)
        else:
            inputs = self._ensure_inputs_are_compute_dtype(inputs)
            inputs, meta_data = self._format_inputs(inputs)
            output = inputs
            output["images"] = self.inference_resizing(inputs["images"])
            return self._format_output(output, meta_data)

    def get_random_transformation(self, image=None, **kwargs):
        original_image_shape = tf.shape(image)
        image_shape = tf.cast(original_image_shape[0:2], tf.float32)

        scaled_size = tf.round(image_shape * self.scale_factor())
        scale = tf.minimum(
            scaled_size[0] / image_shape[0], scaled_size[1] / image_shape[1]
        )

        scaled_size = tf.round(image_shape * scale)
        image_scale = scaled_size / image_shape

        max_offset = scaled_size - self.crop_size
        max_offset = tf.where(
            tf.less(max_offset, 0), tf.zeros_like(max_offset), max_offset
        )
        offset = max_offset * tf.random.uniform([2], minval=0, maxval=1)
        offset = tf.cast(offset, tf.int32)

        return {
            "original_size": original_image_shape,
            "image_scale": image_scale,
            "scaled_size": scaled_size,
            "offset": offset,
        }

    def compute_image_signature(self, images):
        return tf.TensorSpec(
            shape=list(self.target_size) + [images.shape[-1]],
            dtype=self.compute_dtype,
        )

    def augment_image(self, image, transformation, **kwargs):
        # unpackage augmentation arguments
        scaled_size = transformation["scaled_size"]
        offset = transformation["offset"]
        target_size = self.target_size
        crop_size = self.crop_size

        scaled_image = tf.image.resize(
            image, tf.cast(scaled_size, tf.int32), method=self.interpolation
        )
        scaled_image = scaled_image[
            offset[0] : offset[0] + crop_size[0],
            offset[1] : offset[1] + crop_size[1],
            :,
        ]
        scaled_image = tf.image.pad_to_bounding_box(
            scaled_image, 0, 0, target_size[0], target_size[1]
        )
        return tf.cast(scaled_image, self.compute_dtype)

    def augment_bounding_boxes(self, bounding_boxes, transformation, **kwargs):
        if self.bounding_box_format is None:
            raise ValueError(
                "Please provide a `bounding_box_format` when augmenting "
                "bounding boxes with `JitteredResize()`."
            )

        image_scale = tf.cast(transformation["image_scale"], self.compute_dtype)
        offset = tf.cast(transformation["offset"], self.compute_dtype)
        original_size = transformation["original_size"]

        bounding_boxes = keras_cv.bounding_box.convert_format(
            bounding_boxes,
            image_shape=original_size,
            source=self.bounding_box_format,
            target="yxyx",
        )

        # Adjusts box coordinates based on image_scale and offset.
        yxyx = bounding_boxes[:, :4]
        rest = bounding_boxes[:, 4:]
        yxyx *= tf.tile(tf.expand_dims(image_scale, axis=0), [1, 2])
        yxyx -= tf.tile(tf.expand_dims(offset, axis=0), [1, 2])

        bounding_boxes = tf.concat([yxyx, rest], axis=-1)
        bounding_boxes = keras_cv.bounding_box.clip_to_image(
            bounding_boxes,
            image_shape=self.target_size + (3,),
            bounding_box_format="yxyx",
        )
        bounding_boxes = keras_cv.bounding_box.filter_sentinels(bounding_boxes)
        return keras_cv.bounding_box.convert_format(
            bounding_boxes,
            image_shape=self.target_size + (3,),
            source="yxyx",
            target=self.bounding_box_format,
        )

    def augment_label(self, label, transformation, **kwargs):
        return label

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
