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

import tensorflow as tf
from tensorflow import keras

from keras_cv import bounding_box
from keras_cv.layers.preprocessing.vectorized_base_image_augmentation_layer import (
    BOUNDING_BOXES,
)
from keras_cv.layers.preprocessing.vectorized_base_image_augmentation_layer import (
    IMAGES,
)
from keras_cv.layers.preprocessing.vectorized_base_image_augmentation_layer import (
    LABELS,
)
from keras_cv.layers.preprocessing.vectorized_base_image_augmentation_layer import (
    VectorizedBaseImageAugmentationLayer,
)

# In order to support both unbatched and batched inputs, the horizontal
# and verticle axis is reverse indexed
H_AXIS = -3
W_AXIS = -2


@keras.utils.register_keras_serializable(package="keras_cv")
class RandomCrop(VectorizedBaseImageAugmentationLayer):
    """A preprocessing layer which randomly crops images during training.

    During training, this layer will randomly choose a location to crop images
    down to a target size.

    At inference time, and during training if an input image is smaller than the
    target size, the input will be resized and cropped so as to return the
    largest possible window in the image that matches the target aspect ratio.
    If you need to apply random cropping at inference time, set `training` to
    True when calling the layer.

    Input pixel values can be of any range (e.g. `[0., 1.)` or `[0, 255]`) and
    of interger or floating point dtype.

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
    """

    def __init__(
        self, height, width, seed=None, bounding_box_format=None, **kwargs
    ):
        super().__init__(
            **kwargs, autocast=False, seed=seed, force_generator=True
        )
        self.height = height
        self.width = width
        self.seed = seed
        self.bounding_box_format = bounding_box_format

    def compute_ragged_image_signature(self, images):
        ragged_spec = tf.RaggedTensorSpec(
            shape=(self.height, self.width, images.shape[-1]),
            ragged_rank=1,
            dtype=self.compute_dtype,
        )
        return ragged_spec

    def get_random_transformation_batch(self, batch_size, **kwargs):
        tops = self._random_generator.random_uniform(
            shape=(batch_size, 1), minval=0, maxval=1, dtype=tf.float32
        )
        lefts = self._random_generator.random_uniform(
            shape=(batch_size, 1), minval=0, maxval=1, dtype=tf.float32
        )
        return {"tops": tops, "lefts": lefts}

    def augment_ragged_image(self, image, transformation, **kwargs):
        image = tf.expand_dims(image, axis=0)
        tops = transformation["tops"]
        lefts = transformation["lefts"]
        transformation = {
            "tops": tf.expand_dims(tops, axis=0),
            "lefts": tf.expand_dims(lefts, axis=0),
        }
        image = self.augment_images(
            images=image, transformations=transformation, **kwargs
        )
        return tf.squeeze(image, axis=0)

    def augment_images(self, images, transformations, **kwargs):
        image_shape = tf.shape(images)
        h_diff = image_shape[H_AXIS] - self.height
        w_diff = image_shape[W_AXIS] - self.width
        if h_diff >= 0 and w_diff >= 0:
            return self._crop_images(images, transformations)
        else:
            return self._resize_images(images)

    def augment_labels(self, labels, transformations, **kwargs):
        return labels

    def augment_bounding_boxes(
        self, bounding_boxes, transformations, images=None, **kwargs
    ):
        if self.bounding_box_format is None:
            raise ValueError(
                "`RandomCrop()` was called with bounding boxes,"
                "but no `bounding_box_format` was specified in the constructor."
                "Please specify a bounding box format in the constructor. i.e."
                "`RandomCrop(bounding_box_format='xyxy')`"
            )
        if isinstance(bounding_boxes["boxes"], tf.RaggedTensor):
            bounding_boxes = bounding_box.to_dense(
                bounding_boxes, default_value=-1
            )
        image_shape = tf.shape(images)

        bounding_boxes = bounding_box.convert_format(
            bounding_boxes,
            source=self.bounding_box_format,
            target="xyxy",
            images=images,
        )
        h_diff = image_shape[H_AXIS] - self.height
        w_diff = image_shape[W_AXIS] - self.width
        if h_diff >= 0 and w_diff >= 0:
            bounding_boxes = self._crop_bounding_boxes(
                images, bounding_boxes, transformations
            )
        else:
            bounding_boxes = self._resize_bounding_boxes(
                images,
                bounding_boxes,
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
            image_shape=(self.height, self.width, image_shape[-1]),
        )
        return bounding_boxes

    def _crop_images(self, images, transformations):
        image_shape = tf.shape(images)
        batch_size = image_shape[0]
        image_height = tf.cast(image_shape[H_AXIS], dtype=tf.float32)
        image_width = tf.cast(image_shape[W_AXIS], dtype=tf.float32)

        tops = transformations["tops"]
        lefts = transformations["lefts"]
        x1s = lefts * (image_width - self.width)
        y1s = tops * (image_height - self.height)
        x2s = x1s + self.width
        y2s = y1s + self.height
        # normalize
        x1s /= image_width
        y1s /= image_height
        x2s /= image_width
        y2s /= image_height
        boxes = tf.concat([y1s, x1s, y2s, x2s], axis=-1)

        images = tf.image.crop_and_resize(
            images,
            boxes,
            tf.range(batch_size),
            [self.height, self.width],
            method="nearest",
        )
        return tf.cast(images, dtype=self.compute_dtype)

    def _resize_images(self, images):
        resizing_layer = keras.layers.Resizing(self.height, self.width)
        outputs = resizing_layer(images)
        return tf.cast(outputs, dtype=self.compute_dtype)

    def _crop_bounding_boxes(self, images, bounding_boxes, transformation):
        outputs = bounding_boxes.copy()
        image_shape = tf.shape(images)
        tops = transformation["tops"]
        lefts = transformation["lefts"]
        image_height = tf.cast(image_shape[H_AXIS], dtype=tf.float32)
        image_width = tf.cast(image_shape[W_AXIS], dtype=tf.float32)

        # compute offsets for xyxy bounding_boxes
        top_offsets = tf.cast(
            tf.math.round(tops * (image_height - self.height)),
            dtype=self.compute_dtype,
        )
        left_offsets = tf.cast(
            tf.math.round(lefts * (image_width - self.width)),
            dtype=self.compute_dtype,
        )

        x1s, y1s, x2s, y2s = tf.split(bounding_boxes["boxes"], 4, axis=-1)
        x1s -= tf.expand_dims(left_offsets, axis=1)
        y1s -= tf.expand_dims(top_offsets, axis=1)
        x2s -= tf.expand_dims(left_offsets, axis=1)
        y2s -= tf.expand_dims(top_offsets, axis=1)
        outputs["boxes"] = tf.concat([x1s, y1s, x2s, y2s], axis=-1)
        return outputs

    def _resize_bounding_boxes(self, images, bounding_boxes):
        outputs = bounding_boxes.copy()
        image_shape = tf.shape(images)
        x_scale = tf.cast(
            self.width / image_shape[W_AXIS], dtype=self.compute_dtype
        )
        y_scale = tf.cast(
            self.height / image_shape[H_AXIS], dtype=self.compute_dtype
        )
        x1s, y1s, x2s, y2s = tf.split(bounding_boxes["boxes"], 4, axis=-1)
        outputs["boxes"] = tf.concat(
            [
                x1s * x_scale,
                y1s * y_scale,
                x2s * x_scale,
                y2s * y_scale,
            ],
            axis=-1,
        )
        return outputs

    def _batch_augment(self, inputs):
        # overwrite _batch_augment to support raw_images for
        # augment_bounding_boxes
        images = inputs.get(IMAGES, None)
        raw_images = images  # needs raw_images for augment_bounding_boxes
        labels = inputs.get(LABELS, None)
        bounding_boxes = inputs.get(BOUNDING_BOXES, None)
        batch_size = tf.shape(images)[0]
        transformations = self.get_random_transformation_batch(
            batch_size,
        )

        if isinstance(images, tf.RaggedTensor):
            inputs_for_raggeds = {"transformations": transformations, **inputs}
            print("inputs_for_raggeds", inputs_for_raggeds)
            print(
                "self._unwrap_ragged_image_call", self._unwrap_ragged_image_call
            )
            images = tf.map_fn(
                self._unwrap_ragged_image_call,
                inputs_for_raggeds,
                fn_output_signature=self.compute_ragged_image_signature(images),
            )
        else:
            images = self.augment_images(
                images, transformations=transformations
            )

        result = {IMAGES: images}
        if labels is not None:
            labels = self.augment_targets(
                labels, transformations=transformations
            )
            result[LABELS] = labels

        if bounding_boxes is not None:
            bounding_boxes = self.augment_bounding_boxes(
                bounding_boxes,
                transformations=transformations,
                images=raw_images,
            )
            bounding_boxes = bounding_box.to_ragged(bounding_boxes)
            result[BOUNDING_BOXES] = bounding_boxes

        # preserve any additional inputs unmodified by this layer.
        for key in inputs.keys() - result.keys():
            result[key] = inputs[key]
        return result

    def get_config(self):
        config = {
            "height": self.height,
            "width": self.width,
            "seed": self.seed,
            "bounding_box_format": self.bounding_box_format,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)
