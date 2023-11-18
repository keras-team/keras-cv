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

import time
import unittest

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras_cv import bounding_box
from keras_cv.layers import RandomCrop
from keras_cv.layers.preprocessing.base_image_augmentation_layer import (
    BaseImageAugmentationLayer,
)
from tensorflow import keras

H_AXIS = -3
W_AXIS = -2


class OldRandomCrop(BaseImageAugmentationLayer):
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
    of integer or floating point dtype. By default, the layer will output
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
        self.auto_vectorize = False
        self.bounding_box_format = bounding_box_format

    def get_random_transformation(self, image=None, **kwargs):
        image_shape = tf.shape(image)
        h_diff = image_shape[H_AXIS] - self.height
        w_diff = image_shape[W_AXIS] - self.width
        dtype = image_shape.dtype
        rands = self._random_generator.uniform([2], 0, dtype.max, dtype)
        h_start = rands[0] % (h_diff + 1)
        w_start = rands[1] % (w_diff + 1)
        return {"top": h_start, "left": w_start}

    def augment_image(self, image, transformation, **kwargs):
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


class RandomCropTest(tf.test.TestCase):
    def test_consistency_with_old_impl_crop(self):
        ori_height, ori_width = 256, 256
        height, width = 128, 128
        input_image = np.random.random((ori_height, ori_width, 3)).astype(
            np.float32
        )
        bboxes = {
            "boxes": tf.convert_to_tensor([[100, 100, 200, 200]]),
            "classes": tf.convert_to_tensor([1]),
        }
        input = {"images": input_image, "bounding_boxes": bboxes}

        layer = RandomCrop(
            height=height, width=width, bounding_box_format="xyxy"
        )
        old_layer = OldRandomCrop(
            height=height, width=width, bounding_box_format="xyxy"
        )

        # manually set height_offset and width_offset
        height_offset = 20
        width_offset = 30
        transformations = {
            "tops": tf.ones((1, 1)) * (height_offset / (ori_height - height)),
            "lefts": tf.ones((1, 1)) * (width_offset / (ori_width - width)),
        }
        old_transformation = {
            "top": tf.convert_to_tensor(height_offset, dtype=tf.int32),
            "left": tf.convert_to_tensor(width_offset, dtype=tf.int32),
        }

        with unittest.mock.patch.object(
            layer,
            "get_random_transformation_batch",
            return_value=transformations,
        ):
            output = layer(input, training=True)
        with unittest.mock.patch.object(
            old_layer,
            "get_random_transformation",
            return_value=old_transformation,
        ):
            old_output = old_layer(input, training=True)

        self.assertAllClose(
            output["bounding_boxes"]["boxes"],
            old_output["bounding_boxes"]["boxes"].to_tensor(-1),
        )
        self.assertAllClose(output["images"], old_output["images"])

    def test_consistency_with_old_impl_resize(self):
        input_image = np.random.random((256, 256, 3)).astype(np.float32)
        bboxes = {
            "boxes": tf.convert_to_tensor([[100, 100, 200, 200]]),
            "classes": tf.convert_to_tensor([1]),
        }
        input = {"images": input_image, "bounding_boxes": bboxes}

        layer = RandomCrop(height=512, width=512, bounding_box_format="xyxy")
        old_layer = OldRandomCrop(
            height=512, width=512, bounding_box_format="xyxy"
        )

        output = layer(input, training=True)
        old_output = old_layer(input, training=True)

        self.assertAllClose(
            output["bounding_boxes"]["boxes"],
            old_output["bounding_boxes"]["boxes"].to_tensor(-1),
        )
        self.assertAllClose(output["images"], old_output["images"])


if __name__ == "__main__":
    # Run benchmark
    (x_train, _), _ = keras.datasets.cifar10.load_data()
    x_train = x_train.astype(np.float32)

    num_images = [100, 200, 500, 1000]
    num_classes = 10
    results = {}
    aug_candidates = [RandomCrop, OldRandomCrop]
    aug_args = {"height": 16, "width": 16}

    for aug in aug_candidates:
        # Eager Mode
        c = aug.__name__
        layer = aug(**aug_args)
        runtimes = []
        print(f"Timing {c}")

        for n_images in num_images:
            # warmup
            layer(x_train[:n_images])

            t0 = time.time()
            r1 = layer(x_train[:n_images])
            t1 = time.time()
            runtimes.append(t1 - t0)
            print(f"Runtime for {c}, n_images={n_images}: {t1-t0}")
        results[c] = runtimes

        # Graph Mode
        c = aug.__name__ + " Graph Mode"
        layer = aug(**aug_args)

        @tf.function()
        def apply_aug(inputs):
            return layer(inputs)

        runtimes = []
        print(f"Timing {c}")

        for n_images in num_images:
            # warmup
            apply_aug(x_train[:n_images])

            t0 = time.time()
            r1 = apply_aug(x_train[:n_images])
            t1 = time.time()
            runtimes.append(t1 - t0)
            print(f"Runtime for {c}, n_images={n_images}: {t1-t0}")
        results[c] = runtimes

        # XLA Mode
        # cannot run tf.image.crop_and_resize on XLA

    plt.figure()
    for key in results:
        plt.plot(num_images, results[key], label=key)
        plt.xlabel("Number images")

    plt.ylabel("Runtime (seconds)")
    plt.legend()
    plt.savefig("comparison.png")

    # So we can actually see more relevant margins
    del results[aug_candidates[1].__name__]
    plt.figure()
    for key in results:
        plt.plot(num_images, results[key], label=key)
        plt.xlabel("Number images")

    plt.ylabel("Runtime (seconds)")
    plt.legend()
    plt.savefig("comparison_no_old_eager.png")

    # Run unit tests
    tf.test.main()
