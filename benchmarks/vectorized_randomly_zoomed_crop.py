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

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

from keras_cv import core
from keras_cv.layers import RandomlyZoomedCrop
from keras_cv.layers.preprocessing.base_image_augmentation_layer import (
    BaseImageAugmentationLayer,
)
from keras_cv.utils import preprocessing as preprocessing_utils


class OldRandomlyZoomedCrop(BaseImageAugmentationLayer):
    """Randomly crops a part of an image and zooms it by a provided amount size.

    This implementation takes a distortion-oriented approach, which means the
    amount of distortion in the image is proportional to the `zoom_factor`
    argument. To do this, we first sample a random value for `zoom_factor` and
    `aspect_ratio_factor`. Further we deduce a `crop_size` which abides by the
    calculated aspect ratio. Finally we do the actual cropping operation and
    resize the image to `(height, width)`.

    Args:
        height: The height of the output shape.
        width: The width of the output shape.
        zoom_factor: A tuple of two floats, ConstantFactorSampler or
            UniformFactorSampler. Represents the area relative to the original
            image of the cropped image before resizing it to `(height, width)`.
        aspect_ratio_factor: A tuple of two floats, ConstantFactorSampler or
            UniformFactorSampler. Aspect ratio means the ratio of width to
            height of the cropped image. In the context of this layer, the
            aspect ratio sampled represents a value to distort the aspect ratio
            by.
            Represents the lower and upper bound for the aspect ratio of the
            cropped image before resizing it to `(height, width)`. For most
            tasks, this should be `(3/4, 4/3)`. To perform a no-op provide the
            value `(1.0, 1.0)`.
        interpolation: (Optional) A string specifying the sampling method for
            resizing, defaults to "bilinear".
        seed: (Optional) Used to create a random seed, defaults to `None`.
    """

    def __init__(
        self,
        height,
        width,
        zoom_factor,
        aspect_ratio_factor,
        interpolation="bilinear",
        seed=None,
        **kwargs,
    ):
        super().__init__(seed=seed, **kwargs)
        self.height = height
        self.width = width
        self.aspect_ratio_factor = preprocessing_utils.parse_factor(
            aspect_ratio_factor,
            min_value=0.0,
            max_value=None,
            param_name="aspect_ratio_factor",
            seed=seed,
        )
        self.zoom_factor = preprocessing_utils.parse_factor(
            zoom_factor,
            min_value=0.0,
            max_value=None,
            param_name="zoom_factor",
            seed=seed,
        )

        self._check_class_arguments(
            height, width, zoom_factor, aspect_ratio_factor
        )
        self.force_output_dense_images = True
        self.interpolation = interpolation
        self.seed = seed

    def get_random_transformation(
        self, image=None, label=None, bounding_box=None, **kwargs
    ):
        zoom_factor = self.zoom_factor()
        aspect_ratio = self.aspect_ratio_factor()

        original_height = tf.cast(tf.shape(image)[-3], tf.float32)
        original_width = tf.cast(tf.shape(image)[-2], tf.float32)

        crop_size = (
            tf.round(self.height / zoom_factor),
            tf.round(self.width / zoom_factor),
        )

        new_height = crop_size[0] / tf.sqrt(aspect_ratio)

        new_width = crop_size[1] * tf.sqrt(aspect_ratio)

        height_offset = self._random_generator.random_uniform(
            (),
            minval=tf.minimum(0.0, original_height - new_height),
            maxval=tf.maximum(0.0, original_height - new_height),
            dtype=tf.float32,
        )

        width_offset = self._random_generator.random_uniform(
            (),
            minval=tf.minimum(0.0, original_width - new_width),
            maxval=tf.maximum(0.0, original_width - new_width),
            dtype=tf.float32,
        )

        new_height = new_height / original_height
        new_width = new_width / original_width

        height_offset = height_offset / original_height
        width_offset = width_offset / original_width

        return (new_height, new_width, height_offset, width_offset)

    def call(self, inputs, training=True):
        if training:
            return super().call(inputs, training)
        else:
            inputs = self._ensure_inputs_are_compute_dtype(inputs)
            inputs, meta_data = self._format_inputs(inputs)
            output = inputs
            # self._resize() returns valid results for both batched and
            # unbatched
            output["images"] = self._resize(inputs["images"])

            return self._format_output(output, meta_data)

    def augment_image(self, image, transformation, **kwargs):
        image_shape = tf.shape(image)

        height = tf.cast(image_shape[-3], tf.float32)
        width = tf.cast(image_shape[-2], tf.float32)

        image = tf.expand_dims(image, axis=0)
        new_height, new_width, height_offset, width_offset = transformation

        transform = OldRandomlyZoomedCrop._format_transform(
            [
                new_width,
                0.0,
                width_offset * width,
                0.0,
                new_height,
                height_offset * height,
                0.0,
                0.0,
            ]
        )

        image = preprocessing_utils.transform(
            images=image,
            transforms=transform,
            output_shape=(self.height, self.width),
            interpolation=self.interpolation,
            fill_mode="reflect",
        )

        return tf.squeeze(image, axis=0)

    @staticmethod
    def _format_transform(transform):
        transform = tf.convert_to_tensor(transform, dtype=tf.float32)
        return transform[tf.newaxis]

    def _resize(self, image):
        outputs = keras.preprocessing.image.smart_resize(
            image, (self.height, self.width)
        )
        # smart_resize will always output float32, so we need to re-cast.
        return tf.cast(outputs, self.compute_dtype)

    def _check_class_arguments(
        self, height, width, zoom_factor, aspect_ratio_factor
    ):
        if not isinstance(height, int):
            raise ValueError(
                "`height` must be an integer. Received height={height}"
            )

        if not isinstance(width, int):
            raise ValueError(
                "`width` must be an integer. Received width={width}"
            )

        if (
            not isinstance(zoom_factor, (tuple, list, core.FactorSampler))
            or isinstance(zoom_factor, float)
            or isinstance(zoom_factor, int)
        ):
            raise ValueError(
                "`zoom_factor` must be tuple of two positive floats"
                " or keras_cv.core.FactorSampler instance. Received "
                f"zoom_factor={zoom_factor}"
            )

        if (
            not isinstance(
                aspect_ratio_factor, (tuple, list, core.FactorSampler)
            )
            or isinstance(aspect_ratio_factor, float)
            or isinstance(aspect_ratio_factor, int)
        ):
            raise ValueError(
                "`aspect_ratio_factor` must be tuple of two positive floats or "
                "keras_cv.core.FactorSampler instance. Received "
                f"aspect_ratio_factor={aspect_ratio_factor}"
            )

    def augment_target(self, augment_target, **kwargs):
        return augment_target

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "height": self.height,
                "width": self.width,
                "zoom_factor": self.zoom_factor,
                "aspect_ratio_factor": self.aspect_ratio_factor,
                "interpolation": self.interpolation,
                "seed": self.seed,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        if isinstance(config["zoom_factor"], dict):
            config["zoom_factor"] = keras.utils.deserialize_keras_object(
                config["zoom_factor"]
            )
        if isinstance(config["aspect_ratio_factor"], dict):
            config[
                "aspect_ratio_factor"
            ] = keras.utils.deserialize_keras_object(
                config["aspect_ratio_factor"]
            )
        return cls(**config)

    def _crop_and_resize(self, image, transformation, method=None):
        image = tf.expand_dims(image, axis=0)
        boxes = transformation

        # See bit.ly/tf_crop_resize for more details
        augmented_image = tf.image.crop_and_resize(
            image,  # image shape: [B, H, W, C]
            boxes,  # boxes: (1, 4) in this case; represents area
            # to be cropped from the original image
            [0],  # box_indices: maps boxes to images along batch axis
            # [0] since there is only one image
            (self.height, self.width),  # output size
            method=method or self.interpolation,
        )

        return tf.squeeze(augmented_image, axis=0)


class RandomlyZoomedCropTest(tf.test.TestCase):
    def test_consistency_with_old_impl(self):
        image_shape = (1, 64, 64, 3)
        height, width = 32, 32
        fixed_zoom_factor = (0.8, 0.8)
        fixed_aspect_ratio_factor = (3.0 / 4.0, 3.0 / 4.0)
        fixed_seed = 2023
        image = tf.random.uniform(shape=image_shape) * 255.0

        layer = RandomlyZoomedCrop(
            height,
            width,
            fixed_zoom_factor,
            fixed_aspect_ratio_factor,
            seed=fixed_seed,
        )
        old_layer = OldRandomlyZoomedCrop(
            height,
            width,
            fixed_zoom_factor,
            fixed_aspect_ratio_factor,
            seed=fixed_seed,
        )

        output = layer(image)
        old_output = old_layer(image)

        self.assertAllClose(old_output, output)


if __name__ == "__main__":
    # Run benchmark
    (x_train, _), _ = keras.datasets.cifar10.load_data()
    x_train = x_train.astype(np.float32)

    num_images = [1000, 2000, 5000, 10000]
    results = {}
    aug_candidates = [RandomlyZoomedCrop, OldRandomlyZoomedCrop]
    aug_args = {
        "height": 16,
        "width": 16,
        "zoom_factor": (0.8, 1.2),
        "aspect_ratio_factor": (3.0 / 4.0, 4.0 / 3.0),
    }

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
        # cannot run tf.raw_ops.ImageProjectiveTransformV3 on XLA
        # for more information please refer:
        # https://github.com/tensorflow/tensorflow/issues/55194

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
