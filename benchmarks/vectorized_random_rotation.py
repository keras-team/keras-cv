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
from keras_cv import bounding_box
from keras_cv.layers import RandomRotation
from keras_cv.layers.preprocessing.base_image_augmentation_layer import (
    BaseImageAugmentationLayer,
)
from keras_cv.utils import preprocessing as preprocessing_utils
from tensorflow import keras

H_AXIS = -3
W_AXIS = -2


class OldRandomRotation(BaseImageAugmentationLayer):
    """A preprocessing layer which randomly rotates images during training.

    This layer will apply random rotations to each image, filling empty space
    according to `fill_mode`.

    By default, random rotations are only applied during training.
    At inference time, the layer does nothing. If you need to apply random
    rotations at inference time, set `training` to True when calling the layer.

    Input pixel values can be of any range (e.g. `[0., 1.)` or `[0, 255]`) and
    of integer or floating point dtype. By default, the layer will output
    floats.

    Input shape:
      3D (unbatched) or 4D (batched) tensor with shape:
      `(..., height, width, channels)`, in `"channels_last"` format

    Output shape:
      3D (unbatched) or 4D (batched) tensor with shape:
      `(..., height, width, channels)`, in `"channels_last"` format

    Arguments:
      factor: a float represented as fraction of 2 Pi, or a tuple of size 2
        representing lower and upper bound for rotating clockwise and
        counter-clockwise. A positive values means rotating counter clock-wise,
        while a negative value means clock-wise. When represented as a single
        float, this value is used for both the upper and lower bound. For
        instance, `factor=(-0.2, 0.3)` results in an output rotation by a random
        amount in the range `[-20% * 2pi, 30% * 2pi]`. `factor=0.2` results in
        an output rotating by a random amount in the range
        `[-20% * 2pi, 20% * 2pi]`.
      fill_mode: Points outside the boundaries of the input are filled according
        to the given mode (one of `{"constant", "reflect", "wrap", "nearest"}`).
        - *reflect*: `(d c b a | a b c d | d c b a)` The input is extended by
          reflecting about the edge of the last pixel.
        - *constant*: `(k k k k | a b c d | k k k k)` The input is extended by
          filling all values beyond the edge with the same constant value k = 0.
        - *wrap*: `(a b c d | a b c d | a b c d)` The input is extended by
          wrapping around to the opposite edge.
        - *nearest*: `(a a a a | a b c d | d d d d)` The input is extended by
          the nearest pixel.
      interpolation: Interpolation mode. Supported values: `"nearest"`,
        `"bilinear"`.
      seed: Integer. Used to create a random seed.
      fill_value: a float represents the value to be filled outside the
        boundaries when `fill_mode="constant"`.
      bounding_box_format: The format of bounding boxes of input dataset. Refer
        https://github.com/keras-team/keras-cv/blob/master/keras_cv/bounding_box/converters.py
        for more details on supported bounding box formats.
      segmentation_classes: an optional integer with the number of classes in
        the input segmentation mask. Required iff augmenting data with sparse
        (non one-hot) segmentation masks. Include the background class in this
        count (e.g. for segmenting dog vs background, this should be set to 2).
    """

    def __init__(
        self,
        factor,
        fill_mode="reflect",
        interpolation="bilinear",
        seed=None,
        fill_value=0.0,
        bounding_box_format=None,
        segmentation_classes=None,
        **kwargs,
    ):
        super().__init__(seed=seed, **kwargs)
        self.factor = factor
        if isinstance(factor, (tuple, list)):
            self.lower = factor[0]
            self.upper = factor[1]
        else:
            self.lower = -factor
            self.upper = factor
        if self.upper < self.lower:
            raise ValueError(
                "Factor cannot have negative values, " "got {}".format(factor)
            )
        preprocessing_utils.check_fill_mode_and_interpolation(
            fill_mode, interpolation
        )
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.interpolation = interpolation
        self.seed = seed
        self.bounding_box_format = bounding_box_format
        self.segmentation_classes = segmentation_classes

    def get_random_transformation(self, **kwargs):
        min_angle = self.lower * 2.0 * np.pi
        max_angle = self.upper * 2.0 * np.pi
        angle = self._random_generator.uniform(
            shape=[1], minval=min_angle, maxval=max_angle
        )
        return {"angle": angle}

    def augment_image(self, image, transformation, **kwargs):
        return self._rotate_image(image, transformation)

    def _rotate_image(self, image, transformation):
        image = preprocessing_utils.ensure_tensor(image, self.compute_dtype)
        original_shape = image.shape
        image = tf.expand_dims(image, 0)
        image_shape = tf.shape(image)
        img_hd = tf.cast(image_shape[H_AXIS], tf.float32)
        img_wd = tf.cast(image_shape[W_AXIS], tf.float32)
        angle = transformation["angle"]
        output = preprocessing_utils.transform(
            image,
            preprocessing_utils.get_rotation_matrix(angle, img_hd, img_wd),
            fill_mode=self.fill_mode,
            fill_value=self.fill_value,
            interpolation=self.interpolation,
        )
        output = tf.squeeze(output, 0)
        output.set_shape(original_shape)
        return output

    def augment_bounding_boxes(
        self, bounding_boxes, transformation, image=None, **kwargs
    ):
        if self.bounding_box_format is None:
            raise ValueError(
                "`RandomRotation()` was called with bounding boxes, "
                "but no `bounding_box_format` was specified in the "
                "constructor. Please specify a bounding box format in the "
                "constructor. i.e. "
                "`RandomRotation(bounding_box_format='xyxy')`"
            )

        bounding_boxes = bounding_box.convert_format(
            bounding_boxes,
            source=self.bounding_box_format,
            target="xyxy",
            images=image,
        )
        image_shape = tf.shape(image)
        h = image_shape[H_AXIS]
        w = image_shape[W_AXIS]

        # origin coordinates, all the points on the image are rotated around
        # this point
        origin_x, origin_y = tf.cast(w / 2, dtype=self.compute_dtype), tf.cast(
            h / 2, dtype=self.compute_dtype
        )
        angle = transformation["angle"]
        angle = -angle
        # calculate coordinates of all four corners of the bounding box
        boxes = bounding_boxes["boxes"]
        point = tf.stack(
            [
                tf.stack([boxes[:, 0], boxes[:, 1]], axis=1),
                tf.stack([boxes[:, 2], boxes[:, 1]], axis=1),
                tf.stack([boxes[:, 2], boxes[:, 3]], axis=1),
                tf.stack([boxes[:, 0], boxes[:, 3]], axis=1),
            ],
            axis=1,
        )
        # point_x : x coordinates of all corners of the bounding box
        point_x = tf.gather(point, [0], axis=2)
        # point_y : y coordinates of all corners of the bounding box
        point_y = tf.gather(point, [1], axis=2)
        # rotated bounding box coordinates
        # new_x : new position of x coordinates of corners of bounding box
        new_x = (
            origin_x
            + tf.multiply(
                tf.cos(angle), tf.cast((point_x - origin_x), dtype=tf.float32)
            )
            - tf.multiply(
                tf.sin(angle), tf.cast((point_y - origin_y), dtype=tf.float32)
            )
        )
        # new_y : new position of y coordinates of corners of bounding box
        new_y = (
            origin_y
            + tf.multiply(
                tf.sin(angle), tf.cast((point_x - origin_x), dtype=tf.float32)
            )
            + tf.multiply(
                tf.cos(angle), tf.cast((point_y - origin_y), dtype=tf.float32)
            )
        )
        # rotated bounding box coordinates
        out = tf.concat([new_x, new_y], axis=2)
        # find readjusted coordinates of bounding box to represent it in corners
        # format
        min_coordinates = tf.math.reduce_min(out, axis=1)
        max_coordinates = tf.math.reduce_max(out, axis=1)
        boxes = tf.concat([min_coordinates, max_coordinates], axis=1)

        bounding_boxes = bounding_boxes.copy()
        bounding_boxes["boxes"] = boxes
        bounding_boxes = bounding_box.clip_to_image(
            bounding_boxes,
            bounding_box_format="xyxy",
            images=image,
        )
        # coordinates cannot be float values, it is casted to int32
        bounding_boxes = bounding_box.convert_format(
            bounding_boxes,
            source="xyxy",
            target=self.bounding_box_format,
            dtype=self.compute_dtype,
            images=image,
        )
        return bounding_boxes

    def augment_label(self, label, transformation, **kwargs):
        return label

    def augment_segmentation_mask(
        self, segmentation_mask, transformation, **kwargs
    ):
        # If segmentation_classes is specified, we have a dense segmentation
        # mask. We therefore one-hot encode before rotation to avoid bad
        # interpolation during the rotation transformation. We then make the
        # mask sparse again using tf.argmax.
        if self.segmentation_classes:
            one_hot_mask = tf.one_hot(
                tf.squeeze(segmentation_mask, axis=-1),
                self.segmentation_classes,
            )
            rotated_one_hot_mask = self._rotate_image(
                one_hot_mask, transformation
            )
            rotated_mask = tf.argmax(rotated_one_hot_mask, axis=-1)
            return tf.expand_dims(rotated_mask, axis=-1)
        else:
            if segmentation_mask.shape[-1] == 1:
                raise ValueError(
                    "Segmentation masks must be one-hot encoded, or "
                    "RandomRotate must be initialized with "
                    "`segmentation_classes`. `segmentation_classes` was not "
                    f"specified, and mask has shape {segmentation_mask.shape}"
                )
            rotated_mask = self._rotate_image(segmentation_mask, transformation)
            # Round because we are in one-hot encoding, and we may have
            # pixels with ambiguous value due to floating point math for
            # rotation.
            return tf.round(rotated_mask)

    def get_config(self):
        config = {
            "factor": self.factor,
            "fill_mode": self.fill_mode,
            "fill_value": self.fill_value,
            "interpolation": self.interpolation,
            "bounding_box_format": self.bounding_box_format,
            "segmentation_classes": self.segmentation_classes,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class RandomRotationTest(tf.test.TestCase):
    def test_consistency_with_old_implementation_bounding_boxes(self):
        input_image = np.random.random((2, 20, 20, 3)).astype(np.float32)
        bboxes = {
            "boxes": tf.ragged.constant(
                [[[2, 2, 4, 4], [1, 1, 3, 3]], [[2, 2, 4, 4]]],
                dtype=tf.float32,
            ),
            "classes": tf.ragged.constant(
                [[0, 1], [0]],
                dtype=tf.float32,
            ),
        }
        input = {
            "images": input_image,
            "bounding_boxes": bboxes,
        }

        layer = RandomRotation(factor=(0.5, 0.5), bounding_box_format="xyxy")
        old_layer = OldRandomRotation(
            factor=(0.5, 0.5), bounding_box_format="xyxy"
        )

        output = layer(input, training=True)
        old_output = old_layer(input, training=True)

        self.assertAllClose(output["images"], old_output["images"])
        self.assertAllClose(
            output["bounding_boxes"]["classes"],
            old_output["bounding_boxes"]["classes"],
        )
        self.assertAllClose(
            output["bounding_boxes"]["boxes"].to_tensor(),
            old_output["bounding_boxes"]["boxes"].to_tensor(),
        )

    def test_consistency_with_old_implementation_segmentation_masks(self):
        num_classes = 10
        input_image = np.random.random((2, 20, 20, 3)).astype(np.float32)
        masks = np.random.randint(2, size=(2, 20, 20, 1)) * (num_classes - 1)
        input = {
            "images": input_image,
            "segmentation_masks": masks,
        }

        layer = RandomRotation(
            factor=(0.5, 0.5),
            segmentation_classes=num_classes,
        )
        old_layer = OldRandomRotation(
            factor=(0.5, 0.5),
            segmentation_classes=num_classes,
        )

        output = layer(input, training=True)
        old_output = old_layer(input, training=True)

        self.assertAllClose(output["images"], old_output["images"])
        self.assertAllClose(
            output["segmentation_masks"], old_output["segmentation_masks"]
        )


if __name__ == "__main__":
    # Run benchmark
    (x_train, _), _ = keras.datasets.cifar10.load_data()
    x_train = x_train.astype(np.float32)

    num_images = [100, 200, 500, 1000]
    num_classes = 10
    results = {}
    aug_candidates = [RandomRotation, OldRandomRotation]
    aug_args = {"factor": 0.5}

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
