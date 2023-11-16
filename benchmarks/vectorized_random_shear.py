import time
import warnings
from unittest.mock import MagicMock

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

import keras_cv
from keras_cv import bounding_box
from keras_cv.layers import RandomShear
from keras_cv.layers.preprocessing.base_image_augmentation_layer import (
    BaseImageAugmentationLayer,
)
from keras_cv.utils import preprocessing


# Copied from:
# https://github.com/keras-team/keras-cv/blob/cd12204b1f6df37b15359b6adf222b9ef0f67dc8/keras_cv/layers/preprocessing/random_shear.py#L27
class OldRandomShear(BaseImageAugmentationLayer):
    """A preprocessing layer which randomly shears images during training.
    This layer will apply random shearings to each image, filling empty space
    according to `fill_mode`.
    By default, random shears are only applied during training.
    At inference time, the layer does nothing. If you need to apply random
    shear at inference time, set `training` to True when calling the layer.
    Input pixel values can be of any range and any data type.
    Input shape:
      3D (unbatched) or 4D (batched) tensor with shape:
      `(..., height, width, channels)`, in `"channels_last"` format
    Output shape:
      3D (unbatched) or 4D (batched) tensor with shape:
      `(..., height, width, channels)`, in `"channels_last"` format
    Args:
        x_factor: A tuple of two floats, a single float or a
            `keras_cv.FactorSampler`. For each augmented image a value is
            sampled from the provided range. If a float is passed, the range is
            interpreted as `(0, x_factor)`. Values represent a percentage of the
            image to shear over. For example, 0.3 shears pixels up to 30% of the
            way across the image. All provided values should be positive. If
            `None` is passed, no shear occurs on the X axis. Defaults to `None`.
        y_factor: A tuple of two floats, a single float or a
            `keras_cv.FactorSampler`. For each augmented image a value is
            sampled from the provided range. If a float is passed, the range is
            interpreted as `(0, y_factor)`. Values represent a percentage of the
            image to shear over. For example, 0.3 shears pixels up to 30% of the
            way across the image. All provided values should be positive. If
            `None` is passed, no shear occurs on the Y axis. Defaults to `None`.
        interpolation: interpolation method used in the
            `ImageProjectiveTransformV3` op. Supported values are `"nearest"`
            and `"bilinear"`. Defaults to `"bilinear"`.
        fill_mode: fill_mode in the `ImageProjectiveTransformV3` op. Supported
            values are `"reflect"`, `"wrap"`, `"constant"`, and `"nearest"`.
            Defaults to `"reflect"`.
        fill_value: fill_value in the `ImageProjectiveTransformV3` op.
            A `Tensor` of type `float32`. The value to be filled when fill_mode
            is constant". Defaults to `0.0`.
        bounding_box_format: The format of bounding boxes of input dataset.
            Refer to
            https://github.com/keras-team/keras-cv/blob/master/keras_cv/bounding_box/converters.py
            for more details on supported bounding box formats.
        seed: Integer. Used to create a random seed.
    """

    def __init__(
        self,
        x_factor=None,
        y_factor=None,
        interpolation="bilinear",
        fill_mode="reflect",
        fill_value=0.0,
        bounding_box_format=None,
        seed=None,
        **kwargs,
    ):
        super().__init__(seed=seed, **kwargs)
        if x_factor is not None:
            self.x_factor = preprocessing.parse_factor(
                x_factor, max_value=None, param_name="x_factor", seed=seed
            )
        else:
            self.x_factor = x_factor
        if y_factor is not None:
            self.y_factor = preprocessing.parse_factor(
                y_factor, max_value=None, param_name="y_factor", seed=seed
            )
        else:
            self.y_factor = y_factor
        if x_factor is None and y_factor is None:
            warnings.warn(
                "RandomShear received both `x_factor=None` and "
                "`y_factor=None`. As a result, the layer will perform no "
                "augmentation."
            )
        self.interpolation = interpolation
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.seed = seed
        self.bounding_box_format = bounding_box_format

    def get_random_transformation(self, **kwargs):
        x = self._get_shear_amount(self.x_factor)
        y = self._get_shear_amount(self.y_factor)
        return (x, y)

    def _get_shear_amount(self, constraint):
        if constraint is None:
            return None

        invert = preprocessing.random_inversion(self._seed_generator)
        return invert * constraint()

    def augment_image(self, image, transformation=None, **kwargs):
        image = tf.expand_dims(image, axis=0)

        x, y = transformation

        if x is not None:
            transform_x = OldRandomShear._format_transform(
                [1.0, x, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
            )
            image = preprocessing.transform(
                images=image,
                transforms=transform_x,
                interpolation=self.interpolation,
                fill_mode=self.fill_mode,
                fill_value=self.fill_value,
            )

        if y is not None:
            transform_y = OldRandomShear._format_transform(
                [1.0, 0.0, 0.0, y, 1.0, 0.0, 0.0, 0.0]
            )
            image = preprocessing.transform(
                images=image,
                transforms=transform_y,
                interpolation=self.interpolation,
                fill_mode=self.fill_mode,
                fill_value=self.fill_value,
            )

        return tf.squeeze(image, axis=0)

    def augment_label(self, label, transformation=None, **kwargs):
        return label

    def augment_bounding_boxes(
        self, bounding_boxes, transformation, image=None, **kwargs
    ):
        if self.bounding_box_format is None:
            raise ValueError(
                "`RandomShear()` was called with bounding boxes, "
                "but no `bounding_box_format` was specified in the "
                "constructor. Please specify a bounding box format in the "
                "constructor. i.e. `RandomShear(bounding_box_format='xyxy')`"
            )
        bounding_boxes = keras_cv.bounding_box.convert_format(
            bounding_boxes,
            source=self.bounding_box_format,
            target="rel_xyxy",
            images=image,
            dtype=self.compute_dtype,
        )
        x, y = transformation
        extended_boxes = self._convert_to_extended_corners_format(
            bounding_boxes["boxes"]
        )
        if x is not None:
            extended_boxes = (
                self._apply_horizontal_transformation_to_bounding_box(
                    extended_boxes, x
                )
            )
        # apply vertical shear
        if y is not None:
            extended_boxes = (
                self._apply_vertical_transformation_to_bounding_box(
                    extended_boxes, y
                )
            )

        boxes = self._convert_to_four_coordinate(extended_boxes, x, y)
        bounding_boxes = bounding_boxes.copy()
        bounding_boxes["boxes"] = boxes
        bounding_boxes = bounding_box.clip_to_image(
            bounding_boxes, images=image, bounding_box_format="rel_xyxy"
        )
        bounding_boxes = keras_cv.bounding_box.convert_format(
            bounding_boxes,
            source="rel_xyxy",
            target=self.bounding_box_format,
            images=image,
            dtype=self.compute_dtype,
        )
        return bounding_boxes

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "x_factor": self.x_factor,
                "y_factor": self.y_factor,
                "interpolation": self.interpolation,
                "fill_mode": self.fill_mode,
                "fill_value": self.fill_value,
                "bounding_box_format": self.bounding_box_format,
                "seed": self.seed,
            }
        )
        return config

    @staticmethod
    def _format_transform(transform):
        transform = tf.convert_to_tensor(transform, dtype=tf.float32)
        return transform[tf.newaxis]

    @staticmethod
    def _convert_to_four_coordinate(extended_bboxes, x, y):
        """convert from extended coordinates to 4 coordinates system"""
        (
            top_left_x,
            top_left_y,
            bottom_right_x,
            bottom_right_y,
            top_right_x,
            top_right_y,
            bottom_left_x,
            bottom_left_y,
        ) = tf.split(extended_bboxes, 8, axis=1)

        # choose x1,x2 when x>0
        def positive_case_x():
            final_x1 = bottom_left_x
            final_x2 = top_right_x
            return final_x1, final_x2

        # choose x1,x2 when x<0
        def negative_case_x():
            final_x1 = top_left_x
            final_x2 = bottom_right_x
            return final_x1, final_x2

        if x is not None:
            final_x1, final_x2 = tf.cond(
                tf.less(x, 0), negative_case_x, positive_case_x
            )
        else:
            final_x1, final_x2 = top_left_x, bottom_right_x

        # choose y1,y2 when y > 0
        def positive_case_y():
            final_y1 = top_right_y
            final_y2 = bottom_left_y
            return final_y1, final_y2

        # choose y1,y2 when y < 0
        def negative_case_y():
            final_y1 = top_left_y
            final_y2 = bottom_right_y
            return final_y1, final_y2

        if y is not None:
            final_y1, final_y2 = tf.cond(
                tf.less(y, 0), negative_case_y, positive_case_y
            )
        else:
            final_y1, final_y2 = top_left_y, bottom_right_y
        return tf.concat(
            [final_x1, final_y1, final_x2, final_y2],
            axis=1,
        )

    @staticmethod
    def _apply_horizontal_transformation_to_bounding_box(
        extended_bounding_boxes, x
    ):
        # create transformation matrix [1,4]
        matrix = tf.stack([1.0, -x, 0, 1.0], axis=0)
        # reshape it to [2,2]
        matrix = tf.reshape(matrix, (2, 2))
        # reshape unnormalized bboxes from [N,8] -> [N*4,2]
        new_bboxes = tf.reshape(extended_bounding_boxes, (-1, 2))
        # [[1,x`],[y`,1]]*[x,y]->[new_x,new_y]
        transformed_bboxes = tf.reshape(
            tf.einsum("ij,kj->ki", matrix, new_bboxes), (-1, 8)
        )
        return transformed_bboxes

    @staticmethod
    def _apply_vertical_transformation_to_bounding_box(
        extended_bounding_boxes, y
    ):
        # create transformation matrix [1,4]
        matrix = tf.stack([1.0, 0, -y, 1.0], axis=0)
        # reshape it to [2,2]
        matrix = tf.reshape(matrix, (2, 2))
        # reshape unnormalized bboxes from [N,8] -> [N*4,2]
        new_bboxes = tf.reshape(extended_bounding_boxes, (-1, 2))
        # [[1,x`],[y`,1]]*[x,y]->[new_x,new_y]
        transformed_bboxes = tf.reshape(
            tf.einsum("ij,kj->ki", matrix, new_bboxes), (-1, 8)
        )
        return transformed_bboxes

    @staticmethod
    def _convert_to_extended_corners_format(boxes):
        """splits corner boxes top left,bottom right to 4 corners top left,
        bottom right,top right and bottom left"""
        x1, y1, x2, y2 = tf.split(boxes, [1, 1, 1, 1], axis=-1)
        new_boxes = tf.concat(
            [x1, y1, x2, y2, x2, y1, x1, y2],
            axis=-1,
        )
        return new_boxes


# End copy


class RandomShearTest(tf.test.TestCase):
    def test_consistency_with_old_implementation(self):
        # Prepare inputs
        batch_size = 2
        images = tf.random.uniform(shape=(batch_size, 64, 64, 3))
        shear_x = tf.random.uniform(shape=())
        shear_y = tf.random.uniform(shape=())

        bounding_boxes = {
            "boxes": tf.constant(
                [
                    [[10.0, 20.0, 40.0, 50.0], [12.0, 22.0, 42.0, 54.0]],
                    [[15.0, 16.0, 17, 18], [12.0, 22.0, 42.0, 54.0]],
                ],
                dtype=tf.float32,
            ),
            "classes": tf.constant([[0, 0], [0, 0]], dtype=tf.float32),
        }

        # Build layers
        old_layer = OldRandomShear(
            x_factor=(shear_x, shear_x),
            y_factor=(shear_y, shear_y),
            seed=1234,
            bounding_box_format="xyxy",
        )
        new_layer = RandomShear(
            x_factor=(shear_x, shear_x),
            y_factor=(shear_y, shear_y),
            seed=1234,
            bounding_box_format="xyxy",
        )

        # Disable random negation to get deterministic factor
        old_layer.get_random_transformation = MagicMock(
            return_value=(
                old_layer.x_factor(),
                old_layer.y_factor(),
            )
        )
        new_layer.get_random_transformation_batch = MagicMock(
            return_value={
                "shear_x": new_layer.x_factor((batch_size, 1)),
                "shear_y": new_layer.y_factor((batch_size, 1)),
            }
        )

        # Run inference + compare outputs:
        old_output = old_layer(
            {"images": images, "bounding_boxes": bounding_boxes}
        )
        output = new_layer({"images": images, "bounding_boxes": bounding_boxes})

        self.assertAllClose(output["images"], old_output["images"])
        self.assertAllClose(
            output["bounding_boxes"]["boxes"].to_tensor(),
            old_output["bounding_boxes"]["boxes"].to_tensor(),
        )
        self.assertAllClose(
            output["bounding_boxes"]["classes"],
            old_output["bounding_boxes"]["classes"],
        )


if __name__ == "__main__":
    # Run benchmark
    (x_train, _), _ = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype(np.float32)

    num_images = [1000, 2000, 5000, 10000]
    results = {}
    aug_candidates = [RandomShear, OldRandomShear]
    aug_args = {"x_factor": (5, 5), "y_factor": (5, 5)}

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
            print(f"Runtime for {c}, n_images={n_images}: {t1 - t0}")
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
            print(f"Runtime for {c}, n_images={n_images}: {t1 - t0}")
        results[c] = runtimes

        # Not running with XLA as it does not support ImageProjectiveTransformV3

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
