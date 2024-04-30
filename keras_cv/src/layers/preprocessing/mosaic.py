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

from keras_cv.src import bounding_box
from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.layers.preprocessing.vectorized_base_image_augmentation_layer import (  # noqa: E501
    BATCHED,
)
from keras_cv.src.layers.preprocessing.vectorized_base_image_augmentation_layer import (  # noqa: E501
    BOUNDING_BOXES,
)
from keras_cv.src.layers.preprocessing.vectorized_base_image_augmentation_layer import (  # noqa: E501
    IMAGES,
)
from keras_cv.src.layers.preprocessing.vectorized_base_image_augmentation_layer import (  # noqa: E501
    LABELS,
)
from keras_cv.src.layers.preprocessing.vectorized_base_image_augmentation_layer import (  # noqa: E501
    SEGMENTATION_MASKS,
)
from keras_cv.src.layers.preprocessing.vectorized_base_image_augmentation_layer import (  # noqa: E501
    VectorizedBaseImageAugmentationLayer,
)
from keras_cv.src.utils import preprocessing as preprocessing_utils


@keras_cv_export("keras_cv.layers.Mosaic")
class Mosaic(VectorizedBaseImageAugmentationLayer):
    """Mosaic implements the mosaic data augmentation technique.

    Mosaic data augmentation first takes 4 images from the batch and makes a
    grid. After that based on the offset, a crop is taken to form the mosaic
    image. Labels are in the same ratio as the area of their images in the
    output image. Bounding boxes are translated according to the position of the
    4 images.

    Args:
        offset: A tuple of two floats, a single float or
            `keras_cv.FactorSampler`. `offset` is used to determine the offset
            of the mosaic center from the top-left corner of the mosaic. If a
            tuple is used, the x and y coordinates of the mosaic center are
            sampled between the two values for every image augmented. If a
            single float is used, a value between `0.0` and the passed float is
            sampled. In order to ensure the value is always the same, please
            pass a tuple with two identical floats: `(0.5, 0.5)`. Defaults to
            (0.25, 0.75).
        bounding_box_format: a case-insensitive string (for example, "xyxy") to
            be passed if bounding boxes are being augmented by this layer. Each
            bounding box is defined by at least these 4 values. The inputs may
            contain additional information such as classes and confidence after
            these 4 values but these values will be ignored and returned as is.
            For detailed information on the supported formats, see the
            [KerasCV bounding box documentation](https://keras.io/api/keras_cv/bounding_box/formats/).
            Defaults to None.
        seed: integer, used to create a random seed.

    References:
        - [Yolov4 paper](https://arxiv.org/pdf/2004.10934).
        - [Yolov5 implementation](https://github.com/ultralytics/yolov5).
        - [YoloX implementation](https://github.com/Megvii-BaseDetection/YOLOX)

    Example:
    ```python
    (images, labels), _ = keras.datasets.cifar10.load_data()
    labels = tf.one_hot(labels,10)
    labels = tf.cast(tf.squeeze(labels), tf.float32)
    mosaic = keras_cv.layers.preprocessing.Mosaic()
    output = mosaic({'images': images, 'labels': labels})
    # output == {'images': updated_images, 'labels': updated_labels}
    ```
    """  # noqa: E501

    def __init__(
        self, offset=(0.25, 0.75), bounding_box_format=None, seed=None, **kwargs
    ):
        super().__init__(seed=seed, **kwargs)
        self.offset = offset
        self.bounding_box_format = bounding_box_format
        self.center_sampler = preprocessing_utils.parse_factor(
            offset, param_name="offset", seed=seed
        )
        self.seed = seed

    def get_random_transformation_batch(self, batch_size, **kwargs):
        # pick 3 indices for every batch to create the mosaic output with.
        permutation_order = self._random_generator.uniform(
            (batch_size, 3),
            minval=0,
            maxval=batch_size,
            dtype=tf.int32,
        )
        # concatenate the batches with permutation order to get all 4 images of
        # the mosaic
        permutation_order = tf.concat(
            [tf.expand_dims(tf.range(batch_size), axis=-1), permutation_order],
            axis=-1,
        )

        mosaic_centers_x = self.center_sampler(
            shape=(batch_size,), dtype=self.compute_dtype
        )
        mosaic_centers_y = self.center_sampler(
            shape=(batch_size,), dtype=self.compute_dtype
        )
        mosaic_centers = tf.stack((mosaic_centers_x, mosaic_centers_y), axis=-1)

        return {
            "permutation_order": permutation_order,
            "mosaic_centers": mosaic_centers,
        }

    def augment_ragged_image(self, image, transformation, **kwargs):
        raise ValueError(
            "Mosaic received ragged images to `call`. The layer relies on "
            "combining multiple examples with same size, and as such will not "
            "behave as expected. Please call the layer with dense images with "
            "same size. This is an implementation constraint, not an algorithm "
            "constraint. If you find this method helpful, please open an issue "
            "on KerasCV."
        )

    def augment_images(
        self, images, transformations, resize_method="bilinear", **kwargs
    ):
        batch_size = tf.shape(images)[0]
        input_height, input_width, _ = images.shape[1:]

        # forms mosaic for one image from the batch
        permutation_order = transformations["permutation_order"]
        mosaic_images = tf.gather(images, permutation_order)

        tops = tf.concat([mosaic_images[:, 0], mosaic_images[:, 1]], axis=2)
        bottoms = tf.concat([mosaic_images[:, 2], mosaic_images[:, 3]], axis=2)
        outputs = tf.concat([tops, bottoms], axis=1)

        # cropping coordinates for the mosaic
        mosaic_centers = transformations["mosaic_centers"]
        mosaic_centers_x = mosaic_centers[..., 0] * input_width
        mosaic_centers_y = mosaic_centers[..., 1] * input_height
        x1s = (input_width - mosaic_centers_x) / (input_width * 2 - 1)
        y1s = (input_height - mosaic_centers_y) / (input_height * 2 - 1)
        x2s = x1s + (input_width) / (input_width * 2 - 1)
        y2s = y1s + (input_height) / (input_height * 2 - 1)
        cropping_boxes = tf.stack([y1s, x1s, y2s, x2s], axis=-1)

        # helps avoid retracing caused by slicing, inspired by RRC
        # implementation
        # boxes must be type tf.float32
        outputs = tf.image.crop_and_resize(
            outputs,
            tf.cast(cropping_boxes, tf.float32),
            tf.range(batch_size),
            [input_height, input_width],
            method=resize_method,
        )
        # tf.image.crop_and_resize will always output float32, so we need to
        # recast tf.image.crop_and_resize outputs
        # [num_boxes, crop_height, crop_width, depth] since num_boxes is always
        # one we squeeze axis 0
        outputs = tf.cast(outputs, self.compute_dtype)
        return outputs

    def augment_segmentation_masks(
        self, segmentation_masks, transformations, **kwargs
    ):
        return self.augment_images(
            segmentation_masks, transformations, resize_method="nearest"
        )

    def augment_labels(self, labels, transformations, images=None, **kwargs):
        input_height, input_width, _ = images.shape[1:]
        # updates labels for one output mosaic
        permutation_order = transformations["permutation_order"]
        labels_for_mosaic = tf.gather(labels, permutation_order)

        mosaic_centers = transformations["mosaic_centers"]
        center_x = mosaic_centers[..., 0] * input_width
        center_y = mosaic_centers[..., 1] * input_height

        area = input_height * input_width

        # labels are in the same ratio as the area of the images
        top_left_ratio = (center_x * center_y) / area
        top_right_ratio = ((input_width - center_x) * center_y) / area
        bottom_left_ratio = (center_x * (input_height - center_y)) / area
        bottom_right_ratio = (
            (input_width - center_x) * (input_height - center_y)
        ) / area
        labels = (
            labels_for_mosaic[:, 0] * top_left_ratio[:, tf.newaxis]
            + labels_for_mosaic[:, 1] * top_right_ratio[:, tf.newaxis]
            + labels_for_mosaic[:, 2] * bottom_left_ratio[:, tf.newaxis]
            + labels_for_mosaic[:, 3] * bottom_right_ratio[:, tf.newaxis]
        )
        return labels

    def augment_bounding_boxes(
        self, bounding_boxes, transformations, images=None, **kwargs
    ):
        batch_size = tf.shape(images)[0]
        input_height, input_width, _ = images.shape[1:]
        bounding_boxes = bounding_box.to_dense(bounding_boxes)
        bounding_boxes = bounding_box.convert_format(
            bounding_boxes,
            source=self.bounding_box_format,
            target="xyxy",
            images=images,
            dtype=self.compute_dtype,
        )
        boxes, classes = bounding_boxes["boxes"], bounding_boxes["classes"]

        # values to translate the boxes by in the mosaic image
        mosaic_centers = transformations["mosaic_centers"]
        mosaic_centers_x = mosaic_centers[..., 0] * input_width
        mosaic_centers_y = mosaic_centers[..., 1] * input_height
        translate_x = tf.stack(
            [
                mosaic_centers_x - input_width,
                mosaic_centers_x,
                mosaic_centers_x - input_width,
                mosaic_centers_x,
            ],
            axis=-1,
        )
        translate_y = tf.stack(
            [
                mosaic_centers_y - input_height,
                mosaic_centers_y - input_height,
                mosaic_centers_y,
                mosaic_centers_y,
            ],
            axis=-1,
        )
        # updates bounding_boxes for one output mosaic
        permutation_order = transformations["permutation_order"]
        classes_for_mosaic = tf.gather(classes, permutation_order)
        boxes_for_mosaic = tf.gather(boxes, permutation_order)

        # stacking translate values such that the shape is (B, 4, 1, 4) or
        # (batch_size, num_images, broadcast dim, coordinates)
        translate_values = tf.stack(
            [translate_x, translate_y, translate_x, translate_y], axis=-1
        )
        translate_values = tf.expand_dims(translate_values, axis=2)
        # translating boxes
        boxes_for_mosaic = boxes_for_mosaic + translate_values
        boxes_for_mosaic = tf.reshape(boxes_for_mosaic, [batch_size, -1, 4])
        classes_for_mosaic = tf.reshape(classes_for_mosaic, [batch_size, -1])
        boxes_for_mosaic = {
            "boxes": boxes_for_mosaic,
            "classes": classes_for_mosaic,
        }
        boxes_for_mosaic = bounding_box.clip_to_image(
            boxes_for_mosaic,
            bounding_box_format="xyxy",
            images=images,
        )
        boxes_for_mosaic = bounding_box.convert_format(
            boxes_for_mosaic,
            source="xyxy",
            target=self.bounding_box_format,
            images=images,
            dtype=self.compute_dtype,
        )
        return boxes_for_mosaic

    def _batch_augment(self, inputs):
        self._validate_inputs(inputs)
        return super()._batch_augment(inputs)

    def call(self, inputs):
        _, metadata = self._format_inputs(inputs)
        if metadata[BATCHED] is not True:
            raise ValueError(
                "Mosaic received a single image to `call`. The "
                "layer relies on combining multiple examples, and as such "
                "will not behave as expected. Please call the layer with 4 "
                "or more samples."
            )
        return super().call(inputs=inputs)

    def _validate_inputs(self, inputs):
        images = inputs.get(IMAGES, None)
        labels = inputs.get(LABELS, None)
        bounding_boxes = inputs.get(BOUNDING_BOXES, None)
        segmentation_masks = inputs.get(SEGMENTATION_MASKS, None)
        if images is None or (
            labels is None
            and bounding_boxes is None
            and segmentation_masks is None
        ):
            raise ValueError(
                "Mosaic expects inputs in a dictionary with format "
                '{"images": images, "labels": labels} or'
                '{"images": images, "bounding_boxes": bounding_boxes} or'
                '{"images": images, "segmentation_masks": masks}.'
                f"Got: inputs = {inputs}"
            )
        if labels is not None and not labels.dtype.is_floating:
            raise ValueError(
                f"Mosaic received labels with type {labels.dtype}. "
                "Labels must be of type float."
            )
        if bounding_boxes is not None and self.bounding_box_format is None:
            raise ValueError(
                "Mosaic received bounding boxes but no bounding_box_format. "
                "Please pass a bounding_box_format from the supported list."
            )

    def get_config(self):
        config = {
            "offset": self.offset,
            "bounding_box_format": self.bounding_box_format,
            "seed": self.seed,
        }
        base_config = super().get_config()

        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)
