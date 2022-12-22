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

from keras_cv import bounding_box
from keras_cv.layers.preprocessing.base_image_augmentation_layer import (
    BaseImageAugmentationLayer,
)
from keras_cv.utils import preprocessing


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class Mosaic(BaseImageAugmentationLayer):
    """Mosaic implements the mosaic data augmentation technique.

    Mosaic data augmentation first takes 4 images from the batch and makes a grid.
    After that based on the offset, a crop is taken to form the mosaic image. Labels
    are in the same ratio as the the area of their images in the output image. Bounding
    boxes are translated according to the position of the 4 images.

    Args:
        offset: A tuple of two floats, a single float or `keras_cv.FactorSampler`.
            `offset` is used to determine the offset of the mosaic center from the
            top-left corner of the mosaic. If a tuple is used, the x and y coordinates
            of the mosaic center are sampled between the two values for every image
            augmented. If a single float is used, a value between `0.0` and the passed
            float is sampled.  In order to ensure the value is always the same, please
            pass a tuple with two identical floats: `(0.5, 0.5)`. Defaults to
            (0.25, 0.75).
        bounding_box_format: a case-insensitive string (for example, "xyxy") to be
            passed if bounding boxes are being augmented by this layer.
            Each bounding box is defined by at least these 4 values. The inputs
            may contain additional information such as classes and confidence after
            these 4 values but these values will be ignored and returned as is. For
            detailed information on the supported formats, see the
            [KerasCV bounding box documentation](https://keras.io/api/keras_cv/bounding_box/formats/).
            Defualts to None.
        seed: Integer. Used to create a random seed.

    References:
        - [Yolov4 paper](https://arxiv.org/pdf/2004.10934).
        - [Yolov5 implementation](https://github.com/ultralytics/yolov5).
        - [YoloX implementation](https://github.com/Megvii-BaseDetection/YOLOX)

    Sample usage:
    ```python
    (images, labels), _ = tf.keras.datasets.cifar10.load_data()
    labels = tf.one_hot(labels,10)
    labels = tf.cast(tf.squeeze(labels), tf.float32)
    mosaic = keras_cv.layers.preprocessing.Mosaic()
    output = mosaic({'images': images, 'labels': labels})
    # output == {'images': updated_images, 'labels': updated_labels}
    ```
    """

    def __init__(
        self, offset=(0.25, 0.75), bounding_box_format=None, seed=None, **kwargs
    ):
        super().__init__(seed=seed, **kwargs)
        self.offset = offset
        self.bounding_box_format = bounding_box_format
        self.center_sampler = preprocessing.parse_factor(
            offset, param_name="offset", seed=seed
        )
        self.seed = seed

    def _batch_augment(self, inputs):
        self._validate_inputs(inputs)
        images = inputs.get("images", None)
        labels = inputs.get("labels", None)
        bounding_boxes = inputs.get("bounding_boxes", None)

        batch_size = tf.shape(images)[0]
        # pick 3 indices for every batch to create the mosaic output with.
        permutation_order = tf.random.uniform(
            (batch_size, 3),
            minval=0,
            maxval=batch_size,
            dtype=tf.int32,
            seed=self._random_generator.make_legacy_seed(),
        )
        # concatenate the batches with permutation order to get all 4 images of the mosaic
        permutation_order = tf.concat(
            [tf.expand_dims(tf.range(batch_size), axis=-1), permutation_order], axis=-1
        )

        input_height, input_width, _ = images.shape[1:]

        mosaic_centers_x = (
            self.center_sampler(
                tf.expand_dims(batch_size, axis=0), dtype=self.compute_dtype
            )
            * input_width
        )
        mosaic_centers_y = (
            self.center_sampler(
                shape=tf.expand_dims(batch_size, axis=0), dtype=self.compute_dtype
            )
            * input_height
        )
        mosaic_centers = tf.stack((mosaic_centers_x, mosaic_centers_y), axis=-1)

        # return the mosaics
        images = tf.vectorized_map(
            lambda index: self._update_image(
                images, permutation_order, mosaic_centers, index
            ),
            tf.range(batch_size),
        )

        if labels is not None:
            labels = tf.vectorized_map(
                lambda index: self._update_label(
                    images, labels, permutation_order, mosaic_centers, index
                ),
                tf.range(batch_size),
            )
            inputs["labels"] = labels

        if bounding_boxes is not None:
            # values to translate the boxes by in the mosaic image
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

            if isinstance(bounding_boxes, tf.RaggedTensor):
                bounding_boxes = bounding_boxes.to_tensor(-1)

            bounding_boxes = tf.vectorized_map(
                lambda index: self._update_bounding_box(
                    images,
                    bounding_boxes,
                    permutation_order,
                    translate_x,
                    translate_y,
                    index,
                ),
                tf.range(batch_size),
            )
            bounding_boxes = bounding_box.filter_sentinels(bounding_boxes)
            inputs["bounding_boxes"] = bounding_boxes
        inputs["images"] = images
        return inputs

    def _augment(self, inputs):
        raise ValueError(
            "Mosaic received a single image to `call`.  The layer relies on "
            "combining multiple examples, and as such will not behave as "
            "expected.  Please call the layer with 4 or more samples."
        )

    def _update_image(self, images, permutation_order, mosaic_centers, index):
        # forms mosaic for one image from the batch
        input_height, input_width, _ = images.shape[1:]
        mosaic_images = tf.gather(images, permutation_order[index])

        top = tf.concat([mosaic_images[0], mosaic_images[1]], axis=1)
        bottom = tf.concat([mosaic_images[2], mosaic_images[3]], axis=1)
        output = tf.concat([top, bottom], axis=0)

        # cropping coordinates for the mosaic
        x1 = (input_width - mosaic_centers[index][0]) / (input_width * 2 - 1)
        y1 = (input_height - mosaic_centers[index][1]) / (input_height * 2 - 1)
        x2 = x1 + (input_width) / (input_width * 2 - 1)
        y2 = y1 + (input_height) / (input_height * 2 - 1)

        # helps avoid retracing caused by slicing, inspired by RRC implementation
        output = tf.image.crop_and_resize(
            tf.expand_dims(output, axis=0),
            [[y1, x1, y2, x2]],
            [0],
            [input_height, input_width],
        )
        # tf.image.crop_and_resize will always output float32, so we need to recast
        output = tf.cast(output, self.compute_dtype)
        return tf.squeeze(output)

    def _update_label(self, images, labels, permutation_order, mosaic_centers, index):
        # updates labels for one output mosaic
        input_height, input_width, _ = images.shape[1:]
        labels_for_mosaic = tf.gather(labels, permutation_order[index])
        center_x = mosaic_centers[index][0]
        center_y = mosaic_centers[index][1]

        area = input_height * input_width

        # labels are in the same ratio as the area of the images
        top_left_ratio = (center_x * center_y) / area
        top_right_ratio = ((input_width - center_x) * center_y) / area
        bottom_left_ratio = (center_x * (input_height - center_y)) / area
        bottom_right_ratio = (
            (input_width - center_x) * (input_height - center_y)
        ) / area
        label = (
            labels_for_mosaic[0] * top_left_ratio
            + labels_for_mosaic[1] * top_right_ratio
            + labels_for_mosaic[2] * bottom_left_ratio
            + labels_for_mosaic[3] * bottom_right_ratio
        )
        return label

    def _update_bounding_box(
        self, images, bounding_boxes, permutation_order, translate_x, translate_y, index
    ):
        # updates bboxes for one output mosaic
        bounding_boxes = bounding_box.convert_format(
            bounding_boxes,
            source=self.bounding_box_format,
            target="xyxy",
            images=images,
            dtype=self.compute_dtype,
        )
        boxes_for_mosaic = tf.gather(bounding_boxes, permutation_order[index])
        if isinstance(boxes_for_mosaic, tf.RaggedTensor):
            boxes_for_mosaic = boxes_for_mosaic.to_tensor(-1, shape=[None, 5])
        boxes_for_mosaic, rest = tf.split(
            boxes_for_mosaic, [4, boxes_for_mosaic.shape[-1] - 4], axis=-1
        )

        # stacking translate values such that the shape is (4, 1, 4) or (num_images, broadcast dim, coordinates)
        translate_values = tf.stack(
            [
                translate_x[index],
                translate_y[index],
                translate_x[index],
                translate_y[index],
            ],
            axis=-1,
        )
        translate_values = tf.expand_dims(translate_values, axis=1)

        # translating boxes
        boxes_for_mosaic = boxes_for_mosaic + translate_values
        boxes_for_mosaic = tf.concat([boxes_for_mosaic, rest], axis=-1)
        boxes_for_mosaic = tf.reshape(boxes_for_mosaic, [-1, bounding_boxes.shape[-1]])

        boxes_for_mosaic = bounding_box.clip_to_image(
            boxes_for_mosaic,
            bounding_box_format="xyxy",
            images=images[index],
        )
        boxes_for_mosaic = bounding_box.filter_sentinels(boxes_for_mosaic)
        boxes_for_mosaic = bounding_box.convert_format(
            boxes_for_mosaic,
            source="xyxy",
            target=self.bounding_box_format,
            images=images[index],
            dtype=self.compute_dtype,
        )
        if isinstance(boxes_for_mosaic, tf.RaggedTensor):
            boxes_for_mosaic = boxes_for_mosaic.to_tensor(-1, shape=[None, 5])
        return boxes_for_mosaic

    def _validate_inputs(self, inputs):
        images = inputs.get("images", None)
        labels = inputs.get("labels", None)
        bounding_boxes = inputs.get("bounding_boxes", None)
        if images is None or (labels is None and bounding_boxes is None):
            raise ValueError(
                "Mosaic expects inputs in a dictionary with format "
                '{"images": images, "labels": labels}. or'
                '{"images": images, "bounding_boxes": bounding_boxes}'
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
